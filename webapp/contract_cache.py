"""
contract_cache.py
=================
A persistent, process-wide cache for the Guo-Parlar-Zhang contract solution.

Why this exists
---------------
Profiling the model shows that essentially ALL of its runtime is one function:
``contract.solve_effort_cached``. A cold 200-farmer, 4-season run spends ~19 of
its ~20 seconds solving principal-agent problems; the rest of the ABM is
noise by comparison.

The solver is already memoised with ``lru_cache``, and agents quantise
(c1, c2, gamma, u_min) before calling, so one run collapses onto a few hundred
distinct problems. But an ``lru_cache`` dies with the process and does not
survive a server restart -- and, worse, a NEW SEED draws new risk-aversion
values that quantise onto new grid points, so the cache does not transfer
between runs either. Measured: baseline seed 1 costs 66 s cold and 0.5 s warm,
but seed 7 immediately costs 66 s again.

For an interactive app that is fatal. This module replaces the in-process
memo with a dict that is persisted to disk, so the cache accumulates across
every run, every scenario, every seed and every server restart. The solved
contract for a given (c1, c2, gamma, u_min) pair is a pure mathematical fact
-- it does not depend on the scenario, the policy or the weather -- so a hit
computed for one user's scenario is valid for every other's.

The store is keyed on the full argument tuple including `n_grid` and `n_quad`,
so a change to solver resolution can never silently return a coarser answer.
"""

from __future__ import annotations

import atexit
import pickle
import threading
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_FILE = CACHE_DIR / "contract_solutions.pkl"

# Persist every N new solutions. Solving is ~0.1 s each and writing the store
# is milliseconds, so a small interval costs nothing and means a crashed or
# killed server loses almost no accumulated work.
_FLUSH_EVERY = 50


class PersistentContractCache:
    """Disk-backed memo wrapping the original solver."""

    def __init__(self, solver, path: Path = CACHE_FILE) -> None:
        self._solver = solver
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: dict = {}
        self._unsaved = 0
        self.hits = 0
        self.misses = 0
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("rb") as fh:
                loaded = pickle.load(fh)
            if isinstance(loaded, dict):
                self._data = loaded
        except Exception:
            # A corrupt or half-written store must never stop the app: the
            # cache is an optimisation, and the solver can always recompute.
            self._data = {}

    def save(self) -> None:
        with self._lock:
            if not self._unsaved:
                return
            data = dict(self._data)
            self._unsaved = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        # Write-then-rename so a crash mid-write cannot corrupt the store that
        # every future run depends on.
        with tmp.open("wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._path)

    # ------------------------------------------------------------------
    def __call__(self, c1, c2, gamma, u_min, scale=1.0, e_max=1.0,
                 n_grid=13, n_quad=121):
        key = (round(float(c1), 6), round(float(c2), 6), round(float(gamma), 6),
               round(float(u_min), 6), round(float(scale), 6),
               round(float(e_max), 6), int(n_grid), int(n_quad))
        with self._lock:
            hit = self._data.get(key)
        if hit is not None:
            self.hits += 1
            return hit

        # Solved OUTSIDE the lock: solving takes ~0.1 s and holding the lock
        # would serialise every concurrent request behind one solve.  A benign
        # race just solves the same problem twice and stores the same answer.
        value = self._solver(c1, c2, gamma, u_min, scale, e_max, n_grid, n_quad)

        with self._lock:
            self._data[key] = value
            self._unsaved += 1
            due = self._unsaved >= _FLUSH_EVERY
        self.misses += 1
        if due:
            self.save()
        return value

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._data), "hits": self.hits,
                    "misses": self.misses}


_installed: PersistentContractCache | None = None


def install() -> PersistentContractCache:
    """Swap the persistent cache in for the lru_cache, in every module.

    ``agents.py`` imported ``solve_effort_cached`` by name, so rebinding it on
    ``contract`` alone would leave the agents calling the original. Both
    references are replaced.
    """
    global _installed
    if _installed is not None:
        return _installed

    from smartagri import agents as _agents
    from smartagri import contract as _contract

    raw = _contract.solve_effort_cached
    # Unwrap the lru_cache so we do not keep a second full copy in memory.
    inner = getattr(raw, "__wrapped__", raw)

    cache = PersistentContractCache(inner)
    _contract.solve_effort_cached = cache
    _agents.solve_effort_cached = cache
    atexit.register(cache.save)
    _installed = cache
    return cache
