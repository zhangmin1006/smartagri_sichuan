"""Pre-warm the persistent contract cache.

The first run a user launches would otherwise pay the full cold-solve cost
(tens of seconds). Sweeping seeds and scenarios here populates the store with
the contract problems that realistic populations actually generate, so the
shipped app answers the first request in about a second.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

import contract_cache
cache = contract_cache.install()

from smartagri.model import SmartAgriModel

SCEN = ["baseline", "voucher_plus_capacity", "integrated", "subsidy",
        "insurance", "training_maintenance", "warning_response",
        "mountain_equity", "voucher"]

t0 = time.time()
n = 0
for seed in range(1, 26):
    for sc in SCEN:
        m = SmartAgriModel(scenario=sc, seed=seed, n_farmers=260, seasons=6)
        m.run()
        n += 1
        st = cache.stats()
        print(f"[{time.time()-t0:7.1f}s] seed={seed:3d} {sc:22s} "
              f"entries={st['entries']:6d} hits={st['hits']:7d} "
              f"misses={st['misses']:6d}", flush=True)
    cache.save()
cache.save()
print("DONE", cache.stats(), flush=True)
