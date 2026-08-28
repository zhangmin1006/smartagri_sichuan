import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smartagri.agents import STATE_EFFECTIVE
from smartagri.contract import (ContractProblem, LinearTiltDensity,
                                normalised_contract_payment)
from smartagri.model import SmartAgriModel
from smartagri.monte_carlo import paired_differences
from smartagri.shocks import annual_to_period_probability


def test_annual_probability_is_not_applied_twice():
    annual = 0.35
    seasonal = annual_to_period_probability(annual, 2)
    assert np.isclose(1.0 - (1.0 - seasonal) ** 2, annual)
    assert seasonal < annual


def test_disadopted_or_none_mode_is_not_effective_use():
    model = SmartAgriModel("baseline", seed=11, n_farmers=12, seasons=1)
    farmer = model.farmers[0]
    for tech in farmer.mode:
        farmer.mode[tech] = "none"
        farmer.state[tech] = STATE_EFFECTIVE
    assert model._effective_use_rate([farmer]) == 0.0


def test_provider_capacities_reconcile_and_assignment_load_balances():
    model = SmartAgriModel("baseline", seed=12, n_farmers=60, seasons=1)
    assert np.isclose(sum(p.season_capacity_mu for p in model.providers),
                      model.sd.season_capacity_mu)
    county = next(c for c in model.counties
                  if sum(p.county == c for p in model.providers) >= 2)
    farmers = [f for f in model.farmers if f.county == county][:8]
    assigned = {p.uid: 0.0 for p in model.providers}
    provider_ids = []
    for farmer in farmers:
        provider = model._assign_provider(farmer, "T3", assigned)
        provider_ids.append(provider.uid)
        assigned[provider.uid] += farmer.area_mu
    assert len(set(provider_ids)) >= 2


def test_contract_anchor_changes_realised_payment_but_preserves_scale():
    problem = ContractProblem(0.55, 0.30, LinearTiltDensity(),
                              gamma=0.12, u_min=0.28)
    solution = problem.solve()
    low = normalised_contract_payment(
        0.35, solution.b_star, solution.principal_expected_payment, 0.55, 0.30)
    high = normalised_contract_payment(
        0.90, solution.b_star, solution.principal_expected_payment, 0.55, 0.30)
    assert 0.25 <= low <= 2.50
    assert 0.25 <= high <= 2.50
    assert high > low


def test_insurance_claim_is_paid_in_a_later_season():
    model = SmartAgriModel("insurance", seed=13, n_farmers=20, seasons=2)
    model.cfg["params"]["production"]["yield_cv"] = 0.0
    d2 = model.disrupt_spec["D2"]["model"]
    d2["severity_distribution"] = "beta"
    d2["severity_params"] = [100.0, 1.0]
    d2["yield_damage_max"] = 1.0
    model.shockgen.specs["D2"]["model"].update(d2)
    model.step(1, force_shocks=["D2"])
    pending = sum(amount for f in model.farmers
                  for _due, amount in f.pending_indemnities)
    assert pending > 0.0
    assert sum(f.last_insurance_receipt for f in model.farmers) == 0.0
    model.step(2, force_shocks=[])
    receipts = sum(f.last_insurance_receipt for f in model.farmers)
    assert np.isclose(receipts, pending)


def test_paired_monte_carlo_summary_uses_within_seed_differences():
    raw = pd.DataFrame([
        {"rep": 0, "scenario": "baseline", "mitigation_rate": 0.10},
        {"rep": 0, "scenario": "policy", "mitigation_rate": 0.13},
        {"rep": 1, "scenario": "baseline", "mitigation_rate": 0.20},
        {"rep": 1, "scenario": "policy", "mitigation_rate": 0.23},
    ])
    for metric in ("effective_use_rate", "mean_wait_days", "income_p10",
                   "equity_gap", "mountain_gap", "fiscal_cumulative"):
        raw[metric] = 1.0
    summary = paired_differences(raw)
    row = summary[(summary.scenario == "policy")
                  & (summary.metric == "mitigation_rate")].iloc[0]
    assert np.isclose(row.mean_difference, 0.03)
    assert np.isclose(row.mcse_difference, 0.0)
    assert row.probability_better == 1.0


def test_common_seed_keeps_shock_path_identical_across_policies():
    paths = []
    for scenario in ("baseline", "voucher", "voucher_plus_capacity"):
        model = SmartAgriModel(scenario, seed=991, n_farmers=20, seasons=3)
        model.run()
        paths.append([record.shocks for record in model.records])
    assert paths[0] == paths[1] == paths[2]
