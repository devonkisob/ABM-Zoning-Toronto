"""
simulation.py
-------------
Monte Carlo simulation orchestrator for the Toronto Missing Middle Zoning ABM.

Runs N realisations of each scenario over T time steps (years), recording
affordability indices and housing supply metrics at each step.

Usage (from notebook or script):
    from src.simulation import run_scenario, run_all_scenarios

    # Run a single scenario
    results = run_scenario("S0", N=100, T=10)

    # Run all four scenarios
    all_results = run_all_scenarios(N=100, T=10)

Output format:
    Dict with keys: "ai_own", "ai_rent", "units_total", "delta_units_added"
    Each value is a numpy array of shape (N, T) — one row per realisation,
    one column per time step.
"""

import numpy as np
import copy
from typing import Dict

from src.agents import (
    CensusTractAgent,
    PolicyModel,
    DemandAllocationModel,
    InfrastructureModel,
    load_agents,
    get_transit_ctuids,
)
from src.calibration import load_models, predict_development

# ── Global simulation parameters ───────────────────────────────────────────────
from src.config import DEFAULT_CONFIG as cfg

T_DEFAULT = cfg.T
N_DEFAULT = cfg.N

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
SCENARIOS = ["S0", "S1", "S2", "S3"]

def run_scenario(
    scenario: str,
    N: int = N_DEFAULT,
    T: int = T_DEFAULT,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run N Monte Carlo realisations of a single scenario over T years.

    Args:
        scenario — one of "S0", "S1", "S2", "S3"
        N        — number of realisations
        T        — number of time steps (years)
        seed     — base random seed (each realisation uses seed + i)
        verbose  — print progress

    Returns:
        dict with arrays of shape (N, T):
            "ai_own"          — mean ownership affordability index across CTs
            "ai_rent"         — mean rental affordability index across CTs
            "units_total"     — total housing units across all CTs
            "units_added"     — new units added this time step
            "mean_home_price" — mean home price across CTs
            "mean_rent"       — mean annual rent across CTs
            "mean_strain"     — mean infrastructure strain across CTs
    """
    if verbose:
        print(f"\nRunning scenario {scenario} | N={N} realisations | T={T} years")

    # Load static data once (shared across all realisations)
    base_agents    = load_agents()
    transit_ctuids = get_transit_ctuids()
    all_ctuids     = [a.ctuid for a in base_agents]
    stage1, stage2, scaler, features = load_models()

    # Build policy model for this scenario (same across all realisations)
    policy = PolicyModel.from_scenario(scenario, all_ctuids, transit_ctuids)

    if verbose:
        print(f"  Policy: {scenario} | "
              f"Eligible CTs: {len(policy.eligible_ctuids)} | "
              f"Incentive level: {policy.incentive_level}")

    # Output arrays — shape (N, T)
    out_ai_own    = np.zeros((N, T))
    out_ai_rent   = np.zeros((N, T))
    out_units     = np.zeros((N, T))
    out_added     = np.zeros((N, T))
    out_price     = np.zeros((N, T))
    out_rent      = np.zeros((N, T))
    out_strain    = np.zeros((N, T))

    # ── Monte Carlo loop ────────────────────────────────────────────────────
    for i in range(N):
        if verbose and (i % 20 == 0):
            print(f"  Realisation {i+1}/{N}...")

        # Each realisation gets its own RNG (reproducible but independent)
        rng = np.random.default_rng(seed + i)

        # Deep copy agents so each realisation starts from 2021 baseline
        cts = copy.deepcopy(base_agents)

        # Initialise pseudo-agents
        demand_model = DemandAllocationModel(
            base_demand=cfg.base_demand,
            demand_growth=cfg.demand_growth,
            rng=rng,
        )
        infra_model = InfrastructureModel(
            omega0=cfg.omega0,
            omega1=cfg.omega1,
            g_base=cfg.g_base,
            lambda_incent=cfg.lambda_incent,
        )

        # ── Time step loop ──────────────────────────────────────────────────
        for t in range(T):

            # 1. Demand allocation
            demand_model.allocate(cts, t)

            # 2. Development model — predict and apply new units per CT
            units_added_this_step = 0
            for ct in cts:
                units_added = predict_development(
                    ct=ct,
                    policy=policy,
                    stage1=stage1,
                    stage2=stage2,
                    scaler=scaler,
                    features=features,
                    rng=rng,
                )
                if units_added > 0:
                    ct.apply_development(units_added)
                    units_added_this_step += units_added

            # 3. Market update — prices, rents, vacancy
            for ct in cts:
                ct.update_market(cfg.price_kappa, cfg.rent_kappa, cfg.v_star, cfg.vacancy_eq)


            # 4. Infrastructure update
            for ct in cts:
                infra_model.step(ct, policy)

            # 5. Record outputs
            ai_own_vals  = [ct.affordability_own()  for ct in cts]
            ai_rent_vals = [ct.affordability_rent()  for ct in cts]
            prices       = [ct.home_price            for ct in cts]
            rents        = [ct.annual_rent           for ct in cts]
            strains      = [ct.strain                for ct in cts]

            out_ai_own[i, t]  = np.mean(ai_own_vals)
            out_ai_rent[i, t] = np.mean(ai_rent_vals)
            out_units[i, t]   = sum(ct.units_total for ct in cts)
            out_added[i, t]   = units_added_this_step
            out_price[i, t]   = np.mean(prices)
            out_rent[i, t]    = np.mean(rents)
            out_strain[i, t]  = np.mean(strains)

    if verbose:
        print(f"  Done {scenario}. Mean final AI_own:  {out_ai_own[:, -1].mean():.4f}")
        print(f"         Mean final AI_rent: {out_ai_rent[:, -1].mean():.4f}")
        print(f"         Mean units added/yr: {out_added.mean():.0f}")

    return {
        "ai_own":          out_ai_own,
        "ai_rent":         out_ai_rent,
        "units_total":     out_units,
        "units_added":     out_added,
        "mean_home_price": out_price,
        "mean_rent":       out_rent,
        "mean_strain":     out_strain,
    }


def run_all_scenarios(
    N: int = N_DEFAULT,
    T: int = T_DEFAULT,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run all four scenarios and return results in a nested dict.

    Returns:
        {scenario: {metric: array(N, T)}}
        e.g. results["S1"]["ai_own"] → array of shape (N, T)
    """
    results = {}
    for scenario in SCENARIOS:
        results[scenario] = run_scenario(
            scenario=scenario,
            N=N,
            T=T,
            seed=seed,
            verbose=verbose,
        )
    return results


def summarise(results: Dict[str, np.ndarray], metric: str = "ai_own"):
    """
    Compute summary statistics for a metric across realisations.

    Returns dict with keys: mean, median, p25, p75, p5, p95
    Each value is a 1D array of length T.
    """
    arr = results[metric]
    return {
        "mean":   arr.mean(axis=0),
        "median": np.median(arr, axis=0),
        "p25":    np.percentile(arr, 25, axis=0),
        "p75":    np.percentile(arr, 75, axis=0),
        "p5":     np.percentile(arr, 5,  axis=0),
        "p95":    np.percentile(arr, 95, axis=0),
    }