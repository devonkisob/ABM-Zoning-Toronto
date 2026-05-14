"""
scripts/make_spatial_heatmap.py
--------------------------------
Runs a single deterministic realisation of S0 and S1, records
per-CT final affordability, and produces a choropleth of the change.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import copy

from src.agents import load_agents, get_transit_ctuids, PolicyModel, DemandAllocationModel, InfrastructureModel
from src.calibration import load_models, predict_development
from src.config import DEFAULT_CONFIG as cfg
from src.visualization import plot_spatial_heatmap
from src.paths import FIGURES_DIR, AGENTS_CSV

SEED = 42
T    = 40

def run_single(scenario: str, base_agents, stage1, stage2, scaler, features) -> pd.DataFrame:
    """Run one deterministic realisation, return per-CT final AI_own."""
    rng            = np.random.default_rng(SEED)
    cts            = copy.deepcopy(base_agents)
    transit        = get_transit_ctuids()
    all_ctuids     = [a.ctuid for a in cts]
    policy         = PolicyModel.from_scenario(scenario, all_ctuids, transit)
    demand_model   = DemandAllocationModel(cfg.base_demand, cfg.demand_growth, rng)
    infra_model    = InfrastructureModel(cfg.omega0, cfg.omega1, cfg.g_base, cfg.lambda_incent)

    for t in range(T):
        demand_model.allocate(cts, t)
        for ct in cts:
            units = predict_development(ct, policy, stage1, stage2, scaler, features, rng)
            if units > 0:
                ct.apply_development(units)
        for ct in cts:
            ct.update_market(cfg.price_kappa, cfg.rent_kappa, cfg.v_star, cfg.vacancy_eq)
        for ct in cts:
            infra_model.step(ct, policy)

    return pd.DataFrame({
        "ctuid":    [ct.ctuid for ct in cts],
        "ai_own":   [ct.affordability_own() for ct in cts],
        "ai_rent":  [ct.affordability_rent() for ct in cts],
        "home_price": [ct.home_price for ct in cts],
        "units_total": [ct.units_total for ct in cts],
    })


if __name__ == "__main__":
    print("Loading agents and models...")
    base_agents = load_agents()
    stage1, stage2, scaler, features = load_models()

    print("Running S0...")
    df_s0 = run_single("S0", base_agents, stage1, stage2, scaler, features)

    print("Running S1...")
    df_s1 = run_single("S1", base_agents, stage1, stage2, scaler, features)

    print("Running S2...")
    df_s2 = run_single("S2", base_agents, stage1, stage2, scaler, features)

    # Compute per-CT change
    merged = df_s0[["ctuid", "ai_own"]].rename(columns={"ai_own": "ai_own_s0"})
    merged = merged.merge(df_s1[["ctuid", "ai_own"]].rename(columns={"ai_own": "ai_own_s1"}), on="ctuid")
    merged = merged.merge(df_s2[["ctuid", "ai_own"]].rename(columns={"ai_own": "ai_own_s2"}), on="ctuid")

    merged["ai_own_change_s1"] = merged["ai_own_s1"] - merged["ai_own_s0"]
    merged["ai_own_change_s2"] = merged["ai_own_s2"] - merged["ai_own_s0"]

    # Save CT-level results
    merged.to_csv("data/processed/ct_level_ai_change.csv", index=False)
    print(f"Saved CT-level results → data/processed/ct_level_ai_change.csv")

    # Plot S1 vs S0
    fig1 = plot_spatial_heatmap(
        ct_ai_change=merged.rename(columns={"ai_own_change_s1": "ai_own_change"}),
        value_col="ai_own_change",
        title="Change in Ownership Affordability Index: S1 vs S0 (Year 10)",
        save_path=FIGURES_DIR / "heatmap_s1_vs_s0.png",
    )

    # Plot S2 vs S0
    fig2 = plot_spatial_heatmap(
        ct_ai_change=merged.rename(columns={"ai_own_change_s2": "ai_own_change"}),
        value_col="ai_own_change",
        title="Change in Ownership Affordability Index: S2 vs S0 (Year 10)",
        save_path=FIGURES_DIR / "heatmap_s2_vs_s0.png",
    )

    print("Done.")