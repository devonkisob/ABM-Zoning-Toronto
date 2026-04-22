"""
sensitivity_analysis.py
------------------------
One-factor-at-a-time (OFAT) sensitivity analysis for the Toronto Missing
Middle Zoning ABM. Produces a tornado chart showing the influence of
each parameter on the primary output metric (AI_own at year 10).

Parameters analysed (5 minimum required, we test 7):
    1. price_kappa       — price elasticity w.r.t. vacancy gap
    2. rent_kappa        — rent elasticity w.r.t. vacancy gap
    3. omega0            — baseline infrastructure load
    4. omega1            — infrastructure load sensitivity to density
    5. base_demand       — city-wide housing demand (fraction of stock)
    6. demand_growth     — quarterly demand growth rate
    7. vacancy_eq        — steady-state vacancy in baseline

Method: For each parameter, run the simulation at baseline, +20%, and -20%
of the parameter value. Record the change in mean AI_own at year 10 across
all scenarios. The tornado chart ranks parameters by their total influence.

Usage:
    python scripts/sensitivity_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from dataclasses import replace

from src.agents import (
    load_agents, get_transit_ctuids, PolicyModel,
    DemandAllocationModel, InfrastructureModel,
)
from src.calibration import load_models, predict_development
from src.config import DEFAULT_CONFIG as cfg, SimConfig
from src.paths import FIGURES_DIR, PROCESSED_DIR

# ── Sensitivity analysis settings ─────────────────────────────────────────────
N_SA    = 20    # realisations per run (small for speed; increase for Final Report)
T_SA    = 40    # full 10-year horizon
DELTA   = 0.20  # ±20% perturbation
SEED    = 42
SCENARIO = "S1"  # run sensitivity on baseline (cleanest signal)

# Parameters to test: (name, config_attr, display_label)
PARAMETERS = [
    ("price_kappa",   "price_kappa",   "Price elasticity (κ_p)"),
    ("rent_kappa",    "rent_kappa",    "Rent elasticity (κ_r)"),
    ("omega0",        "omega0",        "Infra baseline load (ω₀)"),
    ("omega1",        "omega1",        "Infra density sensitivity (ω₁)"),
    ("base_demand",   "base_demand",   "Base demand (fraction of stock)"),
    ("demand_growth", "demand_growth", "Demand growth rate"),
    ("vacancy_eq",    "vacancy_eq",    "Steady-state vacancy (v_eq)"),
]


# ── Simulation runner ──────────────────────────────────────────────────────────

def run_with_config(config: SimConfig, base_agents, stage1, stage2,
                    scaler, features, scenario: str = SCENARIO) -> float:
    """
    Run N_SA realisations of a scenario with the given config.
    Returns mean AI_own at final time step.
    """
    transit_ctuids = get_transit_ctuids()
    all_ctuids     = [a.ctuid for a in base_agents]
    policy         = PolicyModel.from_scenario(scenario, all_ctuids, transit_ctuids)

    ai_finals = []

    for i in range(N_SA):
        rng  = np.random.default_rng(SEED + i)
        cts  = copy.deepcopy(base_agents)

        demand_model = DemandAllocationModel(
            base_demand=config.base_demand,
            demand_growth=config.demand_growth,
            rng=rng,
        )
        infra_model = InfrastructureModel(
            omega0=config.omega0,
            omega1=config.omega1,
            g_base=config.g_base,
            lambda_incent=config.lambda_incent,
        )

        for t in range(T_SA):
            demand_model.allocate(cts, t)
            for ct in cts:
                units = predict_development(ct, policy, stage1, stage2, scaler, features, rng)
                if units > 0:
                    ct.apply_development(units)
            for ct in cts:
                ct.update_market(
                    config.price_kappa, config.rent_kappa,
                    config.v_star, config.vacancy_eq,
                )
            for ct in cts:
                infra_model.step(ct, policy)

        ai_finals.append(np.mean([ct.affordability_own() for ct in cts]))

    return float(np.mean(ai_finals))


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading agents and models...")
    base_agents = load_agents()
    stage1, stage2, scaler, features = load_models()

    print(f"\nRunning sensitivity analysis: {len(PARAMETERS)} parameters, "
          f"±{DELTA*100:.0f}% perturbation, N={N_SA}, T={T_SA}")
    print(f"Scenario: {SCENARIO}\n")

    # Baseline run
    print("Running baseline...")
    baseline_ai = run_with_config(cfg, base_agents, stage1, stage2, scaler, features)
    print(f"  Baseline AI_own (year 10): {baseline_ai:.6f}")

    # Parameter sweep
    results = []
    for param_name, config_attr, label in PARAMETERS:
        base_val = getattr(cfg, config_attr)
        val_low  = base_val * (1 - DELTA)
        val_high = base_val * (1 + DELTA)

        print(f"\n  {label}:")
        print(f"    base={base_val:.5f}  low={val_low:.5f}  high={val_high:.5f}")

        cfg_low  = replace(cfg, **{config_attr: val_low})
        cfg_high = replace(cfg, **{config_attr: val_high})

        ai_low  = run_with_config(cfg_low,  base_agents, stage1, stage2, scaler, features)
        ai_high = run_with_config(cfg_high, base_agents, stage1, stage2, scaler, features)

        delta_low  = ai_low  - baseline_ai
        delta_high = ai_high - baseline_ai
        total_swing = abs(delta_high - delta_low)

        print(f"    AI low={ai_low:.6f} (Δ={delta_low:+.6f})")
        print(f"    AI high={ai_high:.6f} (Δ={delta_high:+.6f})")
        print(f"    Total swing: {total_swing:.6f}")

        results.append({
            "param":        param_name,
            "label":        label,
            "base_val":     base_val,
            "val_low":      val_low,
            "val_high":     val_high,
            "ai_baseline":  baseline_ai,
            "ai_low":       ai_low,
            "ai_high":      ai_high,
            "delta_low":    delta_low,
            "delta_high":   delta_high,
            "total_swing":  total_swing,
        })

    df_results = pd.DataFrame(results).sort_values("total_swing", ascending=True)

    # Save results
    sa_csv   = PROCESSED_DIR / f"sensitivity_results_{SCENARIO}.csv"
    df_results.to_csv(sa_csv, index=False)
    print(f"\nSaved sensitivity results → {sa_csv}")

    # ── Tornado chart ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    colors_low  = "#4878CF"
    colors_high = "#D65F5F"

    y_positions = range(len(df_results))

    for i, (_, row) in enumerate(df_results.iterrows()):
        # Low value bar (left of baseline)
        low_val  = row["delta_low"]
        high_val = row["delta_high"]

        # Draw bar from min to max change
        left  = min(low_val, high_val)
        right = max(low_val, high_val)

        # Colour: which end is -20% vs +20%?
        if low_val < high_val:
            # Low parameter → lower AI, High parameter → higher AI
            ax.barh(i, low_val,  left=0, color=colors_low,  alpha=0.8, height=0.6)
            ax.barh(i, high_val, left=0, color=colors_high, alpha=0.8, height=0.6)
        else:
            ax.barh(i, high_val, left=0, color=colors_low,  alpha=0.8, height=0.6)
            ax.barh(i, low_val,  left=0, color=colors_high, alpha=0.8, height=0.6)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df_results["label"].tolist())
    ax.axvline(0, color="black", linewidth=1.2)

    ax.set_xlabel("Change in AI_own at Year 10 relative to baseline")
    ax.set_title(
        f"Tornado Chart: Parameter Sensitivity of Ownership Affordability\n"
        f"(±{DELTA*100:.0f}% perturbation, N={N_SA} realisations, Scenario {SCENARIO})",
        fontsize=11, fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_low,  alpha=0.8, label=f"−{DELTA*100:.0f}% of parameter value"),
        Patch(facecolor=colors_high, alpha=0.8, label=f"+{DELTA*100:.0f}% of parameter value"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR   / f"sensitivity_tornado_{SCENARIO}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved tornado chart → {out_path}")
    plt.show()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n── Sensitivity Analysis Summary ──────────────────────────────────")
    print(f"{'Parameter':<40} {'Swing':>10}")
    for _, row in df_results.sort_values("total_swing", ascending=False).iterrows():
        print(f"  {row['label']:<38} {row['total_swing']:>10.6f}")

    print(f"\nMost influential: {df_results.iloc[-1]['label']}")
    print(f"Least influential: {df_results.iloc[0]['label']}")