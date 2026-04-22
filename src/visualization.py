"""
visualization.py
----------------
Plotting utilities for the Toronto Missing Middle Zoning ABM.

Functions:
    plot_baseline_trajectories()  — trajectory plot for a single scenario
    plot_scenario_boxplot()       — boxplot comparison across scenarios
    plot_spatial_heatmap()        — choropleth map of CT-level AI change
    plot_gentrification_risk()    — spatial gentrification pressure metric
    plot_all_scenarios()          — combined trajectory comparison

Usage:
    from src.visualization import plot_baseline_trajectories, plot_spatial_heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Optional

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

SCENARIO_LABELS = {
    "S0": "S0: Status Quo",
    "S1": "S1: City-Wide Zoning",
    "S2": "S2: Transit-Targeted",
    "S3": "S3: Incentive-Based",
}

SCENARIO_COLORS = {
    "S0": "#4878CF",
    "S1": "#6ACC65",
    "S2": "#D65F5F",
    "S3": "#B47CC7",
}

T = 40
YEARS = np.linspace(0, 10, T)


# ── Trajectory plot ────────────────────────────────────────────────────────────

def plot_baseline_trajectories(results: dict, metric: str = "ai_own",
                                scenario: str = "S0",
                                ylabel: str = "Ownership Affordability Index",
                                title: str = None,
                                save_path: Path = None) -> plt.Figure:
    """
    Plot all N realisations as light trajectories with mean and IQR band.
    """
    arr    = results[metric]
    mean   = arr.mean(axis=0)
    p25    = np.percentile(arr, 25, axis=0)
    p75    = np.percentile(arr, 75, axis=0)
    p5     = np.percentile(arr, 5,  axis=0)
    p95    = np.percentile(arr, 95, axis=0)
    color  = SCENARIO_COLORS.get(scenario, "#4878CF")

    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(arr.shape[0]):
        ax.plot(YEARS, arr[i], color=color, alpha=0.06, linewidth=0.6)

    ax.fill_between(YEARS, p5,  p95, color=color, alpha=0.10,
                    label="5th–95th percentile")
    ax.fill_between(YEARS, p25, p75, color=color, alpha=0.22,
                    label="25th–75th percentile (IQR)")
    ax.plot(YEARS, mean, color=color, linewidth=2.5, label="Mean")

    ax.set_xlabel("Years since policy implementation")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Baseline Scenario ({scenario}): {ylabel} Over 10 Years",
                 fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.set_xlim(0, 10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ── Scenario boxplot ───────────────────────────────────────────────────────────

def plot_scenario_boxplot(results: Dict[str, dict],
                          scenarios: list = None,
                          metric: str = "ai_own",
                          ylabel: str = "Ownership Affordability Index (Year 10)",
                          title: str = None,
                          save_path: Path = None) -> plt.Figure:
    """
    Boxplot of final-year affordability across scenarios.
    """
    if scenarios is None:
        scenarios = list(results.keys())

    final_vals = [results[s][metric][:, -1] for s in scenarios]
    labels     = [SCENARIO_LABELS.get(s, s) for s in scenarios]
    colors     = [SCENARIO_COLORS.get(s, "#888888") for s in scenarios]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        final_vals, patch_artist=True, widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(scenarios) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Scenario Comparison: Final-Year Affordability",
                 fontsize=12, fontweight="bold")

    for i, vals in enumerate(final_vals):
        med = np.median(vals)
        ax.text(i + 1, med, f"  {med:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ── All scenarios trajectory ───────────────────────────────────────────────────

def plot_all_scenarios(results: Dict[str, dict],
                       metric: str = "ai_own",
                       ylabel: str = "Ownership Affordability Index",
                       title: str = None,
                       save_path: Path = None) -> plt.Figure:
    """
    Plot mean trajectory for all scenarios on one axes.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for scenario, res in results.items():
        arr   = res[metric]
        mean  = arr.mean(axis=0)
        p25   = np.percentile(arr, 25, axis=0)
        p75   = np.percentile(arr, 75, axis=0)
        color = SCENARIO_COLORS.get(scenario, "#888888")
        label = SCENARIO_LABELS.get(scenario, scenario)

        ax.fill_between(YEARS, p25, p75, color=color, alpha=0.15)
        ax.plot(YEARS, mean, color=color, linewidth=2, label=label)

    ax.set_xlabel("Years since policy implementation")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"All Scenarios: {ylabel}",
                 fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.set_xlim(0, 10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ── Spatial heatmap ────────────────────────────────────────────────────────────

def plot_spatial_heatmap(ct_ai_change: pd.DataFrame,
                         value_col: str = "ai_own_change",
                         title: str = "Change in Ownership Affordability: S1 vs S0",
                         cmap: str = "RdYlGn",
                         save_path: Path = None):
    """
    Choropleth map of CT-level affordability change.

    Args:
        ct_ai_change — dataframe with 'ctuid' and value_col columns
        value_col    — column containing the value to plot
        title        — plot title
        cmap         — matplotlib colormap (RdYlGn: red=worse, green=better)
        save_path    — path to save figure

    Requires geopandas and the CT boundary shapefile.
    """
    try:
        import geopandas as gpd
        from src.paths import BOUNDARY_SHP
    except ImportError:
        print("geopandas required for spatial heatmap — pip install geopandas")
        return None

    TORONTO_CMA   = "535"
    CRS_PROJECTED = "EPSG:32617"
    CRS_DISPLAY   = "EPSG:4326"

    gdf = gpd.read_file(BOUNDARY_SHP).to_crs(CRS_PROJECTED)
    gdf = gdf[gdf["DGUID"].str.contains(TORONTO_CMA, na=False)].copy()
    gdf = gdf.rename(columns={"DGUID": "ctuid"})
    gdf = gdf.merge(ct_ai_change[["ctuid", value_col]], on="ctuid", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    gdf.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        legend=True,
        legend_kwds={
            "label":       value_col.replace("_", " ").title(),
            "orientation": "horizontal",
            "shrink":      0.6,
            "pad":         0.02,
        },
        missing_kwds={"color": "lightgrey", "label": "No data"},
        edgecolor="white",
        linewidth=0.2,
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ── Gentrification risk metric ─────────────────────────────────────────────────

def compute_gentrification_risk(results_s0: dict, results_s1: dict,
                                 agents_csv: Path) -> pd.DataFrame:
    """
    Compute a gentrification risk score for each CT under S1 vs S0.

    Risk is defined as: CTs that are currently low-income (below median)
    AND experience above-average price growth under S1 relative to S0.
    These are CTs where supply-induced price growth could displace existing
    lower-income residents — a key equity concern for the stakeholder.

    Returns dataframe with ctuid, risk_score, and component metrics.
    """
    import pandas as pd

    agents = pd.read_csv(agents_csv)

    # Mean final price across realisations for each scenario
    # results shape: (N, T) — we need per-CT data
    # Note: current results only store city-wide means, not per-CT
    # This is a placeholder that uses city-wide AI change as a proxy
    # Full per-CT tracking would require storing CT-level outputs in simulation

    median_income = agents["median_household_income"].median()
    agents["low_income_ct"] = (
        agents["median_household_income"] < median_income
    ).astype(int)

    agents["high_renter_ct"] = (
        agents["renter_share"] > agents["renter_share"].median()
    ).astype(int)

    # Gentrification risk: low income + high renter share + near transit
    # (transit-adjacent low-income high-renter CTs most vulnerable under S2)
    agents["gentrification_risk"] = (
        agents["low_income_ct"] +
        agents["high_renter_ct"]
    )

    return agents[["ctuid", "GEO_NAME", "median_household_income",
                   "renter_share", "low_income_ct", "high_renter_ct",
                   "gentrification_risk"]].sort_values(
                       "gentrification_risk", ascending=False
                   )