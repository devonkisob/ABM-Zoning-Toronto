"""
config.py
---------
Global simulation parameters in one place.
Import from here rather than hardcoding values across multiple files.

Usage:
    from src.config import SimConfig
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """
    Simulation configuration parameters.

    frozen=True makes this immutable — parameters cannot be accidentally
    changed during a simulation run.

    To override for a specific run, create a new instance:
        cfg = SimConfig(N=500, T=40)
    """

    # ── Monte Carlo ─────────────────────────────────────────────────────────
    N: int   = 100    # number of realisations (use 500 for Final Report)
    T: int   = 40     # time steps (40 quarters = 10 years)
    seed: int = 42    # base random seed

    # ── Time ────────────────────────────────────────────────────────────────
    steps_per_year: int = 4      # quarterly steps
    horizon_years:  int = 10     # simulation horizon

    # ── Demand model ─────────────────────────────────────────────────────────
    # Base demand as fraction of total housing stock per quarter.
    # Calibrated so city-wide annual demand ≈ 15,000 net new households
    # (CMHC Housing Market Outlook, Toronto CMA).
    base_demand:   float = 0.02    # 2% of total stock per quarter
    demand_growth: float = 0.0025  # 0.25%/quarter ≈ 1%/year

    # ── Market update ────────────────────────────────────────────────────────
    # Elasticity of price/rent to vacancy gap (v_star - vacancy_rate).
    price_kappa: float = 2.149   # ownership price elasticity
    rent_kappa:  float = 1.446   # rental price elasticity
    v_star:      float = 0.03   # target vacancy rate (CMHC Toronto avg ~3%)
    vacancy_eq: float = 0.013   # steady-state vacancy in S0 (~Toronto tight market)

    # ── Infrastructure model ─────────────────────────────────────────────────
    omega0:        float = 0.5    # baseline infra load
    omega1:        float = 0.3    # load sensitivity to density
    g_base:        float = 0.01   # baseline annual infra capacity growth
    lambda_incent: float = 0.005  # capacity growth penalty per incentive unit

    # ── Development model ────────────────────────────────────────────────────
    # p_dev from the ML model is an annual probability calibrated on 2016-2021
    # data. Dividing by T scales it to a per-quarter probability so that
    # expected development over the simulation period matches the calibration.
    t_horizon: int = 40   # matches T — used to scale p_dev per time step

    # Incentive level for S3 (boost to p_dev for eligible CTs)
    s3_incentive_level: float = 0.5


# Default config instance — import this directly for standard runs
DEFAULT_CONFIG = SimConfig()