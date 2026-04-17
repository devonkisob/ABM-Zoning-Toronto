"""
calibrate_elasticity.py
-----------------------
Calibrates price_kappa and rent_kappa against observed Toronto housing market trends.

Approach:
    We back-calculate the elasticity parameters that reproduce observed
    long-run average price and rent growth rates in the baseline scenario (S0),
    where no new supply is added and demand grows at the configured rate.

Target observations (Teranet-National Bank HPI, Toronto CMA, 2010-2019 average):
    - Home price growth: ~6% per year (pre-pandemic long-run average)
    - Rent growth: ~4% per year (CMHC Rental Market Survey, Toronto CMA)

    Note: We use the pre-pandemic (2010-2019) period rather than 2021-2023
    because the pandemic period was dominated by interest rate shocks and
    temporary demand surges that are not representative of the structural
    housing market dynamics our model aims to capture.

Method:
    1. Run a short S0 simulation (no new supply) with a candidate price_kappa
    2. Measure the implied annual price growth rate
    3. Use bisection search to find price_kappa that matches the target
    4. Repeat for rent_kappa

Usage:
    python scripts/calibrate_elasticity.py

Output:
    Prints calibrated parameters and saves to data/processed/calibrated_params.json
"""

import sys as sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import copy
from src.agents import load_agents, get_transit_ctuids, PolicyModel, DemandAllocationModel, InfrastructureModel
from src.calibration import load_models, predict_development
from src.config import DEFAULT_CONFIG as cfg
from src.paths import PROCESSED_DIR

# ── Calibration targets ────────────────────────────────────────────────────────
# Pre-pandemic (2010-2019) Toronto long-run averages
# Source: Teranet-National Bank HPI (Toronto CMA)
TARGET_ANNUAL_PRICE_GROWTH = 0.06   # 6% per year
TARGET_ANNUAL_RENT_GROWTH  = 0.04   # 4% per year (CMHC Rental Market Survey)

# Implied quarterly targets
TARGET_QUARTERLY_PRICE = (1 + TARGET_ANNUAL_PRICE_GROWTH) ** (1/4) - 1  # ~1.47%
TARGET_QUARTERLY_RENT  = (1 + TARGET_ANNUAL_RENT_GROWTH)  ** (1/4) - 1  # ~0.98%

# Calibration run parameters (short run for speed)
N_CALIB = 20    # realisations
T_CALIB = 20    # 5 years of quarters (enough to measure stable growth)
SEED    = 42


def run_s0_short(price_kappa: float, rent_kappa: float,
                 base_agents, stage1, stage2, scaler, features) -> tuple:
    """
    Run a short S0 simulation with given elasticity parameters.
    Returns (mean_quarterly_price_growth, mean_quarterly_rent_growth).
    """
    transit_ctuids = get_transit_ctuids()
    all_ctuids     = [a.ctuid for a in base_agents]
    policy         = PolicyModel.from_scenario("S0", all_ctuids, transit_ctuids)

    price_growths = []
    rent_growths  = []

    for i in range(N_CALIB):
        rng  = np.random.default_rng(SEED + i)
        cts  = copy.deepcopy(base_agents)

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

        # Record initial values
        init_prices = np.array([ct.home_price for ct in cts])
        init_rents  = np.array([ct.annual_rent for ct in cts])

        for t in range(T_CALIB):
            demand_model.allocate(cts, t)
            for ct in cts:
                predict_development(ct, policy, stage1, stage2, scaler, features, rng)
            for ct in cts:
                ct.update_market(price_kappa, rent_kappa, cfg.v_star, cfg.vacancy_eq)
            for ct in cts:
                infra_model.step(ct, policy)

        # Measure growth over full run
        final_prices = np.array([ct.home_price for ct in cts])
        final_rents  = np.array([ct.annual_rent for ct in cts])

        # Implied quarterly growth rate (geometric mean)
        price_ratio = np.nanmean(final_prices / np.maximum(init_prices, 1))
        rent_ratio  = np.nanmean(final_rents  / np.maximum(init_rents,  1))

        price_growths.append(price_ratio ** (1/T_CALIB) - 1)
        rent_growths.append(rent_ratio  ** (1/T_CALIB) - 1)

    return np.mean(price_growths), np.mean(rent_growths)


def bisect_kappa(target_growth: float, is_price: bool,
                 base_agents, stage1, stage2, scaler, features,
                 lo: float = 0.001, hi: float = 5.0,
                 tol: float = 1e-4, max_iter: int = 20) -> float:
    """
    Use bisection to find kappa that produces target quarterly growth rate.
    """
    for iteration in range(max_iter):
        mid = (lo + hi) / 2
        if is_price:
            growth, _ = run_s0_short(mid, cfg.rent_kappa, base_agents, stage1, stage2, scaler, features)
        else:
            _, growth = run_s0_short(cfg.price_kappa, mid, base_agents, stage1, stage2, scaler, features)

        error = growth - target_growth
        print(f"  iter {iteration+1}: kappa={mid:.4f}, growth={growth*100:.3f}%, "
              f"target={target_growth*100:.3f}%, error={error*100:.4f}%")

        if abs(error) < tol:
            print(f"  Converged after {iteration+1} iterations")
            return mid

        if error < 0:
            lo = mid   # growth too low → increase kappa
        else:
            hi = mid   # growth too high → decrease kappa

    print(f"  Did not fully converge — returning best estimate")
    return (lo + hi) / 2


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading agents and models...")
    base_agents = load_agents()
    stage1, stage2, scaler, features = load_models()

    print(f"\nCalibration targets:")
    print(f"  Annual price growth:    {TARGET_ANNUAL_PRICE_GROWTH*100:.1f}%")
    print(f"  Annual rent growth:     {TARGET_ANNUAL_RENT_GROWTH*100:.1f}%")
    print(f"  Quarterly price target: {TARGET_QUARTERLY_PRICE*100:.3f}%")
    print(f"  Quarterly rent target:  {TARGET_QUARTERLY_RENT*100:.3f}%")
    print(f"  Calibration run: N={N_CALIB}, T={T_CALIB}")

    # ── Step 1: Check what current parameters produce ─────────────────────────
    print(f"\nCurrent parameters: price_kappa={cfg.price_kappa}, rent_kappa={cfg.rent_kappa}")
    current_price_g, current_rent_g = run_s0_short(
        cfg.price_kappa, cfg.rent_kappa, base_agents, stage1, stage2, scaler, features
    )
    print(f"Current implied quarterly growth: price={current_price_g*100:.3f}%, rent={current_rent_g*100:.3f}%")
    print(f"Current implied annual growth:    price={(((1+current_price_g)**4)-1)*100:.2f}%, rent={(((1+current_rent_g)**4)-1)*100:.2f}%")

    # ── Step 2: Calibrate price_kappa ─────────────────────────────────────────
    print(f"\nCalibrating price_kappa (target quarterly: {TARGET_QUARTERLY_PRICE*100:.3f}%)...")
    calibrated_price_kappa = bisect_kappa(
        TARGET_QUARTERLY_PRICE, is_price=True,
        base_agents=base_agents, stage1=stage1, stage2=stage2,
        scaler=scaler, features=features,
    )
    print(f"Calibrated price_kappa: {calibrated_price_kappa:.4f}")

    # ── Step 3: Calibrate rent_kappa ──────────────────────────────────────────
    print(f"\nCalibrating rent_kappa (target quarterly: {TARGET_QUARTERLY_RENT*100:.3f}%)...")
    calibrated_rent_kappa = bisect_kappa(
        TARGET_QUARTERLY_RENT, is_price=False,
        base_agents=base_agents, stage1=stage1, stage2=stage2,
        scaler=scaler, features=features,
    )
    print(f"Calibrated rent_kappa: {calibrated_rent_kappa:.4f}")

    # ── Step 4: Verify ────────────────────────────────────────────────────────
    print(f"\nVerifying calibrated parameters...")
    final_price_g, final_rent_g = run_s0_short(
        calibrated_price_kappa, calibrated_rent_kappa,
        base_agents, stage1, stage2, scaler, features,
    )
    print(f"Verified quarterly growth: price={final_price_g*100:.3f}%, rent={final_rent_g*100:.3f}%")
    print(f"Verified annual growth:    price={(((1+final_price_g)**4)-1)*100:.2f}%, rent={(((1+final_rent_g)**4)-1)*100:.2f}%")

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    params = {
        "price_kappa": round(calibrated_price_kappa, 4),
        "rent_kappa":  round(calibrated_rent_kappa,  4),
        "calibration_target_annual_price_growth": TARGET_ANNUAL_PRICE_GROWTH,
        "calibration_target_annual_rent_growth":  TARGET_ANNUAL_RENT_GROWTH,
        "calibration_source": "Teranet-National Bank HPI Toronto CMA 2010-2019 pre-pandemic average",
        "calibration_n": N_CALIB,
        "calibration_t": T_CALIB,
        "verified_annual_price_growth": round((((1+final_price_g)**4)-1), 4),
        "verified_annual_rent_growth":  round((((1+final_rent_g)**4)-1), 4),
    }

    out_path = PROCESSED_DIR / "calibrated_params.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nSaved → {out_path}")
    print(f"\nAdd to src/config.py:")
    print(f"  price_kappa: float = {params['price_kappa']}")
    print(f"  rent_kappa:  float = {params['rent_kappa']}")
