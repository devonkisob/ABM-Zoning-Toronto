"""
calibration.py
--------------
Trains the two-stage ML model for development probability and magnitude,
calibrated from observed 2016→2021 CT-level housing growth in Toronto.

Stage 1: Logistic regression classifier
  Input:  CT features (zoning proxy, demand, density, income, transit, etc.)
  Output: P(development occurred) — binary target 'dev_occurred'

Stage 2: Linear regression
  Input:  Same features, restricted to CTs where dev_occurred == 1
  Output: E[units_added | development occurred] — continuous target 'delta_units'

Both fitted models are saved to data/processed/ and loaded by the simulation
at runtime via load_models().

Usage:
    python scripts/calibration.py          # train and save models
    from src.calibration import load_models  # load in simulation

Outputs:
    data/processed/stage1_model.pkl
    data/processed/stage2_model.pkl
    data/processed/feature_scaler.pkl
    data/processed/calibration_report.txt
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
CALIB_CSV   = Path("data/processed/ct_calibration.csv")
TRANSIT_CSV = Path("data/processed/ct_transit_indicators.csv")
OUT_DIR     = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_PKL  = OUT_DIR / "stage1_model.pkl"
STAGE2_PKL  = OUT_DIR / "stage2_model.pkl"
SCALER_PKL  = OUT_DIR / "feature_scaler.pkl"
REPORT_TXT  = OUT_DIR / "calibration_report.txt"

# Random seed for reproducibility
SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
# These are the features used in both Stage 1 and Stage 2.
# Must be kept consistent between calibration and simulation.

FEATURES = [
    # Demand-side: income and affordability pressure
    "median_household_income",       # 2021 median HH income ($)
    "median_household_income_2016",  # 2016 median HH income ($)
    "income_growth",                 # % income growth 2015→2020

    # Housing market: price and rent signals
    "home_price",                    # 2021 median dwelling value ($)
    "home_price_2016",               # 2016 median dwelling value ($)
    "home_price_growth",             # % home price growth 2016→2021
    "annual_rent",                   # 2021 annualised median rent ($)
    "rent_growth",                   # % rent growth 2016→2021

    # Tenure composition
    "renter_share",                  # 2021 fraction of renter households
    "renter_share_2016",             # 2016 fraction of renter households
    "renter_share_change",           # change in renter share 2016→2021

    # Density proxy
    "total_households",              # 2021 total households (density proxy)
    "population_2021",               # 2021 population

    # Transit accessibility (Scenario 2 key feature)
    "transit_indicator",             # normalized rapid transit stop count (0-1)
]

# Stage 1 target
TARGET_S1 = "dev_occurred"

# Stage 2 target
TARGET_S2 = "delta_units"


# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND MERGE DATA
# ══════════════════════════════════════════════════════════════════════════════

print("Loading calibration data...")
calib   = pd.read_csv(CALIB_CSV)
transit = pd.read_csv(TRANSIT_CSV)[["ctuid", "transit_indicator"]]

df = pd.merge(calib, transit, on="ctuid", how="left")
df["transit_indicator"] = df["transit_indicator"].fillna(0.0)

print(f"  Loaded {len(df)} CTs")
print(f"  dev_occurred distribution:\n{df[TARGET_S1].value_counts()}")
print(f"  delta_units range: [{df[TARGET_S2].min():.0f}, {df[TARGET_S2].max():.0f}]")


# ══════════════════════════════════════════════════════════════════════════════
# CLEAN AND PREPARE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

# Check which features are available
missing_features = [f for f in FEATURES if f not in df.columns]
if missing_features:
    print(f"\n⚠  Missing features (will be dropped): {missing_features}")
available_features = [f for f in FEATURES if f in df.columns]

# Drop rows with any NaN in features or targets
df_clean = df[available_features + [TARGET_S1, TARGET_S2, "ctuid"]].dropna()
n_dropped = len(df) - len(df_clean)
print(f"\n  Rows after dropping NaN: {len(df_clean)} ({n_dropped} dropped)")

X_all = df_clean[available_features].values
y_s1  = df_clean[TARGET_S1].values
y_s2  = df_clean[TARGET_S2].values

# Scale features — important for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: DEVELOPMENT OCCURRENCE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
# class_weight='balanced' handles the 92% / 8% imbalance noted in calibration.

print("\n── Stage 1: Development Occurrence ──────────────────────────────────")

stage1 = LogisticRegression(
    max_iter=1000,
    random_state=SEED,
    class_weight="balanced",   # handles imbalanced dev_occurred target
)
stage1.fit(X_scaled, y_s1)

# Cross-validated accuracy (5-fold)
cv_scores_s1 = cross_val_score(stage1, X_scaled, y_s1, cv=5, scoring="f1")
print(f"  5-fold CV F1: {cv_scores_s1.mean():.3f} ± {cv_scores_s1.std():.3f}")

y_s1_pred = stage1.predict(X_scaled)
print(f"\n  Classification report (in-sample):")
print(classification_report(y_s1, y_s1_pred, target_names=["No dev", "Dev"]))

# Feature importances (logistic regression coefficients)
coef_df = pd.DataFrame({
    "feature": available_features,
    "coefficient": stage1.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)
print(f"\n  Top 5 features by coefficient magnitude:")
print(coef_df.head(5).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: DEVELOPMENT MAGNITUDE REGRESSOR
# ══════════════════════════════════════════════════════════════════════════════
# Linear regression on CTs where development occurred.

print("\n── Stage 2: Development Magnitude ───────────────────────────────────")

dev_mask   = y_s1 == 1
X_dev      = X_scaled[dev_mask]
y_dev      = y_s2[dev_mask]

print(f"  Training on {dev_mask.sum()} CTs where development occurred")
print(f"  delta_units stats: mean={y_dev.mean():.1f}, std={y_dev.std():.1f}, "
      f"min={y_dev.min():.0f}, max={y_dev.max():.0f}")

stage2 = LinearRegression()
stage2.fit(X_dev, y_dev)

cv_scores_s2 = cross_val_score(stage2, X_dev, y_dev, cv=5, scoring="r2")
print(f"  5-fold CV R²: {cv_scores_s2.mean():.3f} ± {cv_scores_s2.std():.3f}")

y_dev_pred = stage2.predict(X_dev)
print(f"  In-sample MAE:  {mean_absolute_error(y_dev, y_dev_pred):.1f} units")
print(f"  In-sample R²:   {r2_score(y_dev, y_dev_pred):.3f}")

# Note if R² is low — linear regression on delta_units is a simplification
if r2_score(y_dev, y_dev_pred) < 0.2:
    print("\n  ⚠  Low R² — linear model explains limited variance in delta_units.")
    print("     This is expected given CT-level heterogeneity and is documented")
    print("     as a limitation. The model captures the mean trend.")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════

with open(STAGE1_PKL, "wb") as f:
    pickle.dump(stage1, f)

with open(STAGE2_PKL, "wb") as f:
    pickle.dump(stage2, f)

with open(SCALER_PKL, "wb") as f:
    pickle.dump((scaler, available_features), f)

print(f"\nModels saved:")
print(f"  {STAGE1_PKL}")
print(f"  {STAGE2_PKL}")
print(f"  {SCALER_PKL}")


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

report_lines = [
    "CALIBRATION REPORT",
    "=" * 60,
    f"Training CTs (total):          {len(df_clean)}",
    f"Stage 1 training CTs:          {len(df_clean)}",
    f"Stage 2 training CTs:          {dev_mask.sum()}",
    f"Features used:                 {len(available_features)}",
    "",
    "Stage 1 — Logistic Regression (dev_occurred)",
    "-" * 40,
    f"5-fold CV F1:    {cv_scores_s1.mean():.3f} ± {cv_scores_s1.std():.3f}",
    "",
    classification_report(y_s1, y_s1_pred, target_names=["No dev", "Dev"]),
    "",
    "Stage 1 Feature Coefficients:",
    coef_df.to_string(index=False),
    "",
    "Stage 2 — Linear Regression (delta_units | dev_occurred=1)",
    "-" * 40,
    f"5-fold CV R²:    {cv_scores_s2.mean():.3f} ± {cv_scores_s2.std():.3f}",
    f"In-sample MAE:   {mean_absolute_error(y_dev, y_dev_pred):.1f} units",
    f"In-sample R²:    {r2_score(y_dev, y_dev_pred):.3f}",
    "",
    "Known limitations:",
    "  - 128 CTs excluded due to boundary changes (inner join 2016/2021)",
    "  - Stage 1 target imbalanced (92% dev=1); balanced class weights used",
    "  - Stage 2 uses linear model; CT heterogeneity limits R²",
    "  - No explicit zoning eligibility variable (proxy: density + transit)",
    "  - Calibration period 2016-2021 includes COVID-19 (2020-2021)",
]

report_text = "\n".join(report_lines)
with open(REPORT_TXT, "w") as f:
    f.write(report_text)

print(f"\nCalibration report saved → {REPORT_TXT}")
print("\nDone. Load models in simulation with:")
print("  from src.calibration import load_models")
print("  stage1, stage2, scaler, features = load_models()")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — used by simulation.py
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    """
    Load fitted Stage 1 and Stage 2 models from disk.

    Returns:
        stage1   — sklearn LogisticRegression (predicts dev_occurred)
        stage2   — sklearn LinearRegression   (predicts delta_units)
        scaler   — sklearn StandardScaler     (fitted on training features)
        features — list[str]                  (feature names in order)
    """
    with open(STAGE1_PKL, "rb") as f:
        stage1 = pickle.load(f)
    with open(STAGE2_PKL, "rb") as f:
        stage2 = pickle.load(f)
    with open(SCALER_PKL, "rb") as f:
        scaler, features = pickle.load(f)
    return stage1, stage2, scaler, features


def predict_development(ct, policy, stage1, stage2, scaler, features,
                        rng: np.random.Generator):
    """
    Run the two-stage development model for a single CT agent.

    Args:
        ct      — CensusTractAgent instance
        policy  — PolicyModel instance (provides zoning_eligible, incentive_level)
        stage1  — fitted LogisticRegression
        stage2  — fitted LinearRegression
        scaler  — fitted StandardScaler
        features — list of feature names (must match training order)
        rng     — numpy random Generator for stochastic sampling

    Returns:
        units_added (int) — 0 if no development, else predicted magnitude
    """
    # Build feature vector in the same order as training
    feature_map = {
        "median_household_income":      ct.median_income,
        "median_household_income_2016": ct.median_income_2016,
        "income_growth":                ct.income_growth,
        "home_price":                   ct.home_price,
        "home_price_2016":              ct.home_price_2016,
        "home_price_growth":            ct.home_price_growth,
        "annual_rent":                  ct.annual_rent,
        "rent_growth":                  ct.rent_growth,
        "renter_share":                 ct.renter_share,
        "renter_share_2016":            ct.renter_share_2016,
        "renter_share_change":          ct.renter_share - ct.renter_share_2016,
        "total_households":             ct.households,
        "population_2021":              ct.population,
        "transit_indicator":            ct.transit_indicator,
    }

    x = np.array([feature_map.get(f, 0.0) for f in features]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    # Stage 1: sample development occurrence
    p_dev = stage1.predict_proba(x_scaled)[0, 1]

    # Apply policy modifier: incentive level boosts development probability
    # zoning ineligible CTs cannot develop regardless of probability
    if not policy.is_eligible(ct.ctuid):
        return 0

    p_dev_adjusted = min(1.0, p_dev * (1.0 + 0.3 * policy.incentive_level))
    dev_event = rng.random() < p_dev_adjusted

    if not dev_event:
        return 0

    # Stage 2: predict units added, apply floor of 0
    units_added = float(stage2.predict(x_scaled)[0])
    units_added = max(0, round(units_added))

    return int(units_added)