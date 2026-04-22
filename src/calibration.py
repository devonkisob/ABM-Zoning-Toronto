"""
calibration.py
--------------
Two-stage ML model for development probability and magnitude.
Updated for Final Report: compares linear regression vs random forest
for Stage 2 and selects the better model based on cross-validated R².

Stage 1: Logistic regression
Stage 2: Linear regression vs Random Forest — best CV R² wins

Usage:
    python src/calibration.py
    from src.calibration import load_models, predict_development
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

from src.paths import (
    CALIB_CSV, TRANSIT_CSV,
    STAGE1_PKL, STAGE2_PKL, SCALER_PKL,
    CALIB_REPORT as REPORT_TXT,
    PROCESSED_DIR,
)

SEED = 42

FEATURES = [
    "median_household_income",
    "median_household_income_2016",
    "income_growth",
    "home_price",
    "home_price_2016",
    "home_price_growth",
    "annual_rent",
    "rent_growth",
    "renter_share",
    "renter_share_2016",
    "renter_share_change",
    "total_households",
    "population_2021",
    "transit_indicator",
]

TARGET_S1 = "dev_occurred"
TARGET_S2 = "delta_units"


# ── Public API ─────────────────────────────────────────────────────────────────

def load_models():
    """Load fitted Stage 1 and Stage 2 models from disk."""
    with open(STAGE1_PKL, "rb") as f: stage1 = pickle.load(f)
    with open(STAGE2_PKL, "rb") as f: stage2 = pickle.load(f)
    with open(SCALER_PKL, "rb") as f: scaler, features = pickle.load(f)
    return stage1, stage2, scaler, features


def predict_development(ct, policy, stage1, stage2, scaler, features,
                        rng: np.random.Generator):
    """
    Run two-stage development model for a single CT agent.
    Returns units_added (int) — 0 if no development.
    """
    from src.config import DEFAULT_CONFIG as cfg

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
    x = np.nan_to_num(x, nan=0.0)
    x_scaled = scaler.transform(x)

    p_dev = stage1.predict_proba(x_scaled)[0, 1]

    if not policy.is_eligible(ct.ctuid):
        return 0

    p_adj = min(1.0, (p_dev / cfg.t_horizon) * (1.0 + 0.3 * policy.incentive_level))
    if not (rng.random() < p_adj):
        return 0

    units = float(stage2.predict(x_scaled)[0])
    return int(max(0, round(units)))


# ── Training (only runs when called directly) ──────────────────────────────────

if __name__ == "__main__":

    print("Loading calibration data...")
    calib   = pd.read_csv(CALIB_CSV)
    transit = pd.read_csv(TRANSIT_CSV)[["ctuid", "transit_indicator"]]
    df = pd.merge(calib, transit, on="ctuid", how="left")
    df["transit_indicator"] = df["transit_indicator"].fillna(0.0)

    print(f"  Loaded {len(df)} CTs")
    print(f"  dev_occurred distribution:\n{df[TARGET_S1].value_counts()}")
    print(f"  delta_units range: [{df[TARGET_S2].min():.0f}, {df[TARGET_S2].max():.0f}]")

    available_features = [f for f in FEATURES if f in df.columns]
    df_clean = df[available_features + [TARGET_S1, TARGET_S2, "ctuid"]].dropna()
    n_dropped = len(df) - len(df_clean)
    print(f"\n  Rows after dropping NaN: {len(df_clean)} ({n_dropped} dropped)")

    X_all    = df_clean[available_features].values
    y_s1     = df_clean[TARGET_S1].values
    y_s2     = df_clean[TARGET_S2].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Stage 1
    print("\n── Stage 1: Development Occurrence ──────────────────────────────────")
    stage1 = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    stage1.fit(X_scaled, y_s1)
    cv_s1 = cross_val_score(stage1, X_scaled, y_s1, cv=5, scoring="f1")
    print(f"  5-fold CV F1: {cv_s1.mean():.3f} ± {cv_s1.std():.3f}")
    y_s1_pred = stage1.predict(X_scaled)
    print(classification_report(y_s1, y_s1_pred, target_names=["No dev", "Dev"]))

    coef_df = pd.DataFrame({
        "feature": available_features,
        "coefficient": stage1.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)
    print(coef_df.head(5).to_string(index=False))

    # Stage 2
    print("\n── Stage 2: Development Magnitude — Model Comparison ────────────────")
    dev_mask = y_s1 == 1
    X_dev    = X_scaled[dev_mask]
    y_dev    = y_s2[dev_mask]

    lr = LinearRegression()
    lr.fit(X_dev, y_dev)
    cv_lr = cross_val_score(lr, X_dev, y_dev, cv=5, scoring="r2")
    print(f"  Linear Regression:  CV R²={cv_lr.mean():.3f}±{cv_lr.std():.3f}  "
          f"MAE={mean_absolute_error(y_dev, lr.predict(X_dev)):.1f}")

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=SEED, n_jobs=1,
    )
    rf.fit(X_dev, y_dev)
    cv_rf = cross_val_score(rf, X_dev, y_dev, cv=5, scoring="r2")
    print(f"  Random Forest:      CV R²={cv_rf.mean():.3f}±{cv_rf.std():.3f}  "
          f"MAE={mean_absolute_error(y_dev, rf.predict(X_dev)):.1f}")

    fi_df = pd.DataFrame({
        "feature": available_features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(fi_df.head(5).to_string(index=False))

    if cv_rf.mean() > cv_lr.mean():
        stage2 = rf
        stage2_name = "RandomForest"
        cv_s2 = cv_rf
        mae_s2 = mean_absolute_error(y_dev, rf.predict(X_dev))
        print(f"\n  ✓ Selected: Random Forest (CV R² {cv_rf.mean():.3f} > {cv_lr.mean():.3f})")
    else:
        stage2 = lr
        stage2_name = "LinearRegression"
        cv_s2 = cv_lr
        mae_s2 = mean_absolute_error(y_dev, lr.predict(X_dev))
        print(f"\n  ✓ Selected: Linear Regression (CV R² {cv_lr.mean():.3f} >= {cv_rf.mean():.3f})")

    with open(STAGE1_PKL, "wb") as f: pickle.dump(stage1, f)
    with open(STAGE2_PKL, "wb") as f: pickle.dump(stage2, f)
    with open(SCALER_PKL, "wb") as f: pickle.dump((scaler, available_features), f)
    with open(PROCESSED_DIR / "stage2_model_name.txt", "w") as f: f.write(stage2_name)

    print(f"\nModels saved  [{stage2_name} selected for Stage 2]")

    report = "\n".join([
        "CALIBRATION REPORT", "=" * 60,
        f"Training CTs: {len(df_clean)}  |  Stage 2 CTs: {dev_mask.sum()}",
        f"Features: {len(available_features)}",
        "",
        "Stage 1 — Logistic Regression",
        f"  5-fold CV F1: {cv_s1.mean():.3f} ± {cv_s1.std():.3f}",
        classification_report(y_s1, y_s1_pred, target_names=["No dev", "Dev"]),
        "",
        "Stage 2 — Model Comparison",
        f"  Linear Regression: CV R²={cv_lr.mean():.3f}±{cv_lr.std():.3f}",
        f"  Random Forest:     CV R²={cv_rf.mean():.3f}±{cv_rf.std():.3f}",
        f"  Selected: {stage2_name}  CV R²={cv_s2.mean():.3f}  MAE={mae_s2:.1f}",
    ])
    with open(REPORT_TXT, "w") as f: f.write(report)
    print(f"Calibration report → {REPORT_TXT}")