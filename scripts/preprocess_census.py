"""
preprocess_census.py
--------------------
Extracts Toronto Census Tract data from the StatCan 2021 and 2016 Census
Profiles and outputs two files:

1. ct_agents_init.csv    — 2021 CT data to initialize CensusTractAgent objects
2. ct_calibration.csv   — joined 2016/2021 data with delta_units + ML features

Usage:
    python scripts/preprocess_census.py

Inputs:
    data/raw/98-401-X2021007_English_CSV_data.csv
    data/raw/98-401-X2016043_English_CSV_data.csv

Outputs:
    data/processed/ct_agents_init.csv
    data/processed/ct_calibration.csv

Catalogue numbers:
    2021: Statistics Canada, Census Profile 2021, Catalogue no. 98-401-X2021007
    2016: Statistics Canada, Census Profile 2016, Catalogue no. 98-401-X2016043
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
from src.paths import (
    CENSUS_2021 as RAW_2021,
    CENSUS_2016 as RAW_2016,
    AGENTS_CSV  as OUT_AGENTS,
    CALIB_CSV   as OUT_CALIB,
)

TORONTO_CMA = "535"
CHUNK_SIZE  = 200_000

# ── ID format notes ────────────────────────────────────────────────────────────
# 2021 DGUID:       '2021S05075350001.00'
# 2016 GEO_CODE:    '5350001.00'
# Join key:         extract last 10 chars of 2021 DGUID → matches 2016 GEO_CODE
# e.g. '2021S0507' + '5350001.00'  →  '5350001.00'

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN NAME MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════

COL_2021 = {
    "encoding":       "latin-1",
    "geo_id":         "DGUID",
    "char_id":        "CHARACTERISTIC_ID",
    "geo_level":      "GEO_LEVEL",
    "geo_name":       "GEO_NAME",
    "ct_level":       "Census tract",
    "geo_startswith": False,
}

COL_2016 = {
    "encoding":       "utf-8-sig",
    "geo_id":         "GEO_CODE (POR)",
    "char_id":        "Member ID: Profile of Census Tracts (2247)",
    "geo_level":      "GEO_LEVEL",
    "geo_name":       "GEO_NAME",
    "ct_level":       "2",
    "geo_startswith": True,
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def read_toronto_cts(csv_path: Path, target_ids: set, label: str,
                     col: dict) -> pd.DataFrame:
    print(f"\nReading {label} ({csv_path.name}) in chunks...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        encoding=col["encoding"],
        low_memory=False,
    )):
        chunk.columns = [c.lstrip("\ufeff").strip('"') for c in chunk.columns]

        level_mask = chunk[col["geo_level"]].str.strip() == col["ct_level"]

        if col["geo_startswith"]:
            geo_mask = chunk[col["geo_id"]].str.startswith(TORONTO_CMA, na=False)
        else:
            geo_mask = chunk[col["geo_id"]].str.contains(TORONTO_CMA, na=False)

        char_mask = chunk[col["char_id"]].astype(float).isin(target_ids)

        filtered = chunk[level_mask & geo_mask & char_mask].copy()
        filtered = filtered.rename(columns={
            col["geo_id"]:   "DGUID",
            col["char_id"]:  "CHARACTERISTIC_ID",
            col["geo_name"]: "GEO_NAME",
        })

        if not filtered.empty:
            chunks.append(filtered)
        if i % 10 == 0:
            print(f"  ...processed {(i + 1) * CHUNK_SIZE:,} rows")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Done. Rows kept: {len(df):,}  |  Unique CTs: {df['DGUID'].nunique()}")
    return df


def detect_value_col(df: pd.DataFrame) -> str:
    if "C1_COUNT_TOTAL" in df.columns:
        return "C1_COUNT_TOTAL"
    dim_cols = [c for c in df.columns if c.startswith("Dim:")]
    if dim_cols:
        return dim_cols[0]
    meta = {"CENSUS_YEAR", "DGUID", "ALT_GEO_CODE", "GEO_LEVEL", "GEO_NAME",
            "TNR_SF", "TNR_LF", "DATA_QUALITY_FLAG", "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME", "SYMBOL", "NOTES", "GNR", "GNR_LF"}
    candidates = [c for c in df.columns if c not in meta]
    if candidates:
        return candidates[0]
    raise ValueError("Could not detect value column.")


def pivot_to_wide(df: pd.DataFrame, characteristics: dict,
                  value_col: str) -> pd.DataFrame:
    df = df[["DGUID", "GEO_NAME", "CHARACTERISTIC_ID", value_col]].copy()
    df["CHARACTERISTIC_ID"] = df["CHARACTERISTIC_ID"].astype(float).astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    wide = df.pivot_table(
        index=["DGUID", "GEO_NAME"],
        columns="CHARACTERISTIC_ID",
        values=value_col,
        aggfunc="first",
    ).reset_index()
    wide.rename(columns=characteristics, inplace=True)
    wide.columns.name = None
    return wide


# ══════════════════════════════════════════════════════════════════════════════
# 2021 DATA
# ══════════════════════════════════════════════════════════════════════════════

CHARS_2021 = {
    1:    "population_2021",
    4:    "total_private_dwellings",
    50:   "total_households",
    243:  "median_household_income",
    244:  "median_aftertax_income",
    1414: "total_households_by_tenure",
    1415: "owner_households",
    1416: "renter_households",
    1486: "median_monthly_shelter_owned",
    1488: "median_dwelling_value",
    1494: "median_monthly_shelter_rented",
}

raw_2021  = read_toronto_cts(RAW_2021, set(CHARS_2021.keys()), "2021 Census", COL_2021)
val_2021  = detect_value_col(raw_2021)
print(f"Value column (2021): '{val_2021}'")
wide_2021 = pivot_to_wide(raw_2021, CHARS_2021, val_2021)

wide_2021["renter_share"] = (
    wide_2021["renter_households"] / wide_2021["total_households_by_tenure"]
).clip(0, 1)
wide_2021["annual_rent"] = wide_2021["median_monthly_shelter_rented"] * 12
wide_2021["home_price"]  = wide_2021["median_dwelling_value"]
wide_2021["ctuid"]       = wide_2021["DGUID"]

# Extract short CT code for joining with 2016 data
# '2021S05075350001.00' → '5350001.00' (last 10 characters)
wide_2021["ct_code"] = wide_2021["ctuid"].str[-10:]

AGENT_COLS = [
    "ctuid", "ct_code", "GEO_NAME",
    "population_2021", "total_private_dwellings", "total_households",
    "median_household_income", "median_aftertax_income",
    "renter_share", "owner_households", "renter_households",
    "home_price", "annual_rent",
    "median_monthly_shelter_owned", "median_monthly_shelter_rented",
    "median_dwelling_value",
]
agents = wide_2021[[c for c in AGENT_COLS if c in wide_2021.columns]]
agents = agents.dropna(subset=["median_household_income"])

print(f"\nAgent init table: {len(agents)} Census Tracts")
print(f"Sample ct_code values: {agents['ct_code'].head(3).tolist()}")
agents.to_csv(OUT_AGENTS, index=False)
print(f"Saved → {OUT_AGENTS}")


# ══════════════════════════════════════════════════════════════════════════════
# 2016 DATA
# ══════════════════════════════════════════════════════════════════════════════

CHARS_2016 = {
    1:    "population_2016",
    4:    "total_private_dwellings_2016",
    5:    "occupied_dwellings_2016",
    742:  "median_household_income_2016",
    743:  "median_aftertax_income_2016",
    1617: "total_households_by_tenure_2016",
    1618: "owner_households_2016",
    1619: "renter_households_2016",
    1674: "median_monthly_shelter_owned_2016",
    1676: "median_dwelling_value_2016",
    1681: "median_monthly_shelter_rented_2016",
}

if not RAW_2016.exists():
    print(f"\n⚠  2016 file not found at {RAW_2016}")
else:
    raw_2016  = read_toronto_cts(RAW_2016, set(CHARS_2016.keys()), "2016 Census", COL_2016)
    val_2016  = detect_value_col(raw_2016)
    print(f"Value column (2016): '{val_2016}'")
    wide_2016 = pivot_to_wide(raw_2016, CHARS_2016, val_2016)

    # 2016 DGUID is already the short code e.g. '5350001.00' — use directly as join key
    wide_2016["ct_code"] = wide_2016["DGUID"]

    wide_2016["renter_share_2016"] = (
        wide_2016["renter_households_2016"] /
        wide_2016["total_households_by_tenure_2016"]
    ).clip(0, 1)
    wide_2016["annual_rent_2016"] = wide_2016["median_monthly_shelter_rented_2016"] * 12
    wide_2016["home_price_2016"]  = wide_2016["median_dwelling_value_2016"]

    print(f"Sample 2016 ct_code values: {wide_2016['ct_code'].head(3).tolist()}")

    # Join on ct_code (short numeric CT identifier, same in both years)
    calib = pd.merge(
        agents,
        wide_2016[[
            "ct_code", "population_2016", "occupied_dwellings_2016",
            "median_household_income_2016", "renter_share_2016",
            "home_price_2016", "annual_rent_2016",
        ]],
        on="ct_code",
        how="inner",
    )

    n_before  = len(agents)
    n_after   = len(calib)
    n_dropped = n_before - n_after
    print(f"\nCT join: {n_before} 2021 CTs → {n_after} matched "
          f"({n_dropped} dropped — boundary changes or suppression)")

    calib["units_2021"]  = calib["total_private_dwellings"]
    calib["units_2016"]  = calib["occupied_dwellings_2016"]
    calib["delta_units"] = calib["units_2021"] - calib["units_2016"]

    calib["income_growth"] = (
        calib["median_household_income"] - calib["median_household_income_2016"]
    ) / calib["median_household_income_2016"].clip(lower=1)

    calib["rent_growth"] = (
        calib["annual_rent"] - calib["annual_rent_2016"]
    ) / calib["annual_rent_2016"].clip(lower=1)

    calib["home_price_growth"] = (
        calib["home_price"] - calib["home_price_2016"]
    ) / calib["home_price_2016"].clip(lower=1)

    calib["renter_share_change"] = calib["renter_share"] - calib["renter_share_2016"]

    DEV_THRESHOLD_ABS = 10
    DEV_THRESHOLD_PCT = 0.01
    calib["dev_occurred"] = (
        (calib["delta_units"] > DEV_THRESHOLD_ABS) |
        (calib["delta_units"] / calib["units_2016"].clip(lower=1) > DEV_THRESHOLD_PCT)
    ).astype(int)

    print(f"\nStage 1 target distribution:")
    print(calib["dev_occurred"].value_counts())
    print(f"  → {calib['dev_occurred'].mean()*100:.1f}% of CTs saw significant development")
    print(f"\ndelta_units summary:")
    print(calib["delta_units"].describe())

    calib.to_csv(OUT_CALIB, index=False)
    print(f"\nSaved → {OUT_CALIB}")

    print(f"""
── Documentation notes ───────────────────────────────────────────────────
Data sources:
  2021: Statistics Canada, Census Profile 2021, Catalogue no. 98-401-X2021007
  2016: Statistics Canada, Census Profile 2016, Catalogue no. 98-401-X2016043

Calibration:
  delta_units = total_private_dwellings_2021 - occupied_dwellings_2016
  Stage 1 target (dev_occurred=1): delta_units > {DEV_THRESHOLD_ABS} OR growth > {DEV_THRESHOLD_PCT*100:.0f}%
  Stage 2 target (units added):    delta_units for CTs where dev_occurred=1

Known limitations to document:
  1. {n_dropped} CTs excluded due to CT boundary changes between 2016 and 2021
  2. delta_units slightly upward-biased: 2021 includes vacant units, 2016 does not
  3. Transit proximity indicator added separately (compute_transit_indicator.py)
─────────────────────────────────────────────────────────────────────────────
    """)