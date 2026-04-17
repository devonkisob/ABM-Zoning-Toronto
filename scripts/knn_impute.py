"""
knn_impute.py
-------------
Spatial KNN imputation for suppressed CT census values.

Usage:
    from scripts.knn_impute import knn_impute_spatial
    df = knn_impute_spatial(df, cols=['home_price', 'annual_rent'], k=5)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import geopandas as gpd
from src.paths import BOUNDARY_SHP

TORONTO_CMA  = "535"
CRS_PROJECTED = "EPSG:32617"
K_DEFAULT    = 5


def knn_impute_spatial(df: pd.DataFrame,
                       cols: list,
                       k: int = K_DEFAULT) -> pd.DataFrame:
    """
    Impute missing values in `cols` using the mean of the K
    geographically nearest CTs that have non-missing values.

    Args:
        df   — dataframe with 'ctuid' column and columns to impute
        cols — list of column names to impute
        k    — number of nearest neighbours (default 5)

    Returns:
        df with missing values in cols filled
    """
    print(f"\nApplying spatial KNN imputation (K={k}) for: {cols}")

    # Load CT boundary shapefile and compute centroids
    gdf = gpd.read_file(BOUNDARY_SHP).to_crs(CRS_PROJECTED)
    gdf = gdf[gdf["DGUID"].str.contains(TORONTO_CMA, na=False)].copy()
    gdf["cx"] = gdf.geometry.centroid.x
    gdf["cy"] = gdf.geometry.centroid.y
    gdf = gdf[["DGUID", "cx", "cy"]].rename(columns={"DGUID": "ctuid"})

    df = df.merge(gdf, on="ctuid", how="left")
    coords = df[["cx", "cy"]].values

    for col in cols:
        missing_mask = df[col].isna()
        n_missing = missing_mask.sum()

        if n_missing == 0:
            print(f"  {col}: no missing values")
            continue

        print(f"  {col}: imputing {n_missing} missing values...")

        for idx in df.index[missing_mask]:
            loc = df.index.get_loc(idx)
            ct_coord = coords[loc]

            if np.any(np.isnan(ct_coord)):
                df.at[idx, col] = df[col].median()
                continue

            dists = np.sqrt(
                (coords[:, 0] - ct_coord[0]) ** 2 +
                (coords[:, 1] - ct_coord[1]) ** 2
            )

            has_value = ~df[col].isna()
            dists_masked = np.where(has_value, dists, np.inf)
            dists_masked[loc] = np.inf  # exclude self

            k_idx = np.argsort(dists_masked)[:k]
            vals  = df[col].iloc[k_idx].dropna()

            df.at[idx, col] = vals.mean() if len(vals) > 0 else df[col].median()

        print(f"  {col}: done (remaining NaN: {df[col].isna().sum()})")

    df = df.drop(columns=["cx", "cy"], errors="ignore")
    return df