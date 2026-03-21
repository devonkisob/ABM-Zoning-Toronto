"""
compute_transit_indicator.py
-----------------------------
Computes a transit proximity indicator for each Toronto Census Tract
using TTC GTFS stop locations and StatCan CT boundary shapefiles.

The indicator is used as a feature in the ML calibration model and to
define eligibility under Scenario 2 (transit-targeted zoning).

Usage:
    python scripts/compute_transit_indicator.py

Inputs:
    data/raw/ttc_gtfs/stops.txt                     (from TTC GTFS zip)
    data/raw/ttc_gtfs/routes.txt                    (from TTC GTFS zip)
    data/raw/ttc_gtfs/stop_times.txt                (from TTC GTFS zip)
    data/raw/ttc_gtfs/trips.txt                     (from TTC GTFS zip)
    data/raw/ct_boundaries/lct_000b21a_e.shp        (StatCan 2021 CT boundaries)
    data/processed/ct_agents_init.csv               (to get Toronto CTUID list)

Outputs:
    data/processed/ct_transit_indicators.csv

Data sources to cite in M2:
    TTC GTFS: City of Toronto Open Data, "TTC Routes and Schedules"
              https://open.toronto.ca/dataset/ttc-routes-and-schedules/
              Licensed under Open Government Licence - Toronto

    CT Boundaries: Statistics Canada, 2021 Census - Boundary Files
                   https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/
                   Catalogue no. 92-160-X

Installing dependencies (add to requirements.txt):
    geopandas
    shapely
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
GTFS_DIR      = Path("data/raw/ttc_gtfs")
BOUNDARY_SHP  = Path("data/raw/ct_boundaries/lct_000b21a_e/lct_000b21a_e.shp")
AGENTS_CSV    = Path("data/processed/ct_agents_init.csv")
OUT_CSV       = Path("data/processed/ct_transit_indicators.csv")

# ── Parameters ─────────────────────────────────────────────────────────────────
# Radius for "near transit" — 500m is standard TOD (transit-oriented development)
# planning threshold used by City of Toronto Official Plan.
PROXIMITY_RADIUS_M = 500

# For the continuous score, we also count stops within a wider radius.
SCORE_RADIUS_M = 800

# Projected CRS for accurate distance calculations in metres (Ontario)
CRS_PROJECTED = "EPSG:32617"   # UTM Zone 17N


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load TTC stops and classify by service type
# ══════════════════════════════════════════════════════════════════════════════

print("Loading TTC GTFS data...")
stops      = pd.read_csv(GTFS_DIR / "stops.txt")
routes     = pd.read_csv(GTFS_DIR / "routes.txt")
trips      = pd.read_csv(GTFS_DIR / "trips.txt")
stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt", usecols=["trip_id", "stop_id"])

print(f"  {len(stops)} total TTC stops")
print(f"  {len(routes)} routes")

# GTFS route_type codes:
#   0 = Tram/Streetcar
#   1 = Subway/Metro
#   3 = Bus
# For "transit hubs" in Scenario 2, we want subway + high-frequency surface routes.
# We define two tiers:

# Tier 1: Subway stations (route_type == 1)
# Tier 2: All stops on routes with route_type 0 or 1 (subway + streetcar)
# Tier 3: All stops (every TTC stop including bus)

route_types = routes[["route_id", "route_type", "route_short_name"]].copy()

# Join route type through trips → stop_times → stops
trips_routes = trips[["trip_id", "route_id"]].merge(route_types, on="route_id")
stop_routes  = stop_times.merge(trips_routes, on="trip_id")

# For each stop, get the set of route types that serve it
stop_service = (
    stop_routes.groupby("stop_id")["route_type"]
    .apply(lambda x: set(x))
    .reset_index()
    .rename(columns={"route_type": "route_types_set"})
)

stops = stops.merge(stop_service, on="stop_id", how="left")
stops["is_subway"]     = stops["route_types_set"].apply(
    lambda x: 1 if isinstance(x, set) and 1 in x else 0
)
stops["is_rapid"]      = stops["route_types_set"].apply(
    lambda x: 1 if isinstance(x, set) and (0 in x or 1 in x) else 0
)

print(f"  Subway stops:            {stops['is_subway'].sum()}")
print(f"  Rapid transit stops:     {stops['is_rapid'].sum()}")
print(f"  All stops:               {len(stops)}")

# Convert stops to GeoDataFrame
gdf_stops = gpd.GeoDataFrame(
    stops,
    geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
    crs="EPSG:4326",
).to_crs(CRS_PROJECTED)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load CT boundaries and filter to Toronto CTs
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading CT boundary shapefile...")

if not BOUNDARY_SHP.exists():
    print(f"""
⚠  CT boundary shapefile not found at {BOUNDARY_SHP}

To download:
  1. Go to: https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index-eng.cfm
  2. Select: Census tracts > 2021 > Shapefile
  3. Unzip and place contents in data/raw/ct_boundaries/

The main file should be named: lct_000b21a_e.shp
""")
    raise FileNotFoundError(f"Missing: {BOUNDARY_SHP}")

gdf_cts = gpd.read_file(BOUNDARY_SHP).to_crs(CRS_PROJECTED)
print(f"  Loaded {len(gdf_cts)} CTs nationally")

# Filter to Toronto CTs using our agent list (already filtered to Toronto)
agents    = pd.read_csv(AGENTS_CSV)
toronto_ctuids = set(agents["ctuid"].str.strip())

# The shapefile uses DGUID as identifier
gdf_toronto = gdf_cts[gdf_cts["DGUID"].isin(toronto_ctuids)].copy()
print(f"  Toronto CTs after filter: {len(gdf_toronto)}")

# Compute CT centroids for proximity calculation
gdf_toronto["centroid"] = gdf_toronto.geometry.centroid


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Compute transit indicators per CT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nComputing transit proximity indicators...")
print(f"  Proximity radius (binary flag): {PROXIMITY_RADIUS_M}m")
print(f"  Score radius (count):           {SCORE_RADIUS_M}m")

# Buffer each CT centroid and count stops within
subway_stops  = gdf_stops[gdf_stops["is_subway"] == 1]
rapid_stops   = gdf_stops[gdf_stops["is_rapid"] == 1]
all_stops     = gdf_stops

results = []
for _, ct in gdf_toronto.iterrows():
    centroid = ct["centroid"]

    # Distance from CT centroid to each stop type
    dist_subway = subway_stops.geometry.distance(centroid)
    dist_rapid  = rapid_stops.geometry.distance(centroid)
    dist_all    = all_stops.geometry.distance(centroid)

    # Binary flags: is any stop within radius?
    near_subway = int((dist_subway <= PROXIMITY_RADIUS_M).any())
    near_rapid  = int((dist_rapid  <= PROXIMITY_RADIUS_M).any())

    # Continuous scores: count stops within score radius (normalized later)
    n_subway_800  = int((dist_subway <= SCORE_RADIUS_M).sum())
    n_rapid_800   = int((dist_rapid  <= SCORE_RADIUS_M).sum())
    n_all_800     = int((dist_all    <= SCORE_RADIUS_M).sum())

    # Distance to nearest stop of each type (metres)
    nearest_subway = dist_subway.min() if len(dist_subway) > 0 else float("nan")
    nearest_rapid  = dist_rapid.min()  if len(dist_rapid)  > 0 else float("nan")

    results.append({
        "ctuid":              ct["DGUID"],
        # Binary indicators (use for Scenario 2 eligibility)
        "near_subway_500m":   near_subway,
        "near_rapid_500m":    near_rapid,
        # Counts within 800m (use as ML feature)
        "n_subway_stops_800m": n_subway_800,
        "n_rapid_stops_800m":  n_rapid_800,
        "n_all_stops_800m":    n_all_800,
        # Distance to nearest (use as ML feature)
        "dist_nearest_subway_m": nearest_subway,
        "dist_nearest_rapid_m":  nearest_rapid,
    })

out = pd.DataFrame(results)

# Normalise count scores to 0–1 range
for col in ["n_subway_stops_800m", "n_rapid_stops_800m", "n_all_stops_800m"]:
    max_val = out[col].max()
    out[f"{col}_norm"] = out[col] / max_val if max_val > 0 else 0.0

# The main transit_indicator used in simulation and ML:
# continuous score based on rapid transit stops within 800m (normalized)
out["transit_indicator"] = out["n_rapid_stops_800m_norm"]

print(f"\nResults summary:")
print(f"  CTs near a subway station (500m):       {out['near_subway_500m'].sum()} "
      f"({out['near_subway_500m'].mean()*100:.1f}%)")
print(f"  CTs near rapid transit (500m):          {out['near_rapid_500m'].sum()} "
      f"({out['near_rapid_500m'].mean()*100:.1f}%)")
print(f"  Mean transit_indicator:                 {out['transit_indicator'].mean():.3f}")
print(f"  transit_indicator distribution:")
print(out["transit_indicator"].describe())

out.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")

print("""
── For your M2 write-up ──────────────────────────────────────────────────────
Transit proximity input model:

  transit_indicator (continuous, 0–1):
    Number of rapid transit stops (subway + streetcar) within 800m of CT
    centroid, normalized by the maximum value across all Toronto CTs.
    Used as a feature in the Stage 1 and Stage 2 ML calibration models.

  near_rapid_500m (binary, 0/1):
    1 if any rapid transit stop falls within 500m of CT centroid.
    Used to define zoning eligibility under Scenario 2.
    The 500m threshold follows City of Toronto Official Plan guidelines
    for transit-oriented development (TOD) planning areas.

  Data source: City of Toronto Open Data, "TTC Routes and Schedules" (GTFS),
    licensed under the Open Government Licence - Toronto.
    https://open.toronto.ca/dataset/ttc-routes-and-schedules/

  CT centroids computed from Statistics Canada 2021 Census Boundary Files
    (Catalogue no. 92-160-X), projected to UTM Zone 17N (EPSG:32617)
    for accurate metre-based distance calculations.
──────────────────────────────────────────────────────────────────────────────
""")