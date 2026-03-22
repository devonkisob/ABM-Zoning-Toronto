"""
paths.py
--------
Central path constants for the project.
Import from here rather than hardcoding paths in scripts or src modules.

Usage:
    from src.paths import PROCESSED_DIR, RESULTS_DIR
"""

from pathlib import Path

# Repository root (two levels up from this file: src/paths.py → src/ → root)
ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Source directories
SRC_DIR       = ROOT / "src"
SCRIPTS_DIR   = ROOT / "scripts"
NOTEBOOKS_DIR = ROOT / "notebooks"

# Results
RESULTS_DIR   = ROOT / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
TABLES_DIR    = RESULTS_DIR / "tables"

# Docs
DOCS_DIR      = ROOT / "docs"

# Key processed files
AGENTS_CSV    = PROCESSED_DIR / "ct_agents_init.csv"
CALIB_CSV     = PROCESSED_DIR / "ct_calibration.csv"
TRANSIT_CSV   = PROCESSED_DIR / "ct_transit_indicators.csv"
STAGE1_PKL    = PROCESSED_DIR / "stage1_model.pkl"
STAGE2_PKL    = PROCESSED_DIR / "stage2_model.pkl"
SCALER_PKL    = PROCESSED_DIR / "feature_scaler.pkl"
CALIB_REPORT  = PROCESSED_DIR / "calibration_report.txt"

# Raw data files
CENSUS_2021   = RAW_DIR / "98-401-X2021007_English_CSV_data.csv"
CENSUS_2016   = RAW_DIR / "98-401-X2016043_English_CSV_data.csv"
BOUNDARY_SHP  = RAW_DIR / "ct_boundaries" / "lct_000b21a_e" / "lct_000b21a_e.shp"
GTFS_DIR      = RAW_DIR / "ttc_gtfs"

# Ensure output directories exist when this module is imported
for _dir in [PROCESSED_DIR, INTERIM_DIR, FIGURES_DIR, TABLES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)