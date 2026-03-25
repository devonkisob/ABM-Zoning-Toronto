# Toronto Missing Middle ABM

A census-tract-based agent-based simulation of housing affordability under alternative missing-middle zoning policies in Toronto.

## Team
- Devon Kisob
- Chanuth Weeraratna
- Kevin Kim

## Research Question
For the City of Toronto, how do alternative missing-middle zoning implementations change the evolution of housing affordability across census tracts over a 10-year horizon, under infrastructure capacity constraints and uncertain market response?

## Workflow
<img width="2688" height="1000" alt="SYDE 535 Workflow Diagram" src="https://github.com/user-attachments/assets/a118f40d-1730-4239-ad82-7bbe9534fca8" />

## Scenarios
| ID | Description |
|----|-------------|
| S0 | Baseline — status quo zoning, no missing-middle reform |
| S1 | City-wide missing-middle zoning across all CTs |
| S2 | Transit-targeted missing-middle (CTs within 500m of rapid transit only) |
| S3 | Incentive-based reform (city-wide zoning + reduced development charges) |

## Project Structure
```
ABM-Zoning-Toronto/
├── data/
│   ├── raw/                  # Raw census and GTFS files (not in git — see data/README.md)
│   ├── interim/              # Intermediate files
│   └── processed/            # Agent initialization, calibration, and results CSVs
├── src/
│   ├── agents.py             # CensusTractAgent and pseudo-agent classes
│   ├── calibration.py        # Two-stage ML development model
│   ├── simulation.py         # Monte Carlo simulation loop
│   ├── config.py             # Global configuration
│   └── paths.py              # Path constants
├── scripts/
│   ├── preprocess_census.py          # StatCan 2021 + 2016 census preprocessing
│   ├── compute_transit_indicator.py  # TTC GTFS transit proximity per CT
│   └── run_simulation.py             # Run all scenarios (parallel)
├── notebooks/
│   └── milestone2_results.ipynb     # M2 baseline and scenario comparison figures
├── results/
│   ├── figures/              # Generated plots
│   └── tables/               # Summary statistics
└── docs/
    └── notes.md              # Development notes
    └── SYDE 535 - Project Pitch.pdf              # Original Pitch
    └── SYDE 535 Milestone 1.pdf              # Progress Update 1
    └── SYDE 535 Milestone 2.pdf              # Progress Update 2
```

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download raw data
See [`data/README.md`](https://github.com/devonkisob/ABM-Zoning-Toronto/blob/main/data/README.md) for download instructions. Place files in `data/raw/`.

### 3. Preprocess census data
```bash
python scripts/preprocess_census.py
python scripts/compute_transit_indicator.py
```

### 4. Run simulation
```bash
python scripts/run_simulation.py           # N=100, parallel (default)
python scripts/run_simulation.py --fast    # N=10, quick test
python scripts/run_simulation.py --full    # N=500, final report
```

### 5. View results
Open [`notebooks/milestone2_results.ipynb`](https://github.com/devonkisob/ABM-Zoning-Toronto/blob/main/notebooks/milestone2_results.ipynb) in Jupyter or VS Code.

## Current Status
**Milestone 2 complete** (see [`docs/SYDE 535 Milestone 2.pdf`](https://github.com/devonkisob/ABM-Zoning-Toronto/blob/main/docs/SYDE%20535%20Milestone%202.pdf))
- Data pipeline: StatCan 2021/2016 census + TTC GTFS → 1,220 Toronto CT agents
- ML calibration: two-stage logistic + linear regression (Stage 1 F1=0.73, Stage 2 R²=0.31)
- Simulation: N=100 Monte Carlo realisations, T=40 quarterly steps (10 years)
- Results: baseline trajectories and S0 vs S1 scenario comparison figures generated
