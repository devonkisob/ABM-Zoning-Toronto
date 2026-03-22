# Development Notes

## Simulation Parameters (Milestone 2)
| Parameter | Value | Notes |
|-----------|-------|-------|
| N realisations | 100 | Increase to 500 for Final Report |
| T time steps | 40 | 10 years × 4 quarters |
| Base demand | 2% of housing stock/quarter | Relative to total CT units |
| Demand growth | 0.25%/quarter | ~1%/year |
| Price kappa | 0.15 | Placeholder — calibrate in sensitivity analysis |
| Rent kappa | 0.10 | Placeholder — calibrate in sensitivity analysis |
| Target vacancy | 0.03 | CMHC Toronto avg |
| p_dev scaling | 1/40 | Scales annual ML probability to quarterly |
| S3 incentive boost | 0.5 | 50% boost to p_dev for eligible CTs |

## Known Limitations (document in M2)
- **Stage 1 class imbalance:** 92% of CTs had dev_occurred=1 in calibration data; balanced class weights used
- **Stage 2 R²:** Cross-validated R²=0.31 — linear model explains limited variance in delta_units; high-rise outlier CTs dominate
- **128 CTs excluded** from calibration due to CT boundary changes between 2016 and 2021 (inner join)
- **delta_units upward bias:** 2021 uses total private dwellings (includes vacant); 2016 uses occupied dwellings only
- **8 CTs** with suppressed home price values imputed with Toronto CT median ($955,000)
- **11 CTs** with suppressed rent values imputed with Toronto CT median ($18,600/year)
- **Vacancy rate** initialized at 0.05 for all CTs (CMHC Toronto avg); no CT-level vacancy data available from census
- **Median rent** from census reflects existing tenants (including rent-controlled); understates new market rents
- **Transit indicator** uses rapid transit only (subway + streetcar); excludes high-frequency bus routes
- **Market elasticity parameters** (price_kappa, rent_kappa) are placeholders — sensitivity analysis planned for Final Report

## Branch Strategy
- `main` — stable, merged from other branches
- `data` — data pipeline (preprocessing scripts, processed CSVs)
- `sim` — simulation code (agents, calibration, simulation loop)
- `analysis` — notebooks and figures

## TODO for Final Report
- [ ] Sensitivity analysis on price_kappa, rent_kappa, omega0, omega1
- [ ] Add multiprocessing for sensitivity analysis runs (--parallel flag)
- [ ] Refine transit indicator to include TTC Frequent Transit Network
- [ ] Consider random forest for Stage 1/2 and compare to logistic/linear
- [ ] Add spatial affordability heatmaps (geopandas choropleth)
- [ ] Calibrate market elasticity parameters against observed Toronto price trends
- [ ] Run N=500 for final results
