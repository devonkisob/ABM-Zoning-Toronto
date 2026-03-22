"""
src/
----
Toronto Missing Middle Zoning ABM — source package.

Public API:
    from src import load_agents, run_scenario, run_all_scenarios
    from src.config import SimConfig, DEFAULT_CONFIG
    from src.paths import PROCESSED_DIR, FIGURES_DIR
"""

from src.agents import (
    CensusTractAgent,
    PolicyModel,
    DemandAllocationModel,
    InfrastructureModel,
    load_agents,
    get_transit_ctuids,
)

from src.calibration import (
    load_models,
    predict_development,
)

from src.simulation import (
    run_scenario,
    run_all_scenarios,
    summarise,
    SCENARIOS,
)

from src.config import SimConfig, DEFAULT_CONFIG
from src.paths import (
    PROCESSED_DIR,
    FIGURES_DIR,
    AGENTS_CSV,
    CALIB_CSV,
    TRANSIT_CSV,
)

__all__ = [
    # Agents
    "CensusTractAgent",
    "PolicyModel",
    "DemandAllocationModel",
    "InfrastructureModel",
    "load_agents",
    "get_transit_ctuids",
    # Calibration
    "load_models",
    "predict_development",
    # Simulation
    "run_scenario",
    "run_all_scenarios",
    "summarise",
    "SCENARIOS",
    # Config and paths
    "SimConfig",
    "DEFAULT_CONFIG",
    "PROCESSED_DIR",
    "FIGURES_DIR",
    "AGENTS_CSV",
    "CALIB_CSV",
    "TRANSIT_CSV",
]