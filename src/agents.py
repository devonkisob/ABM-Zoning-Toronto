"""
agents.py
---------
Defines all agent classes for the Toronto Missing Middle Zoning ABM.

Agents:
    CensusTractAgent   — primary agent; represents a neighbourhood CT
    PolicyModel        — pseudo-agent; applies zoning rules by scenario
    DemandAllocationModel — pseudo-agent; distributes city-wide housing demand
    InfrastructureModel   — pseudo-agent; updates infra strain per CT

Usage:
    from src.agents import CensusTractAgent, load_agents
    agents = load_agents()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from src.paths import AGENTS_CSV, TRANSIT_CSV, CALIB_CSV



# ══════════════════════════════════════════════════════════════════════════════
# CENSUS TRACT AGENT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CensusTractAgent:
    """
    Represents a single Census Tract (CT) as an agent in the simulation.

    Initialized from 2021 StatCan Census data (ct_agents_init.csv) and
    TTC transit proximity indicators (ct_transit_indicators.csv).

    Attributes are divided into:
      - Static:  set at initialization, do not change during simulation
      - Dynamic: updated each time step by pseudo-agent models
    """

    # ── Static attributes (from census / transit data) ─────────────────────
    ctuid:             str    # Statistics Canada CT identifier (DGUID)
    geo_name:          str    # CT name/label
    population:        float  # 2021 population
    households:        int    # 2021 total private households
    median_income:     float  # 2021 median total household income ($)
    transit_indicator: float  # normalized rapid transit proximity (0–1)

    # 2016 baseline values (used as ML features in predict_development)
    median_income_2016:  float = 0.0
    home_price_2016:     float = 0.0
    annual_rent_2016:    float = 0.0
    renter_share_2016:   float = 0.0
    income_growth:       float = 0.0
    home_price_growth:   float = 0.0
    rent_growth:         float = 0.0

    # ── Dynamic attributes (evolve during simulation) ──────────────────────
    units_total:    int   = 0      # total private dwellings
    units_rent:     int   = 0      # rental units
    units_own:      int   = 0      # ownership units
    vacancy_rate:   float = 0.05   # vacancy rate (0–1); Toronto avg ~5%
    home_price:     float = 0.0    # current median dwelling value ($)
    annual_rent:    float = 0.0    # current annualised median rent ($)
    renter_share:   float = 0.0    # current fraction of renter households

    # Demand and infrastructure state
    demand_pressure:  float = 0.0  # allocated housing demand (households/yr)
    infra_capacity:   float = 1.0  # infrastructure capacity index
    infra_load:       float = 0.0  # infrastructure load index
    strain:           float = 0.0  # infra_load / infra_capacity

    # Baseline units (set post-init, used for strain proxy)
    baseline_units: int = field(init=False, default=0)

    def __post_init__(self):
        self.baseline_units = max(self.units_total, 1)

    # ── Behaviour: apply development ───────────────────────────────────────
    def apply_development(self, units_added: int) -> None:
        """
        Add new housing units to the CT, split by tenure.

        Tenure split follows current renter_share, clipped to [0.2, 0.8]
        to prevent degenerate all-rental or all-ownership outcomes.

        Input model:
            units_rent_added = round(units_added * clip(renter_share, 0.2, 0.8))
            units_own_added  = units_added - units_rent_added
        """
        units_added = int(max(0, round(units_added)))
        if units_added == 0:
            return

        rent_share  = float(np.clip(self.renter_share, 0.2, 0.8))
        rent_added  = int(round(units_added * rent_share))
        own_added   = units_added - rent_added

        self.units_total += units_added
        self.units_rent  += rent_added
        self.units_own   += own_added

    # ── Behaviour: update market (prices, rents, vacancy) ─────────────────
    def update_market(self, price_kappa: float, rent_kappa: float,
                      v_star: float) -> None:
        """
        Update home price, rent, and vacancy rate based on supply-demand gap.

        Input model:
            vacancy responds to demand pressure relative to units
            prices/rents respond to vacancy gap from target vacancy v_star

            demand_per_unit = demand_pressure / max(units_total, 1)
            vacancy_rate   += 0.02 - 0.5 * demand_per_unit   (clipped 0–0.25)
            home_price     *= 1 + price_kappa * (v_star - vacancy_rate)
            annual_rent    *= 1 + rent_kappa  * (v_star - vacancy_rate)

        Parameters:
            price_kappa — price elasticity w.r.t. vacancy gap (default 0.05)
            rent_kappa  — rent elasticity w.r.t. vacancy gap  (default 0.04)
            v_star      — target vacancy rate (default 0.03, Toronto avg)

        Justification: vacancy-price relationship follows standard hedonic
        housing model structure. Parameters are placeholder values to be
        calibrated in sensitivity analysis for the Final Report.
        """
        # Vacancy rises when supply exceeds demand, falls when demand exceeds supply
        demand_per_unit = self.demand_pressure / max(self.units_total, 1)

        self.vacancy_rate = float(np.clip(
            self.vacancy_rate - 0.02 + 0.5 * demand_per_unit,
            0.0, 0.25
        ))

        self.home_price  *= (1.0 + price_kappa * (v_star - self.vacancy_rate))
        self.annual_rent *= (1.0 + rent_kappa  * (v_star - self.vacancy_rate))

    # ── Behaviour: update infrastructure ──────────────────────────────────
    def update_infrastructure(self, omega0: float, omega1: float) -> None:
        """
        Update infrastructure load and strain based on housing density.

        Input model:
            density_proxy = units_total / max(households, 1)
            infra_load    = omega0 + omega1 * density_proxy
            strain        = infra_load / max(infra_capacity, 1e-6)

        Parameters:
            omega0 — baseline infrastructure load (default 0.5)
            omega1 — load sensitivity to density  (default 0.3)

        Note: infra_capacity is updated by InfrastructureModel each step.
        """
        density_proxy  = self.units_total / max(self.households, 1)
        self.infra_load = omega0 + omega1 * density_proxy
        self.strain     = self.infra_load / max(self.infra_capacity, 1e-6)

    # ── Output metrics: affordability indices ──────────────────────────────
    def affordability_own(self) -> float:
        """
        Ownership Affordability Index = median_income / home_price.
        Higher values indicate more affordable ownership.
        """
        return self.median_income / max(self.home_price, 1.0)

    def affordability_rent(self) -> float:
        """
        Rental Affordability Index = median_income / annual_rent.
        Higher values indicate more affordable renting.
        """
        return self.median_income / max(self.annual_rent, 1.0)

    def __repr__(self) -> str:
        return (f"CensusTractAgent(ctuid={self.ctuid}, "
                f"units={self.units_total}, "
                f"AI_own={self.affordability_own():.3f}, "
                f"AI_rent={self.affordability_rent():.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# POLICY MODEL (pseudo-agent)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyModel:
    """
    Defines zoning rules and incentive levels for a given scenario.

    Scenarios:
        S0 — Baseline: status quo, no missing-middle zoning
        S1 — City-wide missing-middle zoning
        S2 — Targeted missing-middle near rapid transit (near_rapid_500m == 1)
        S3 — Incentive-based reform (S1 + reduced development charges)

    Attributes:
        scenario_name    — one of {"S0", "S1", "S2", "S3"}
        eligible_ctuids  — set of CTUIDs eligible for missing-middle development
        incentive_level  — development cost multiplier reduction (0.0 = none)
    """
    scenario_name:   str
    eligible_ctuids: set
    incentive_level: float = 0.0   # 0.0 = no incentive; 1.0 = max incentive

    def is_eligible(self, ctuid: str) -> bool:
        """Return True if this CT is eligible for missing-middle development."""
        return ctuid in self.eligible_ctuids

    @classmethod
    def from_scenario(cls, scenario: str,
                      all_ctuids: list,
                      transit_ctuids: set) -> "PolicyModel":
        """
        Factory method — create a PolicyModel for a given scenario.

        Args:
            scenario:       "S0", "S1", "S2", or "S3"
            all_ctuids:     list of all Toronto CTUIDs
            transit_ctuids: set of CTUIDs with near_rapid_500m == 1
        """
        if scenario == "S0":
            # Baseline: no CTs eligible for missing-middle
            return cls(
                scenario_name="S0",
                eligible_ctuids=set(),
                incentive_level=0.0,
            )
        elif scenario == "S1":
            # City-wide: all CTs eligible
            return cls(
                scenario_name="S1",
                eligible_ctuids=set(all_ctuids),
                incentive_level=0.0,
            )
        elif scenario == "S2":
            # Transit-targeted: only CTs near rapid transit eligible
            return cls(
                scenario_name="S2",
                eligible_ctuids=transit_ctuids,
                incentive_level=0.0,
            )
        elif scenario == "S3":
            # Incentive-based: all CTs eligible + development cost reduction
            return cls(
                scenario_name="S3",
                eligible_ctuids=set(all_ctuids),
                incentive_level=0.5,   # 50% incentive boost to p_dev
            )
        else:
            raise ValueError(f"Unknown scenario '{scenario}'. "
                             f"Must be one of S0, S1, S2, S3.")


# ══════════════════════════════════════════════════════════════════════════════
# DEMAND ALLOCATION MODEL (pseudo-agent)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DemandAllocationModel:
    """
    Distributes city-wide housing demand across CT agents each time step.

    City-level demand is exogenous and grows at a fixed rate with
    log-normal stochastic shocks (captures uncertainty in household formation).

    Attractiveness: cheaper CTs (lower home_price) attract more demand.
    This is a placeholder — the Final Report will add transit and
    affordability-based attractiveness weights.

    Input model:
        city_demand(t) = total_units * base_demand * (1 + growth_rate)^t * lognormal(0, sigma)
        demand_i = city_demand * (attractiveness_i / sum(attractiveness))
        attractiveness_i = 1 / max(home_price_i, 1)

    Data source for base_demand and growth_rate:
        Canada Mortgage and Housing Corporation (CMHC) Housing Market Outlook,
        Toronto CMA — household formation projections.
        Placeholder value: 15,000 net new households/year (Toronto avg 2016-2021)
    """
    base_demand:  float              # baseline city-wide demand (households/yr)
    demand_growth: float             # annual demand growth rate
    rng:          np.random.Generator

    def city_demand(self, t: int, total_units: int) -> float:
        shock = self.rng.lognormal(mean=0.0, sigma=0.1)
        return total_units * self.base_demand * ((1 + self.demand_growth) ** t) * shock

    def allocate(self, cts: List[CensusTractAgent], t: int) -> None:
        """
        Distribute city demand across CTs proportional to attractiveness.
        Updates ct.demand_pressure in place.
        """
        total_units = sum(ct.units_total for ct in cts)
        total = self.city_demand(t, total_units)
        prices = np.array([ct.home_price for ct in cts], dtype=float)
        attractiveness = 1.0 / np.maximum(prices, 1.0)
        weights = attractiveness / max(attractiveness.sum(), 1e-9)
        for ct, w in zip(cts, weights):
            ct.demand_pressure = float(total * w)


# ══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE MODEL (pseudo-agent)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class InfrastructureModel:
    """
    Updates infrastructure capacity and strain for each CT each time step.

    Capacity grows annually at g_base, reduced by incentive_level (since
    incentives reduce development charges → less municipal revenue for infra).

    Input model:
        g_cap = max(0, g_base - lambda_incent * incentive_level)
        infra_capacity *= (1 + g_cap)
        infra_load = omega0 + omega1 * density_proxy
        strain = infra_load / infra_capacity

    Parameters:
        omega0        — baseline infra load (default 0.5)
        omega1        — load sensitivity to density (default 0.3)
        g_base        — baseline annual infra capacity growth (default 0.01)
        lambda_incent — capacity growth penalty per incentive unit (default 0.005)

    Note: these are placeholder values. Sensitivity analysis in the Final
    Report will test model sensitivity to omega0, omega1, and g_base.
    """
    omega0:        float = 0.5
    omega1:        float = 0.3
    g_base:        float = 0.01
    lambda_incent: float = 0.005

    def step(self, ct: CensusTractAgent, policy: PolicyModel) -> None:
        """Update infrastructure capacity and strain for one CT."""
        g_cap = max(0.0, self.g_base - self.lambda_incent * policy.incentive_level)
        ct.infra_capacity *= (1.0 + g_cap)
        ct.update_infrastructure(self.omega0, self.omega1)


# ══════════════════════════════════════════════════════════════════════════════
# LOADER — initialize agents from processed CSV files
# ══════════════════════════════════════════════════════════════════════════════

def load_agents(agents_csv=AGENTS_CSV, transit_csv=TRANSIT_CSV) -> List[CensusTractAgent]:
    """
    Load and initialize all CensusTractAgent objects from processed data.

    Merges ct_agents_init.csv with ct_transit_indicators.csv and
    ct_calibration.csv (for 2016 baseline values used as ML features).

    Returns:
        list of CensusTractAgent — one per Toronto CT (~1220 agents)
    """
    agents_df  = pd.read_csv(agents_csv)
    transit_df = pd.read_csv(transit_csv)[["ctuid", "transit_indicator",
                                            "near_rapid_500m"]]

    # Try to merge 2016 baseline values from calibration table if available
    calib_path = CALIB_CSV
    if calib_path.exists():
        calib_df = pd.read_csv(calib_path)[[
            "ctuid", "median_household_income_2016",
            "home_price_2016", "annual_rent_2016", "renter_share_2016",
            "income_growth", "home_price_growth", "rent_growth",
        ]]
        agents_df = agents_df.merge(calib_df, on="ctuid", how="left")

    df = agents_df.merge(transit_df, on="ctuid", how="left")
    df["transit_indicator"] = df["transit_indicator"].fillna(0.0)
    df["near_rapid_500m"]   = df["near_rapid_500m"].fillna(0).astype(int)

    # Impute median for suppressed CT values before agent initialization
    median_home_price  = df["home_price"].median()
    median_annual_rent = df["annual_rent"].median()
    df["home_price"]   = df["home_price"].fillna(median_home_price)
    df["annual_rent"]  = df["annual_rent"].fillna(median_annual_rent)

    agents = []
    for _, row in df.iterrows():
        ct = CensusTractAgent(
            # identifiers
            ctuid         = str(row["ctuid"]),
            geo_name      = str(row.get("GEO_NAME", "")),
            # static census attributes
            population    = float(row.get("population_2021", 0)),
            households    = int(row.get("total_households", 1)),
            median_income = float(row.get("median_household_income", 0)),
            transit_indicator = float(row.get("transit_indicator", 0.0)),
            # 2016 baseline (for ML features)
            median_income_2016 = float(row.get("median_household_income_2016", 0)),
            home_price_2016    = float(row.get("home_price_2016", 0)),
            annual_rent_2016   = float(row.get("annual_rent_2016", 0)),
            renter_share_2016  = float(row.get("renter_share_2016", 0)),
            income_growth      = float(row.get("income_growth", 0)),
            home_price_growth  = float(row.get("home_price_growth", 0)),
            rent_growth        = float(row.get("rent_growth", 0)),
            # dynamic housing state (initialized from 2021 census)
            units_total   = int(row.get("total_private_dwellings", 0)),
            units_rent    = int(row.get("renter_households", 0)),
            units_own     = int(row.get("owner_households", 0)),
            vacancy_rate  = 0.05,   # Toronto avg vacancy rate ~5% (CMHC 2021)
            home_price    = float(row.get("home_price", 0)),
            annual_rent   = float(row.get("annual_rent", 0)),
            renter_share  = float(row.get("renter_share", 0)),
        )
        agents.append(ct)

    print(f"Loaded {len(agents)} CensusTractAgent objects")
    return agents


def get_transit_ctuids(transit_csv: Path = TRANSIT_CSV) -> set:
    """
    Return the set of CTUIDs with near_rapid_500m == 1.
    Used by PolicyModel.from_scenario() for Scenario 2.
    """
    df = pd.read_csv(transit_csv)
    return set(df.loc[df["near_rapid_500m"] == 1, "ctuid"].astype(str))