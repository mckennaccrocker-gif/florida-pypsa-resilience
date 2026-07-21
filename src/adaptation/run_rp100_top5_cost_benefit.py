"""Planning-level cost-benefit analysis for the RP100 top-five pilot.

This script reads the completed RP100 top-five flood-adaptation pilot outputs
and estimates illustrative outage-avoidance economics. It does not run PyPSA
or alter any pilot simulation results.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_ROOT = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_ROOT / "data" / "Electricity"
ANALYSIS_ROOT = ELECTRICITY_DIR / "flood_adaptation_analysis"
PILOT_DIR = ANALYSIS_ROOT / "rp100_top5_pilot"
OUTPUT_DIR = ANALYSIS_ROOT / "rp100_top5_cost_benefit"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
NETWORK_PATH = ELECTRICITY_DIR / "pypsa_florida_network" / "florida_network.nc"

SELECTED_ASSETS = PILOT_DIR / "selected_top5_assets.csv"
PROTECTION_DESIGN = PILOT_DIR / "rp100_protection_design.csv"
SCENARIO_COMPARISON = PILOT_DIR / "top5_portfolio_scenario_comparison.csv"
ANNUALIZED_RISK = PILOT_DIR / "pilot_annualized_risk_comparison.csv"
ASSET_DAMAGE = PILOT_DIR / "top5_asset_damage_comparison.csv"
PILOT_INTERPRETATION = PILOT_DIR / "pilot_interpretation.md"


@dataclass(frozen=True)
class EconomicAssumptions:
    """Economic assumptions used in the planning-level calculation."""

    lifetime_years: int
    discount_rate: float
    annual_maintenance_percent_of_capex: float
    annual_benefit_growth_rate: float
    residual_value_percent: float
    low_voll_usd_per_mwh: float
    central_voll_usd_per_mwh: float
    high_voll_usd_per_mwh: float


def setup_logging() -> None:
    """Create output folders and configure logging."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "run_rp100_top5_cost_benefit.log", mode="w", encoding="utf-8"),
        ],
    )


def read_csv_required(path: Path) -> pd.DataFrame:
    """Read a required CSV and fail clearly if it is absent."""
    if not path.exists():
        raise FileNotFoundError(f"Required pilot file is missing: {path}")
    return pd.read_csv(path)


def verify_pilot_results() -> tuple[dict[str, pd.DataFrame], float]:
    """Load pilot outputs and return the verified avoided EENS value."""
    files = {
        "selected_assets": SELECTED_ASSETS,
        "protection_design": PROTECTION_DESIGN,
        "scenario_comparison": SCENARIO_COMPARISON,
        "annualized_risk": ANNUALIZED_RISK,
        "asset_damage": ASSET_DAMAGE,
    }
    tables = {name: read_csv_required(path) for name, path in files.items()}
    if not PILOT_INTERPRETATION.exists():
        raise FileNotFoundError(f"Required pilot interpretation is missing: {PILOT_INTERPRETATION}")

    annualized = tables["annualized_risk"]
    value_col = "pilot_avoided_eens_mwh_per_year"
    if value_col not in annualized.columns or annualized[value_col].dropna().empty:
        raise ValueError(f"Missing verified avoided EENS column in {ANNUALIZED_RISK}: {value_col}")
    avoided_eens = float(annualized[value_col].dropna().iloc[0])
    verification = pd.DataFrame(
        [
            {
                "input_name": name,
                "source_file": str(path),
                "rows": len(tables[name]),
                "status": "loaded",
            }
            for name, path in files.items()
        ]
        + [
            {
                "input_name": "pilot_interpretation",
                "source_file": str(PILOT_INTERPRETATION),
                "rows": np.nan,
                "status": "loaded",
            },
            {
                "input_name": "verified_avoided_eens_mwh_per_year",
                "source_file": str(ANNUALIZED_RISK),
                "rows": np.nan,
                "status": avoided_eens,
            },
        ]
    )
    verification.to_csv(OUTPUT_DIR / "pilot_results_verified.csv", index=False)
    logging.info("Verified avoided EENS: %.4f MWh/year", avoided_eens)
    return tables, avoided_eens


def load_network_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PyPSA network component tables for metadata only."""
    if not NETWORK_PATH.exists():
        logging.warning("Network file not found for metadata: %s", NETWORK_PATH)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        import pypsa
    except ImportError as exc:
        raise ImportError("pypsa is required to read network metadata, but no solves are run.") from exc

    network = pypsa.Network(str(NETWORK_PATH))
    return network.generators.copy(), network.lines.copy(), network.buses.copy()


def bus_xy(bus_id: str, buses: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return x/y coordinates for a bus when available."""
    if not bus_id or bus_id not in buses.index:
        return None, None
    row = buses.loc[bus_id]
    return float(row.get("x", np.nan)), float(row.get("y", np.nan))


def build_asset_characteristics(
    selected: pd.DataFrame,
    design: pd.DataFrame,
    generators: pd.DataFrame,
    lines: pd.DataFrame,
    buses: pd.DataFrame,
) -> pd.DataFrame:
    """Create the modeled asset characteristics table."""
    design_lookup = design.set_index("asset_id")
    rows: list[dict[str, object]] = []
    for _, asset in selected.iterrows():
        asset_id = str(asset["asset_id"])
        asset_type = str(asset["asset_type"])
        base = {
            "asset_id": asset_id,
            "asset_name": asset.get("asset_name", asset_id),
            "asset_type": asset_type,
            "modeled_component": "unknown",
            "modeled_capacity": np.nan,
            "capacity_units": "",
            "associated_bus": "",
            "bus0": "",
            "bus1": "",
            "x": np.nan,
            "y": np.nan,
            "line_length_km": np.nan,
            "voltage_kv": np.nan,
            "line_capacity_mva": np.nan,
            "number_of_segments": np.nan,
            "endpoint0_lon": np.nan,
            "endpoint0_lat": np.nan,
            "endpoint1_lon": np.nan,
            "endpoint1_lat": np.nan,
            "verified_substation_identity": False,
            "RP100_flood_depth_m": np.nan,
            "design_depth_m": np.nan,
            "notes": "",
        }
        if asset_id in design_lookup.index:
            drow = design_lookup.loc[asset_id]
            base["RP100_flood_depth_m"] = float(drow.get("RP100_flood_depth_m", np.nan))
            base["design_depth_m"] = float(drow.get("design_depth_m", np.nan))

        if asset_type == "generator" and asset_id in generators.index:
            grow = generators.loc[asset_id]
            bus = str(grow.get("bus", ""))
            x, y = bus_xy(bus, buses)
            base.update(
                {
                    "modeled_component": "generator facility",
                    "modeled_capacity": float(grow.get("p_nom", np.nan)),
                    "capacity_units": "MW",
                    "associated_bus": bus,
                    "x": x,
                    "y": y,
                    "notes": "Adaptation represents protection of modeled generating facility and associated electrical equipment; no verified substation identity is assigned.",
                }
            )
        elif asset_type == "line" and asset_id in lines.index:
            lrow = lines.loc[asset_id]
            bus0 = str(lrow.get("bus0", ""))
            bus1 = str(lrow.get("bus1", ""))
            x0, y0 = bus_xy(bus0, buses)
            x1, y1 = bus_xy(bus1, buses)
            base.update(
                {
                    "modeled_component": "transmission line",
                    "modeled_capacity": float(lrow.get("s_nom", np.nan)),
                    "capacity_units": "MVA",
                    "bus0": bus0,
                    "bus1": bus1,
                    "line_length_km": float(lrow.get("length", np.nan)),
                    "voltage_kv": float(lrow.get("v_nom", np.nan)),
                    "line_capacity_mva": float(lrow.get("s_nom", np.nan)),
                    "number_of_segments": 1,
                    "endpoint0_lon": x0,
                    "endpoint0_lat": y0,
                    "endpoint1_lon": x1,
                    "endpoint1_lat": y1,
                    "x": np.nanmean([x0, x1]),
                    "y": np.nanmean([y0, y1]),
                    "notes": "Model stores a PyPSA line between endpoint buses; no assumption is made that an entire longer corridor must be upgraded.",
                }
            )
        else:
            base["notes"] = "Asset was selected in pilot outputs but was not found in the metadata network table."
        rows.append(base)

    characteristics = pd.DataFrame(rows)
    characteristics.to_csv(OUTPUT_DIR / "adaptation_asset_characteristics.csv", index=False)
    return characteristics


def build_preliminary_designs(characteristics: pd.DataFrame) -> pd.DataFrame:
    """Assign conceptual adaptation categories for each modeled asset."""
    rows: list[dict[str, object]] = []
    for _, row in characteristics.iterrows():
        if row["asset_type"] == "generator":
            adaptation = (
                "equipment elevation; local drainage and pumping; dry floodproofing of control equipment; "
                "protection of transformers, switchgear, and control systems"
            )
            target = "modeled generating facility and associated electrical equipment"
            uncertainty = "high: conceptual screening only; site layout and protected perimeter are not engineered"
        elif row["asset_type"] == "line":
            adaptation = (
                "flood-resistant pole or tower foundation; localized drainage or berm protection; "
                "protection of associated terminal equipment"
            )
            target = "modeled exposed line segment and endpoint electrical equipment where represented"
            uncertainty = "high: model line does not specify exact exposed structures or detailed civil works"
        else:
            adaptation = "conceptual flood hardening"
            target = "modeled component"
            uncertainty = "high"
        rows.append(
            {
                "asset_id": row["asset_id"],
                "asset_name": row["asset_name"],
                "asset_type": row["asset_type"],
                "conceptual_adaptation": adaptation,
                "protection_target": target,
                "design_depth_m": row["design_depth_m"],
                "engineering_uncertainty": uncertainty,
                "notes": "Conceptual adaptation category, not a detailed engineering design.",
            }
        )
    designs = pd.DataFrame(rows)
    designs.to_csv(OUTPUT_DIR / "preliminary_adaptation_designs.csv", index=False)
    return designs


def write_default_cost_assumptions(characteristics: pd.DataFrame, path: Path) -> None:
    """Create editable low/central/high planning cost assumptions if absent."""
    if path.exists():
        return
    rows: list[dict[str, object]] = []
    for _, row in characteristics.iterrows():
        if row["asset_type"] == "generator":
            low, central, high = 2_000_000, 10_000_000, 35_000_000
            basis = "generic planning assumption: modest equipment elevation/drainage to major perimeter protection"
        elif row["asset_type"] == "line":
            low, central, high = 500_000, 2_000_000, 8_000_000
            basis = "generic planning assumption: localized structure/foundation hardening to exposed segment reconstruction"
        else:
            low, central, high = 500_000, 2_000_000, 10_000_000
            basis = "generic planning placeholder"
        rows.append(
            {
                "asset_id": row["asset_id"],
                "asset_name": row["asset_name"],
                "asset_type": row["asset_type"],
                "low_cost_usd": low,
                "central_cost_usd": central,
                "high_cost_usd": high,
                "cost_basis": basis,
                "cost_evidence_source": "No credible project-specific flood-protection unit cost found in local files; editable planning assumption.",
                "editable": True,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_default_economic_assumptions(path: Path) -> None:
    """Create editable economic assumptions if absent."""
    if path.exists():
        return
    pd.DataFrame(
        [
            {
                "analysis_lifetime_years": 30,
                "real_discount_rate": 0.03,
                "low_voll_usd_per_mwh": 5000,
                "central_voll_usd_per_mwh": 10000,
                "high_voll_usd_per_mwh": 20000,
                "annual_maintenance_percent_of_capex": 0.01,
                "annual_benefit_growth_rate": 0.0,
                "residual_value_percent": 0.0,
                "notes": "VOLL values are illustrative sensitivity assumptions, not verified Florida-specific values.",
            }
        ]
    ).to_csv(path, index=False)


def load_economic_assumptions(path: Path) -> EconomicAssumptions:
    """Read economic assumptions from CSV."""
    row = pd.read_csv(path).iloc[0]
    return EconomicAssumptions(
        lifetime_years=int(row["analysis_lifetime_years"]),
        discount_rate=float(row["real_discount_rate"]),
        annual_maintenance_percent_of_capex=float(row["annual_maintenance_percent_of_capex"]),
        annual_benefit_growth_rate=float(row["annual_benefit_growth_rate"]),
        residual_value_percent=float(row["residual_value_percent"]),
        low_voll_usd_per_mwh=float(row["low_voll_usd_per_mwh"]),
        central_voll_usd_per_mwh=float(row["central_voll_usd_per_mwh"]),
        high_voll_usd_per_mwh=float(row["high_voll_usd_per_mwh"]),
    )


def annuity_factor(rate: float, years: int, growth: float = 0.0) -> float:
    """Present-value factor for a growing annual stream."""
    if years <= 0:
        return 0.0
    if abs(rate - growth) < 1e-12:
        return years / (1.0 + rate)
    return (1.0 - ((1.0 + growth) / (1.0 + rate)) ** years) / (rate - growth)


def discounted_payback_year(capital_cost: float, net_annual_benefit: float, rate: float, years: int) -> float:
    """Return discounted payback year, or NaN if not achieved."""
    cumulative = 0.0
    for year in range(1, years + 1):
        cumulative += net_annual_benefit / ((1.0 + rate) ** year)
        if cumulative >= capital_cost:
            return float(year)
    return math.nan


def annual_benefits(avoided_eens: float, assumptions: EconomicAssumptions) -> pd.DataFrame:
    """Calculate annual avoided outage value under VOLL sensitivity values."""
    rows = []
    for label, voll in {
        "low": assumptions.low_voll_usd_per_mwh,
        "central": assumptions.central_voll_usd_per_mwh,
        "high": assumptions.high_voll_usd_per_mwh,
    }.items():
        rows.append(
            {
                "voll_scenario": label,
                "voll_usd_per_mwh": voll,
                "avoided_eens_mwh_per_year": avoided_eens,
                "annual_avoided_outage_cost_usd": avoided_eens * voll,
                "included_benefits": "avoided service interruption only",
                "excluded_benefits": "avoided physical repair costs are excluded because they are not in the pilot outputs",
            }
        )
    benefits = pd.DataFrame(rows)
    benefits.to_csv(OUTPUT_DIR / "annual_benefit_estimates.csv", index=False)
    return benefits


def package_costs(costs: pd.DataFrame) -> pd.DataFrame:
    """Summarize total package costs for low, central, and high cases."""
    rows = []
    for label in ["low", "central", "high"]:
        rows.append({"package_cost_scenario": label, "capital_cost_usd": float(costs[f"{label}_cost_usd"].sum())})
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "package_cost_summary.csv", index=False)
    return result


def calculate_cost_benefit(
    costs: pd.DataFrame,
    benefits: pd.DataFrame,
    assumptions: EconomicAssumptions,
) -> pd.DataFrame:
    """Calculate lifecycle economics for all package-cost and VOLL combinations."""
    package = package_costs(costs)
    benefit_factor = annuity_factor(
        assumptions.discount_rate, assumptions.lifetime_years, assumptions.annual_benefit_growth_rate
    )
    maintenance_factor = annuity_factor(assumptions.discount_rate, assumptions.lifetime_years, 0.0)
    rows: list[dict[str, object]] = []
    for _, cost_row in package.iterrows():
        capex = float(cost_row["capital_cost_usd"])
        annual_maintenance = capex * assumptions.annual_maintenance_percent_of_capex
        pv_maintenance = annual_maintenance * maintenance_factor
        for _, benefit_row in benefits.iterrows():
            annual_benefit = float(benefit_row["annual_avoided_outage_cost_usd"])
            net_annual = annual_benefit - annual_maintenance
            pv_gross_benefits = annual_benefit * benefit_factor
            pv_net_benefits = pv_gross_benefits - pv_maintenance
            total_cost_pv = capex + pv_maintenance
            npv = pv_gross_benefits - total_cost_pv
            bcr = pv_gross_benefits / total_cost_pv if total_cost_pv > 0 else np.nan
            breakeven = pv_gross_benefits / (1.0 + assumptions.annual_maintenance_percent_of_capex * maintenance_factor)
            rows.append(
                {
                    "package_cost_scenario": cost_row["package_cost_scenario"],
                    "voll_scenario": benefit_row["voll_scenario"],
                    "capital_cost_usd": capex,
                    "annual_maintenance_cost_usd": annual_maintenance,
                    "annual_avoided_outage_benefit_usd": annual_benefit,
                    "net_annual_benefit_usd": net_annual,
                    "pv_gross_benefits_usd": pv_gross_benefits,
                    "pv_maintenance_costs_usd": pv_maintenance,
                    "pv_net_benefits_usd": pv_net_benefits,
                    "net_present_value_usd": npv,
                    "benefit_cost_ratio": bcr,
                    "simple_payback_years": capex / net_annual if net_annual > 0 else np.nan,
                    "discounted_payback_years": discounted_payback_year(
                        capex, net_annual, assumptions.discount_rate, assumptions.lifetime_years
                    ),
                    "cost_per_avoided_annual_mwh_usd_per_mwh_year": capex / float(benefit_row["avoided_eens_mwh_per_year"]),
                    "lifecycle_cost_per_avoided_mwh_usd_per_mwh": total_cost_pv
                    / (float(benefit_row["avoided_eens_mwh_per_year"]) * maintenance_factor),
                    "break_even_capital_cost_usd": breakeven,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "cost_benefit_results.csv", index=False)
    return result


def sensitivity_analysis(
    avoided_eens: float,
    package: pd.DataFrame,
    maintenance_percent: float,
    benefit_growth: float,
) -> pd.DataFrame:
    """Calculate sensitivity grid over rate, lifetime, VOLL, and package cost."""
    rows: list[dict[str, object]] = []
    for rate in [0.02, 0.03, 0.05, 0.07]:
        for lifetime in [20, 30, 40]:
            benefit_factor = annuity_factor(rate, lifetime, benefit_growth)
            maintenance_factor = annuity_factor(rate, lifetime, 0.0)
            for voll in [5000, 10000, 20000]:
                annual_benefit = avoided_eens * voll
                pv_benefits = annual_benefit * benefit_factor
                for _, cost_row in package.iterrows():
                    capex = float(cost_row["capital_cost_usd"])
                    pv_maintenance = capex * maintenance_percent * maintenance_factor
                    total_cost = capex + pv_maintenance
                    rows.append(
                        {
                            "discount_rate": rate,
                            "lifetime_years": lifetime,
                            "voll_usd_per_mwh": voll,
                            "package_cost_scenario": cost_row["package_cost_scenario"],
                            "capital_cost_usd": capex,
                            "benefit_cost_ratio": pv_benefits / total_cost if total_cost > 0 else np.nan,
                            "net_present_value_usd": pv_benefits - total_cost,
                            "minimum_avoided_eens_for_bcr_1_mwh_per_year": total_cost / (voll * benefit_factor),
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "sensitivity_results.csv", index=False)
    return result


def illustrative_asset_allocation(
    characteristics: pd.DataFrame,
    asset_damage: pd.DataFrame,
    avoided_eens: float,
) -> pd.DataFrame:
    """Create clearly marked illustrative asset benefit allocations."""
    equal_share = avoided_eens / len(characteristics)
    proxy = characteristics[["asset_id", "modeled_capacity"]].copy()
    proxy["modeled_capacity"] = pd.to_numeric(proxy["modeled_capacity"], errors="coerce").fillna(0.0)
    if proxy["modeled_capacity"].sum() <= 0:
        proxy["proxy_weight"] = 1.0 / len(proxy)
    else:
        proxy["proxy_weight"] = proxy["modeled_capacity"] / proxy["modeled_capacity"].sum()

    damage_cols = [c for c in asset_damage.columns if "avoided" in c.lower() or "reduction" in c.lower()]
    rows: list[dict[str, object]] = []
    for _, row in characteristics.iterrows():
        weight = float(proxy.loc[proxy["asset_id"] == row["asset_id"], "proxy_weight"].iloc[0])
        rows.append(
            {
                "asset_id": row["asset_id"],
                "asset_name": row["asset_name"],
                "asset_type": row["asset_type"],
                "allocation_method": "equal allocation",
                "illustrative_avoided_eens_mwh_per_year": equal_share,
                "warning": "Illustrative only; portfolio benefit was not simulated separately by asset.",
            }
        )
        rows.append(
            {
                "asset_id": row["asset_id"],
                "asset_name": row["asset_name"],
                "asset_type": row["asset_type"],
                "allocation_method": "modeled capacity proxy allocation",
                "illustrative_avoided_eens_mwh_per_year": avoided_eens * weight,
                "warning": "Illustrative only; asset-level BCRs are not model-derived. Damage table columns checked: "
                + ", ".join(damage_cols[:6]),
            }
        )
    allocation = pd.DataFrame(rows)
    allocation.to_csv(OUTPUT_DIR / "illustrative_asset_benefit_allocation.csv", index=False)
    return allocation


def savefig(name: str) -> None:
    """Save the active matplotlib figure as PNG and PDF."""
    plt.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_annual_benefits(benefits: pd.DataFrame) -> None:
    """Plot annual avoided outage value by VOLL assumption."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#88bde6", "#2f6fbb", "#183b73"]
    values = benefits["annual_avoided_outage_cost_usd"] / 1e6
    ax.bar(benefits["voll_scenario"].str.title(), values, color=colors)
    for i, value in enumerate(values):
        ax.text(i, value, f"${value:.1f}M/yr", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Annual avoided outage value (million USD/year)")
    ax.set_title("Avoided Outage Value Depends Strongly on VOLL")
    ax.grid(axis="y", alpha=0.25)
    savefig("01_annual_avoided_outage_value")


def plot_matrix(data: pd.DataFrame, value_col: str, title: str, filename: str, fmt: str, scale: float = 1.0) -> None:
    """Plot a cost-scenario by VOLL-scenario matrix."""
    pivot = data.pivot(index="package_cost_scenario", columns="voll_scenario", values=value_col)
    pivot = pivot.loc[["low", "central", "high"], ["low", "central", "high"]]
    shown = pivot / scale
    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(shown.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(shown.columns)), [c.title() for c in shown.columns])
    ax.set_yticks(range(len(shown.index)), [i.title() for i in shown.index])
    ax.set_xlabel("VOLL assumption")
    ax.set_ylabel("Package cost assumption")
    ax.set_title(title)
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            raw = pivot.iloc[i, j]
            label = fmt.format(raw / scale if scale != 1.0 else raw)
            ax.text(j, i, label, ha="center", va="center", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(value_col.replace("_", " "))
    savefig(filename)


def plot_break_even(results: pd.DataFrame, package: pd.DataFrame) -> None:
    """Plot break-even capital cost compared with package cost assumptions."""
    central_cost_rows = results[results["package_cost_scenario"] == "central"].copy()
    order = ["low", "central", "high"]
    breakeven = central_cost_rows.set_index("voll_scenario").loc[order, "break_even_capital_cost_usd"] / 1e6
    package_values = package.set_index("package_cost_scenario").loc[order, "capital_cost_usd"] / 1e6
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(order))
    ax.bar(x - 0.18, breakeven.values, width=0.36, label="Break-even upfront cost", color="#2f6fbb")
    ax.bar(x + 0.18, package_values.values, width=0.36, label="Assumed package cost", color="#d77a22")
    ax.set_xticks(x, [o.title() for o in order])
    ax.set_ylabel("Capital cost (million USD)")
    ax.set_xlabel("VOLL or cost scenario")
    ax.set_title("Break-Even Cost Is Far Above the Planning Cost Range")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    savefig("04_break_even_package_cost")


def plot_bcr_sensitivity(results: pd.DataFrame) -> None:
    """Plot BCR sensitivity to package capital cost."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for voll_label, group in results.groupby("voll_scenario"):
        group = group.sort_values("capital_cost_usd")
        ax.plot(
            group["capital_cost_usd"] / 1e6,
            group["benefit_cost_ratio"],
            marker="o",
            linewidth=2.5,
            label=f"{voll_label.title()} VOLL",
        )
    ax.axhline(1, color="#333333", linestyle="--", linewidth=1.2, label="BCR = 1")
    ax.set_xlabel("Package capital cost (million USD)")
    ax.set_ylabel("Benefit-cost ratio")
    ax.set_title("BCR Remains Sensitive to Both VOLL and Package Cost")
    ax.grid(alpha=0.25)
    ax.legend()
    savefig("05_bcr_sensitivity_to_cost")


def write_policy_interpretation(
    avoided_eens: float,
    risk_reduction: float,
    costs: pd.DataFrame,
    assumptions: EconomicAssumptions,
    results: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    """Write a concise planning-level policy interpretation."""
    central = results[
        (results["package_cost_scenario"] == "central") & (results["voll_scenario"] == "central")
    ].iloc[0]
    cost_effective = results[results["benefit_cost_ratio"] > 1][
        ["package_cost_scenario", "voll_scenario", "benefit_cost_ratio"]
    ]
    cost_effective_lines = "\n".join(
        f"- {row.package_cost_scenario} cost / {row.voll_scenario} VOLL: BCR {row.benefit_cost_ratio:.2f}"
        for row in cost_effective.itertuples(index=False)
    )
    break_even_central = float(central["break_even_capital_cost_usd"])
    min_bcr = sensitivity["benefit_cost_ratio"].min()
    max_bcr = sensitivity["benefit_cost_ratio"].max()
    text = f"""# Preliminary Policy Interpretation

This is a preliminary economic assessment of the RP100 top-five flood-adaptation pilot. The five-asset package reduces pilot annualized EENS by approximately {avoided_eens:,.1f} MWh/year, or {risk_reduction:.2f}% of the pilot baseline flood risk. Benefits are monetized only as avoided service interruption, so avoided physical repair costs are excluded and total benefits may be underestimated.

## Main Central Case

- Central package capital cost: ${float(costs.loc[costs['package_cost_scenario'] == 'central', 'capital_cost_usd'].iloc[0]) / 1e6:,.1f} million.
- Central VOLL assumption: ${assumptions.central_voll_usd_per_mwh:,.0f}/MWh.
- Central-case BCR: {float(central['benefit_cost_ratio']):.2f}.
- Central-case NPV: ${float(central['net_present_value_usd']) / 1e6:,.1f} million.
- Central simple payback: {float(central['simple_payback_years']):.1f} years.
- Central break-even upfront capital cost: ${break_even_central / 1e6:,.1f} million.

## Policy Questions

1. Under the illustrative assumptions, the package is cost-effective for these cost and VOLL combinations:

{cost_effective_lines}

2. The central-case BCR is {float(central['benefit_cost_ratio']):.2f}, and the central-case NPV is ${float(central['net_present_value_usd']) / 1e6:,.1f} million.

3. The central-VOLL break-even package cost is about ${break_even_central / 1e6:,.1f} million after accounting for 1% annual maintenance.

4. Across the sensitivity grid, BCR ranges from {min_bcr:.2f} to {max_bcr:.2f}. Results improve with longer lifetimes and lower discount rates, and weaken with shorter lifetimes and higher discount rates.

5. The package is promising enough to justify running the full return-period suite, but this pilot annualization is not final because it only uses RP10, RP100, and RP500.

6. The most useful next information would be site-specific adaptation cost estimates, verified VOLL assumptions for the affected customer mix, more complete return-period results, and engineering confirmation of what each modeled asset physically represents.

## Limitations

- This is a planning-level estimate, not an engineering cost estimate.
- VOLL values are illustrative sensitivity assumptions, not verified Florida-specific values.
- Portfolio benefits are not allocated to individual assets as model-derived BCRs.
- EENS reductions come from the transmission-scale PyPSA model and do not capture all distribution-level outage benefits.
- Avoided physical repair costs are excluded because they were not available in the pilot outputs.
"""
    (OUTPUT_DIR / "preliminary_policy_interpretation.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the full cost-benefit workflow."""
    setup_logging()
    tables, avoided_eens = verify_pilot_results()
    risk_reduction = float(tables["annualized_risk"]["pilot_risk_reduction_percent"].iloc[0])

    generators, lines, buses = load_network_metadata()
    characteristics = build_asset_characteristics(
        tables["selected_assets"], tables["protection_design"], generators, lines, buses
    )
    build_preliminary_designs(characteristics)

    cost_path = OUTPUT_DIR / "adaptation_cost_assumptions.csv"
    econ_path = OUTPUT_DIR / "economic_assumptions.csv"
    write_default_cost_assumptions(characteristics, cost_path)
    write_default_economic_assumptions(econ_path)
    cost_assumptions = pd.read_csv(cost_path)
    econ = load_economic_assumptions(econ_path)
    logging.info("Cost assumptions loaded from %s", cost_path)
    logging.info("Economic assumptions loaded from %s", econ_path)

    benefits = annual_benefits(avoided_eens, econ)
    package = package_costs(cost_assumptions)
    results = calculate_cost_benefit(cost_assumptions, benefits, econ)
    sensitivity = sensitivity_analysis(
        avoided_eens, package, econ.annual_maintenance_percent_of_capex, econ.annual_benefit_growth_rate
    )
    illustrative_asset_allocation(characteristics, tables["asset_damage"], avoided_eens)

    plot_annual_benefits(benefits)
    plot_matrix(results, "benefit_cost_ratio", "Benefit-Cost Ratio by Cost and VOLL Assumption", "02_bcr_matrix", "{:.1f}")
    plot_matrix(
        results,
        "net_present_value_usd",
        "Net Present Value by Cost and VOLL Assumption",
        "03_npv_matrix",
        "${:.0f}M",
        scale=1e6,
    )
    plot_break_even(results, package)
    plot_bcr_sensitivity(results)
    write_policy_interpretation(avoided_eens, risk_reduction, package, econ, results, sensitivity)

    central = results[
        (results["package_cost_scenario"] == "central") & (results["voll_scenario"] == "central")
    ].iloc[0]
    combos = results.loc[results["benefit_cost_ratio"] > 1, ["package_cost_scenario", "voll_scenario"]]

    print("\nRP100 top-five cost-benefit workflow complete")
    print(f"Verified avoided EENS: {avoided_eens:,.1f} MWh/year")
    for _, row in benefits.iterrows():
        print(
            f"{str(row['voll_scenario']).title()} annual avoided outage value: "
            f"${float(row['annual_avoided_outage_cost_usd']) / 1e6:,.1f}M/year"
        )
    for _, row in package.iterrows():
        print(f"{str(row['package_cost_scenario']).title()} package cost: ${float(row['capital_cost_usd']) / 1e6:,.1f}M")
    print(f"Central-case BCR: {float(central['benefit_cost_ratio']):.2f}")
    print(f"Central-case NPV: ${float(central['net_present_value_usd']) / 1e6:,.1f}M")
    print(f"Central-case simple payback: {float(central['simple_payback_years']):.1f} years")
    print(f"Central-VOLL break-even package cost: ${float(central['break_even_capital_cost_usd']) / 1e6:,.1f}M")
    print("Cost and VOLL combinations with BCR above 1:")
    for _, row in combos.iterrows():
        print(f"  - {row['package_cost_scenario']} cost / {row['voll_scenario']} VOLL")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print("Major limitations: planning-level costs; illustrative VOLL; three-return-period pilot EENS; no repair-cost savings; no distribution-level outage benefits.")


if __name__ == "__main__":
    main()
