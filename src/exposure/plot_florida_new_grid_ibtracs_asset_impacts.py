"""
Simple, one-purpose plots for newest-grid IBTrACS asset impacts.

No multi-panel figures: each PNG answers one question.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure" / "florida_new_grid_ibtracs_asset_impacts"
CASCADING_DIR = (
    PROJECT_DIR
    / "data"
    / "Electricity"
    / "pypsa_florida_network_extended_tie_lines"
    / "ibtracs_cascading_top5_direct_events"
    / "summary_tables"
)
OUT_DIR = EXPOSURE_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


BLUE = "#2B6CB0"
ORANGE = "#DD6B20"
GREEN = "#2F855A"
RED = "#C53030"
GRID = "#D0D7DE"


def usd_millions(x, _pos):
    return f"${x / 1e6:.0f}M"


def usd_billions(x, _pos):
    return f"${x / 1e9:.1f}B"


def mw_label(x, _pos):
    return f"{x:.0f} MW"


def save_barh(df: pd.DataFrame, y_col: str, x_col: str, title: str, xlabel: str, path: Path, color: str) -> None:
    df = df.copy().sort_values(x_col, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df[y_col], df[x_col], color=color)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_top_powerplants() -> None:
    plants = pd.read_csv(EXPOSURE_DIR / "top_directly_impacted_powerplants_by_all_events.csv")
    plants = plants[pd.to_numeric(plants["cumulative_capacity_loss_mw"], errors="coerce") > 0].copy()
    plants["label"] = plants["name"].fillna(plants["plant_asset_id"])
    plants = plants.head(15)

    save_barh(
        plants,
        "label",
        "cumulative_capacity_loss_mw",
        "Top Directly Impacted Power Plant Polygons",
        "Cumulative modeled capacity loss across IBTrACS events (MW)",
        OUT_DIR / "top_directly_impacted_powerplants.png",
        ORANGE,
    )


def plot_top_substations() -> None:
    substations = pd.read_csv(EXPOSURE_DIR / "top_directly_impacted_substation_buses_by_all_events.csv")
    substations = substations[pd.to_numeric(substations["cumulative_direct_damage_usd"], errors="coerce") > 0].copy()
    substations["label"] = substations["bus_substation_name"].fillna(substations["osm_name"]).fillna(substations["bus"])
    substations = substations.head(15)

    df = substations.sort_values("cumulative_direct_damage_usd", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["label"], df["cumulative_direct_damage_usd"], color=BLUE)
    ax.set_title("Top Directly Impacted Substations", fontsize=14, weight="bold")
    ax.set_xlabel("Cumulative modeled direct damage across IBTrACS events (USD)")
    ax.xaxis.set_major_formatter(FuncFormatter(usd_millions))
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "top_directly_impacted_substations.png", dpi=300)
    plt.close(fig)


def plot_event_direct_damage() -> None:
    events = pd.read_csv(EXPOSURE_DIR / "direct_event_damage_summary_lines_plants_substations.csv")
    events = events.sort_values("total_direct_damage_usd_proxy", ascending=False).head(15).copy()
    events = events.sort_values("total_direct_damage_usd_proxy", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(events["event_id"], events["total_direct_damage_usd_proxy"], color=GREEN)
    ax.set_title("Highest Direct-Damage Historical IBTrACS Events", fontsize=14, weight="bold")
    ax.set_xlabel("Modeled direct grid damage, lines plus substation proxy (USD)")
    ax.set_ylabel("IBTrACS event ID")
    ax.xaxis.set_major_formatter(FuncFormatter(usd_billions))
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "top_direct_damage_ibtracs_events.png", dpi=300)
    plt.close(fig)


def plot_cascading_cost() -> None:
    path = CASCADING_DIR / "storm_level_cascading_operational_summary.csv"
    if not path.exists():
        return
    cascade = pd.read_csv(path)
    cascade = cascade.sort_values("incremental_system_cost_usd", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(cascade["event_id"], cascade["incremental_system_cost_usd"], color=RED)
    ax.set_title("Top-5 Storms: Cascading Operational Cost Increase", fontsize=14, weight="bold")
    ax.set_xlabel("Incremental PyPSA system cost vs no-hazard baseline (USD)")
    ax.set_ylabel("IBTrACS event ID")
    ax.xaxis.set_major_formatter(FuncFormatter(usd_millions))
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "top5_cascading_operational_cost_increase.png", dpi=300)
    plt.close(fig)


def plot_cascading_load_shedding() -> None:
    path = CASCADING_DIR / "storm_level_cascading_operational_summary.csv"
    if not path.exists():
        return
    cascade = pd.read_csv(path).sort_values("event_id")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cascade["event_id"], cascade["incremental_load_shed_mwh"], color=BLUE)
    ax.axhline(0, color="#24292F", linewidth=1)
    ax.set_title("Top-5 Storms: No Additional Load Shedding", fontsize=14, weight="bold")
    ax.set_xlabel("IBTrACS event ID")
    ax.set_ylabel("Incremental load shed vs baseline (MWh)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "top5_cascading_incremental_load_shedding.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_top_powerplants()
    plot_top_substations()
    plot_event_direct_damage()
    plot_cascading_cost()
    plot_cascading_load_shedding()
    print(f"Saved plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
