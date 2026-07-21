"""
Capacity exposure plots for Florida power plant polygons.

Creates:
  - exposed capacity vs return period for JRC flood and STORM TC wind
  - exposed capacity by fuel type for JRC flood and STORM TC wind
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
OUTPUT_DIR = EXPOSURE_DIR / "powerplant_capacity_exposure_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOOD_RP = (
    EXPOSURE_DIR
    / "powerplant_polygon_flood_exposure"
    / "powerplant_polygon_flood_summary_by_return_period.csv"
)
FLOOD_FUEL = (
    EXPOSURE_DIR
    / "powerplant_polygon_flood_exposure"
    / "powerplant_polygon_flood_summary_by_fuel.csv"
)
TC_RP = (
    EXPOSURE_DIR
    / "powerplant_polygon_tc_exposure"
    / "powerplant_polygon_tc_summary_by_dataset.csv"
)
TC_FUEL = (
    EXPOSURE_DIR
    / "powerplant_polygon_tc_exposure"
    / "powerplant_polygon_tc_summary_by_fuel.csv"
)

FUEL_ORDER = [
    "Solar",
    "Gas",
    "Waste",
    "Coal",
    "Oil",
    "Biomass",
    "Cogeneration",
    "Hydro",
    "Nuclear",
    "Storage",
    "Other",
    "Unknown",
]
FUEL_COLORS = {
    "Solar": "#f4c430",
    "Gas": "#8c8c8c",
    "Waste": "#8c6bb1",
    "Coal": "#303030",
    "Oil": "#111111",
    "Biomass": "#93cf8e",
    "Cogeneration": "#7b6fb7",
    "Hydro": "#2b8cbe",
    "Nuclear": "#31a354",
    "Storage": "#fb6a4a",
    "Other": "#bdbdbd",
    "Unknown": "#d9d9d9",
}


def normalize_fuel(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    fuel = str(value).strip()
    if fuel in FUEL_COLORS:
        return fuel
    return "Other"


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def annotate_bar_values(ax: plt.Axes, values: pd.Series, suffix: str = "GW") -> None:
    ymax = max(float(values.max()), 1.0)
    ax.set_ylim(0, ymax * 1.15)
    for idx, value in enumerate(values):
        ax.text(
            idx,
            value + ymax * 0.025,
            f"{value:,.1f} {suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def storm_tc_return_periods(tc: pd.DataFrame) -> pd.DataFrame:
    storm = tc[tc["dataset"].astype(str).str.startswith("storm_rp")].copy()
    storm = storm.dropna(subset=["return_period"])
    storm["return_period"] = storm["return_period"].astype(int)
    return storm.sort_values("return_period")


def plot_capacity_vs_return_period(flood: pd.DataFrame, tc: pd.DataFrame) -> list[Path]:
    storm = storm_tc_return_periods(tc)

    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 5.4))
    flood = flood.sort_values("return_period").copy()
    x = flood["return_period"].astype(int).astype(str)
    y = flood["exposed_capacity_mw"] / 1000
    ax.bar(x, y, color="#2b7bbb", edgecolor="white", linewidth=1.0)
    annotate_bar_values(ax, y)
    ax.set_title("Power Plant Capacity Exposed to JRC Flooding by Return Period", fontsize=13)
    ax.set_xlabel("Flood return period (years)")
    ax.set_ylabel("Exposed capacity (GW)")
    style_axis(ax)
    fig.tight_layout()
    outputs.append(save_figure(fig, "flood_capacity_exposed_vs_return_period.png"))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = storm["return_period"].astype(int).astype(str)
    y = storm["capacity_ge_25ms"] / 1000
    ax.bar(x, y, color="#d95f02", edgecolor="white", linewidth=1.0)
    annotate_bar_values(ax, y)
    ax.set_title("Power Plant Capacity Exposed to STORM TC Winds by Return Period", fontsize=13)
    ax.set_xlabel("TC wind return period (years)")
    ax.set_ylabel("Capacity exposed to >=25 m/s (GW)")
    style_axis(ax)
    fig.tight_layout()
    outputs.append(save_figure(fig, "storm_tc_capacity_exposed_vs_return_period.png"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Power Plant Capacity Exposure by Return Period", fontsize=16, fontweight="bold")

    ax = axes[0]
    x = flood["return_period"].astype(int).astype(str)
    y = flood["exposed_capacity_mw"] / 1000
    ax.bar(x, y, color="#2b7bbb", edgecolor="white", linewidth=1.0)
    annotate_bar_values(ax, y)
    ax.set_title("JRC flood")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Exposed capacity (GW)")
    style_axis(ax)

    ax = axes[1]
    x = storm["return_period"].astype(int).astype(str)
    y = storm["capacity_ge_25ms"] / 1000
    ax.bar(x, y, color="#d95f02", edgecolor="white", linewidth=1.0)
    annotate_bar_values(ax, y)
    ax.set_title("STORM TC wind")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Capacity exposed to >=25 m/s (GW)")
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    outputs.append(save_figure(fig, "flood_and_storm_tc_capacity_exposed_vs_return_period.png"))

    return outputs


def fuel_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = df.copy()
    work["fuel_clean"] = work["primary_fuel"].map(normalize_fuel)
    work["return_period"] = work["return_period"].astype(int)
    table = work.pivot_table(
        index="return_period",
        columns="fuel_clean",
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    ).sort_index()
    cols = [fuel for fuel in FUEL_ORDER if fuel in table.columns and table[fuel].sum() > 0]
    return table[cols] / 1000


def plot_capacity_by_fuel(flood_fuel: pd.DataFrame, tc_fuel: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []

    flood_table = fuel_pivot(flood_fuel, "exposed_capacity_mw")
    storm_fuel = tc_fuel[tc_fuel["dataset"].astype(str).str.startswith("storm_rp")].copy()
    storm_table = fuel_pivot(storm_fuel, "exposed_capacity_mw")

    fig, ax = plt.subplots(figsize=(10, 6))
    flood_table.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[FUEL_COLORS.get(col, "#bdbdbd") for col in flood_table.columns],
        width=0.8,
    )
    totals = flood_table.sum(axis=1)
    annotate_bar_values(ax, totals)
    ax.set_title("JRC Flood-Exposed Power Plant Capacity by Fuel Type", fontsize=13)
    ax.set_xlabel("Flood return period (years)")
    ax.set_ylabel("Exposed capacity (GW)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Fuel type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    style_axis(ax)
    fig.tight_layout()
    outputs.append(save_figure(fig, "flood_capacity_exposed_by_fuel_type.png"))

    fig, ax = plt.subplots(figsize=(10, 6))
    storm_table.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[FUEL_COLORS.get(col, "#bdbdbd") for col in storm_table.columns],
        width=0.8,
    )
    totals = storm_table.sum(axis=1)
    annotate_bar_values(ax, totals)
    ax.set_title("STORM TC Wind-Exposed Power Plant Capacity by Fuel Type", fontsize=13)
    ax.set_xlabel("TC wind return period (years)")
    ax.set_ylabel("Capacity exposed to >=25 m/s (GW)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Fuel type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    style_axis(ax)
    fig.tight_layout()
    outputs.append(save_figure(fig, "storm_tc_capacity_exposed_by_fuel_type.png"))

    return outputs


def main() -> None:
    print(f"Reading {FLOOD_RP}")
    flood = pd.read_csv(FLOOD_RP)
    print(f"Reading {FLOOD_FUEL}")
    flood_fuel = pd.read_csv(FLOOD_FUEL)
    print(f"Reading {TC_RP}")
    tc = pd.read_csv(TC_RP)
    print(f"Reading {TC_FUEL}")
    tc_fuel = pd.read_csv(TC_FUEL)

    outputs = []
    outputs.extend(plot_capacity_vs_return_period(flood, tc))
    outputs.extend(plot_capacity_by_fuel(flood_fuel, tc_fuel))

    print("\nCreated capacity exposure figures:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
