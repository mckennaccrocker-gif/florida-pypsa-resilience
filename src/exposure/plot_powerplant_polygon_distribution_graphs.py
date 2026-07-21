"""
Polygon-only distribution plots for Florida power plant hazard exposure.

These figures mirror the earlier node-exposure histograms, but each observation
is an OSM power plant polygon footprint. Area-weighted exposure percentages are
reported so the plots show both facility counts and footprint area exposure.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
OUTPUT_DIR = EXPOSURE_DIR / "powerplant_polygon_distribution_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOOD_GPKG = (
    EXPOSURE_DIR
    / "powerplant_polygon_flood_exposure"
    / "powerplant_polygon_flood_exposure.gpkg"
)
TC_GPKG = (
    EXPOSURE_DIR
    / "powerplant_polygon_tc_exposure"
    / "powerplant_polygon_tc_exposure.gpkg"
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

FLOOD_CLASSES = ["0 m", "0-0.5 m", "0.5-1 m", "1-2 m", "2-5 m", ">5 m"]
FLOOD_THRESHOLDS = [0.5, 1, 2, 5]
FLOOD_BINS = [0, 0.5, 1, 2, 5, np.inf]
FLOOD_HIST_BINS = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6]

TC_CLASSES = ["<25", "25-30", "30-35", "35-40", ">40"]
TC_THRESHOLDS = [25, 30, 35, 40]
TC_BINS = [0, 25, 30, 35, 40, np.inf]
TC_HIST_BINS = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def normalize_fuel(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    fuel = str(value).strip()
    if not fuel:
        return "Unknown"
    if fuel in FUEL_COLORS:
        return fuel
    return "Other"


def classify_depth(depth: float) -> str:
    if pd.isna(depth) or depth <= 0:
        return "0 m"
    return pd.cut([depth], bins=FLOOD_BINS, labels=FLOOD_CLASSES[1:], include_lowest=True)[0]


def classify_wind(speed: float) -> str:
    if pd.isna(speed):
        return np.nan
    return pd.cut([speed], bins=TC_BINS, labels=TC_CLASSES, include_lowest=True)[0]


def weighted_area_percent(gdf: gpd.GeoDataFrame, fraction_col: str, mask: pd.Series | None = None) -> float:
    if "polygon_area_m2" not in gdf.columns:
        areas = gdf.geometry.area
    else:
        areas = gdf["polygon_area_m2"].fillna(0)
    total_area = areas.sum()
    if total_area <= 0:
        return 0.0
    if mask is None:
        mask = pd.Series(True, index=gdf.index)
    fractions = gdf.loc[mask, fraction_col].fillna(0).clip(lower=0, upper=1)
    exposed_area = (areas.loc[mask] * fractions).sum()
    return float(exposed_area / total_area * 100)


def class_area_percents(gdf: gpd.GeoDataFrame, class_col: str, class_order: list[str]) -> pd.Series:
    if "polygon_area_m2" not in gdf.columns:
        areas = gdf.geometry.area
    else:
        areas = gdf["polygon_area_m2"].fillna(0)
    total_area = areas.sum()
    if total_area <= 0:
        return pd.Series(0.0, index=class_order)
    return pd.Series(
        {
            cls: float(areas[gdf[class_col] == cls].sum() / total_area * 100)
            for cls in class_order
        }
    )


def make_stacked_count_table(
    gdf: gpd.GeoDataFrame, class_col: str, class_order: list[str]
) -> pd.DataFrame:
    work = gdf.copy()
    work["fuel_clean"] = work["primary_fuel"].map(normalize_fuel)
    table = (
        work.groupby([class_col, "fuel_clean"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=class_order, fill_value=0)
    )
    fuel_cols = [fuel for fuel in FUEL_ORDER if fuel in table.columns and table[fuel].sum() > 0]
    return table[fuel_cols]


def annotate_stacked_bars(
    ax: plt.Axes, totals: pd.Series, area_percents: pd.Series, y_offset: float = 0.8
) -> None:
    ymax = max(float(totals.max()), 1.0)
    ax.set_ylim(0, ymax * 1.22)
    for idx, cls in enumerate(totals.index):
        total = totals.loc[cls]
        area_pct = area_percents.loc[cls]
        if total > 0:
            ax.text(
                idx,
                total + y_offset,
                f"{int(total)}\n{area_pct:.1f}% area",
                ha="center",
                va="bottom",
                fontsize=9,
            )


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


def plot_flood_polygon_distribution(return_period: int = 100) -> Path:
    print(f"Reading flood polygons: {FLOOD_GPKG}")
    flood = gpd.read_file(FLOOD_GPKG)
    flood = flood[flood["return_period"] == return_period].copy()
    flood["flood_depth_class"] = flood["max_flood_depth_m"].map(classify_depth)
    flood["flood_depth_class"] = pd.Categorical(
        flood["flood_depth_class"], categories=FLOOD_CLASSES, ordered=True
    )

    exposed_area_pct = weighted_area_percent(flood, "wet_pixel_fraction")
    exposed_polygons = int((flood["max_flood_depth_m"].fillna(0) > 0).sum())
    exposed_capacity_gw = flood.loc[
        flood["max_flood_depth_m"].fillna(0) > 0, "capacity_mw"
    ].fillna(0).sum() / 1000

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle(
        f"Florida Power Plant Polygon Flood Exposure, JRC RP{return_period}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.91,
        (
            f"{exposed_area_pct:.1f}% of total plant polygon footprint area is flooded "
            f"(>0 m); {exposed_polygons:,} polygons and {exposed_capacity_gw:,.1f} GW exposed"
        ),
        ha="center",
        fontsize=11,
    )

    ax = axes[0]
    values = flood["max_flood_depth_m"].fillna(0)
    ax.hist(values, bins=FLOOD_HIST_BINS, color="#4f7daf", edgecolor="white", linewidth=1.1)
    for threshold in FLOOD_THRESHOLDS:
        ax.axvline(threshold, color="#8f8f8f", linestyle="--", linewidth=1)
    ax.set_title("Distribution of polygon maximum flood depth")
    ax.set_xlabel("Maximum flood depth within polygon (m)")
    ax.set_ylabel("Number of power plant polygons")
    style_axis(ax)

    ax = axes[1]
    stacked = make_stacked_count_table(flood, "flood_depth_class", FLOOD_CLASSES)
    colors = [FUEL_COLORS.get(col, "#bdbdbd") for col in stacked.columns]
    stacked.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
    totals = stacked.sum(axis=1)
    area_percents = class_area_percents(flood, "flood_depth_class", FLOOD_CLASSES)
    annotate_stacked_bars(ax, totals, area_percents)
    ax.set_title("Power plant polygons by flood depth class and energy type")
    ax.set_xlabel("Maximum flood depth class (m)")
    ax.set_ylabel("Number of power plant polygons")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Energy type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return save_figure(fig, f"flood_polygon_distribution_rp{return_period}.png")


def plot_tc_polygon_distribution(dataset: str = "open_gira_historical_max") -> Path:
    print(f"Reading TC polygons: {TC_GPKG}")
    tc = gpd.read_file(TC_GPKG)
    tc = tc[tc["dataset"] == dataset].copy()
    tc["tc_wind_class_clean"] = tc["max_wind_speed_ms"].map(classify_wind)
    tc["tc_wind_class_clean"] = pd.Categorical(
        tc["tc_wind_class_clean"], categories=TC_CLASSES, ordered=True
    )

    exposed_area_pct = weighted_area_percent(tc, "fraction_ge_25ms")
    exposed_polygons = int((tc["max_wind_speed_ms"].fillna(0) >= 25).sum())
    exposed_capacity_gw = tc.loc[
        tc["max_wind_speed_ms"].fillna(0) >= 25, "capacity_mw"
    ].fillna(0).sum() / 1000
    if dataset == "open_gira_historical_max":
        label = "OpenGIRA Historical Maximum"
    elif dataset.startswith("storm_rp"):
        label = f"STORM RP{dataset.removeprefix('storm_rp')}"
    else:
        label = dataset

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle(
        f"Florida Power Plant Polygon TC Wind Exposure, {label}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.91,
        (
            f"{exposed_area_pct:.1f}% of total plant polygon footprint area experiences "
            f">=25 m/s winds; {exposed_polygons:,} polygons and {exposed_capacity_gw:,.1f} GW exposed"
        ),
        ha="center",
        fontsize=11,
    )

    ax = axes[0]
    values = tc["max_wind_speed_ms"].dropna()
    ax.hist(values, bins=TC_HIST_BINS, color="#4f7daf", edgecolor="white", linewidth=1.1)
    for threshold in TC_THRESHOLDS:
        ax.axvline(threshold, color="#8f8f8f", linestyle="--", linewidth=1)
    ax.set_title("Distribution of polygon maximum TC wind exposure")
    ax.set_xlabel("Maximum tropical cyclone wind speed within polygon (m/s)")
    ax.set_ylabel("Number of power plant polygons")
    style_axis(ax)

    ax = axes[1]
    stacked = make_stacked_count_table(tc, "tc_wind_class_clean", TC_CLASSES)
    colors = [FUEL_COLORS.get(col, "#bdbdbd") for col in stacked.columns]
    stacked.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
    totals = stacked.sum(axis=1)
    area_percents = class_area_percents(tc, "tc_wind_class_clean", TC_CLASSES)
    annotate_stacked_bars(ax, totals, area_percents)
    ax.set_title("Power plant polygons by TC wind class and energy type")
    ax.set_xlabel("Maximum tropical cyclone wind speed class (m/s)")
    ax.set_ylabel("Number of power plant polygons")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Energy type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return save_figure(fig, f"tc_polygon_distribution_{dataset}.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 120,
        }
    )

    outputs = [
        plot_flood_polygon_distribution(return_period=100),
        plot_tc_polygon_distribution(dataset="open_gira_historical_max"),
        plot_tc_polygon_distribution(dataset="storm_rp100"),
    ]

    print("\nCreated polygon distribution figures:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
