"""
SNAIL-style distribution plots for Florida transmission line exposure.

SNAIL splits line assets by raster cells, so these figures report both segment
counts and length-weighted exposure percentages.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
OUTPUT_DIR = EXPOSURE_DIR / "snail_line_distribution_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOOD_SEGMENTS = EXPOSURE_DIR / "lines_flood_exposure_segments.gpkg"
TC_SEGMENTS = (
    EXPOSURE_DIR
    / "florida_clean_assets_tc_snail_intersection"
    / "snail_fred_open_gira_florida_overhead_lines.gpkg"
)
TC_STORM_RP100_SEGMENTS = (
    EXPOSURE_DIR
    / "florida_clean_assets_tc_snail_intersection"
    / "snail_storm_rp100_florida_overhead_lines.gpkg"
)

FLOOD_CLASSES = ["0 m", "0-0.5 m", "0.5-1 m", "1-2 m", "2-5 m", ">5 m"]
FLOOD_BINS = [0, 0.5, 1, 2, 5, np.inf]
FLOOD_HIST_BINS = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6]
FLOOD_COLORS = {
    "0 m": "#eeeeee",
    "0-0.5 m": "#bfe3ef",
    "0.5-1 m": "#7fb9d6",
    "1-2 m": "#4779b3",
    "2-5 m": "#f6a05d",
    ">5 m": "#c43b3b",
}

TC_CLASSES = ["<25", "25-30", "30-35", "35-40", ">40"]
TC_BINS = [0, 25, 30, 35, 40, np.inf]
TC_HIST_BINS = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60]
TC_COLORS = {
    "<25": "#4169a8",
    "25-30": "#98d4df",
    "30-35": "#f7e98b",
    "35-40": "#f5a65b",
    ">40": "#c63d3d",
}

VOLTAGE_ORDER = ["<100 kV", "100-161 kV", "220-287 kV", "345 kV", "500 kV", "Unknown"]
VOLTAGE_COLORS = {
    "<100 kV": "#7fb3d5",
    "100-161 kV": "#4f7daf",
    "220-287 kV": "#f5a65b",
    "345 kV": "#c43b3b",
    "500 kV": "#5c4d7d",
    "Unknown": "#bdbdbd",
}
TYPE_COLORS = {
    "AC; OVERHEAD": "#4f7daf",
    "OVERHEAD": "#7fb3d5",
    "AC; UNDERGROUND": "#5c4d7d",
    "UNDERGROUND": "#8c6bb1",
    "Other": "#bdbdbd",
}


def classify_depth(depth: float) -> str:
    if pd.isna(depth) or depth <= 0:
        return "0 m"
    return pd.cut([depth], bins=FLOOD_BINS, labels=FLOOD_CLASSES[1:], include_lowest=True)[0]


def classify_wind(speed: float) -> str:
    if pd.isna(speed):
        return np.nan
    return pd.cut([speed], bins=TC_BINS, labels=TC_CLASSES, include_lowest=True)[0]


def classify_voltage(voltage: float) -> str:
    if pd.isna(voltage) or voltage <= 0:
        return "Unknown"
    if voltage < 100:
        return "<100 kV"
    if voltage <= 161:
        return "100-161 kV"
    if voltage <= 287:
        return "220-287 kV"
    if voltage <= 400:
        return "345 kV"
    return "500 kV"


def normalize_type(value: object) -> str:
    if pd.isna(value):
        return "Other"
    text = str(value).strip()
    if text in TYPE_COLORS:
        return text
    if "UNDERGROUND" in text:
        return "UNDERGROUND"
    if "OVERHEAD" in text:
        return "OVERHEAD"
    return "Other"


def length_percent(gdf: gpd.GeoDataFrame, mask: pd.Series) -> float:
    total = gdf["length_km"].fillna(0).sum()
    if total <= 0:
        return 0.0
    return float(gdf.loc[mask, "length_km"].fillna(0).sum() / total * 100)


def class_length_percents(
    gdf: gpd.GeoDataFrame, class_col: str, class_order: list[str]
) -> pd.Series:
    total = gdf["length_km"].fillna(0).sum()
    if total <= 0:
        return pd.Series(0.0, index=class_order)
    return pd.Series(
        {
            cls: float(gdf.loc[gdf[class_col] == cls, "length_km"].fillna(0).sum() / total * 100)
            for cls in class_order
        }
    )


def make_stacked_length_table(
    gdf: gpd.GeoDataFrame,
    class_col: str,
    stack_col: str,
    class_order: list[str],
    stack_order: list[str],
) -> pd.DataFrame:
    table = (
        gdf.pivot_table(
            index=class_col,
            columns=stack_col,
            values="length_km",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(index=class_order, fill_value=0)
        .fillna(0)
    )
    cols = [col for col in stack_order if col in table.columns and table[col].sum() > 0]
    return table[cols]


def annotate_length_bars(ax: plt.Axes, totals: pd.Series, percents: pd.Series) -> None:
    ymax = max(float(totals.max()), 1.0)
    ax.set_ylim(0, ymax * 1.2)
    for idx, cls in enumerate(totals.index):
        total = totals.loc[cls]
        if total > 0:
            ax.text(
                idx,
                total + ymax * 0.025,
                f"{total:,.0f} km\n{percents.loc[cls]:.1f}%",
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


def plot_flood_snail_distribution() -> Path:
    print(f"Reading SNAIL flood line segments: {FLOOD_SEGMENTS}")
    flood = gpd.read_file(FLOOD_SEGMENTS)
    flood["flood_depth_class_clean"] = flood["flood_depth_m"].map(classify_depth)
    flood["line_type_clean"] = flood["TYPE"].map(normalize_type)

    exposed_mask = flood["flood_depth_m"].fillna(0) > 0
    exposed_length_pct = length_percent(flood, exposed_mask)
    exposed_length_km = flood.loc[exposed_mask, "length_km"].fillna(0).sum()
    total_length_km = flood["length_km"].fillna(0).sum()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle("SNAIL Transmission Line Flood Exposure", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.91,
        (
            f"{exposed_length_pct:.1f}% of split line length is exposed to >0 m flooding "
            f"({exposed_length_km:,.0f} of {total_length_km:,.0f} km)"
        ),
        ha="center",
        fontsize=11,
    )

    ax = axes[0]
    ax.hist(
        flood["flood_depth_m"].fillna(0),
        bins=FLOOD_HIST_BINS,
        weights=flood["length_km"].fillna(0),
        color="#4f7daf",
        edgecolor="white",
        linewidth=1.1,
    )
    for threshold in [0.5, 1, 2, 5]:
        ax.axvline(threshold, color="#8f8f8f", linestyle="--", linewidth=1)
    ax.set_title("Length-weighted distribution of line flood depth")
    ax.set_xlabel("Flood depth on SNAIL line segment (m)")
    ax.set_ylabel("Transmission line length (km)")
    style_axis(ax)

    ax = axes[1]
    stack_order = [key for key in TYPE_COLORS if key != "Other"] + ["Other"]
    stacked = make_stacked_length_table(
        flood,
        "flood_depth_class_clean",
        "line_type_clean",
        FLOOD_CLASSES,
        stack_order,
    )
    colors = [TYPE_COLORS.get(col, "#bdbdbd") for col in stacked.columns]
    stacked.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
    totals = stacked.sum(axis=1)
    percents = class_length_percents(flood, "flood_depth_class_clean", FLOOD_CLASSES)
    annotate_length_bars(ax, totals, percents)
    ax.set_title("Line length by flood depth class and line type")
    ax.set_xlabel("Flood depth class (m)")
    ax.set_ylabel("Transmission line length (km)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Line type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return save_figure(fig, "snail_flood_line_distribution.png")


def find_wind_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = [
        "fred_tc_wind_speed_ms",
        "storm_tc_wind_speed_ms",
        "tc_wind_speed_ms",
        "tc_wind_speed_max_ms",
        "wind_speed_ms",
    ]
    for col in candidates:
        if col in gdf.columns:
            return col
    raise KeyError(f"Could not find a TC wind speed column. Columns: {list(gdf.columns)}")


def plot_tc_snail_distribution(
    segment_path: Path = TC_SEGMENTS,
    label: str = "OpenGIRA Historical Max",
    output_suffix: str = "open_gira_historical_max",
) -> Path:
    print(f"Reading SNAIL TC line segments: {segment_path}")
    tc = gpd.read_file(segment_path)
    wind_col = find_wind_column(tc)
    tc["tc_wind_class_clean"] = tc[wind_col].map(classify_wind)
    tc["voltage_class_clean"] = tc["voltage_kv"].map(classify_voltage)

    exposed_mask = tc[wind_col].fillna(0) >= 25
    exposed_length_pct = length_percent(tc, exposed_mask)
    exposed_length_km = tc.loc[exposed_mask, "length_km"].fillna(0).sum()
    total_length_km = tc["length_km"].fillna(0).sum()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle(
        f"SNAIL Transmission Line TC Wind Exposure, {label}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.91,
        (
            f"{exposed_length_pct:.1f}% of split line length is exposed to >=25 m/s winds "
            f"({exposed_length_km:,.0f} of {total_length_km:,.0f} km)"
        ),
        ha="center",
        fontsize=11,
    )

    ax = axes[0]
    valid = tc[wind_col].notna()
    ax.hist(
        tc.loc[valid, wind_col],
        bins=TC_HIST_BINS,
        weights=tc.loc[valid, "length_km"].fillna(0),
        color="#4f7daf",
        edgecolor="white",
        linewidth=1.1,
    )
    for threshold in [25, 30, 35, 40]:
        ax.axvline(threshold, color="#8f8f8f", linestyle="--", linewidth=1)
    ax.set_title("Length-weighted distribution of line TC wind exposure")
    ax.set_xlabel("Tropical cyclone wind speed on SNAIL segment (m/s)")
    ax.set_ylabel("Transmission line length (km)")
    style_axis(ax)

    ax = axes[1]
    stacked = make_stacked_length_table(
        tc.dropna(subset=["tc_wind_class_clean"]),
        "tc_wind_class_clean",
        "voltage_class_clean",
        TC_CLASSES,
        VOLTAGE_ORDER,
    )
    colors = [VOLTAGE_COLORS.get(col, "#bdbdbd") for col in stacked.columns]
    stacked.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
    totals = stacked.sum(axis=1)
    percents = class_length_percents(tc, "tc_wind_class_clean", TC_CLASSES)
    annotate_length_bars(ax, totals, percents)
    ax.set_title("Line length by TC wind class and voltage class")
    ax.set_xlabel("Tropical cyclone wind speed class (m/s)")
    ax.set_ylabel("Transmission line length (km)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Voltage class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return save_figure(fig, f"snail_tc_line_distribution_{output_suffix}.png")


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
        plot_flood_snail_distribution(),
        plot_tc_snail_distribution(),
        plot_tc_snail_distribution(
            segment_path=TC_STORM_RP100_SEGMENTS,
            label="STORM RP100",
            output_suffix="storm_rp100",
        ),
    ]
    print("\nCreated SNAIL line distribution figures:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
