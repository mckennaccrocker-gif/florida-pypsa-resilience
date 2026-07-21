"""
Polished power plant polygon hazard maps.

The map uses plant footprint representative points sized by capacity and colored
by hazard class. Panel subtitles keep the area-based interpretation: percentage
of total plant polygon footprint area exposed to the relevant hazard threshold.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
OUTPUT_DIR = EXPOSURE_DIR / "powerplant_polygon_hazard_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOOD_GPKG = EXPOSURE_DIR / "powerplant_polygon_flood_exposure" / "powerplant_polygon_flood_exposure.gpkg"
TC_GPKG = EXPOSURE_DIR / "powerplant_polygon_tc_exposure" / "powerplant_polygon_tc_exposure.gpkg"
LINES_GPKG = ELECTRICITY_DIR / "florida_lines_with_s_nom.gpkg"

FLOOD_CLASSES = ["0 m", "0-0.5 m", "0.5-1 m", "1-2 m", "2-5 m", ">5 m"]
FLOOD_COLORS = {
    "0 m": "#f5f5f5",
    "0-0.5 m": "#bfe3ef",
    "0.5-1 m": "#79b7d4",
    "1-2 m": "#3f76b5",
    "2-5 m": "#f39b52",
    ">5 m": "#c53a32",
}
TC_CLASSES = ["<25", "25-30", "30-35", "35-40", ">40"]
TC_COLORS = {
    "<25": "#3f6fa8",
    "25-30": "#8bd0df",
    "30-35": "#f5e57a",
    "35-40": "#f4a259",
    ">40": "#c73e3a",
}


def weighted_area_percent(gdf: gpd.GeoDataFrame, fraction_col: str) -> float:
    areas = gdf["polygon_area_m2"].fillna(0)
    total_area = areas.sum()
    if total_area <= 0:
        return 0.0
    fractions = gdf[fraction_col].fillna(0).clip(lower=0, upper=1)
    return float((areas * fractions).sum() / total_area * 100)


def exposed_capacity_gw(gdf: gpd.GeoDataFrame, mask: pd.Series) -> float:
    return float(gdf.loc[mask, "capacity_mw"].fillna(0).sum() / 1000)


def capacity_marker_size(capacity_mw: pd.Series) -> pd.Series:
    capacity = capacity_mw.fillna(0).clip(lower=0)
    return np.clip(np.sqrt(capacity) * 4.0, 12, 210)


def add_context_lines(ax: plt.Axes, lines: gpd.GeoDataFrame, target_crs: object) -> None:
    if lines.empty:
        return
    lines.to_crs(target_crs).plot(
        ax=ax,
        color="#d5d8d9",
        linewidth=0.32,
        alpha=0.75,
        zorder=1,
    )


def add_marker_size_legend(ax: plt.Axes) -> None:
    capacities = [100, 1000, 4000]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=0.8,
            markersize=np.sqrt(capacity_marker_size(pd.Series([cap])).iloc[0]),
            label=f"{cap:,} MW",
        )
        for cap in capacities
    ]
    legend = ax.legend(
        handles=handles,
        title="Plant capacity",
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        borderpad=0.7,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d0d0d0")


def color_legend(
    ax: plt.Axes,
    colors: dict[str, str],
    title: str,
    loc: str = "lower left",
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="#333333",
            label=label,
            markersize=8,
        )
        for label, color in colors.items()
    ]
    legend = ax.legend(
        handles=handles,
        title=title,
        loc=loc,
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        borderpad=0.7,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d0d0d0")
    ax.add_artist(legend)


def plot_panel(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    class_col: str,
    colors: dict[str, str],
    title: str,
    subtitle: str,
    legend_title: str,
) -> None:
    add_context_lines(ax, lines, gdf.crs)

    points = gdf.copy()
    points["geometry"] = points.geometry.representative_point()
    points["marker_size"] = capacity_marker_size(points["capacity_mw"])
    points["marker_color"] = points[class_col].astype(object).map(colors).fillna("#f5f5f5")

    ax.scatter(
        points.geometry.x,
        points.geometry.y,
        s=points["marker_size"],
        c=points["marker_color"],
        edgecolors="#202020",
        linewidths=0.42,
        alpha=0.92,
        zorder=3,
    )

    # Draw the real footprint outlines lightly under the markers so the map is
    # still explicitly polygon-based even at statewide scale.
    gdf.boundary.plot(ax=ax, color="#222222", linewidth=0.18, alpha=0.25, zorder=2)

    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx - 0.45, maxx + 0.45)
    ax.set_ylim(miny - 0.35, maxy + 0.25)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(
        0.5,
        1.085,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        0.5,
        1.045,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4a4a4a",
    )
    color_legend(ax, colors, legend_title)
    add_marker_size_legend(ax)


def make_hazard_map(tc_dataset: str, tc_label: str, output_name: str) -> Path:
    print(f"Reading flood polygons: {FLOOD_GPKG}")
    flood = gpd.read_file(FLOOD_GPKG)
    flood = flood[flood["return_period"] == 100].copy()
    flood["flood_class"] = pd.Categorical(flood["flood_class"], FLOOD_CLASSES, ordered=True)

    print(f"Reading TC polygons: {TC_GPKG}")
    tc = gpd.read_file(TC_GPKG)
    tc = tc[tc["dataset"] == tc_dataset].copy()
    tc["tc_wind_class"] = pd.Categorical(tc["tc_wind_class"], TC_CLASSES, ordered=True)

    lines = gpd.read_file(LINES_GPKG) if LINES_GPKG.exists() else gpd.GeoDataFrame()

    flood_area_pct = weighted_area_percent(flood, "wet_pixel_fraction")
    flood_poly_count = int((flood["max_flood_depth_m"].fillna(0) > 0).sum())
    flood_capacity = exposed_capacity_gw(flood, flood["max_flood_depth_m"].fillna(0) > 0)

    tc_area_pct = weighted_area_percent(tc, "fraction_ge_25ms")
    tc_poly_count = int((tc["max_wind_speed_ms"].fillna(0) >= 25).sum())
    tc_capacity = exposed_capacity_gw(tc, tc["max_wind_speed_ms"].fillna(0) >= 25)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Florida Power Plant Polygon Hazard Exposure",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.94,
        "Markers are plant polygon representative points sized by capacity; percentages are area-weighted across plant footprints.",
        ha="center",
        fontsize=11,
        color="#4a4a4a",
    )

    plot_panel(
        axes[0],
        flood,
        lines,
        "flood_class",
        FLOOD_COLORS,
        "JRC RP100 Flood Depth",
        f"{flood_area_pct:.1f}% of total plant footprint area flooded; {flood_poly_count} polygons, {flood_capacity:,.1f} GW",
        "Max depth",
    )
    plot_panel(
        axes[1],
        tc,
        lines,
        "tc_wind_class",
        TC_COLORS,
        f"{tc_label} TC Wind",
        f"{tc_area_pct:.1f}% of total plant footprint area >=25 m/s; {tc_poly_count} polygons, {tc_capacity:,.1f} GW",
        "Wind class (m/s)",
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.91], w_pad=3.5)
    path = OUTPUT_DIR / output_name
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def main() -> None:
    outputs = [
        make_hazard_map(
            tc_dataset="open_gira_historical_max",
            tc_label="OpenGIRA Historical Max",
            output_name="powerplant_polygon_hazard_map_open_gira_historical_max.png",
        ),
        make_hazard_map(
            tc_dataset="storm_rp100",
            tc_label="STORM RP100",
            output_name="powerplant_polygon_hazard_map_storm_rp100.png",
        ),
    ]

    print("\nCreated polished hazard maps:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
