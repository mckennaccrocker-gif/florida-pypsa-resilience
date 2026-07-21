"""
Single-storm direct-damage and cascading-impact maps/plots.

Default storm: 2022266N12294, Hurricane Ian.
Each saved PNG answers one question. A GeoPackage is also written for QGIS.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from shapely.geometry import LineString, Point


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EVENT_ID = "2022266N12294"
EVENT_NAME = "Hurricane Ian"

EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
ASSET_IMPACT_DIR = EXPOSURE_DIR / "florida_new_grid_ibtracs_asset_impacts"
LINE_DAMAGE_DIR = EXPOSURE_DIR / "florida_new_grid_tc_event_damage_snail"
GRID_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_extended_tie_lines"
SCENARIO_DIR = GRID_DIR / "ibtracs_cascading_top5_direct_events" / f"ibtracs_{EVENT_ID}_tc_direct_damage"
OUT_DIR = ASSET_IMPACT_DIR / f"single_storm_{EVENT_ID}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_GPKG = OUT_DIR / f"{EVENT_ID}_direct_and_cascade_layers.gpkg"

GRID_QGIS = GRID_DIR / "qgis" / "florida_extended_tie_lines_for_qgis.gpkg"
LINE_GEOMETRY = LINE_DAMAGE_DIR / "new_grid_lines_used_for_snail.gpkg"
PLANT_POLYGONS = PROJECT_DIR / "data" / "Electricity" / "florida_osm_powerplant_polygons_with_point_attributes.gpkg"
IBTRACS = PROJECT_DIR / "Florida" / "ibtracs.NA.list.v04r01.csv"
US_STATE_BOUNDARIES = PROJECT_DIR / "data" / "Boundaries" / "cb_2024_us_state_500k" / "cb_2024_us_state_500k.shp"

MAP_BOUNDS = (-88.2, -79.3, 24.0, 32.0)
GRAY = "#A0A7B0"
BLUE = "#2B6CB0"
ORANGE = "#DD6B20"
RED = "#C53030"
PURPLE = "#6B46C1"
GREEN = "#2F855A"
OUTLINE = "#111827"


def usd_millions(x, _pos):
    return f"${x / 1e6:.0f}M"


def style_map(ax, title: str) -> None:
    ax.set_xlim(MAP_BOUNDS[0], MAP_BOUNDS[1])
    ax.set_ylim(MAP_BOUNDS[2], MAP_BOUNDS[3])
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#D0D7DE", linewidth=0.6, alpha=0.55)
    ax.set_aspect("equal", adjustable="box")


def load_florida_outline() -> gpd.GeoDataFrame:
    states = gpd.read_file(US_STATE_BOUNDARIES).to_crs("EPSG:4326")
    return states[states["STUSPS"].eq("FL")].copy()


def draw_florida_outline(ax, florida_outline: gpd.GeoDataFrame) -> None:
    if florida_outline.empty:
        return
    florida_outline.boundary.plot(ax=ax, color=OUTLINE, linewidth=1.8, alpha=0.95, zorder=20)


def load_storm_track() -> gpd.GeoDataFrame:
    tracks = pd.read_csv(IBTRACS, low_memory=False)
    storm = tracks[tracks["SID"].astype(str).eq(EVENT_ID)].copy()
    storm["LAT"] = pd.to_numeric(storm["LAT"], errors="coerce")
    storm["LON"] = pd.to_numeric(storm["LON"], errors="coerce")
    storm["USA_WIND"] = pd.to_numeric(storm["USA_WIND"], errors="coerce")
    storm = storm.dropna(subset=["LAT", "LON"])
    points = [Point(xy) for xy in zip(storm["LON"], storm["LAT"])]
    line = LineString(points)
    return gpd.GeoDataFrame(
        [
            {
                "event_id": EVENT_ID,
                "name": storm["NAME"].dropna().iloc[0] if storm["NAME"].notna().any() else EVENT_NAME,
                "season": storm["SEASON"].dropna().iloc[0] if storm["SEASON"].notna().any() else 2022,
                "max_usa_wind_kt": storm["USA_WIND"].max(),
                "track_points": len(storm),
                "geometry": line,
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def load_base_grid() -> gpd.GeoDataFrame:
    lines = gpd.read_file(LINE_GEOMETRY).to_crs("EPSG:4326")
    return lines


def load_direct_damaged_lines(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    damaged = pd.read_csv(SCENARIO_DIR / "damaged_lines.csv")
    damaged = damaged[pd.to_numeric(damaged["damage_ratio"], errors="coerce") > 0].copy()
    merged = lines.merge(damaged, left_on="asset_line_id", right_on="line", how="inner")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=lines.crs)


def load_direct_damaged_substations() -> gpd.GeoDataFrame:
    buses = pd.read_csv(SCENARIO_DIR / "bus_substation_deratings.csv")
    buses = buses[pd.to_numeric(buses["damage_ratio"], errors="coerce") > 0].copy()
    buses["geometry"] = [Point(xy) for xy in zip(buses["x"], buses["y"])]
    return gpd.GeoDataFrame(buses, geometry="geometry", crs="EPSG:4326")


def load_powerplant_exposure() -> gpd.GeoDataFrame:
    impacts = pd.read_csv(ASSET_IMPACT_DIR / "direct_powerplant_polygon_event_damage.csv")
    impacts = impacts[impacts["event_id"].astype(str).eq(EVENT_ID)].copy()
    impacts["max_wind_speed_ms"] = pd.to_numeric(impacts["max_wind_speed_ms"], errors="coerce")
    impacts["damage_ratio"] = pd.to_numeric(impacts["damage_ratio"], errors="coerce").fillna(0.0)
    impacts = impacts[impacts["max_wind_speed_ms"].fillna(0) >= 25].copy()

    plants = gpd.read_file(PLANT_POLYGONS).to_crs("EPSG:4326")
    merged = plants.merge(
        impacts[
            [
                "plant_asset_id",
                "max_wind_speed_ms",
                "damage_ratio",
                "capacity_loss_mw",
                "assigned_curve_id",
                "assigned_curve_description",
            ]
        ],
        left_on="polygon_osm_id",
        right_on="plant_asset_id",
        how="inner",
    )
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


def load_cascade_load_shedding() -> gpd.GeoDataFrame:
    shedding = pd.read_csv(SCENARIO_DIR / "load_shedding_by_bus.csv")
    buses = gpd.read_file(GRID_QGIS, layer="cleaned_buses").to_crs("EPSG:4326")
    merged = buses.merge(shedding, left_on="name", right_on="bus", how="inner")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


def load_cascade_loaded_lines(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    loading = pd.read_csv(SCENARIO_DIR / "line_loading.csv")
    loading = loading[pd.to_numeric(loading["max_loading_pu"], errors="coerce") >= 0.90].copy()
    merged = lines.merge(loading, left_on="asset_line_id", right_on="line", how="inner")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=lines.crs)


def write_qgis_layers(layers: dict[str, gpd.GeoDataFrame]) -> None:
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    for layer_name, gdf in layers.items():
        if gdf.empty:
            continue
        gdf.to_file(OUT_GPKG, layer=layer_name, driver="GPKG")


def plot_direct_damage_map(base_lines, track, damaged_lines, damaged_substations, florida_outline) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    base_lines.plot(ax=ax, color="#D8DEE4", linewidth=0.25, alpha=0.75)
    damaged_lines.plot(
        ax=ax,
        column="damage_ratio",
        cmap="YlOrRd",
        linewidth=1.2,
        legend=True,
        legend_kwds={"label": "Line damage ratio", "shrink": 0.55},
    )
    damaged_substations.plot(
        ax=ax,
        markersize=(damaged_substations["damage_ratio"].clip(lower=0.001) * 550) + 12,
        color=PURPLE,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    draw_florida_outline(ax, florida_outline)
    track.plot(ax=ax, color="#111827", linewidth=2.2)
    style_map(ax, f"{EVENT_NAME} {EVENT_ID}: Directly Damaged Lines and Substations")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#111827", lw=2, label="Storm track"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE, label="Damaged substation bus", markersize=8),
        ],
        loc="lower left",
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "map_direct_damaged_lines_and_substations.png", dpi=300)
    plt.close(fig)


def plot_powerplant_exposure_map(base_lines, track, plants, florida_outline) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    base_lines.plot(ax=ax, color="#E5E7EB", linewidth=0.25, alpha=0.65)
    if not plants.empty:
        plants.plot(
            ax=ax,
            column="max_wind_speed_ms",
            cmap="Oranges",
            edgecolor="#7C2D12",
            linewidth=0.35,
            alpha=0.8,
            legend=True,
            legend_kwds={"label": "Max wind over plant polygon (m/s)", "shrink": 0.55},
        )
    draw_florida_outline(ax, florida_outline)
    track.plot(ax=ax, color="#111827", linewidth=2.2)
    style_map(ax, f"{EVENT_NAME}: Power Plant Polygons Exposed to >=25 m/s")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "map_powerplant_polygons_exposed_ge25ms.png", dpi=300)
    plt.close(fig)


def plot_cascade_load_shedding_map(base_lines, track, load_shed, florida_outline) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    base_lines.plot(ax=ax, color="#E5E7EB", linewidth=0.25, alpha=0.65)
    sizes = (load_shed["total_load_shed_mwh"] / load_shed["total_load_shed_mwh"].max() * 260) + 20
    load_shed.plot(ax=ax, markersize=sizes, color=RED, alpha=0.75, edgecolor="white", linewidth=0.45)
    draw_florida_outline(ax, florida_outline)
    track.plot(ax=ax, color="#111827", linewidth=2.2)
    style_map(ax, f"{EVENT_NAME}: Cascading Load-Shedding Buses After Damage")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#111827", lw=2, label="Storm track"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, label="Load-shedding bus", markersize=8),
        ],
        loc="lower left",
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "map_cascade_load_shedding_buses.png", dpi=300)
    plt.close(fig)


def plot_cascade_loaded_lines_map(base_lines, track, loaded_lines, florida_outline) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    base_lines.plot(ax=ax, color="#E5E7EB", linewidth=0.25, alpha=0.65)
    loaded_lines.plot(
        ax=ax,
        column="max_loading_pu",
        cmap="Blues",
        linewidth=1.3,
        legend=True,
        legend_kwds={"label": "Max loading (p.u.)", "shrink": 0.55},
    )
    draw_florida_outline(ax, florida_outline)
    track.plot(ax=ax, color="#111827", linewidth=2.2)
    style_map(ax, f"{EVENT_NAME}: Highly Loaded Lines After Damage")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "map_cascade_highly_loaded_lines.png", dpi=300)
    plt.close(fig)


def plot_line_capacity_loss_bar(damaged_lines) -> None:
    line_loss = float(damaged_lines["capacity_loss_mva"].sum())
    original_capacity = float(damaged_lines["original_s_nom_mva"].sum())
    percent_loss = (line_loss / original_capacity * 100) if original_capacity > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    bar = ax.bar(["Transmission lines"], [line_loss], color=BLUE, width=0.45)
    ax.set_title(f"{EVENT_NAME}: Transmission Line Capacity Lost", fontsize=14, weight="bold")
    ax.set_ylabel("Capacity loss from direct wind damage (MVA)")
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        bar[0].get_x() + bar[0].get_width() / 2,
        line_loss,
        f"{line_loss:,.0f} MVA\n{percent_loss:.1f}% of directly damaged-line capacity",
        ha="center",
        va="bottom",
        fontsize=10,
        weight="bold",
    )
    ax.set_ylim(0, line_loss * 1.22 if line_loss > 0 else 1)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "bar_line_capacity_lost_ian.png", dpi=300)
    plt.close(fig)


def plot_cascade_incremental_cost_bar() -> None:
    baseline = pd.read_csv(GRID_DIR / "baseline_calibrated_no_hazard" / "baseline_summary.csv").iloc[0]
    scenario = pd.read_csv(SCENARIO_DIR / "scenario_summary.csv").iloc[0]
    incremental_cost = float(scenario["total_system_cost_usd"]) - float(baseline["total_system_cost_usd"])
    values = pd.DataFrame(
        {
            "case": [f"{EVENT_NAME} damage"],
            "incremental_system_cost_usd": [incremental_cost],
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(values["case"], values["incremental_system_cost_usd"], color=RED, width=0.45)
    ax.set_title(f"{EVENT_NAME}: Incremental PyPSA System Cost", fontsize=14, weight="bold")
    ax.set_ylabel("Extra 24-hour system cost vs no-hazard baseline (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(usd_millions))
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"${value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )
    ax.set_ylim(0, max(incremental_cost * 1.25, 1))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "bar_cascade_incremental_system_cost.png", dpi=300)
    plt.close(fig)


def plot_load_shedding_line() -> None:
    baseline = pd.read_csv(GRID_DIR / "baseline_calibrated_no_hazard" / "baseline_load_shedding.csv")
    scenario = pd.read_csv(SCENARIO_DIR / "load_shedding.csv")
    baseline["snapshot"] = pd.to_datetime(baseline["snapshot"])
    scenario["snapshot"] = pd.to_datetime(scenario["snapshot"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(baseline["snapshot"], baseline["load_shed_mw"], color=GRAY, linewidth=2, label="No-hazard baseline")
    ax.plot(scenario["snapshot"], scenario["load_shed_mw"], color=RED, linewidth=2, linestyle="--", label=f"After {EVENT_NAME} damage")
    ax.set_title(f"{EVENT_NAME}: Hourly Load Shedding Before and After Damage", fontsize=14, weight="bold")
    ax.set_xlabel("Snapshot")
    ax.set_ylabel("Load shed (MW)")
    ax.grid(color="#D0D7DE", linewidth=0.8, alpha=0.6)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "line_load_shedding_baseline_vs_storm.png", dpi=300)
    plt.close(fig)


def main() -> None:
    base_lines = load_base_grid()
    florida_outline = load_florida_outline()
    storm_track = load_storm_track()
    damaged_lines = load_direct_damaged_lines(base_lines)
    damaged_substations = load_direct_damaged_substations()
    exposed_plants = load_powerplant_exposure()
    load_shed = load_cascade_load_shedding()
    loaded_lines = load_cascade_loaded_lines(base_lines)

    write_qgis_layers(
        {
            "storm_track": storm_track,
            "florida_state_outline": florida_outline,
            "direct_damaged_lines": damaged_lines,
            "direct_damaged_substations": damaged_substations,
            "powerplant_polygons_exposed_ge25ms": exposed_plants,
            "cascade_load_shedding_buses": load_shed,
            "cascade_highly_loaded_lines_ge90pct": loaded_lines,
        }
    )

    plot_direct_damage_map(base_lines, storm_track, damaged_lines, damaged_substations, florida_outline)
    plot_powerplant_exposure_map(base_lines, storm_track, exposed_plants, florida_outline)
    plot_cascade_load_shedding_map(base_lines, storm_track, load_shed, florida_outline)
    plot_cascade_loaded_lines_map(base_lines, storm_track, loaded_lines, florida_outline)
    plot_line_capacity_loss_bar(damaged_lines)
    plot_cascade_incremental_cost_bar()
    plot_load_shedding_line()

    summary = pd.DataFrame(
        [
            {
                "event_id": EVENT_ID,
                "event_name": EVENT_NAME,
                "direct_damaged_lines": len(damaged_lines),
                "direct_damaged_substations": len(damaged_substations),
                "powerplant_polygons_exposed_ge25ms": len(exposed_plants),
                "modeled_powerplant_capacity_loss_mw": float(exposed_plants["capacity_loss_mw"].sum()) if not exposed_plants.empty else 0.0,
                "cascade_load_shedding_buses": len(load_shed),
                "cascade_highly_loaded_lines_ge90pct": len(loaded_lines),
                "output_dir": str(OUT_DIR),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "single_storm_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved QGIS layers: {OUT_GPKG}")


if __name__ == "__main__":
    main()
