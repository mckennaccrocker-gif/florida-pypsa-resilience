"""
Intersect Florida electricity assets with the open-gira tropical cyclone wind grid.

This script uses:
  - data/Electricty/Nodes2.gpkg
  - data/Electricty/TransmissionLines2.gpkg
  - open_gira_outputs/max_wind_field_FLORIDA_IBTrACS_0.nc

Outputs are written to:
  - data/Exposure/florida_open_gira_tc_exposure/

For transmission lines, the default is to keep only above-ground / overhead lines,
matching the earlier tropical cyclone exposure assumption.
"""

from __future__ import annotations

import os
import re
import sys
from math import erf, log, sqrt
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
from shapely import contains_xy
from shapely.geometry import LineString, MultiLineString


PROJECT_DIR = Path(__file__).resolve().parents[2]
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricty"  # project folder is misspelled
COST_DIR = PROJECT_DIR / "data" / "Cost"
OUTPUT_DIR = PROJECT_DIR / "data" / "Exposure" / "florida_open_gira_tc_exposure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODES_FILE = ELECTRICITY_DIR / "Nodes2.gpkg"
LINES_FILE = ELECTRICITY_DIR / "TransmissionLines2.gpkg"
WIND_NC = PROJECT_DIR / "open_gira_outputs" / "max_wind_field_FLORIDA_IBTrACS_0.nc"
OLD_STORM_RASTER = (
    PROJECT_DIR
    / "data"
    / "Hazards"
    / "Tropical_cyclones"
    / "STORM_constant_100yr_US_crop.tif"
)
STORM_RP_RASTERS = {
    10: PROJECT_DIR / "data" / "Hazards" / "Tropical_cyclones" / "STORM_constant_RP10_US_crop.tif",
    20: PROJECT_DIR / "data" / "Hazards" / "Tropical_cyclones" / "STORM_constant_RP20_US_crop.tif",
    50: PROJECT_DIR / "data" / "Hazards" / "Tropical_cyclones" / "STORM_constant_RP50_US_crop.tif",
    100: PROJECT_DIR / "data" / "Hazards" / "Tropical_cyclones" / "STORM_constant_RP100_US_crop.tif",
    200: PROJECT_DIR / "data" / "Hazards" / "Tropical_cyclones" / "STORM_constant_RP200_US_crop.tif",
    500: PROJECT_DIR / "data" / "Hazards" / "Tropical_cyclones" / "STORM_constant_RP500_US_crop.tif",
}
NHESS_W310_CURVE = COST_DIR / "nhess_w310_power_tower_160kmh_urban_curve.csv"

# This boundary is created by the open-gira workflow in WSL.
FLORIDA_BOUNDARIES = (
    Path.home()
    / "projects"
    / "open-gira"
    / "results"
    / "input"
    / "admin-boundaries"
    / "gadm36_levels.gpkg"
)

LINE_TYPE_FILTER = "OVERHEAD"
LINE_SEGMENT_SPACING_M = 5000

LAT_NAME = "latitude"
LON_NAME = "longitude"
EVENT_NAME = "event_id"
WIND_VAR = "max_wind_speed"

TC_BINS = [0, 25, 30, 35, 40, np.inf]
TC_LABELS = ["<25", "25-30", "30-35", "35-40", ">40"]
TC_CLASS_COLORS = {
    "<25": "#2c7bb6",
    "25-30": "#abd9e9",
    "30-35": "#ffffbf",
    "35-40": "#fdae61",
    ">40": "#d7191c",
}
FUEL_COLORS = {
    "Solar": "#f4c430",
    "Gas": "#7f7f7f",
    "Hydro": "#2b8cbe",
    "Wind": "#41ab5d",
    "Oil": "#252525",
    "Waste": "#8c6bb1",
    "Coal": "#4d4d4d",
    "Biomass": "#a1d99b",
    "Storage": "#fb6a4a",
    "Geothermal": "#d95f0e",
    "Nuclear": "#31a354",
    "Cogeneration": "#756bb1",
    "Other": "#969696",
    "Petcoke": "#8c510a",
    "Unknown": "#bdbdbd",
}

STANDARD_VOLTAGES = np.array([69, 115, 161, 230, 345, 500, 765])
LINE_COST_PER_MILE_USD = {
    69: 1_500_000,
    115: 2_500_000,
    161: 2_750_000,
    230: 3_000_000,
    345: 3_050_000,
    500: 3_600_000,
    765: 5_900_000,
}


# Help GDAL/pyproj find projection metadata when run from the Pixi environment.
os.environ.setdefault("PROJ_LIB", str(Path(sys.prefix) / "share" / "proj"))


def load_florida_geometry(boundaries_path: Path):
    if not boundaries_path.exists():
        raise FileNotFoundError(
            f"Florida boundary source not found: {boundaries_path}\n"
            "Run this from WSL after the open-gira workflow has created the "
            "GADM admin-boundary file."
        )

    states = gpd.read_file(boundaries_path, layer="level1")
    florida = states[(states["GID_0"] == "USA") & (states["NAME_1"] == "Florida")]
    if florida.empty:
        raise ValueError("Could not find Florida in GADM level1 boundaries.")

    return florida.to_crs("EPSG:4326").geometry.union_all()


def filter_to_florida(gdf: gpd.GeoDataFrame, florida_geometry) -> gpd.GeoDataFrame:
    gdf_4326 = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = florida_geometry.bounds
    gdf_4326 = gdf_4326.cx[minx:maxx, miny:maxy].copy()
    return gdf_4326[gdf_4326.intersects(florida_geometry)].copy()


def filter_lines_by_type(lines: gpd.GeoDataFrame, target: str | None) -> gpd.GeoDataFrame:
    if not target:
        return lines.copy()

    type_text = lines["TYPE"].fillna("").astype(str)
    return lines[type_text.str.contains(target, case=False, na=False)].copy()


def classify_wind_speed(speed):
    if pd.isna(speed):
        return np.nan
    return pd.cut(
        [speed],
        bins=TC_BINS,
        labels=TC_LABELS,
        include_lowest=True,
    )[0]


def sample_event_winds(wind: xr.DataArray, lon_values, lat_values) -> xr.DataArray:
    sample_index = np.arange(len(lon_values))
    return wind.sel(
        {
            LAT_NAME: xr.DataArray(lat_values, dims="sample", coords={"sample": sample_index}),
            LON_NAME: xr.DataArray(lon_values, dims="sample", coords={"sample": sample_index}),
        },
        method="nearest",
    )


def sample_nodes(nodes: gpd.GeoDataFrame, wind: xr.DataArray) -> gpd.GeoDataFrame:
    coords = nodes.geometry
    sampled = sample_event_winds(wind, coords.x.to_numpy(), coords.y.to_numpy())

    nodes_out = nodes.copy()
    max_by_node = sampled.max(dim=EVENT_NAME, skipna=True)
    event_index = sampled.fillna(-np.inf).argmax(dim=EVENT_NAME)
    event_ids = wind[EVENT_NAME].values

    nodes_out["tc_wind_speed_max_ms"] = max_by_node.values
    nodes_out["tc_event_id_at_max"] = [
        event_ids[int(index)] if np.isfinite(value) else None
        for index, value in zip(event_index.values, max_by_node.values)
    ]
    nodes_out["tc_wind_class"] = nodes_out["tc_wind_speed_max_ms"].apply(classify_wind_speed)
    nodes_out["tc_wind_class"] = pd.Categorical(
        nodes_out["tc_wind_class"],
        categories=TC_LABELS,
        ordered=True,
    )

    return nodes_out


def iter_line_parts(geometry):
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def make_line_segments(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lines_metric = lines.to_crs("EPSG:5070")
    records = []

    for line_id, row in lines_metric.reset_index(drop=True).iterrows():
        for part_id, part in enumerate(iter_line_parts(row.geometry)):
            if part is None or part.is_empty or part.length == 0:
                continue

            n_segments = max(1, int(np.ceil(part.length / LINE_SEGMENT_SPACING_M)))
            distances = np.linspace(0, part.length, n_segments + 1)
            points = [part.interpolate(distance) for distance in distances]

            for segment_id, (start, end) in enumerate(zip(points[:-1], points[1:])):
                segment = LineString([start, end])
                record = row.drop(labels="geometry").to_dict()
                record.update(
                    {
                        "source_line_id": line_id,
                        "source_part_id": part_id,
                        "source_segment_id": segment_id,
                        "length_km": segment.length / 1000,
                        "geometry": segment,
                    }
                )
                records.append(record)

    return gpd.GeoDataFrame(records, crs=lines_metric.crs)


def sample_line_segments(segments: gpd.GeoDataFrame, wind: xr.DataArray) -> gpd.GeoDataFrame:
    segments_out = segments.copy()
    midpoint_metric = segments_out.geometry.interpolate(0.5, normalized=True)

    transformer = Transformer.from_crs(segments_out.crs, "EPSG:4326", always_xy=True)
    lon_values, lat_values = transformer.transform(midpoint_metric.x.to_numpy(), midpoint_metric.y.to_numpy())

    sampled = sample_event_winds(wind, np.asarray(lon_values), np.asarray(lat_values))
    max_by_segment = sampled.max(dim=EVENT_NAME, skipna=True)
    event_index = sampled.fillna(-np.inf).argmax(dim=EVENT_NAME)
    event_ids = wind[EVENT_NAME].values

    segments_out["tc_wind_speed_max_ms"] = max_by_segment.values
    segments_out["tc_event_id_at_max"] = [
        event_ids[int(index)] if np.isfinite(value) else None
        for index, value in zip(event_index.values, max_by_segment.values)
    ]
    segments_out["tc_wind_class"] = segments_out["tc_wind_speed_max_ms"].apply(classify_wind_speed)
    segments_out["tc_wind_class"] = pd.Categorical(
        segments_out["tc_wind_class"],
        categories=TC_LABELS,
        ordered=True,
    )

    return segments_out.to_crs("EPSG:4326")


def sample_old_storm_raster_at_points(points: gpd.GeoDataFrame, raster_path: Path) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        points_raster = points.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in points_raster.geometry]
        values = np.array([value[0] for value in src.sample(coords)], dtype=float)

        if src.nodata is not None:
            values[values == src.nodata] = np.nan

    return values


def sample_old_storm_raster_at_line_midpoints(
    segments: gpd.GeoDataFrame,
    raster_path: Path,
) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        segments_metric = segments.to_crs("EPSG:5070")
        midpoints = segments_metric.geometry.interpolate(0.5, normalized=True)
        midpoint_gdf = gpd.GeoDataFrame(geometry=midpoints, crs=segments_metric.crs).to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in midpoint_gdf.geometry]
        values = np.array([value[0] for value in src.sample(coords)], dtype=float)

        if src.nodata is not None:
            values[values == src.nodata] = np.nan

    return values


def summarize_count_by_class(frame: pd.DataFrame, class_column: str, value_name: str) -> pd.DataFrame:
    summary = (
        frame.dropna(subset=[class_column])
        .groupby(class_column, observed=False)
        .size()
        .reset_index(name=value_name)
    )
    return summary


def summarize_length_by_class(frame: pd.DataFrame, class_column: str) -> pd.DataFrame:
    return (
        frame.dropna(subset=[class_column])
        .groupby(class_column, observed=False)["length_km"]
        .sum()
        .reset_index()
    )


def add_percent(summary: pd.DataFrame, value_column: str) -> pd.DataFrame:
    total = summary[value_column].sum()
    summary["percent"] = np.where(total > 0, summary[value_column] / total * 100, 0)
    return summary


def save_bar_with_percent(summary, class_column, value_column, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary[class_column].astype(str), summary[value_column])
    max_value = summary[value_column].max()
    label_offset = max_value * 0.02 if max_value > 0 else 0.1

    for bar, percent in zip(bars, summary["percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{percent:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Maximum tropical cyclone wind speed (m/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max_value * 1.14 if max_value > 0 else 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_stacked_node_fuel_chart(nodes_out: gpd.GeoDataFrame, output_path: Path):
    nodes_plot = nodes_out.copy()
    nodes_plot["primary_fuel"] = nodes_plot["primary_fuel"].fillna("Unknown")

    summary = (
        nodes_plot.dropna(subset=["tc_wind_class"])
        .groupby(["tc_wind_class", "primary_fuel"], observed=False)
        .size()
        .reset_index(name="node_count")
    )
    summary.to_csv(OUTPUT_DIR / "florida_nodes_open_gira_tc_by_class_and_fuel.csv", index=False)

    wide = (
        summary.pivot_table(
            index="tc_wind_class",
            columns="primary_fuel",
            values="node_count",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(TC_LABELS)
    )
    fuel_order = wide.sum(axis=0).sort_values(ascending=False).index
    wide = wide[fuel_order]
    wide["total"] = wide.sum(axis=1)
    wide = wide.reset_index()
    wide.to_csv(OUTPUT_DIR / "florida_nodes_open_gira_tc_by_class_and_fuel_wide.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(wide))
    x = np.arange(len(wide))
    fuel_columns = [column for column in wide.columns if column not in ["tc_wind_class", "total"]]

    for fuel in fuel_columns:
        values = wide[fuel].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=fuel,
            color=FUEL_COLORS.get(fuel, FUEL_COLORS["Unknown"]),
        )
        bottom += values

    max_total = wide["total"].max()
    label_offset = max_total * 0.02 if max_total > 0 else 0.1
    for index, total in enumerate(wide["total"]):
        ax.text(index, total + label_offset, f"{int(total):,}", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(wide["tc_wind_class"].astype(str))
    ax.set_xlabel("Maximum tropical cyclone wind speed (m/s)")
    ax.set_ylabel("Number of electricity nodes")
    ax.set_title("Florida Electricity Nodes by TC Wind Class and Energy Type")
    ax.set_ylim(0, max_total * 1.16 if max_total > 0 else 1)
    ax.legend(title="Energy type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_line_map(segments: gpd.GeoDataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 9))

    for tc_class in TC_LABELS:
        class_lines = segments[segments["tc_wind_class"].astype(str) == tc_class]
        if class_lines.empty:
            continue
        class_lines.plot(
            ax=ax,
            color=TC_CLASS_COLORS[tc_class],
            linewidth=0.55,
            label=tc_class,
        )

    ax.set_title("Florida Overhead Transmission Lines by TC Wind Class")
    ax.set_axis_off()
    ax.legend(title="TC wind speed\n(m/s)", loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_max_wind_footprint_map(
    wind: xr.DataArray,
    florida_geometry,
    nodes: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    output_path: Path,
):
    max_wind = wind.max(dim=EVENT_NAME, skipna=True)
    florida_boundary = gpd.GeoDataFrame(geometry=[florida_geometry], crs="EPSG:4326")

    minx, miny, maxx, maxy = florida_boundary.total_bounds

    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    mesh = ax.pcolormesh(
        max_wind[LON_NAME],
        max_wind[LAT_NAME],
        max_wind,
        cmap="YlOrRd",
        shading="auto",
    )
    florida_boundary.boundary.plot(ax=ax, color="black", linewidth=1.2)
    lines.plot(ax=ax, color="#3b3b3b", linewidth=0.22, alpha=0.35)
    nodes.plot(ax=ax, color="#1f78b4", markersize=9, alpha=0.78)

    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.035)
    colorbar.set_label("Maximum wind speed (m/s)")
    ax.set_title("Florida Maximum Tropical Cyclone Wind Footprint")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(minx - 0.25, maxx + 0.25)
    ax.set_ylim(miny - 0.25, maxy + 0.25)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_node_wind_histogram(nodes: gpd.GeoDataFrame, output_path: Path):
    values = nodes["tc_wind_speed_max_ms"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        values,
        bins=np.arange(20, 51, 2.5),
        color="#4c78a8",
        edgecolor="white",
    )

    for threshold in [25, 30, 35, 40]:
        ax.axvline(threshold, color="#333333", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xlabel("Maximum tropical cyclone wind speed at node (m/s)")
    ax.set_ylabel("Number of electricity nodes")
    ax.set_title("Distribution of Florida Electricity Node TC Wind Exposure")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def normal_cdf(value):
    return 0.5 * (1 + erf(value / sqrt(2)))


def baseline_expected_damage_ratio(wind_speed_ms):
    if pd.isna(wind_speed_ms) or wind_speed_ms <= 0:
        return 0.0

    theta_values = np.array([30.0, 42.0, 55.0, 67.0])
    damage_ratios = np.array([0.05, 0.20, 0.50, 1.00])
    beta = 0.25

    exceedance = np.array(
        [
            normal_cdf(log(float(wind_speed_ms) / theta) / beta)
            for theta in theta_values
        ]
    )
    state_probabilities = exceedance - np.append(exceedance[1:], 0.0)
    state_probabilities = np.clip(state_probabilities, 0, 1)
    return float(np.sum(state_probabilities * damage_ratios))


def parse_voltage_kv(value):
    """Return the highest listed voltage in kV."""
    if pd.isna(value):
        return np.nan

    text = str(value).upper()
    if "DC" in text and not re.search(r"\d", text):
        return np.nan

    numbers = [
        float(match)
        for match in re.findall(r"\d+(?:\.\d+)?", text)
    ]
    if not numbers:
        return np.nan

    voltages_kv = [
        number / 1000 if number >= 1000 else number
        for number in numbers
        if number > 0
    ]
    if not voltages_kv:
        return np.nan

    return max(voltages_kv)


def snap_voltage_to_cost_class(voltage_kv):
    if pd.isna(voltage_kv):
        return np.nan
    if voltage_kv >= STANDARD_VOLTAGES.max():
        return int(STANDARD_VOLTAGES.max())
    nearest_index = np.argmin(np.abs(STANDARD_VOLTAGES - voltage_kv))
    return int(STANDARD_VOLTAGES[nearest_index])


def add_line_cost_damage(line_segments_tc: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    damage = line_segments_tc.copy()

    voltage_source = "VOLT_CLASS" if "VOLT_CLASS" in damage.columns else "VOLTAGE"
    damage["voltage_kv_raw"] = damage[voltage_source].apply(parse_voltage_kv)
    damage["voltage_kv_cost_class"] = damage["voltage_kv_raw"].apply(
        snap_voltage_to_cost_class
    )
    damage["line_cost_per_mile_usd"] = damage["voltage_kv_cost_class"].map(
        LINE_COST_PER_MILE_USD
    )
    damage["length_miles"] = damage["length_km"] / 1.609344
    damage["replacement_cost_usd"] = (
        damage["length_miles"]
        * damage["line_cost_per_mile_usd"]
    )

    damage["old_tc_damage_ratio"] = damage["old_tc_wind_speed_ms"].apply(
        baseline_expected_damage_ratio
    )
    damage["new_tc_damage_ratio"] = damage["tc_wind_speed_max_ms"].apply(
        baseline_expected_damage_ratio
    )
    damage["old_tc_damage_usd"] = (
        damage["replacement_cost_usd"].fillna(0)
        * damage["old_tc_damage_ratio"]
    )
    damage["new_tc_damage_usd"] = (
        damage["replacement_cost_usd"].fillna(0)
        * damage["new_tc_damage_ratio"]
    )
    damage["damage_difference_usd"] = (
        damage["new_tc_damage_usd"]
        - damage["old_tc_damage_usd"]
    )
    return damage


def save_damage_comparison_outputs(line_damage: gpd.GeoDataFrame):
    output_gpkg = OUTPUT_DIR / "florida_lines_old_storm_vs_open_gira_tc_damage.gpkg"
    line_damage.to_file(
        output_gpkg,
        layer="florida_lines_old_storm_vs_open_gira_tc_damage",
        driver="GPKG",
    )
    line_damage.drop(columns="geometry").to_csv(
        OUTPUT_DIR / "florida_lines_old_storm_vs_open_gira_tc_damage.csv",
        index=False,
    )

    total_replacement = line_damage["replacement_cost_usd"].sum()
    old_total = line_damage["old_tc_damage_usd"].sum()
    new_total = line_damage["new_tc_damage_usd"].sum()

    summary = pd.DataFrame(
        {
            "dataset": [
                "Old STORM 100-year raster",
                "New Fred/open-gira historical IBTrACS maximum",
            ],
            "replacement_cost_usd": [total_replacement, total_replacement],
            "tc_damage_usd": [old_total, new_total],
            "mean_damage_ratio_weighted_by_replacement_cost": [
                old_total / total_replacement if total_replacement > 0 else 0,
                new_total / total_replacement if total_replacement > 0 else 0,
            ],
        }
    )
    summary["difference_vs_old_usd"] = summary["tc_damage_usd"] - old_total
    summary["difference_vs_old_percent"] = np.where(
        old_total > 0,
        summary["difference_vs_old_usd"] / old_total * 100,
        0,
    )
    summary.to_csv(
        OUTPUT_DIR / "florida_lines_old_storm_vs_open_gira_tc_damage_summary.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Old STORM\n100-year", "New Fred/open-gira\nhistorical max"],
        summary["tc_damage_usd"] / 1e9,
        color=["#6baed6", "#fb6a4a"],
    )
    max_value = (summary["tc_damage_usd"] / 1e9).max()
    for bar, value in zip(bars, summary["tc_damage_usd"] / 1e9):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_value * 0.02,
            f"${value:.2f}B",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Expected transmission line damage (billion USD)")
    ax.set_title("Florida TC Wind Damage Cost: Old STORM vs New Fred/open-gira")
    ax.set_ylim(0, max_value * 1.16 if max_value > 0 else 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "florida_lines_old_storm_vs_open_gira_tc_damage_cost.png", dpi=300)
    plt.close()

    voltage_summary = (
        line_damage
        .groupby("voltage_kv_cost_class", dropna=False)
        .agg(
            replacement_cost_usd=("replacement_cost_usd", "sum"),
            old_tc_damage_usd=("old_tc_damage_usd", "sum"),
            new_tc_damage_usd=("new_tc_damage_usd", "sum"),
            length_km=("length_km", "sum"),
        )
        .reset_index()
    )
    voltage_summary.to_csv(
        OUTPUT_DIR / "florida_lines_old_storm_vs_open_gira_tc_damage_by_voltage.csv",
        index=False,
    )

    plot_voltage = voltage_summary.dropna(subset=["voltage_kv_cost_class"]).copy()
    plot_voltage["voltage_kv_cost_class"] = (
        plot_voltage["voltage_kv_cost_class"].astype(int).astype(str)
    )
    x = np.arange(len(plot_voltage))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        plot_voltage["old_tc_damage_usd"] / 1e6,
        width,
        label="Old STORM 100-year",
        color="#6baed6",
    )
    ax.bar(
        x + width / 2,
        plot_voltage["new_tc_damage_usd"] / 1e6,
        width,
        label="New Fred/open-gira historical max",
        color="#fb6a4a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_voltage["voltage_kv_cost_class"])
    ax.set_xlabel("Voltage cost class (kV)")
    ax.set_ylabel("Expected damage (million USD)")
    ax.set_title("Florida TC Wind Damage by Voltage Class")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "florida_lines_old_storm_vs_open_gira_tc_damage_by_voltage.png", dpi=300)
    plt.close()

    return summary, voltage_summary


def save_storm_return_period_damage_curve(
    line_damage: gpd.GeoDataFrame,
    fred_damage_usd: float,
):
    records = []

    for return_period, raster_path in sorted(STORM_RP_RASTERS.items()):
        if not raster_path.exists():
            print(f"Skipping STORM RP{return_period}; missing raster: {raster_path}")
            continue

        wind_values = sample_old_storm_raster_at_line_midpoints(line_damage, raster_path)
        damage_ratios = np.array(
            [baseline_expected_damage_ratio(value) for value in wind_values],
            dtype=float,
        )
        damage_usd = line_damage["replacement_cost_usd"].fillna(0).to_numpy() * damage_ratios

        records.append(
            {
                "return_period": return_period,
                "annual_exceedance_probability": 1 / return_period,
                "storm_damage_usd": float(np.nansum(damage_usd)),
                "mean_wind_speed_ms": float(np.nanmean(wind_values)),
                "max_wind_speed_ms": float(np.nanmax(wind_values)),
                "mean_damage_ratio": float(np.nanmean(damage_ratios)),
            }
        )

    rp_damage = pd.DataFrame(records).sort_values("return_period")
    rp_damage["fred_open_gira_historical_max_damage_usd"] = fred_damage_usd
    rp_damage["difference_vs_fred_usd"] = (
        rp_damage["storm_damage_usd"]
        - rp_damage["fred_open_gira_historical_max_damage_usd"]
    )
    rp_damage.to_csv(
        OUTPUT_DIR / "florida_storm_return_period_damage_vs_fred.csv",
        index=False,
    )

    if rp_damage.empty:
        return rp_damage

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        rp_damage["return_period"],
        rp_damage["storm_damage_usd"] / 1e9,
        marker="o",
        linewidth=2,
        color="#2c7fb8",
        label="STORM return-period rasters",
    )
    ax.axhline(
        fred_damage_usd / 1e9,
        color="#d95f0e",
        linestyle="--",
        linewidth=2,
        label="Fred/open-gira historical IBTrACS maximum",
    )

    for row in rp_damage.itertuples(index=False):
        ax.text(
            row.return_period,
            row.storm_damage_usd / 1e9,
            f"${row.storm_damage_usd / 1e9:.2f}B",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.text(
        rp_damage["return_period"].max(),
        fred_damage_usd / 1e9,
        f"  Fred/open-gira ${fred_damage_usd / 1e9:.2f}B",
        va="center",
        ha="left",
        color="#d95f0e",
    )

    ax.set_xscale("log")
    ax.set_xticks(rp_damage["return_period"])
    ax.set_xticklabels(rp_damage["return_period"].astype(str))
    ax.set_xlabel("STORM return period (years)")
    ax.set_ylabel("Expected transmission line damage (billion USD)")
    ax.set_title("Florida TC Wind Damage: STORM Return Periods vs Fred/open-gira")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "florida_storm_return_period_damage_vs_fred.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    rp_by_aep = rp_damage.sort_values("annual_exceedance_probability")
    ax.plot(
        rp_by_aep["annual_exceedance_probability"],
        rp_by_aep["storm_damage_usd"] / 1e9,
        marker="o",
        linewidth=2,
        color="#2c7fb8",
    )
    ax.fill_between(
        rp_by_aep["annual_exceedance_probability"],
        rp_by_aep["storm_damage_usd"] / 1e9,
        alpha=0.2,
        color="#2c7fb8",
    )
    ax.axhline(
        fred_damage_usd / 1e9,
        color="#d95f0e",
        linestyle="--",
        linewidth=2,
        label="Fred/open-gira historical max reference",
    )
    ax.set_xlabel("Annual exceedance probability (1 / return period)")
    ax.set_ylabel("Expected transmission line damage (billion USD)")
    ax.set_title("Florida TC Damage-Probability Curve with Fred/open-gira Reference")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "florida_storm_damage_probability_curve_with_fred_reference.png",
        dpi=300,
    )
    plt.close()

    return rp_damage


def save_existing_fragility_curve_plot(output_path: Path):
    if not NHESS_W310_CURVE.exists():
        print("Skipping fragility curve graph; missing:", NHESS_W310_CURVE)
        return

    nhess_curve = pd.read_csv(NHESS_W310_CURVE)
    wind_grid = np.linspace(0.01, 100, 400)
    baseline_values = [
        baseline_expected_damage_ratio(wind_speed)
        for wind_speed in wind_grid
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        wind_grid,
        baseline_values,
        label="Baseline lognormal",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        nhess_curve["wind_speed_ms"],
        nhess_curve["nhess_w310_damage_ratio"],
        label="NHESS W3.10",
        linewidth=2,
    )

    ax.set_xlabel("Tropical cyclone wind speed (m/s)")
    ax.set_ylabel("Damage ratio")
    ax.set_title("Existing TC Wind Fragility / Damage-Ratio Curves")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_grouped_comparison_bar(
    comparison: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
):
    x = np.arange(len(comparison))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    old_bars = ax.bar(
        x - width / 2,
        comparison[f"old_{value_column}"],
        width,
        label="Old STORM 100-year raster",
        color="#6baed6",
    )
    new_bars = ax.bar(
        x + width / 2,
        comparison[f"new_{value_column}"],
        width,
        label="New Fred/open-gira IBTrACS footprint",
        color="#fb6a4a",
    )

    max_value = max(
        comparison[f"old_{value_column}"].max(),
        comparison[f"new_{value_column}"].max(),
    )
    label_offset = max_value * 0.015 if max_value > 0 else 0.1

    for bars in [old_bars, new_bars]:
        for bar in bars:
            height = bar.get_height()
            label = f"{height:,.0f}" if value_column == "node_count" else f"{height:,.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + label_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(comparison["tc_wind_class"].astype(str))
    ax.set_xlabel("Tropical cyclone wind speed class (m/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_asset_wind_scatter(
    frame: pd.DataFrame,
    old_column: str,
    new_column: str,
    title: str,
    output_path: Path,
):
    plot_data = frame[[old_column, new_column]].dropna()
    if plot_data.empty:
        print("Skipping scatter plot; no overlapping wind values:", output_path)
        return

    max_value = float(plot_data.max().max())
    min_value = float(plot_data.min().min())
    axis_min = max(0, min_value - 2)
    axis_max = max_value + 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        plot_data[old_column],
        plot_data[new_column],
        s=14,
        alpha=0.35,
        color="#4c78a8",
        edgecolors="none",
    )
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_xlabel("Old STORM 100-year wind speed (m/s)")
    ax.set_ylabel("New Fred/open-gira wind speed (m/s)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_old_vs_new_comparisons(
    nodes_tc: gpd.GeoDataFrame,
    line_segments_tc: gpd.GeoDataFrame,
):
    node_old = summarize_count_by_class(
        nodes_tc,
        "old_tc_wind_class",
        "node_count",
    ).rename(columns={"node_count": "old_node_count", "old_tc_wind_class": "tc_wind_class"})
    node_new = summarize_count_by_class(
        nodes_tc,
        "tc_wind_class",
        "node_count",
    ).rename(columns={"node_count": "new_node_count"})

    node_comparison = (
        pd.DataFrame({"tc_wind_class": pd.Categorical(TC_LABELS, categories=TC_LABELS, ordered=True)})
        .merge(node_old, on="tc_wind_class", how="left")
        .merge(node_new, on="tc_wind_class", how="left")
        .fillna({"old_node_count": 0, "new_node_count": 0})
    )
    node_comparison["old_percent"] = np.where(
        node_comparison["old_node_count"].sum() > 0,
        node_comparison["old_node_count"] / node_comparison["old_node_count"].sum() * 100,
        0,
    )
    node_comparison["new_percent"] = np.where(
        node_comparison["new_node_count"].sum() > 0,
        node_comparison["new_node_count"] / node_comparison["new_node_count"].sum() * 100,
        0,
    )
    node_comparison.to_csv(
        OUTPUT_DIR / "florida_nodes_old_storm_vs_open_gira_tc_comparison.csv",
        index=False,
    )
    save_grouped_comparison_bar(
        node_comparison,
        "node_count",
        "Number of electricity nodes",
        "Florida Node TC Wind Exposure: Old STORM vs New Fred/open-gira",
        OUTPUT_DIR / "florida_nodes_old_storm_vs_open_gira_tc_by_class.png",
    )
    save_asset_wind_scatter(
        nodes_tc,
        "old_tc_wind_speed_ms",
        "tc_wind_speed_max_ms",
        "Florida Nodes: Old STORM vs New Fred/open-gira Wind",
        OUTPUT_DIR / "florida_nodes_old_storm_vs_open_gira_tc_scatter.png",
    )

    line_old = summarize_length_by_class(
        line_segments_tc,
        "old_tc_wind_class",
    ).rename(columns={"length_km": "old_length_km", "old_tc_wind_class": "tc_wind_class"})
    line_new = summarize_length_by_class(
        line_segments_tc,
        "tc_wind_class",
    ).rename(columns={"length_km": "new_length_km"})

    line_comparison = (
        pd.DataFrame({"tc_wind_class": pd.Categorical(TC_LABELS, categories=TC_LABELS, ordered=True)})
        .merge(line_old, on="tc_wind_class", how="left")
        .merge(line_new, on="tc_wind_class", how="left")
        .fillna({"old_length_km": 0, "new_length_km": 0})
    )
    line_comparison["old_percent"] = np.where(
        line_comparison["old_length_km"].sum() > 0,
        line_comparison["old_length_km"] / line_comparison["old_length_km"].sum() * 100,
        0,
    )
    line_comparison["new_percent"] = np.where(
        line_comparison["new_length_km"].sum() > 0,
        line_comparison["new_length_km"] / line_comparison["new_length_km"].sum() * 100,
        0,
    )
    line_comparison.to_csv(
        OUTPUT_DIR / "florida_lines_old_storm_vs_open_gira_tc_comparison.csv",
        index=False,
    )
    save_grouped_comparison_bar(
        line_comparison,
        "length_km",
        "Overhead transmission line length (km)",
        "Florida Line TC Wind Exposure: Old STORM vs New Fred/open-gira",
        OUTPUT_DIR / "florida_lines_old_storm_vs_open_gira_tc_by_class.png",
    )
    save_asset_wind_scatter(
        line_segments_tc,
        "old_tc_wind_speed_ms",
        "tc_wind_speed_max_ms",
        "Florida Line Segments: Old STORM vs New Fred/open-gira Wind",
        OUTPUT_DIR / "florida_lines_old_storm_vs_open_gira_tc_scatter.png",
    )


def main() -> None:
    for path in [NODES_FILE, LINES_FILE, WIND_NC, OLD_STORM_RASTER, FLORIDA_BOUNDARIES]:
        print(path, "exists?", path.exists())

    florida_geometry = load_florida_geometry(FLORIDA_BOUNDARIES)
    wind = xr.open_dataset(WIND_NC)[WIND_VAR]

    print("\nLoading and clipping electricity assets to Florida...")
    nodes = gpd.read_file(NODES_FILE)
    nodes = nodes[nodes["country"] == "USA"].copy()
    florida_nodes = filter_to_florida(nodes, florida_geometry)

    lines = gpd.read_file(LINES_FILE)
    lines = filter_lines_by_type(lines, LINE_TYPE_FILTER)
    florida_lines = filter_to_florida(lines, florida_geometry)

    florida_nodes.to_file(OUTPUT_DIR / "florida_nodes2.gpkg", layer="florida_nodes2", driver="GPKG")
    florida_lines.to_file(
        OUTPUT_DIR / "florida_overhead_transmissionlines2.gpkg",
        layer="florida_overhead_transmissionlines2",
        driver="GPKG",
    )

    print("Florida nodes:", len(florida_nodes))
    print(f"Florida {LINE_TYPE_FILTER.lower()} lines:", len(florida_lines))

    print("\nSampling node exposure...")
    nodes_tc = sample_nodes(florida_nodes, wind)
    nodes_tc["old_tc_wind_speed_ms"] = sample_old_storm_raster_at_points(
        nodes_tc,
        OLD_STORM_RASTER,
    )
    nodes_tc["old_tc_wind_class"] = nodes_tc["old_tc_wind_speed_ms"].apply(classify_wind_speed)
    nodes_tc["old_tc_wind_class"] = pd.Categorical(
        nodes_tc["old_tc_wind_class"],
        categories=TC_LABELS,
        ordered=True,
    )
    nodes_tc.to_file(
        OUTPUT_DIR / "florida_nodes_open_gira_tc_exposure.gpkg",
        layer="florida_nodes_open_gira_tc_exposure",
        driver="GPKG",
    )

    node_summary = (
        nodes_tc.dropna(subset=["tc_wind_class"])
        .groupby("tc_wind_class", observed=False)
        .size()
        .reset_index(name="node_count")
    )
    node_summary = add_percent(node_summary, "node_count")
    node_summary.to_csv(OUTPUT_DIR / "florida_nodes_open_gira_tc_by_class.csv", index=False)
    save_bar_with_percent(
        node_summary,
        "tc_wind_class",
        "node_count",
        "Number of electricity nodes",
        "Florida Electricity Node Exposure to Tropical Cyclone Wind",
        OUTPUT_DIR / "florida_nodes_open_gira_tc_by_class.png",
    )
    save_stacked_node_fuel_chart(
        nodes_tc,
        OUTPUT_DIR / "florida_nodes_open_gira_tc_by_class_and_fuel.png",
    )
    save_node_wind_histogram(
        nodes_tc,
        OUTPUT_DIR / "florida_nodes_open_gira_tc_wind_histogram.png",
    )

    print("\nSegmenting and sampling transmission lines...")
    line_segments = make_line_segments(florida_lines)
    line_segments_tc = sample_line_segments(line_segments, wind)
    line_segments_tc["old_tc_wind_speed_ms"] = sample_old_storm_raster_at_line_midpoints(
        line_segments_tc,
        OLD_STORM_RASTER,
    )
    line_segments_tc["old_tc_wind_class"] = line_segments_tc["old_tc_wind_speed_ms"].apply(
        classify_wind_speed
    )
    line_segments_tc["old_tc_wind_class"] = pd.Categorical(
        line_segments_tc["old_tc_wind_class"],
        categories=TC_LABELS,
        ordered=True,
    )
    line_segments_tc.to_file(
        OUTPUT_DIR / "florida_overhead_lines_open_gira_tc_exposure_segments.gpkg",
        layer="florida_overhead_lines_open_gira_tc_exposure_segments",
        driver="GPKG",
    )

    line_summary = (
        line_segments_tc.dropna(subset=["tc_wind_class"])
        .groupby("tc_wind_class", observed=False)["length_km"]
        .sum()
        .reset_index()
    )
    line_summary = add_percent(line_summary, "length_km")
    line_summary.to_csv(OUTPUT_DIR / "florida_overhead_lines_open_gira_tc_length_by_class.csv", index=False)
    save_bar_with_percent(
        line_summary,
        "tc_wind_class",
        "length_km",
        "Transmission line length (km)",
        "Florida Overhead Transmission Line Exposure to Tropical Cyclone Wind",
        OUTPUT_DIR / "florida_overhead_lines_open_gira_tc_length_by_class.png",
    )
    save_line_map(
        line_segments_tc,
        OUTPUT_DIR / "florida_overhead_lines_open_gira_tc_by_class_map.png",
    )
    save_max_wind_footprint_map(
        wind,
        florida_geometry,
        nodes_tc,
        florida_lines,
        OUTPUT_DIR / "florida_open_gira_tc_max_wind_footprint_map.png",
    )
    save_existing_fragility_curve_plot(
        OUTPUT_DIR / "florida_existing_tc_wind_fragility_curves.png",
    )
    save_old_vs_new_comparisons(nodes_tc, line_segments_tc)
    damage_summary, voltage_damage_summary = save_damage_comparison_outputs(
        add_line_cost_damage(line_segments_tc)
    )
    fred_damage_usd = float(
        damage_summary.loc[
            damage_summary["dataset"] == "New Fred/open-gira historical IBTrACS maximum",
            "tc_damage_usd",
        ].iloc[0]
    )
    return_period_damage = save_storm_return_period_damage_curve(
        add_line_cost_damage(line_segments_tc),
        fred_damage_usd,
    )

    print("\nSaved outputs in:", OUTPUT_DIR)
    print("\nNode exposure by TC wind class:")
    print(node_summary)
    print("\nTransmission line exposure by TC wind class:")
    print(line_summary)
    print("\nTransmission line TC wind damage cost comparison:")
    print(damage_summary)
    print("\nTransmission line TC wind damage by voltage class:")
    print(voltage_damage_summary)
    print("\nSTORM return-period TC wind damage with Fred/open-gira reference:")
    print(return_period_damage)


if __name__ == "__main__":
    main()
