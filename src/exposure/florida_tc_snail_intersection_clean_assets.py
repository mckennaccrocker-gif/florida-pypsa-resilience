"""
Use nismod-snail to intersect cleaned Florida overhead transmission lines with TC wind rasters.

This is a more formal line/raster intersection workflow than midpoint sampling:
snail splits line geometries along raster grid cells, assigns raster cell indices
to each split, and reads the hazard value from the corresponding cell.

Inputs:
  - data/Electricity/florida_lines_with_s_nom.gpkg
  - data/Hazards/Tropical_cyclones/STORM_constant_RP*.tif
  - open_gira_outputs/max_wind_field_FLORIDA_IBTrACS_0.nc

Outputs:
  - data/Exposure/florida_clean_assets_tc_snail_intersection/
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
import snail.intersection
import xarray as xr
from pyproj import Geod
from rasterio.transform import from_origin


PROJECT_DIR = Path(__file__).resolve().parents[2]
TC_DIR = PROJECT_DIR / "data" / "Hazards" / "Tropical_cyclones"
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure" / "florida_clean_assets_tc_snail_intersection"
OUTPUT_DIR = EXPOSURE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLORIDA_OVERHEAD_LINES = ELECTRICITY_DIR / "florida_lines_with_s_nom.gpkg"
FRED_NETCDF = PROJECT_DIR / "open_gira_outputs" / "max_wind_field_FLORIDA_IBTrACS_0.nc"
FRED_GEOTIFF = OUTPUT_DIR / "fred_open_gira_ibtracs_historical_max_wind_florida.tif"

STORM_RP_RASTERS = {
    10: TC_DIR / "STORM_constant_RP10_US_crop.tif",
    20: TC_DIR / "STORM_constant_RP20_US_crop.tif",
    50: TC_DIR / "STORM_constant_RP50_US_crop.tif",
    100: TC_DIR / "STORM_constant_RP100_US_crop.tif",
    200: TC_DIR / "STORM_constant_RP200_US_crop.tif",
    500: TC_DIR / "STORM_constant_RP500_US_crop.tif",
}

TC_BINS = [0, 25, 30, 35, 40, np.inf]
TC_LABELS = ["<25", "25-30", "30-35", "35-40", ">40"]
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

os.environ.setdefault("PROJ_LIB", str(Path(sys.prefix) / "share" / "proj"))


def classify_wind(value):
    if pd.isna(value):
        return np.nan
    return pd.cut([value], bins=TC_BINS, labels=TC_LABELS, include_lowest=True)[0]


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


def add_replacement_costs(splits: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    costed = splits.copy()
    voltage_source = "VOLT_CLASS" if "VOLT_CLASS" in costed.columns else "VOLTAGE"
    costed["voltage_kv_raw"] = costed[voltage_source].apply(parse_voltage_kv)
    costed["voltage_kv_cost_class"] = costed["voltage_kv_raw"].apply(
        snap_voltage_to_cost_class
    )
    costed["line_cost_per_mile_usd"] = costed["voltage_kv_cost_class"].map(
        LINE_COST_PER_MILE_USD
    )
    costed["length_miles"] = costed["length_km"] / 1.609344
    costed["replacement_cost_usd"] = (
        costed["length_miles"] * costed["line_cost_per_mile_usd"]
    )
    return costed


def calculate_damage_usd(splits: gpd.GeoDataFrame, wind_column: str) -> pd.Series:
    damage_ratios = splits[wind_column].apply(baseline_expected_damage_ratio)
    return splits["replacement_cost_usd"].fillna(0) * damage_ratios


def create_fred_geotiff() -> Path:
    """Convert the Fred/open-gira NetCDF historical maximum into a GeoTIFF."""
    if FRED_GEOTIFF.exists():
        return FRED_GEOTIFF

    ds = xr.open_dataset(FRED_NETCDF)
    wind = ds["max_wind_speed"].max(dim="event_id", skipna=True)

    lon = wind["longitude"].values
    lat = wind["latitude"].values
    dx = float(np.median(np.diff(lon)))
    dy = float(np.median(np.diff(lat)))

    west = float(lon.min() - dx / 2)
    north = float(lat.max() + dy / 2)
    transform = from_origin(west, north, abs(dx), abs(dy))

    # Raster rows must run north-to-south, while the NetCDF latitude is ascending.
    data = wind.sortby("latitude", ascending=False).values.astype("float32")
    nodata = -9999.0
    data = np.where(np.isfinite(data), data, nodata).astype("float32")

    with rasterio.open(
        FRED_GEOTIFF,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(
            description="Fred/open-gira IBTrACS historical maximum TC wind speed",
            units="m/s",
        )

    return FRED_GEOTIFF


def read_florida_lines_for_snail(raster_path: Path) -> gpd.GeoDataFrame:
    lines = gpd.read_file(FLORIDA_OVERHEAD_LINES).reset_index(drop=True)
    print("\nCleaned Florida line columns:")
    print(", ".join(lines.columns.astype(str)))

    type_text = lines.get("TYPE", pd.Series("", index=lines.index)).fillna("").astype(str).str.upper()
    lines = lines[type_text.str.contains("OVERHEAD", na=False)].copy()
    if "STATUS" in lines.columns:
        status_text = lines["STATUS"].fillna("").astype(str).str.upper().str.strip()
        lines = lines[status_text.eq("IN SERVICE")].copy()

    if "florida_line_id" not in lines.columns:
        lines["florida_line_id"] = np.arange(len(lines))
    lines["source_line_id"] = lines["florida_line_id"]
    lines = lines.explode(index_parts=False, ignore_index=True)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].copy()

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    return lines.to_crs(raster_crs)


def snail_intersect_lines_with_raster(
    lines: gpd.GeoDataFrame,
    raster_path: Path,
    value_column: str,
) -> gpd.GeoDataFrame:
    grid = snail.intersection.GridDefinition.from_raster(raster_path)

    splits = snail.intersection.split_linestrings(lines.reset_index(drop=True), grid)
    splits = snail.intersection.apply_indices(
        splits,
        grid,
        index_i="raster_i",
        index_j="raster_j",
    )

    geod = Geod(ellps="WGS84")
    splits = splits.to_crs("EPSG:4326")
    splits["length_km"] = splits.geometry.apply(geod.geometry_length) / 1000

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        values = snail.intersection.get_raster_values_for_splits(
            splits,
            data,
            index_i="raster_i",
            index_j="raster_j",
        ).astype(float)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan

    splits[value_column] = values
    splits["tc_wind_class"] = splits[value_column].apply(classify_wind)
    splits["tc_wind_class"] = pd.Categorical(
        splits["tc_wind_class"],
        categories=TC_LABELS,
        ordered=True,
    )

    return splits


def assign_raster_values_to_snail_splits(
    splits: gpd.GeoDataFrame,
    raster_path: Path,
    value_column: str,
) -> gpd.GeoDataFrame:
    """Assign another raster's values to already Snail-split line segments."""
    grid = snail.intersection.GridDefinition.from_raster(raster_path)

    with rasterio.open(raster_path) as src:
        working = splits.to_crs(src.crs).copy()
        working = snail.intersection.apply_indices(
            working,
            grid,
            index_i=f"{value_column}_raster_i",
            index_j=f"{value_column}_raster_j",
        )
        data = src.read(1)
        values = snail.intersection.get_raster_values_for_splits(
            working,
            data,
            index_i=f"{value_column}_raster_i",
            index_j=f"{value_column}_raster_j",
        ).astype(float)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan

    out = working.to_crs("EPSG:4326")
    out[value_column] = values
    return out


def summarize_length_by_class(splits: gpd.GeoDataFrame, value_column: str) -> pd.DataFrame:
    summary = (
        splits.dropna(subset=["tc_wind_class"])
        .groupby("tc_wind_class", observed=False)["length_km"]
        .sum()
        .reset_index()
    )
    summary["percent"] = np.where(
        summary["length_km"].sum() > 0,
        summary["length_km"] / summary["length_km"].sum() * 100,
        0,
    )
    summary["mean_wind_speed_ms"] = [
        splits.loc[splits["tc_wind_class"].astype(str) == str(label), value_column].mean()
        for label in summary["tc_wind_class"]
    ]
    return summary


def plot_return_period_summary(storm_summary: pd.DataFrame, fred_summary: pd.DataFrame):
    wide = storm_summary.pivot_table(
        index="return_period",
        columns="tc_wind_class",
        values="length_km",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    ).reindex(columns=TC_LABELS)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(wide))
    x = np.arange(len(wide))

    colors = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
    for label, color in zip(TC_LABELS, colors):
        values = wide[label].to_numpy()
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values

    ax.set_ylim(0, bottom.max() * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(wide.index.astype(str))
    ax.set_xlabel("STORM return period (years)")
    ax.set_ylabel("Snail-split overhead line length (km)")
    ax.set_title("Snail TC Wind Intersection by STORM Return Period")
    ax.legend(title="Wind speed (m/s)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snail_storm_return_period_line_length_by_wind_class.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    fred_plot = fred_summary.copy()
    ax.bar(
        fred_plot["tc_wind_class"].astype(str),
        fred_plot["length_km"],
        color=colors,
    )
    ax.set_ylim(0, fred_plot["length_km"].max() * 1.15)
    ax.set_xlabel("Fred/open-gira wind speed class (m/s)")
    ax.set_ylabel("Snail-split overhead line length (km)")
    ax.set_title("Snail TC Wind Intersection: Fred/open-gira Historical Maximum")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snail_fred_line_length_by_wind_class.png", dpi=300)
    plt.close()


def plot_snail_old_vs_fred_comparison(
    storm_rp100_splits: gpd.GeoDataFrame,
    fred_raster: Path,
) -> None:
    comparison = assign_raster_values_to_snail_splits(
        storm_rp100_splits,
        fred_raster,
        value_column="fred_tc_wind_speed_ms",
    )
    comparison = comparison.rename(
        columns={"tc_wind_speed_ms": "old_storm_rp100_wind_speed_ms"}
    )
    comparison["old_storm_wind_class"] = comparison[
        "old_storm_rp100_wind_speed_ms"
    ].apply(classify_wind)
    comparison["fred_wind_class"] = comparison["fred_tc_wind_speed_ms"].apply(
        classify_wind
    )

    save_columns = comparison.copy()
    save_columns["old_storm_wind_class"] = save_columns[
        "old_storm_wind_class"
    ].astype(str)
    save_columns["fred_wind_class"] = save_columns["fred_wind_class"].astype(str)
    save_columns.to_file(
        OUTPUT_DIR / "snail_old_storm_rp100_vs_fred_line_segments.gpkg",
        layer="snail_old_storm_rp100_vs_fred_line_segments",
        driver="GPKG",
    )
    save_columns.drop(columns="geometry").to_csv(
        OUTPUT_DIR / "snail_old_storm_rp100_vs_fred_line_segments.csv",
        index=False,
    )

    old_summary = (
        comparison.dropna(subset=["old_storm_wind_class"])
        .groupby("old_storm_wind_class", observed=False)["length_km"]
        .sum()
        .reindex(TC_LABELS, fill_value=0)
    )
    fred_summary = (
        comparison.dropna(subset=["fred_wind_class"])
        .groupby("fred_wind_class", observed=False)["length_km"]
        .sum()
        .reindex(TC_LABELS, fill_value=0)
    )
    summary = pd.DataFrame(
        {
            "tc_wind_class": TC_LABELS,
            "old_storm_rp100_length_km": old_summary.to_numpy(),
            "fred_open_gira_length_km": fred_summary.to_numpy(),
        }
    )
    summary.to_csv(
        OUTPUT_DIR / "snail_old_storm_rp100_vs_fred_by_class.csv",
        index=False,
    )

    x = np.arange(len(TC_LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.8))
    old_bars = ax.bar(
        x - width / 2,
        summary["old_storm_rp100_length_km"],
        width,
        label="Old STORM 100-year raster",
        color="#6baed6",
    )
    fred_bars = ax.bar(
        x + width / 2,
        summary["fred_open_gira_length_km"],
        width,
        label="New Fred/open-gira IBTrACS footprint",
        color="#fb6a4a",
    )
    max_height = max(
        summary["old_storm_rp100_length_km"].max(),
        summary["fred_open_gira_length_km"].max(),
    )
    ax.set_ylim(0, max_height * 1.18)
    for bars in [old_bars, fred_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max_height * 0.015,
                f"{height:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(TC_LABELS)
    ax.set_xlabel("Tropical cyclone wind speed class (m/s)")
    ax.set_ylabel("Snail-split overhead transmission line length (km)")
    ax.set_title("Snail Florida Line TC Wind Exposure: STORM RP100 vs Fred/open-gira")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snail_old_storm_rp100_vs_fred_by_class.png", dpi=300)
    plt.close()

    valid = comparison.dropna(
        subset=["old_storm_rp100_wind_speed_ms", "fred_tc_wind_speed_ms"]
    ).copy()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        valid["old_storm_rp100_wind_speed_ms"],
        valid["fred_tc_wind_speed_ms"],
        s=9,
        alpha=0.35,
        color="#4c78a8",
        edgecolors="none",
    )
    min_value = np.floor(
        min(
            valid["old_storm_rp100_wind_speed_ms"].min(),
            valid["fred_tc_wind_speed_ms"].min(),
        )
    )
    max_value = np.ceil(
        max(
            valid["old_storm_rp100_wind_speed_ms"].max(),
            valid["fred_tc_wind_speed_ms"].max(),
        )
    )
    ax.plot([min_value, max_value], [min_value, max_value], "k--", linewidth=1)
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    ax.set_xlabel("Old STORM 100-year wind speed (m/s)")
    ax.set_ylabel("New Fred/open-gira wind speed (m/s)")
    ax.set_title("Snail Florida Line Segments: STORM RP100 vs Fred/open-gira Wind")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snail_old_storm_rp100_vs_fred_scatter.png", dpi=300)
    plt.close()


def plot_snail_damage_comparisons(
    storm_splits_by_rp: dict[int, gpd.GeoDataFrame],
    fred_raster: Path,
) -> pd.DataFrame:
    storm_rp100 = add_replacement_costs(storm_splits_by_rp[100])
    comparison = assign_raster_values_to_snail_splits(
        storm_rp100,
        fred_raster,
        value_column="fred_tc_wind_speed_ms",
    ).rename(columns={"tc_wind_speed_ms": "old_storm_rp100_wind_speed_ms"})

    comparison["old_storm_rp100_damage_usd"] = calculate_damage_usd(
        comparison,
        "old_storm_rp100_wind_speed_ms",
    )
    comparison["fred_open_gira_damage_usd"] = calculate_damage_usd(
        comparison,
        "fred_tc_wind_speed_ms",
    )

    old_total = float(comparison["old_storm_rp100_damage_usd"].sum())
    fred_total = float(comparison["fred_open_gira_damage_usd"].sum())
    total_replacement = float(comparison["replacement_cost_usd"].sum())

    damage_summary = pd.DataFrame(
        {
            "dataset": [
                "Old STORM 100-year raster",
                "New Fred/open-gira historical IBTrACS maximum",
            ],
            "replacement_cost_usd": [total_replacement, total_replacement],
            "tc_damage_usd": [old_total, fred_total],
            "mean_damage_ratio_weighted_by_replacement_cost": [
                old_total / total_replacement if total_replacement > 0 else 0,
                fred_total / total_replacement if total_replacement > 0 else 0,
            ],
        }
    )
    damage_summary["difference_vs_old_usd"] = (
        damage_summary["tc_damage_usd"] - old_total
    )
    damage_summary["difference_vs_old_percent"] = np.where(
        old_total > 0,
        damage_summary["difference_vs_old_usd"] / old_total * 100,
        0,
    )
    damage_summary.to_csv(
        OUTPUT_DIR / "snail_old_storm_rp100_vs_fred_damage_summary.csv",
        index=False,
    )
    comparison.drop(columns="geometry").to_csv(
        OUTPUT_DIR / "snail_old_storm_rp100_vs_fred_damage_segments.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Old STORM\n100-year", "New Fred/open-gira\nhistorical max"],
        damage_summary["tc_damage_usd"] / 1e9,
        color=["#6baed6", "#fb6a4a"],
    )
    max_value = (damage_summary["tc_damage_usd"] / 1e9).max()
    ax.set_ylim(0, max_value * 1.16 if max_value > 0 else 1)
    for bar, value in zip(bars, damage_summary["tc_damage_usd"] / 1e9):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_value * 0.02,
            f"${value:.2f}B",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Expected transmission line damage (billion USD)")
    ax.set_title("Snail Florida TC Wind Damage Cost: STORM RP100 vs Fred/open-gira")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "snail_old_storm_rp100_vs_fred_damage_cost.png",
        dpi=300,
    )
    plt.close()

    records = []
    for return_period, splits in sorted(storm_splits_by_rp.items()):
        costed = add_replacement_costs(splits)
        damage_usd = calculate_damage_usd(costed, "tc_wind_speed_ms")
        records.append(
            {
                "return_period": return_period,
                "annual_exceedance_probability": 1 / return_period,
                "storm_damage_usd": float(damage_usd.sum()),
                "mean_wind_speed_ms": float(costed["tc_wind_speed_ms"].mean()),
                "max_wind_speed_ms": float(costed["tc_wind_speed_ms"].max()),
            }
        )

    rp_damage = pd.DataFrame(records).sort_values("return_period")
    rp_damage["fred_open_gira_historical_max_damage_usd"] = fred_total
    rp_damage["difference_vs_fred_usd"] = (
        rp_damage["storm_damage_usd"]
        - rp_damage["fred_open_gira_historical_max_damage_usd"]
    )
    rp_damage.to_csv(
        OUTPUT_DIR / "snail_storm_return_period_damage_vs_fred.csv",
        index=False,
    )

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
        fred_total / 1e9,
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
        fred_total / 1e9,
        f"  Fred/open-gira ${fred_total / 1e9:.2f}B",
        va="center",
        ha="left",
        color="#d95f0e",
    )
    ax.set_xscale("log")
    ax.set_xticks(rp_damage["return_period"])
    ax.set_xticklabels(rp_damage["return_period"].astype(str))
    ax.set_xlabel("STORM return period (years)")
    ax.set_ylabel("Expected transmission line damage (billion USD)")
    ax.set_title("Snail Florida TC Wind Damage: STORM Return Periods vs Fred/open-gira")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "snail_storm_return_period_damage_vs_fred.png",
        dpi=300,
    )
    plt.close()

    return rp_damage


def main() -> None:
    print("Checking inputs...")
    for path in [FLORIDA_OVERHEAD_LINES, FRED_NETCDF, *STORM_RP_RASTERS.values()]:
        print(path, "exists?", path.exists())

    fred_raster = create_fred_geotiff()
    print("Fred GeoTIFF:", fred_raster)

    storm_summaries = []
    storm_rp100_splits = None
    storm_splits_by_rp = {}

    # STORM return-period rasters share a grid, so we split once on the first raster.
    base_storm_raster = next(iter(STORM_RP_RASTERS.values()))
    storm_lines = read_florida_lines_for_snail(base_storm_raster)

    for return_period, raster_path in sorted(STORM_RP_RASTERS.items()):
        print(f"\nSnail intersecting STORM RP{return_period}: {raster_path}")
        splits = snail_intersect_lines_with_raster(
            storm_lines,
            raster_path,
            value_column="tc_wind_speed_ms",
        )
        splits["return_period"] = return_period
        storm_splits_by_rp[return_period] = splits.copy()
        if return_period == 100:
            storm_rp100_splits = splits.copy()
        output_gpkg = OUTPUT_DIR / f"snail_storm_rp{return_period}_florida_overhead_lines.gpkg"
        splits.to_file(output_gpkg, layer=f"snail_storm_rp{return_period}", driver="GPKG")

        summary = summarize_length_by_class(splits, "tc_wind_speed_ms")
        summary["return_period"] = return_period
        storm_summaries.append(summary)
        print(summary)

    storm_summary = pd.concat(storm_summaries, ignore_index=True)
    storm_summary.to_csv(
        OUTPUT_DIR / "snail_storm_return_period_line_length_by_wind_class.csv",
        index=False,
    )

    print("\nSnail intersecting Fred/open-gira historical max:", fred_raster)
    fred_lines = read_florida_lines_for_snail(fred_raster)
    fred_splits = snail_intersect_lines_with_raster(
        fred_lines,
        fred_raster,
        value_column="fred_tc_wind_speed_ms",
    )
    fred_splits.to_file(
        OUTPUT_DIR / "snail_fred_open_gira_florida_overhead_lines.gpkg",
        layer="snail_fred_open_gira_florida_overhead_lines",
        driver="GPKG",
    )
    fred_summary = summarize_length_by_class(fred_splits, "fred_tc_wind_speed_ms")
    fred_summary.to_csv(
        OUTPUT_DIR / "snail_fred_line_length_by_wind_class.csv",
        index=False,
    )
    print(fred_summary)

    plot_return_period_summary(storm_summary, fred_summary)
    if storm_rp100_splits is not None:
        plot_snail_old_vs_fred_comparison(storm_rp100_splits, fred_raster)
    if 100 in storm_splits_by_rp:
        rp_damage = plot_snail_damage_comparisons(storm_splits_by_rp, fred_raster)
        print(rp_damage)

    print("\nSaved snail outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
