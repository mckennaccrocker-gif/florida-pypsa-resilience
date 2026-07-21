"""
Snail event-by-event tropical cyclone damage analysis for Florida transmission lines.

This is the Snail version of the advisor-requested workflow:

1. Load each historical open-gira/IBTrACS event wind footprint.
2. Split Florida overhead transmission lines by the raster grid using nismod-snail.
3. Read the wind value for each line-grid split.
4. For each event and each original transmission line, take the maximum wind speed
   experienced anywhere along that line.
5. Apply the TC wind vulnerability/damage-ratio curve.
6. Convert damage ratios to dollar damage using voltage-based replacement costs.
7. Sum line damages to get total Florida network damage for each historical event.

Run this with the open-gira Pixi/WSL environment if Snail is not installed on Windows:

    wsl -d Ubuntu -- bash -lc "cd /mnt/c/oxford_tc_project && \
    /home/mckennacroc/.pixi/bin/pixi run --manifest-path /home/mckennacroc/projects/open-gira/pixi.toml \
    python data/Exposure/open_gira_ibtracs_event_damage_snail.py"

Outputs:
    data/Exposure/florida_open_gira_tc_event_damage_snail/
"""

from __future__ import annotations

import platform
import re
from math import erf, log, sqrt
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import snail.intersection
from pyproj import Geod


def project_root() -> Path:
    """Return the project folder on either Windows or WSL."""
    if platform.system().lower() == "windows":
        return Path(r"C:\oxford_tc_project")
    return Path("/mnt/c/oxford_tc_project")


PROJECT_DIR = project_root()
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
OUTPUT_DIR = PROJECT_DIR / "data" / "Exposure" / "florida_open_gira_tc_event_damage_snail"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIND_NETCDF = PROJECT_DIR / "open_gira_outputs" / "max_wind_field_FLORIDA_IBTrACS_0.nc"
WIND_GEOTIFF = OUTPUT_DIR / "open_gira_ibtracs_florida_event_wind_stack.tif"
FLORIDA_LINES = ELECTRICITY_DIR / "florida_transmission_lines.gpkg"

STORAGE_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3086"
NODATA_VALUE = -9999.0
KEEP_ONLY_IN_SERVICE = True

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


def normal_cdf(value: float) -> float:
    return 0.5 * (1 + erf(value / sqrt(2)))


def baseline_expected_damage_ratio(wind_speed_ms: float) -> float:
    """Baseline TC wind expected damage-ratio curve used elsewhere in the project."""
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


def parse_voltage_kv(value) -> float:
    """Extract a numeric voltage in kV from HIFLD voltage text fields."""
    if pd.isna(value):
        return np.nan

    text = str(value).upper()
    if "DC" in text and not re.search(r"\d", text):
        return np.nan

    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return np.nan

    voltages_kv = [number / 1000 if number >= 1000 else number for number in numbers]
    voltages_kv = [number for number in voltages_kv if number > 0]
    return max(voltages_kv) if voltages_kv else np.nan


def snap_voltage_to_cost_class(voltage_kv: float) -> float:
    """Map observed voltage to the closest replacement-cost class."""
    if pd.isna(voltage_kv):
        return np.nan
    if voltage_kv >= STANDARD_VOLTAGES.max():
        return int(STANDARD_VOLTAGES.max())
    nearest_index = int(np.argmin(np.abs(STANDARD_VOLTAGES - voltage_kv)))
    return int(STANDARD_VOLTAGES[nearest_index])


def parse_event_ids(src: rasterio.io.DatasetReader) -> list[str]:
    """Extract IBTrACS event IDs from the open-gira NetCDF tags."""
    tag_value = src.tags().get("NETCDF_DIM_event_id_VALUES", "")
    match = re.search(r"\{(.+)\}", tag_value)
    if not match:
        return [f"event_{i:03d}" for i in range(1, src.count + 1)]

    event_ids = [part.strip() for part in match.group(1).split(",") if part.strip()]
    if len(event_ids) != src.count:
        return [f"event_{i:03d}" for i in range(1, src.count + 1)]
    return event_ids


def create_multiband_event_geotiff() -> tuple[Path, list[str]]:
    """
    Convert the open-gira NetCDF into a Snail-friendly multiband GeoTIFF.

    The NetCDF uses latitude/longitude but has no CRS tag when opened by rasterio.
    We write the same grid as EPSG:4326 so Snail can split line geometries against it.
    """
    with rasterio.open(WIND_NETCDF) as src:
        event_ids = parse_event_ids(src)

        if WIND_GEOTIFF.exists():
            return WIND_GEOTIFF, event_ids

        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": src.count,
            "dtype": "float32",
            "crs": STORAGE_CRS,
            "transform": src.transform,
            "nodata": NODATA_VALUE,
            "compress": "deflate",
            "tiled": False,
        }

        with rasterio.open(WIND_GEOTIFF, "w", **profile) as dst:
            for band_number in range(1, src.count + 1):
                data = src.read(band_number, masked=True).filled(np.nan).astype("float32")
                data = np.where(np.isfinite(data), data, NODATA_VALUE).astype("float32")
                dst.write(data, band_number)
                dst.set_band_description(band_number, event_ids[band_number - 1])

    return WIND_GEOTIFF, event_ids


def load_florida_overhead_lines() -> gpd.GeoDataFrame:
    """Load Florida HIFLD lines and keep overhead, in-service transmission lines."""
    lines = gpd.read_file(FLORIDA_LINES).to_crs(STORAGE_CRS)
    lines = lines.reset_index(drop=True)

    type_text = lines.get("TYPE", pd.Series("", index=lines.index)).fillna("").str.upper()
    lines = lines[type_text.str.contains("OVERHEAD", na=False)].copy()

    if KEEP_ONLY_IN_SERVICE and "STATUS" in lines.columns:
        status_text = lines["STATUS"].fillna("").str.upper().str.strip()
        lines = lines[status_text.eq("IN SERVICE")].copy()

    lines = lines.explode(index_parts=False, ignore_index=True)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
    lines["analysis_line_id"] = np.arange(len(lines))

    projected = lines.to_crs(PROJECTED_CRS)
    lines["length_m"] = projected.length
    lines["length_km"] = lines["length_m"] / 1000
    lines["length_miles"] = lines["length_km"] / 1.609344

    voltage_source = "VOLT_CLASS" if "VOLT_CLASS" in lines.columns else "VOLTAGE"
    lines["voltage_kv_raw"] = lines[voltage_source].apply(parse_voltage_kv)
    lines["voltage_kv_cost_class"] = lines["voltage_kv_raw"].apply(
        snap_voltage_to_cost_class
    )
    lines["line_cost_per_mile_usd"] = lines["voltage_kv_cost_class"].map(
        LINE_COST_PER_MILE_USD
    )
    lines["replacement_cost_usd"] = (
        lines["length_miles"] * lines["line_cost_per_mile_usd"]
    )

    return lines.to_crs(STORAGE_CRS)


def snail_split_lines(lines: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    """Use Snail to split transmission lines by the event wind raster grid."""
    grid = snail.intersection.GridDefinition.from_raster(str(raster_path))
    splits = snail.intersection.split_linestrings(lines.reset_index(drop=True), grid)
    splits = snail.intersection.apply_indices(
        splits,
        grid,
        index_i="raster_i",
        index_j="raster_j",
    )

    # Length of each Snail split segment is useful for QA and optional segment summaries.
    geod = Geod(ellps="WGS84")
    splits = splits.to_crs(STORAGE_CRS)
    splits["split_length_km"] = splits.geometry.apply(geod.geometry_length) / 1000
    return splits


def line_event_damage_from_splits(
    splits: gpd.GeoDataFrame,
    raster_path: Path,
    event_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loop through event bands, get split values, then max wind and damage per line."""
    static_cols = [
        "analysis_line_id",
        "florida_line_id",
        "TYPE",
        "STATUS",
        "OWNER",
        "VOLTAGE",
        "VOLT_CLASS",
        "SUB_1",
        "SUB_2",
        "length_km",
        "length_miles",
        "voltage_kv_raw",
        "voltage_kv_cost_class",
        "line_cost_per_mile_usd",
        "replacement_cost_usd",
    ]
    static_cols = [col for col in static_cols if col in splits.columns]
    line_static = (
        splits[static_cols]
        .drop_duplicates(subset=["analysis_line_id"])
        .sort_values("analysis_line_id")
        .reset_index(drop=True)
    )

    event_records = []
    line_event_records = []
    total_replacement_cost = float(line_static["replacement_cost_usd"].fillna(0).sum())

    with rasterio.open(raster_path) as src:
        for band_number, event_id in enumerate(event_ids, start=1):
            data = src.read(band_number).astype("float64")
            data[data == NODATA_VALUE] = np.nan

            split_values = snail.intersection.get_raster_values_for_splits(
                splits,
                data,
                index_i="raster_i",
                index_j="raster_j",
            ).astype(float)

            split_values = split_values.replace(NODATA_VALUE, np.nan)
            split_table = pd.DataFrame(
                {
                    "analysis_line_id": splits["analysis_line_id"].to_numpy(),
                    "split_wind_speed_ms": split_values.to_numpy(),
                }
            )

            line_max_wind = (
                split_table.groupby("analysis_line_id")["split_wind_speed_ms"]
                .max()
                .reset_index(name="line_max_wind_ms")
            )

            line_event = line_static.merge(
                line_max_wind,
                on="analysis_line_id",
                how="left",
            )
            line_event.insert(0, "event_id", event_id)
            line_event.insert(1, "event_year", int(str(event_id)[:4]))
            line_event["damage_ratio"] = line_event["line_max_wind_ms"].apply(
                baseline_expected_damage_ratio
            )
            line_event["damage_usd"] = (
                line_event["replacement_cost_usd"].fillna(0)
                * line_event["damage_ratio"]
            )

            event_records.append(
                {
                    "event_id": event_id,
                    "event_year": int(str(event_id)[:4]),
                    "band_number": band_number,
                    "lines_intersected": int(line_event["line_max_wind_ms"].notna().sum()),
                    "lines_with_wind_ge_25ms": int(
                        (line_event["line_max_wind_ms"].fillna(0) >= 25).sum()
                    ),
                    "mean_line_max_wind_ms": float(line_event["line_max_wind_ms"].mean()),
                    "max_line_wind_ms": float(line_event["line_max_wind_ms"].max()),
                    "total_replacement_cost_usd": total_replacement_cost,
                    "total_network_damage_usd": float(line_event["damage_usd"].sum()),
                    "mean_damage_ratio": float(line_event["damage_ratio"].mean()),
                    "replacement_cost_weighted_damage_ratio": (
                        float(line_event["damage_usd"].sum()) / total_replacement_cost
                        if total_replacement_cost > 0
                        else np.nan
                    ),
                }
            )
            line_event_records.append(line_event)

            if band_number % 25 == 0 or band_number == len(event_ids):
                print(f"Processed Snail event {band_number:,}/{len(event_ids):,}")

    event_damage = pd.DataFrame(event_records).sort_values(
        "total_network_damage_usd",
        ascending=False,
    )
    line_event_damage = pd.concat(line_event_records, ignore_index=True)
    return event_damage, line_event_damage


def make_graphs(event_damage: pd.DataFrame) -> None:
    """Save core event-damage graphs."""
    top_events = event_damage.head(20).sort_values("total_network_damage_usd")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        top_events["event_id"],
        top_events["total_network_damage_usd"] / 1e9,
        color="#1f77b4",
    )
    for bar, value in zip(bars, top_events["total_network_damage_usd"] / 1e9):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" ${value:.2f}B", va="center", fontsize=8)
    ax.set_xlabel("Total transmission line damage (billion USD)")
    ax.set_ylabel("IBTrACS event ID")
    ax.set_title("Snail: Top 20 IBTrACS Events by Florida Transmission Line Damage")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snail_ibtracs_top20_event_damage.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(event_damage["total_network_damage_usd"] / 1e9, bins=30, color="#4c78a8")
    ax.set_xlabel("Total transmission line damage (billion USD)")
    ax.set_ylabel("Number of historical events")
    ax.set_title("Snail: Distribution of Historical IBTrACS Event Damages")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snail_ibtracs_event_damage_histogram.png", dpi=300)
    plt.close()

    exceedance = event_damage.sort_values("total_network_damage_usd", ascending=False).copy()
    exceedance["rank"] = np.arange(1, len(exceedance) + 1)
    exceedance["empirical_exceedance_probability"] = exceedance["rank"] / (
        len(exceedance) + 1
    )
    exceedance.to_csv(OUTPUT_DIR / "snail_ibtracs_event_damage_exceedance_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        exceedance["empirical_exceedance_probability"],
        exceedance["total_network_damage_usd"] / 1e9,
        marker="o",
        markersize=3,
        linewidth=1.2,
    )
    ax.set_xlabel("Empirical exceedance probability among historical events")
    ax.set_ylabel("Total transmission line damage (billion USD)")
    ax.set_title("Snail: Historical Event Damage Exceedance Curve")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snail_ibtracs_event_damage_exceedance_curve.png", dpi=300)
    plt.close()


def main() -> None:
    raster_path, event_ids = create_multiband_event_geotiff()
    lines = load_florida_overhead_lines()
    print(f"Loaded Florida overhead in-service lines: {len(lines):,}")
    print(f"Using event raster stack: {raster_path}")
    print(f"IBTrACS event bands: {len(event_ids):,}")

    splits = snail_split_lines(lines, raster_path)
    splits_path = OUTPUT_DIR / "snail_florida_overhead_lines_split_by_wind_grid.gpkg"
    splits.to_file(splits_path, layer="snail_line_grid_splits", driver="GPKG")
    print(f"Snail split line segments: {len(splits):,}")
    print(f"Saved Snail split layer: {splits_path}")

    event_damage, line_event_damage = line_event_damage_from_splits(
        splits,
        raster_path,
        event_ids,
    )

    event_damage_path = OUTPUT_DIR / "snail_ibtracs_event_network_damage.csv"
    line_event_damage_path = OUTPUT_DIR / "snail_ibtracs_line_event_damage.csv"
    event_damage.to_csv(event_damage_path, index=False)
    line_event_damage.to_csv(line_event_damage_path, index=False)
    make_graphs(event_damage)

    print(f"Saved event damage summary: {event_damage_path}")
    print(f"Saved line-event damage table: {line_event_damage_path}")
    print("\nTop 10 damaging Snail events:")
    print(
        event_damage[
            [
                "event_id",
                "event_year",
                "max_line_wind_ms",
                "lines_with_wind_ge_25ms",
                "total_network_damage_usd",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
