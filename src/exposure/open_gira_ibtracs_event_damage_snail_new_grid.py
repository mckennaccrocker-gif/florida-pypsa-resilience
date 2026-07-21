"""
Snail event-by-event TC damage/exceedance curves for the newest Florida grid.

Uses the newest transmission geometry:
    data/Electricity/pypsa_florida_network_extended_tie_lines/qgis/
        florida_extended_tie_lines_for_qgis.gpkg

Workflow:
1. Convert open-gira IBTrACS event NetCDF to a Snail-friendly multiband GeoTIFF.
2. Split the newest grid's transmission line geometries by the wind raster grid.
3. For each event band, assign wind to each split segment.
4. Aggregate to max wind per line, apply the TC wind damage curve, and sum damage.
5. Rank events and plot empirical exceedance probability curves.

Run in the OpenGIRA/Snail environment, for example:
    wsl -d Ubuntu -- bash -lc "cd /mnt/c/oxford_tc_project && \
    /home/mckennacroc/.pixi/bin/pixi run --manifest-path /home/mckennacroc/projects/open-gira/pixi.toml \
    python data/Exposure/open_gira_ibtracs_event_damage_snail_new_grid.py"
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
    if platform.system().lower() == "windows":
        return Path(r"C:\oxford_tc_project")
    return Path("/mnt/c/oxford_tc_project")


PROJECT_DIR = project_root()
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
OUTPUT_DIR = PROJECT_DIR / "data" / "Exposure" / "florida_new_grid_tc_event_damage_snail"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIND_NETCDF = PROJECT_DIR / "open_gira_outputs" / "max_wind_field_FLORIDA_IBTrACS_0.nc"
WIND_GEOTIFF = OUTPUT_DIR / "open_gira_ibtracs_florida_event_wind_stack.tif"
NEW_GRID_GPKG = (
    ELECTRICITY_DIR
    / "pypsa_florida_network_extended_tie_lines"
    / "qgis"
    / "florida_extended_tie_lines_for_qgis.gpkg"
)

STORAGE_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3086"
NODATA_VALUE = -9999.0

STANDARD_VOLTAGES = np.array([69, 100, 115, 138, 161, 230, 345, 500, 765])
LINE_COST_PER_MILE_USD = {
    69: 1_500_000,
    100: 2_000_000,
    115: 2_500_000,
    138: 2_650_000,
    161: 2_750_000,
    230: 3_000_000,
    345: 3_050_000,
    500: 3_600_000,
    765: 5_900_000,
}


def normal_cdf(value: float) -> float:
    return 0.5 * (1 + erf(value / sqrt(2)))


def baseline_expected_damage_ratio(wind_speed_ms: float) -> float:
    if pd.isna(wind_speed_ms) or wind_speed_ms <= 0:
        return 0.0
    theta_values = np.array([30.0, 42.0, 55.0, 67.0])
    damage_ratios = np.array([0.05, 0.20, 0.50, 1.00])
    beta = 0.25
    exceedance = np.array(
        [normal_cdf(log(float(wind_speed_ms) / theta) / beta) for theta in theta_values]
    )
    state_probabilities = np.clip(exceedance - np.append(exceedance[1:], 0.0), 0, 1)
    return float(np.sum(state_probabilities * damage_ratios))


def snap_voltage_to_cost_class(voltage_kv: float) -> float:
    if pd.isna(voltage_kv):
        return np.nan
    if voltage_kv >= STANDARD_VOLTAGES.max():
        return int(STANDARD_VOLTAGES.max())
    return int(STANDARD_VOLTAGES[np.argmin(np.abs(STANDARD_VOLTAGES - voltage_kv))])


def parse_event_ids(src: rasterio.io.DatasetReader) -> list[str]:
    tag_value = src.tags().get("NETCDF_DIM_event_id_VALUES", "")
    match = re.search(r"\{(.+)\}", tag_value)
    if not match:
        return [f"event_{i:03d}" for i in range(1, src.count + 1)]
    event_ids = [part.strip() for part in match.group(1).split(",") if part.strip()]
    if len(event_ids) != src.count:
        return [f"event_{i:03d}" for i in range(1, src.count + 1)]
    return event_ids


def create_multiband_event_geotiff() -> tuple[Path, list[str]]:
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


def load_new_grid_lines() -> gpd.GeoDataFrame:
    original = gpd.read_file(NEW_GRID_GPKG, layer="cleaned_original_transmission_lines").to_crs(STORAGE_CRS)
    extensions = gpd.read_file(NEW_GRID_GPKG, layer="external_tie_line_extensions").to_crs(STORAGE_CRS)

    type_text = original.get("TYPE", pd.Series("", index=original.index)).fillna("").str.upper()
    status_text = original.get("STATUS", pd.Series("", index=original.index)).fillna("").str.upper().str.strip()
    original = original[type_text.str.contains("OVERHEAD", na=False) & status_text.eq("IN SERVICE")].copy()
    original["asset_source"] = "reviewed_florida_original_line"
    original["asset_line_id"] = original["pypsa_line"].astype(str)
    original["hifld_id"] = original["ID"]
    original["voltage_kv_raw"] = pd.to_numeric(original["pypsa_v_nom"], errors="coerce")
    original["s_nom_mva"] = pd.to_numeric(original["pypsa_s_nom"], errors="coerce")

    extensions = extensions.copy()
    extensions["asset_source"] = "external_tie_extension"
    extensions["asset_line_id"] = extensions["line"].astype(str)
    extensions["hifld_id"] = extensions["hifld_id"]
    extensions["TYPE"] = "AC; OVERHEAD"
    extensions["STATUS"] = "IN SERVICE"
    extensions["OWNER"] = extensions.get("owner", "")
    extensions["VOLTAGE"] = extensions["v_nom"]
    extensions["VOLT_CLASS"] = ""
    extensions["SUB_1"] = extensions["boundary_bus"]
    extensions["SUB_2"] = extensions["external_bus"]
    extensions["voltage_kv_raw"] = pd.to_numeric(extensions["v_nom"], errors="coerce")
    extensions["s_nom_mva"] = pd.to_numeric(extensions["s_nom"], errors="coerce")

    keep_cols = [
        "asset_line_id",
        "asset_source",
        "hifld_id",
        "TYPE",
        "STATUS",
        "OWNER",
        "VOLTAGE",
        "VOLT_CLASS",
        "SUB_1",
        "SUB_2",
        "voltage_kv_raw",
        "s_nom_mva",
        "geometry",
    ]
    lines = pd.concat([original[keep_cols], extensions[keep_cols]], ignore_index=True)
    lines = gpd.GeoDataFrame(lines, geometry="geometry", crs=STORAGE_CRS)
    lines = lines.explode(index_parts=False, ignore_index=True)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
    lines["analysis_line_id"] = np.arange(len(lines))

    projected = lines.to_crs(PROJECTED_CRS)
    lines["length_m"] = projected.length
    lines["length_km"] = lines["length_m"] / 1000
    lines["length_miles"] = lines["length_km"] / 1.609344
    lines["voltage_kv_cost_class"] = lines["voltage_kv_raw"].apply(snap_voltage_to_cost_class)
    lines["line_cost_per_mile_usd"] = lines["voltage_kv_cost_class"].map(LINE_COST_PER_MILE_USD)
    lines["replacement_cost_usd"] = lines["length_miles"] * lines["line_cost_per_mile_usd"]
    return lines.to_crs(STORAGE_CRS)


def snail_split_lines(lines: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    grid = snail.intersection.GridDefinition.from_raster(str(raster_path))
    splits = snail.intersection.split_linestrings(lines.reset_index(drop=True), grid)
    splits = snail.intersection.apply_indices(splits, grid, index_i="raster_i", index_j="raster_j")
    geod = Geod(ellps="WGS84")
    splits = splits.to_crs(STORAGE_CRS)
    splits["split_length_km"] = splits.geometry.apply(geod.geometry_length) / 1000
    return splits


def line_event_damage_from_splits(
    splits: gpd.GeoDataFrame,
    raster_path: Path,
    event_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    static_cols = [
        "analysis_line_id",
        "asset_line_id",
        "asset_source",
        "hifld_id",
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
        "s_nom_mva",
        "line_cost_per_mile_usd",
        "replacement_cost_usd",
    ]
    line_static = (
        splits[static_cols]
        .drop_duplicates(subset=["analysis_line_id"])
        .sort_values("analysis_line_id")
        .reset_index(drop=True)
    )
    total_replacement_cost = float(line_static["replacement_cost_usd"].fillna(0).sum())

    event_records = []
    line_event_records = []
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
            line_event = line_static.merge(line_max_wind, on="analysis_line_id", how="left")
            line_event.insert(0, "event_id", event_id)
            line_event.insert(1, "event_year", int(str(event_id)[:4]))
            line_event["damage_ratio"] = line_event["line_max_wind_ms"].apply(
                baseline_expected_damage_ratio
            )
            line_event["damage_usd"] = line_event["replacement_cost_usd"].fillna(0) * line_event["damage_ratio"]

            event_records.append(
                {
                    "event_id": event_id,
                    "event_year": int(str(event_id)[:4]),
                    "band_number": band_number,
                    "lines_intersected": int(line_event["line_max_wind_ms"].notna().sum()),
                    "lines_with_wind_ge_25ms": int((line_event["line_max_wind_ms"].fillna(0) >= 25).sum()),
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
                print(f"Processed event {band_number:,}/{len(event_ids):,}")

    event_damage = pd.DataFrame(event_records).sort_values("total_network_damage_usd", ascending=False)
    line_event_damage = pd.concat(line_event_records, ignore_index=True)
    return event_damage, line_event_damage


def make_exceedance_outputs(event_damage: pd.DataFrame) -> pd.DataFrame:
    ranked = event_damage.sort_values("total_network_damage_usd", ascending=False).reset_index(drop=True)
    ranked["event_rank"] = np.arange(1, len(ranked) + 1)
    ranked["empirical_exceedance_probability"] = ranked["event_rank"] / (len(ranked) + 1)
    ranked["damage_billion_usd"] = ranked["total_network_damage_usd"] / 1e9
    ranked.to_csv(OUTPUT_DIR / "new_grid_ibtracs_event_damage_exceedance_curve.csv", index=False)

    top_events = ranked.head(20).sort_values("damage_billion_usd")
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top_events["event_id"], top_events["damage_billion_usd"], color="#1f77b4")
    bars[-1].set_color("#d62728")
    for bar, value in zip(bars, top_events["damage_billion_usd"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" ${value:.2f}B", va="center", fontsize=8)
    ax.set_xlabel("Total network damage (billion USD)")
    ax.set_ylabel("IBTrACS event ID")
    ax.set_title("Top 20 IBTrACS Events by New-Grid Transmission Damage")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "new_grid_ibtracs_top20_event_damage.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ranked["empirical_exceedance_probability"], ranked["damage_billion_usd"], marker="o", markersize=3, linewidth=1.2, color="#d62728")
    ax.set_xlabel("Empirical exceedance probability, rank / (N + 1)")
    ax.set_ylabel("Total network damage (billion USD)")
    ax.set_title("New-Grid Historical IBTrACS Damage Exceedance Curve")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "new_grid_ibtracs_damage_exceedance_probability.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(ranked["empirical_exceedance_probability"], ranked["total_network_damage_usd"].clip(lower=1), marker="o", markersize=3, linewidth=1.2, color="#d62728")
    ax.set_xlabel("Empirical exceedance probability, rank / (N + 1)")
    ax.set_ylabel("Total network damage (USD, log scale)")
    ax.set_title("New-Grid Historical IBTrACS Damage Exceedance Curve, Log Scale")
    ax.grid(alpha=0.25, which="both")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "new_grid_ibtracs_damage_exceedance_probability_log.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ranked["event_rank"], ranked["damage_billion_usd"], marker="o", markersize=3, linewidth=1.2, color="#d62728")
    ax.set_xlabel("Event rank, 1 = highest damage")
    ax.set_ylabel("Total network damage (billion USD)")
    ax.set_title("New-Grid Damage vs. Event Rank")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "new_grid_ibtracs_damage_vs_event_rank.png", dpi=300)
    plt.close()
    return ranked


def main() -> None:
    raster_path, event_ids = create_multiband_event_geotiff()
    lines = load_new_grid_lines()
    print(f"Loaded newest-grid overhead in-service lines: {len(lines):,}")
    print(f"IBTrACS event bands: {len(event_ids):,}")
    print(f"Event wind raster stack: {raster_path}")

    lines.to_file(OUTPUT_DIR / "new_grid_lines_used_for_snail.gpkg", layer="new_grid_lines_used_for_snail", driver="GPKG")
    splits = snail_split_lines(lines, raster_path)
    splits.to_file(OUTPUT_DIR / "new_grid_lines_split_by_wind_grid.gpkg", layer="snail_line_grid_splits", driver="GPKG")
    print(f"Snail split line segments: {len(splits):,}")

    event_damage, line_event_damage = line_event_damage_from_splits(splits, raster_path, event_ids)
    event_damage.to_csv(OUTPUT_DIR / "new_grid_ibtracs_event_network_damage.csv", index=False)
    line_event_damage.to_csv(OUTPUT_DIR / "new_grid_ibtracs_line_event_damage.csv", index=False)
    ranked = make_exceedance_outputs(event_damage)

    summary = pd.DataFrame(
        [
            {
                "events": len(ranked),
                "lines_used": len(lines),
                "snail_split_segments": len(splits),
                "max_event_damage_usd": ranked["total_network_damage_usd"].max(),
                "max_event_damage_billion_usd": ranked["damage_billion_usd"].max(),
                "damaging_events_gt_1m_usd": int((ranked["total_network_damage_usd"] > 1_000_000).sum()),
                "top_event_id": ranked.iloc[0]["event_id"],
                "output_dir": str(OUTPUT_DIR),
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "new_grid_ibtracs_event_damage_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nTop 10 events:")
    print(ranked[["event_rank", "event_id", "event_year", "damage_billion_usd", "empirical_exceedance_probability"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
