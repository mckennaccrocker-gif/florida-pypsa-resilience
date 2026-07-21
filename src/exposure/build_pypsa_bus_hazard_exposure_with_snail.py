"""
Build PyPSA-bus/substation hazard exposure tables with nismod-snail.

The calibrated PyPSA model represents substations as PyPSA buses. This script
uses SNAIL point/raster indexing to assign each bus a flood depth or tropical
cyclone wind speed for each available return-period raster.

Outputs:
  - data/Exposure/pypsa_bus_hazard_exposure/pypsa_bus_flood_exposure_by_return_period.csv
  - data/Exposure/pypsa_bus_hazard_exposure/pypsa_bus_tc_exposure_by_return_period.csv
"""

from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import snail.intersection


PROJECT_DIR = Path(r"C:\oxford_tc_project")
PYPSA_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network"
FLOOD_DIR = PROJECT_DIR / "data" / "Hazards" / "Flood"
TC_DIR = PROJECT_DIR / "data" / "Hazards" / "Tropical_cyclones"
FLOOD_CROP_DIR = (
    PROJECT_DIR
    / "data"
    / "Exposure"
    / "line_flood_exposure_with_ids"
    / "snail_flood_cropped_rasters"
)
OUTPUT_DIR = PROJECT_DIR / "data" / "Exposure" / "pypsa_bus_hazard_exposure"

BUSES_FILE = PYPSA_DIR / "buses.csv"


def return_period_from_name(path: Path) -> int | None:
    match = re.search(r"RP(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_pypsa_buses() -> gpd.GeoDataFrame:
    buses = pd.read_csv(BUSES_FILE)
    required = {"name", "x", "y"}
    missing = required.difference(buses.columns)
    if missing:
        raise ValueError(f"{BUSES_FILE} missing columns: {sorted(missing)}")
    buses = buses.dropna(subset=["name", "x", "y"]).copy()
    buses["x"] = pd.to_numeric(buses["x"], errors="coerce")
    buses["y"] = pd.to_numeric(buses["y"], errors="coerce")
    buses = buses.dropna(subset=["x", "y"])
    return gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    )


def raster_value_at_snail_point(points: gpd.GeoDataFrame, raster_path: Path) -> pd.Series:
    with rasterio.open(raster_path) as src:
        working = points.to_crs(src.crs).copy()
        grid = snail.intersection.GridDefinition.from_raster(raster_path)
        indexed = snail.intersection.apply_indices(
            working,
            grid,
            index_i="raster_i",
            index_j="raster_j",
        )
        data = src.read(1)
        raster_i = pd.to_numeric(indexed["raster_i"], errors="coerce").fillna(-1).astype(int)
        raster_j = pd.to_numeric(indexed["raster_j"], errors="coerce").fillna(-1).astype(int)
        in_bounds = (
            raster_i.ge(0)
            & raster_j.ge(0)
            & raster_i.lt(data.shape[1])
            & raster_j.lt(data.shape[0])
        )
        values = np.full(len(indexed), np.nan, dtype=float)
        values[in_bounds.to_numpy()] = data[
            raster_j.loc[in_bounds].to_numpy(),
            raster_i.loc[in_bounds].to_numpy(),
        ].astype(float)
        if src.nodata is not None:
            values[values == src.nodata] = 0.0
    return pd.Series(values, index=points.index).fillna(0.0).clip(lower=0.0)


def discover_flood_rasters() -> list[tuple[int, Path]]:
    rasters: dict[int, Path] = {}
    for path in FLOOD_CROP_DIR.glob("JRC_RP*_USA_assets_florida_snail_crop.tif"):
        rp = return_period_from_name(path)
        if rp is not None:
            rasters[rp] = path
    if not rasters:
        for path in FLOOD_DIR.glob("JRC_RP*_USA_assets.vrt"):
            rp = return_period_from_name(path)
            if rp is not None:
                rasters[rp] = path
    return sorted(rasters.items())


def discover_tc_rasters() -> list[tuple[int, Path]]:
    rasters: dict[int, Path] = {}
    for path in TC_DIR.glob("STORM_constant_RP*_US_crop.tif"):
        rp = return_period_from_name(path)
        if rp is not None:
            rasters[rp] = path
    return sorted(rasters.items())


def build_flood_exposure(buses: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for return_period, raster_path in discover_flood_rasters():
        print(f"Assigning PyPSA-bus flood exposure with SNAIL: RP{return_period} {raster_path.name}", flush=True)
        values = raster_value_at_snail_point(buses, raster_path)
        table = buses.drop(columns="geometry").copy()
        table["return_period"] = return_period
        table["max_flood_depth_m"] = values.to_numpy()
        table["mean_flood_depth_m"] = values.to_numpy()
        table["hazard_source"] = str(raster_path.relative_to(PROJECT_DIR))
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def build_tc_exposure(buses: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for return_period, raster_path in discover_tc_rasters():
        print(f"Assigning PyPSA-bus TC wind exposure with SNAIL: RP{return_period} {raster_path.name}", flush=True)
        values = raster_value_at_snail_point(buses, raster_path)
        table = buses.drop(columns="geometry").copy()
        table["return_period"] = return_period
        table["dataset"] = f"storm_rp{return_period}"
        table["tc_wind_speed_ms"] = values.to_numpy()
        table["max_wind_speed_ms"] = values.to_numpy()
        table["hazard_source"] = str(raster_path.relative_to(PROJECT_DIR))
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    buses = read_pypsa_buses()
    print(f"Loaded {len(buses):,} PyPSA buses from {BUSES_FILE}")

    flood = build_flood_exposure(buses)
    tc = build_tc_exposure(buses)

    flood_path = OUTPUT_DIR / "pypsa_bus_flood_exposure_by_return_period.csv"
    tc_path = OUTPUT_DIR / "pypsa_bus_tc_exposure_by_return_period.csv"
    flood.to_csv(flood_path, index=False)
    tc.to_csv(tc_path, index=False)

    print("\nSaved PyPSA-bus SNAIL exposure tables:")
    print(flood_path)
    print(tc_path)
    print("\nFlood summary:")
    print(
        flood.groupby("return_period")["max_flood_depth_m"]
        .agg(["count", "mean", "max"])
        .to_string(float_format=lambda value: f"{value:,.3f}")
    )
    print("\nTC summary:")
    print(
        tc.groupby("return_period")["tc_wind_speed_ms"]
        .agg(["count", "mean", "max"])
        .to_string(float_format=lambda value: f"{value:,.3f}")
    )


if __name__ == "__main__":
    main()
