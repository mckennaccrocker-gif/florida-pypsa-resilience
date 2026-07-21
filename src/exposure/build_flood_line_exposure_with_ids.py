"""
Build line-level flood exposure tables using nismod-snail.

This script uses SNAIL's raster/line intersection workflow so that transmission
lines are split by flood-raster grid cells, assigned raster cell indices, and
given the flood depth from each intersected cell. The outputs preserve
HIFLD/PyPSA line identifiers so flood exposure can be mapped directly into
PyPSA hazard scenarios.

Inputs:
  - data/Electricity/florida_lines_with_s_nom.gpkg
  - data/Hazards/Flood/JRC_RP*_USA_assets.vrt

Outputs:

  - flood_line_exposure_segments_with_ids.gpkg/csv
  - flood_line_exposure_by_line_return_period.csv

The summary table includes both `florida_line_id` and HIFLD `ID`, making it
usable directly by calibrated PyPSA hazard scenarios.
"""

from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds, transform as window_transform
import snail.intersection
from pyproj import Geod


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
HAZARD_DIR = PROJECT_DIR / "data" / "Hazards" / "Flood"
OUTPUT_DIR = PROJECT_DIR / "data" / "Exposure" / "line_flood_exposure_with_ids"
CROPPED_RASTER_DIR = OUTPUT_DIR / "snail_flood_cropped_rasters"

LINES_FILE = ELECTRICITY_DIR / "florida_lines_with_s_nom.gpkg"
FLOOD_BINS = [0, 0.5, 1, 2, 5, np.inf]
FLOOD_LABELS = ["0-0.5 m", "0.5-1 m", "1-2 m", "2-5 m", ">5 m"]
SNAIL_LINE_CHUNK_SIZE = 100


def return_period_from_path(path: Path) -> int | None:
    match = re.search(r"RP(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def discover_flood_rasters() -> list[tuple[int, Path]]:
    rasters: dict[int, Path] = {}
    for path in HAZARD_DIR.glob("JRC_RP*_USA_assets.vrt"):
        return_period = return_period_from_path(path)
        if return_period is not None:
            rasters[return_period] = path
    return sorted(rasters.items())


def crop_raster_to_lines_extent(raster_path: Path, lines_raster_crs: gpd.GeoDataFrame) -> Path:
    """Write a Florida-sized flood raster for SNAIL intersection."""
    CROPPED_RASTER_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CROPPED_RASTER_DIR / f"{raster_path.stem}_florida_snail_crop.tif"
    if output_path.exists():
        return output_path

    with rasterio.open(raster_path) as src:
        minx, miny, maxx, maxy = lines_raster_crs.total_bounds
        buffer = 0.5 if src.crs and src.crs.is_geographic else 50_000
        raw_window = from_bounds(
            minx - buffer,
            miny - buffer,
            maxx + buffer,
            maxy + buffer,
            transform=src.transform,
        )
        raster_window = Window(0, 0, src.width, src.height)
        window = raw_window.round_offsets().round_lengths().intersection(raster_window)
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": window_transform(window, src.transform),
                "compress": "deflate",
            }
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(
                source=str(raster_path),
                description="Florida crop of JRC flood depth raster for SNAIL line intersection",
            )
    return output_path


def clean_depth(value: float, nodata: float | None) -> float:
    if value is None or np.isnan(value):
        return np.nan
    if nodata is not None and float(value) == float(nodata):
        return 0.0
    return max(float(value), 0.0)


def classify_flood_depth(depth: float) -> str | float:
    if pd.isna(depth):
        return np.nan
    if depth == 0:
        return "0 m"
    return pd.cut([depth], bins=FLOOD_BINS, labels=FLOOD_LABELS, include_lowest=True)[0]


def classify_flood_depth_series(depths: pd.Series) -> pd.Series:
    values = pd.to_numeric(depths, errors="coerce")
    labels = np.full(len(values), np.nan, dtype=object)
    labels[values.eq(0).to_numpy()] = "0 m"
    labels[(values.gt(0) & values.le(0.5)).to_numpy()] = "0-0.5 m"
    labels[(values.gt(0.5) & values.le(1)).to_numpy()] = "0.5-1 m"
    labels[(values.gt(1) & values.le(2)).to_numpy()] = "1-2 m"
    labels[(values.gt(2) & values.le(5)).to_numpy()] = "2-5 m"
    labels[values.gt(5).to_numpy()] = ">5 m"
    return pd.Series(labels, index=depths.index)


def read_lines_for_snail(raster_path: Path) -> gpd.GeoDataFrame:
    lines = gpd.read_file(LINES_FILE)
    required = {"florida_line_id", "ID", "geometry"}
    missing = required.difference(lines.columns)
    if missing:
        raise ValueError(f"{LINES_FILE} missing columns: {sorted(missing)}")

    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
    lines = lines.explode(index_parts=False, ignore_index=True)
    for column in lines.columns:
        if column != "geometry" and (
            pd.api.types.is_string_dtype(lines[column])
            or str(lines[column].dtype).startswith("string")
        ):
            lines[column] = lines[column].astype(object)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    return lines.to_crs(raster_crs)


def snail_intersect_lines_with_flood_raster(
    lines: gpd.GeoDataFrame,
    raster_path: Path,
) -> gpd.GeoDataFrame:
    grid = snail.intersection.GridDefinition.from_raster(raster_path)
    split_parts = []
    total = len(lines)
    for start in range(0, total, SNAIL_LINE_CHUNK_SIZE):
        end = min(start + SNAIL_LINE_CHUNK_SIZE, total)
        print(f"  SNAIL splitting lines {start + 1:,}-{end:,} of {total:,}", flush=True)
        chunk = lines.iloc[start:end].copy().reset_index(drop=True)
        for column in chunk.columns:
            if column != "geometry" and (
                pd.api.types.is_string_dtype(chunk[column])
                or str(chunk[column].dtype).startswith("string")
            ):
                chunk[column] = chunk[column].astype(object)
        chunk_splits = snail.intersection.split_linestrings(chunk, grid)
        chunk_splits = snail.intersection.apply_indices(
            chunk_splits,
            grid,
            index_i="raster_i",
            index_j="raster_j",
        )
        split_parts.append(chunk_splits)

    splits = gpd.GeoDataFrame(pd.concat(split_parts, ignore_index=True), crs=lines.crs)

    geod = Geod(ellps="WGS84")
    splits = splits.to_crs("EPSG:4326")
    splits["length_km"] = splits.geometry.apply(geod.geometry_length) / 1000.0
    return splits


def assign_flood_values_to_snail_splits(
    splits: gpd.GeoDataFrame,
    raster_path: Path,
    return_period: int,
) -> gpd.GeoDataFrame:
    with rasterio.open(raster_path) as src:
        working = splits.to_crs(src.crs).copy()
        working = working.drop(columns=["raster_i", "raster_j"], errors="ignore")
        grid = snail.intersection.GridDefinition.from_raster(raster_path)
        working = snail.intersection.apply_indices(
            working,
            grid,
            index_i="raster_i",
            index_j="raster_j",
        )
        data = src.read(1)
        raster_i = pd.to_numeric(working["raster_i"], errors="coerce").fillna(-1).astype(int)
        raster_j = pd.to_numeric(working["raster_j"], errors="coerce").fillna(-1).astype(int)
        in_bounds = (
            raster_i.ge(0)
            & raster_j.ge(0)
            & raster_i.lt(data.shape[1])
            & raster_j.lt(data.shape[0])
        )
        values = np.full(len(working), np.nan, dtype=float)
        values[in_bounds.to_numpy()] = data[
            raster_j.loc[in_bounds].to_numpy(),
            raster_i.loc[in_bounds].to_numpy(),
        ].astype(float)
        if src.nodata is not None:
            values[values == src.nodata] = 0.0

    out = working.to_crs("EPSG:4326")
    out["return_period"] = return_period
    out["segment_index"] = out.groupby("florida_line_id").cumcount()
    out["flood_depth_m"] = [clean_depth(value, np.nan) for value in values]
    out["flood_class"] = classify_flood_depth_series(out["flood_depth_m"])
    return out


def summarize_by_line(segments: gpd.GeoDataFrame) -> pd.DataFrame:
    exposed_0_5 = segments["flood_depth_m"].fillna(0.0) > 0.5
    exposed_1_0 = segments["flood_depth_m"].fillna(0.0) >= 1.0
    segments = segments.assign(
        exposed_length_gt_0_5m_km=np.where(exposed_0_5, segments["length_km"], 0.0),
        exposed_length_ge_1m_km=np.where(exposed_1_0, segments["length_km"], 0.0),
    )
    summary = (
        segments.groupby(["return_period", "florida_line_id", "ID"], dropna=False)
        .agg(
            TYPE=("TYPE", "first"),
            VOLTAGE=("VOLTAGE", "first"),
            VOLT_CLASS=("VOLT_CLASS", "first"),
            s_nom_mva=("s_nom_mva", "first"),
            segment_count=("segment_index", "size"),
            total_sampled_length_km=("length_km", "sum"),
            max_flood_depth_m=("flood_depth_m", "max"),
            mean_flood_depth_m=("flood_depth_m", "mean"),
            exposed_length_gt_0_5m_km=("exposed_length_gt_0_5m_km", "sum"),
            exposed_length_ge_1m_km=("exposed_length_ge_1m_km", "sum"),
        )
        .reset_index()
    )
    summary["max_flood_class"] = classify_flood_depth_series(summary["max_flood_depth_m"])
    summary["exposed_gt_0_5m"] = summary["max_flood_depth_m"].fillna(0.0) > 0.5
    summary["exposed_ge_1m"] = summary["max_flood_depth_m"].fillna(0.0) >= 1.0
    return summary.sort_values(["return_period", "florida_line_id"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rasters = discover_flood_rasters()
    if not rasters:
        raise FileNotFoundError(f"No JRC_RP*_USA_assets.vrt rasters found in {HAZARD_DIR}")

    base_return_period, base_raster_path = rasters[0]
    base_lines_full_grid = read_lines_for_snail(base_raster_path)
    cropped_rasters = []
    for return_period, raster_path in rasters:
        cropped_raster = crop_raster_to_lines_extent(raster_path, base_lines_full_grid)
        cropped_rasters.append((return_period, cropped_raster))
        print(f"Prepared Florida SNAIL crop for RP{return_period}: {cropped_raster.name}", flush=True)

    base_return_period, base_raster_path = cropped_rasters[0]
    print(
        f"SNAIL splitting Florida lines on base JRC flood grid "
        f"(RP{base_return_period}: {base_raster_path.name})",
        flush=True,
    )
    base_splits_gpkg = OUTPUT_DIR / "flood_line_base_snail_splits.gpkg"
    if base_splits_gpkg.exists():
        print(f"Reading cached SNAIL base splits: {base_splits_gpkg}", flush=True)
        base_splits = gpd.read_file(base_splits_gpkg)
    else:
        lines = read_lines_for_snail(base_raster_path)
        base_splits = snail_intersect_lines_with_flood_raster(lines, base_raster_path)
        base_splits.to_file(base_splits_gpkg, layer="flood_line_base_snail_splits", driver="GPKG")
    print(f"Created {len(base_splits):,} SNAIL line/raster split segments.", flush=True)

    all_segments = []
    for return_period, raster_path in cropped_rasters:
        print(f"Assigning JRC flood depths to SNAIL splits for RP{return_period}: {raster_path.name}", flush=True)
        splits = assign_flood_values_to_snail_splits(base_splits, raster_path, return_period)
        all_segments.append(splits)

    segments = gpd.GeoDataFrame(pd.concat(all_segments, ignore_index=True), crs="EPSG:4326")
    summary = summarize_by_line(segments)

    segments_csv = OUTPUT_DIR / "flood_line_exposure_segments_with_ids.csv"
    segments_gpkg = OUTPUT_DIR / "flood_line_exposure_segments_with_ids.gpkg"
    summary_csv = OUTPUT_DIR / "flood_line_exposure_by_line_return_period.csv"

    segments.drop(columns="geometry").to_csv(segments_csv, index=False)
    segments.to_file(segments_gpkg, layer="flood_line_exposure_segments_with_ids", driver="GPKG")
    summary.to_csv(summary_csv, index=False)

    print("\nSaved flood line exposure outputs:")
    print(segments_csv)
    print(segments_gpkg)
    print(summary_csv)
    print("\nSummary by return period:")
    rp_summary = (
        summary.groupby("return_period")
        .agg(
            lines=("florida_line_id", "size"),
            exposed_lines_ge_1m=("exposed_ge_1m", "sum"),
            max_flood_depth_m=("max_flood_depth_m", "max"),
            exposed_capacity_ge_1m_mva=("s_nom_mva", lambda s: s[summary.loc[s.index, "exposed_ge_1m"]].sum()),
        )
        .reset_index()
    )
    print(rp_summary.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))


if __name__ == "__main__":
    main()
