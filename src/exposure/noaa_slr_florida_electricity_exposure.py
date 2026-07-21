"""
NOAA sea-level-rise flood exposure for Florida electricity assets.

Assets used:
  - data/Electricity/florida_transmission_lines.gpkg
  - data/Electricity/florida_network_nodes_substations.gpkg
  - data/Electricity/florida_osm_powerplant_polygons_with_point_attributes.gpkg

NOAA SLR depth rasters:
  https://coast.noaa.gov/slrdata/Depth_Rasters/FL/index.html

Important NOAA interpretation:
  - File scenario values such as "1.0 ft" are sea-level-rise above MHHW.
  - Raster cell values are flood depths in meters above ground level.
  - The filename scenario is not the same thing as flood depth.

Run:
  python data/Exposure/noaa_slr_florida_electricity_exposure.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from pyproj import Geod
from rasterio.mask import mask
from rasterio.transform import array_bounds
from shapely.geometry import LineString, box

try:
    import snail.intersection

    HAS_SNAIL = True
except Exception:
    HAS_SNAIL = False


PROJECT_DIR = Path(r"C:\oxford_tc_project")
NOAA_URL = "https://coast.noaa.gov/slrdata/Depth_Rasters/FL/index.html"

RASTER_DIR = PROJECT_DIR / "data" / "flood" / "noaa_slr_depth_rasters" / "florida"
OUTPUT_DIR = PROJECT_DIR / "data" / "exposure" / "noaa_slr_hifld_florida"
RASTER_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
TRANSMISSION_LINES = ELECTRICITY_DIR / "florida_transmission_lines.gpkg"
SUBSTATIONS = ELECTRICITY_DIR / "florida_network_nodes_substations.gpkg"
POWER_PLANTS = ELECTRICITY_DIR / "florida_osm_powerplant_polygons_with_point_attributes.gpkg"

STORAGE_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3086"
LINE_SEGMENT_LENGTH_M = 5_000
SELECTED_MAP_SCENARIOS_FT = {1.0, 3.0, 6.0}
DOWNLOAD_ATTEMPTS = 5
MIN_SLR_SCENARIO_FT = 0.0
MAX_SLR_SCENARIO_FT = 10.0

DEPTH_BIN_ORDER = [
    "0-0.1 m",
    "0.1-0.5 m",
    "0.5-1 m",
    "1-2 m",
    "2-3 m",
    "3-5 m",
    ">5 m",
]
DEPTH_BIN_COLORS = {
    "0-0.1 m": "#d9f0f3",
    "0.1-0.5 m": "#a6bddb",
    "0.5-1 m": "#74a9cf",
    "1-2 m": "#2b8cbe",
    "2-3 m": "#fdae61",
    "3-5 m": "#f46d43",
    ">5 m": "#a50026",
}


def safe_name(value) -> str:
    """Make a string safe for filenames/layer names."""
    text = str(value).replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_")


def depth_bin(depth_m: float) -> str | float:
    """Classify positive flood depths into requested NOAA SLR depth bins."""
    if pd.isna(depth_m) or depth_m <= 0:
        return np.nan
    if depth_m <= 0.1:
        return "0-0.1 m"
    if depth_m <= 0.5:
        return "0.1-0.5 m"
    if depth_m <= 1:
        return "0.5-1 m"
    if depth_m <= 2:
        return "1-2 m"
    if depth_m <= 3:
        return "2-3 m"
    if depth_m <= 5:
        return "3-5 m"
    return ">5 m"


def clean_depth(value, nodata) -> float:
    """Convert raster values into valid non-negative flood depths in meters."""
    if value is None or pd.isna(value):
        return np.nan
    value = float(value)
    if nodata is not None and value == float(nodata):
        return np.nan
    return max(value, 0.0)


def download_noaa_rasters() -> tuple[int, int, int]:
    """Scrape NOAA Florida SLR page and download all linked GeoTIFFs."""
    print(f"Scraping NOAA page: {NOAA_URL}")
    response = requests.get(NOAA_URL, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    urls = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.lower().endswith(".tif"):
            urls.append(urljoin(NOAA_URL, href))
    urls = sorted(set(urls))

    downloaded = 0
    skipped = 0
    for url in urls:
        output_path = RASTER_DIR / Path(url).name
        part_path = output_path.with_suffix(output_path.suffix + ".part")

        expected_size = None
        try:
            head = requests.head(url, timeout=30, allow_redirects=True)
            if head.ok and head.headers.get("Content-Length"):
                expected_size = int(head.headers["Content-Length"])
        except Exception:
            expected_size = None

        if output_path.exists() and output_path.stat().st_size > 0:
            if expected_size is None or output_path.stat().st_size == expected_size:
                skipped += 1
                continue
            print(
                f"Existing file looks incomplete, redownloading: {output_path.name} "
                f"({output_path.stat().st_size:,} of {expected_size:,} bytes)"
            )
            output_path.unlink()

        success = download_one_file(url, part_path, output_path.name, expected_size)

        part_path.replace(output_path)
        downloaded += 1

    print(f"NOAA raster links found: {len(urls)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped existing: {skipped}")
    return len(urls), downloaded, skipped


def download_one_file(
    url: str,
    part_path: Path,
    display_name: str,
    expected_size: int | None,
) -> bool:
    """Download one NOAA raster with retries, preferring curl when available."""
    curl = shutil.which("curl.exe") or shutil.which("curl")
    last_error = None

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        if part_path.exists():
            part_path.unlink()

        print(f"Downloading {display_name} (attempt {attempt}/{DOWNLOAD_ATTEMPTS})")
        try:
            if curl:
                command = [
                    curl,
                    "--silent",
                    "--show-error",
                    "-L",
                    "--fail",
                    "--retry",
                    "5",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "30",
                    "--speed-time",
                    "180",
                    "--speed-limit",
                    "1024",
                    "-o",
                    str(part_path),
                    url,
                ]
                subprocess.run(command, check=True)
            else:
                with requests.get(url, stream=True, timeout=(30, 300)) as r:
                    r.raise_for_status()
                    with part_path.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)

            if expected_size is not None and part_path.stat().st_size != expected_size:
                raise IOError(
                    f"Download size mismatch for {display_name}: "
                    f"got {part_path.stat().st_size:,}, expected {expected_size:,}"
                )

            return True
        except (RequestException, IOError, subprocess.CalledProcessError) as exc:
            last_error = exc
            print(f"  Download failed: {exc}")

    raise RuntimeError(f"Failed to download {url}") from last_error


def parse_noaa_filename(path: Path) -> dict:
    """
    Parse NOAA filename like FL_East_1_slr_depth_1_0ft.tif.

    Region can include underscores. Scenario is feet above MHHW, not flood depth.
    """
    match = re.match(
        r"(?P<region>.+?)_slr_depth_(?P<scenario>\d+(?:[._]\d+)?)_?ft\.tif$",
        path.name,
        flags=re.IGNORECASE,
    )
    if not match:
        return {
            "noaa_region": "unknown",
            "slr_scenario_ft": np.nan,
            "slr_scenario_m": np.nan,
            "raster_path": str(path),
            "raster_filename": path.name,
        }
    scenario_ft = float(match.group("scenario").replace("_", "."))
    return {
        "noaa_region": match.group("region"),
        "slr_scenario_ft": scenario_ft,
        "slr_scenario_m": scenario_ft * 0.3048,
        "raster_path": str(path),
        "raster_filename": path.name,
    }


def create_raster_inventory() -> pd.DataFrame:
    """Create and save the NOAA Florida SLR raster inventory."""
    records = [parse_noaa_filename(path) for path in sorted(RASTER_DIR.glob("*.tif"))]
    inventory = pd.DataFrame(records)
    if not inventory.empty:
        inventory = inventory[
            inventory["slr_scenario_ft"].between(
                MIN_SLR_SCENARIO_FT,
                MAX_SLR_SCENARIO_FT,
                inclusive="both",
            )
        ].copy()
        inventory = inventory.sort_values(["noaa_region", "slr_scenario_ft", "raster_filename"])
    output = OUTPUT_DIR / "noaa_fl_slr_raster_inventory.csv"
    inventory.to_csv(output, index=False)
    print(f"Saved raster inventory: {output}")
    return inventory


def print_layer_info(name: str, gdf: gpd.GeoDataFrame) -> None:
    """Print required asset metadata."""
    print(f"\n{name}")
    print(f"  CRS: {gdf.crs}")
    print(f"  Bounds: {tuple(round(v, 6) for v in gdf.total_bounds)}")
    print(f"  Feature count: {len(gdf):,}")
    print(f"  Geometry types: {sorted(gdf.geometry.geom_type.dropna().unique())}")
    print(f"  Columns: {list(gdf.columns)}")


def load_electricity_assets() -> dict[str, gpd.GeoDataFrame]:
    """Load transmission lines, substation nodes, and power plant polygons."""
    assets = {
        "transmission_lines": gpd.read_file(TRANSMISSION_LINES).to_crs(STORAGE_CRS),
        "substations": gpd.read_file(SUBSTATIONS).to_crs(STORAGE_CRS),
        "powerplants": gpd.read_file(POWER_PLANTS).to_crs(STORAGE_CRS),
    }
    for name, gdf in assets.items():
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        assets[name] = gdf
        print_layer_info(name, gdf)
    return assets


def raster_bounds_gdf(src: rasterio.io.DatasetReader) -> gpd.GeoDataFrame:
    """Return raster bounds as a GeoDataFrame in the raster CRS."""
    left, bottom, right, top = array_bounds(src.height, src.width, src.transform)
    return gpd.GeoDataFrame(geometry=[box(left, bottom, right, top)], crs=src.crs)


def filter_to_raster_bounds(
    gdf: gpd.GeoDataFrame,
    src: rasterio.io.DatasetReader,
) -> gpd.GeoDataFrame:
    """Reproject to raster CRS and keep features intersecting raster bounds."""
    raster_crs = src.crs or STORAGE_CRS
    working = gdf.to_crs(raster_crs)
    bounds = raster_bounds_gdf(src)
    minx, miny, maxx, maxy = bounds.total_bounds
    candidates = working.cx[minx:maxx, miny:maxy].copy()
    if candidates.empty:
        return candidates
    return candidates[candidates.intersects(bounds.geometry.iloc[0])].copy()


def point_exposure(
    points: gpd.GeoDataFrame,
    src: rasterio.io.DatasetReader,
    scenario: dict,
) -> gpd.GeoDataFrame:
    """Sample NOAA flood depth at substation points."""
    if points.empty:
        return points.copy()

    coords = [(geom.x, geom.y) for geom in points.geometry]
    values = [value[0] for value in src.sample(coords)]
    out = points.copy()
    out["flood_depth_m"] = [clean_depth(value, src.nodata) for value in values]
    out["is_exposed"] = out["flood_depth_m"].fillna(0) > 0
    out["depth_bin"] = pd.Categorical(
        out["flood_depth_m"].apply(depth_bin),
        categories=DEPTH_BIN_ORDER,
        ordered=True,
    )
    out["noaa_region"] = scenario["noaa_region"]
    out["slr_scenario_ft"] = scenario["slr_scenario_ft"]
    out["raster_filename"] = scenario["raster_filename"]
    return out.to_crs(STORAGE_CRS)


def split_lines_to_segments(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fallback line method: split lines into regular 5 km segments."""
    metric = lines.to_crs(PROJECTED_CRS)
    records = []
    for _, row in metric.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part_id, part in enumerate(parts):
            length = float(part.length)
            if length <= 0:
                continue
            n_segments = max(1, int(np.ceil(length / LINE_SEGMENT_LENGTH_M)))
            distances = np.linspace(0, length, n_segments + 1)
            for segment_id, (start, end) in enumerate(zip(distances[:-1], distances[1:])):
                segment = LineString([part.interpolate(start), part.interpolate(end)])
                record = row.drop(labels="geometry").to_dict()
                record.update(
                    {
                        "line_part_id": part_id,
                        "line_segment_id": segment_id,
                        "segment_length_km": segment.length / 1000,
                        "geometry": segment,
                    }
                )
                records.append(record)
    return gpd.GeoDataFrame(records, crs=PROJECTED_CRS)


def line_exposure_fallback(
    lines: gpd.GeoDataFrame,
    src: rasterio.io.DatasetReader,
    scenario: dict,
) -> gpd.GeoDataFrame:
    """Fallback line exposure by regular segments and midpoint depth sampling."""
    if lines.empty:
        return lines.copy()
    segments = split_lines_to_segments(lines)
    raster_segments = segments.to_crs(src.crs)
    coords = []
    for geom in raster_segments.geometry:
        midpoint = geom.interpolate(0.5, normalized=True)
        coords.append((midpoint.x, midpoint.y))
    values = [value[0] for value in src.sample(coords)]

    out = segments.copy()
    out["flood_depth_m"] = [clean_depth(value, src.nodata) for value in values]
    out["is_exposed"] = out["flood_depth_m"].fillna(0) > 0
    out["depth_bin"] = pd.Categorical(
        out["flood_depth_m"].apply(depth_bin),
        categories=DEPTH_BIN_ORDER,
        ordered=True,
    )
    out["noaa_region"] = scenario["noaa_region"]
    out["slr_scenario_ft"] = scenario["slr_scenario_ft"]
    out["raster_filename"] = scenario["raster_filename"]
    out["line_exposure_method"] = "5km_midpoint_fallback"
    return out.to_crs(STORAGE_CRS)


def line_exposure_snail(
    lines: gpd.GeoDataFrame,
    src: rasterio.io.DatasetReader,
    scenario: dict,
    raster_path: Path,
) -> gpd.GeoDataFrame:
    """Line exposure using Snail grid-cell splitting."""
    grid = snail.intersection.GridDefinition.from_raster(str(raster_path))
    splits = snail.intersection.split_linestrings(lines.reset_index(drop=True), grid)
    splits = snail.intersection.apply_indices(
        splits,
        grid,
        index_i="raster_i",
        index_j="raster_j",
    )

    data = src.read(1).astype(float)
    values = snail.intersection.get_raster_values_for_splits(
        splits,
        data,
        index_i="raster_i",
        index_j="raster_j",
    )
    splits["flood_depth_m"] = [clean_depth(value, src.nodata) for value in values]

    geod = Geod(ellps="WGS84")
    splits = splits.to_crs(STORAGE_CRS)
    splits["segment_length_km"] = splits.geometry.apply(geod.geometry_length) / 1000
    splits["is_exposed"] = splits["flood_depth_m"].fillna(0) > 0
    splits["depth_bin"] = pd.Categorical(
        splits["flood_depth_m"].apply(depth_bin),
        categories=DEPTH_BIN_ORDER,
        ordered=True,
    )
    splits["noaa_region"] = scenario["noaa_region"]
    splits["slr_scenario_ft"] = scenario["slr_scenario_ft"]
    splits["raster_filename"] = scenario["raster_filename"]
    splits["line_exposure_method"] = "snail"
    return splits


def line_exposure(
    lines: gpd.GeoDataFrame,
    src: rasterio.io.DatasetReader,
    scenario: dict,
    raster_path: Path,
) -> gpd.GeoDataFrame:
    """Use Snail if available, otherwise fallback to 5 km line segment sampling."""
    if lines.empty:
        return lines.copy()
    if HAS_SNAIL:
        try:
            return line_exposure_snail(lines, src, scenario, raster_path)
        except Exception as exc:
            print(f"  Snail failed for {raster_path.name}; using fallback. Reason: {exc}")
    return line_exposure_fallback(lines, src, scenario)


def approximate_pixel_area_m2(src: rasterio.io.DatasetReader, geometry) -> float:
    """Approximate raster pixel area in square meters."""
    if src.crs and src.crs.is_projected:
        return abs(src.transform.a * src.transform.e)

    centroid = geometry.representative_point()
    lon = centroid.x
    lat = centroid.y
    dx = abs(src.transform.a)
    dy = abs(src.transform.e)
    geod = Geod(ellps="WGS84")
    pixel = box(lon - dx / 2, lat - dy / 2, lon + dx / 2, lat + dy / 2)
    area, _ = geod.geometry_area_perimeter(pixel)
    return abs(area)


def polygon_depth_stats(src: rasterio.io.DatasetReader, geometry) -> dict:
    """Calculate polygon flood stats using raster pixels, not centroid sampling."""
    try:
        out, _ = mask(src, [geometry], crop=True, filled=False)
        values = np.ma.asarray(out[0]).compressed().astype(float)
        if src.nodata is not None:
            values = values[values != float(src.nodata)]
        values = values[np.isfinite(values)]
        values = np.maximum(values, 0)
    except Exception:
        values = np.array([])

    if len(values) == 0:
        return {
            "max_depth_m": np.nan,
            "mean_depth_m": np.nan,
            "median_depth_m": np.nan,
            "flooded_area_m2": 0.0,
            "percent_flooded": 0.0,
            "flood_pixel_count": 0,
            "missing_or_nodata": True,
        }

    flooded = values > 0
    pixel_area = approximate_pixel_area_m2(src, geometry)
    flooded_area = float(np.sum(flooded) * pixel_area)
    polygon_area = geometry.area if src.crs and src.crs.is_projected else np.nan
    if pd.isna(polygon_area) or polygon_area <= 0:
        # Estimate polygon area in meters when the raster CRS is geographic.
        geod = Geod(ellps="WGS84")
        area, _ = geod.geometry_area_perimeter(geometry)
        polygon_area = abs(area)

    return {
        "max_depth_m": float(np.max(values)),
        "mean_depth_m": float(np.mean(values)),
        "median_depth_m": float(np.median(values)),
        "flooded_area_m2": flooded_area,
        "percent_flooded": flooded_area / polygon_area * 100 if polygon_area > 0 else np.nan,
        "flood_pixel_count": int(np.sum(flooded)),
        "missing_or_nodata": False,
    }


def polygon_exposure(
    polygons: gpd.GeoDataFrame,
    src: rasterio.io.DatasetReader,
    scenario: dict,
) -> gpd.GeoDataFrame:
    """Calculate polygon exposure using raster-mask/zonal statistics."""
    if polygons.empty:
        return polygons.copy()
    records = []
    for idx, row in polygons.iterrows():
        stats = polygon_depth_stats(src, row.geometry)
        base = row.drop(labels="geometry").to_dict()
        base.update(stats)
        base["noaa_region"] = scenario["noaa_region"]
        base["slr_scenario_ft"] = scenario["slr_scenario_ft"]
        base["raster_filename"] = scenario["raster_filename"]
        records.append({**base, "geometry": row.geometry})
    out = gpd.GeoDataFrame(records, crs=polygons.crs)
    out["is_exposed"] = (out["max_depth_m"].fillna(0) > 0) | (out["percent_flooded"].fillna(0) > 0)
    out["depth_bin"] = pd.Categorical(
        out["max_depth_m"].apply(depth_bin),
        categories=DEPTH_BIN_ORDER,
        ordered=True,
    )
    return out.to_crs(STORAGE_CRS)


def summarize_asset_exposure(
    frame: gpd.GeoDataFrame,
    asset_type: str,
) -> dict:
    """Create one scenario/region/asset summary row."""
    if frame.empty:
        return {}
    row = {
        "noaa_region": frame["noaa_region"].iloc[0],
        "slr_scenario_ft": frame["slr_scenario_ft"].iloc[0],
        "asset_type": asset_type,
        "total_assets": len(frame),
        "exposed_assets": int(frame["is_exposed"].sum()),
        "percent_exposed": frame["is_exposed"].mean() * 100 if len(frame) else 0,
    }

    if asset_type == "transmission_lines":
        row.update(
            {
                "mean_flood_depth_m": frame["flood_depth_m"].mean(),
                "max_flood_depth_m": frame["flood_depth_m"].max(),
                "total_exposed_transmission_line_length_km": frame.loc[
                    frame["is_exposed"], "segment_length_km"
                ].sum(),
                "total_flooded_powerplant_area_km2": np.nan,
                "average_percent_flooded_powerplants": np.nan,
            }
        )
    elif asset_type == "substations":
        row.update(
            {
                "mean_flood_depth_m": frame["flood_depth_m"].mean(),
                "max_flood_depth_m": frame["flood_depth_m"].max(),
                "total_exposed_transmission_line_length_km": np.nan,
                "total_flooded_powerplant_area_km2": np.nan,
                "average_percent_flooded_powerplants": np.nan,
            }
        )
    else:
        row.update(
            {
                "mean_flood_depth_m": frame["mean_depth_m"].mean(),
                "max_flood_depth_m": frame["max_depth_m"].max(),
                "total_exposed_transmission_line_length_km": np.nan,
                "total_flooded_powerplant_area_km2": frame["flooded_area_m2"].sum() / 1e6,
                "average_percent_flooded_powerplants": frame["percent_flooded"].mean(),
            }
        )
    return row


def group_optional_summaries(
    lines: gpd.GeoDataFrame,
    substations: gpd.GeoDataFrame,
    powerplants: gpd.GeoDataFrame,
) -> dict[str, pd.DataFrame]:
    """Create optional summaries by voltage, owner/operator, and subtype."""
    outputs = {}
    if not lines.empty:
        lines = lines.copy()
        lines["exposed_length_km_for_summary"] = np.where(
            lines["is_exposed"],
            lines["segment_length_km"],
            0,
        )
        for column in ["VOLT_CLASS", "OWNER", "TYPE", "STATUS"]:
            if column in lines.columns:
                outputs[f"lines_by_{column.lower()}"] = (
                    lines.groupby(["noaa_region", "slr_scenario_ft", column], dropna=False)
                    .agg(
                        total_segments=("geometry", "count"),
                        exposed_segments=("is_exposed", "sum"),
                        exposed_length_km=("exposed_length_km_for_summary", "sum"),
                    )
                    .reset_index()
                )
    if not powerplants.empty:
        for column in ["primary_fuel", "owner", "polygon_attribute_status"]:
            if column in powerplants.columns:
                outputs[f"powerplants_by_{column.lower()}"] = (
                    powerplants.groupby(["noaa_region", "slr_scenario_ft", column], dropna=False)
                    .agg(
                        total_polygons=("geometry", "count"),
                        exposed_polygons=("is_exposed", "sum"),
                        flooded_area_km2=("flooded_area_m2", lambda x: x.sum() / 1e6),
                        mean_percent_flooded=("percent_flooded", "mean"),
                    )
                    .reset_index()
                )
    if not substations.empty:
        for column in ["voltage_count", "is_step_up_down"]:
            if column in substations.columns:
                outputs[f"substations_by_{column.lower()}"] = (
                    substations.groupby(["noaa_region", "slr_scenario_ft", column], dropna=False)
                    .agg(total_substations=("geometry", "count"), exposed_substations=("is_exposed", "sum"))
                    .reset_index()
                )
    return outputs


def create_summary_tables(
    all_lines: list[gpd.GeoDataFrame],
    all_substations: list[gpd.GeoDataFrame],
    all_powerplants: list[gpd.GeoDataFrame],
) -> dict[str, pd.DataFrame]:
    """Create required summary tables."""
    lines = pd.concat(all_lines, ignore_index=True) if all_lines else gpd.GeoDataFrame()
    substations = pd.concat(all_substations, ignore_index=True) if all_substations else gpd.GeoDataFrame()
    powerplants = pd.concat(all_powerplants, ignore_index=True) if all_powerplants else gpd.GeoDataFrame()

    scenario_rows = []
    for frame, asset_type in [
        (lines, "transmission_lines"),
        (substations, "substations"),
        (powerplants, "powerplants"),
    ]:
        if frame.empty:
            continue
        for _, group in frame.groupby(["noaa_region", "slr_scenario_ft"], dropna=False):
            scenario_rows.append(summarize_asset_exposure(group, asset_type))

    by_scenario = pd.DataFrame(scenario_rows).sort_values(
        ["asset_type", "noaa_region", "slr_scenario_ft"]
    )

    by_asset_type = (
        by_scenario.groupby(["asset_type", "slr_scenario_ft"], as_index=False)
        .agg(
            total_assets=("total_assets", "sum"),
            exposed_assets=("exposed_assets", "sum"),
            mean_flood_depth_m=("mean_flood_depth_m", "mean"),
            max_flood_depth_m=("max_flood_depth_m", "max"),
            total_exposed_transmission_line_length_km=(
                "total_exposed_transmission_line_length_km",
                "sum",
            ),
            total_flooded_powerplant_area_km2=("total_flooded_powerplant_area_km2", "sum"),
            average_percent_flooded_powerplants=("average_percent_flooded_powerplants", "mean"),
        )
    )
    by_asset_type["percent_exposed"] = (
        by_asset_type["exposed_assets"] / by_asset_type["total_assets"] * 100
    )

    depth_rows = []
    for frame, asset_type, value_column in [
        (lines, "transmission_lines", "segment_length_km"),
        (substations, "substations", None),
        (powerplants, "powerplants", "flooded_area_m2"),
    ]:
        if frame.empty:
            continue
        grouped = frame.dropna(subset=["depth_bin"]).groupby(
            ["noaa_region", "slr_scenario_ft", "depth_bin"],
            observed=False,
        )
        for keys, group in grouped:
            noaa_region, scenario_ft, bin_name = keys
            depth_rows.append(
                {
                    "asset_type": asset_type,
                    "noaa_region": noaa_region,
                    "slr_scenario_ft": scenario_ft,
                    "depth_bin": bin_name,
                    "asset_count": len(group),
                    "exposed_asset_count": int(group["is_exposed"].sum()),
                    "exposed_line_length_km": group[value_column].sum()
                    if value_column == "segment_length_km"
                    else np.nan,
                    "flooded_powerplant_area_km2": group[value_column].sum() / 1e6
                    if value_column == "flooded_area_m2"
                    else np.nan,
                }
            )
    by_depth_bin = pd.DataFrame(depth_rows)

    outputs = {
        "noaa_slr_hifld_exposure_summary_by_scenario": by_scenario,
        "noaa_slr_hifld_exposure_summary_by_asset_type": by_asset_type,
        "noaa_slr_hifld_exposure_summary_by_depth_bin": by_depth_bin,
    }
    outputs.update(group_optional_summaries(lines, substations, powerplants))
    return outputs


def save_outputs_for_scenario(
    lines: gpd.GeoDataFrame,
    substations: gpd.GeoDataFrame,
    powerplants: gpd.GeoDataFrame,
    scenario: dict,
) -> None:
    """Save per-raster scenario GeoPackages."""
    region = safe_name(scenario["noaa_region"])
    scenario_ft = safe_name(f"{scenario['slr_scenario_ft']}ft")

    lines.to_file(
        OUTPUT_DIR / f"hifld_lines_exposure_{region}_{scenario_ft}.gpkg",
        layer="hifld_lines_exposure",
        driver="GPKG",
    )
    substations.to_file(
        OUTPUT_DIR / f"hifld_substations_exposure_{region}_{scenario_ft}.gpkg",
        layer="hifld_substations_exposure",
        driver="GPKG",
    )
    powerplants.to_file(
        OUTPUT_DIR / f"hifld_powerplants_exposure_{region}_{scenario_ft}.gpkg",
        layer="hifld_powerplants_exposure",
        driver="GPKG",
    )


def plot_results(summaries: dict[str, pd.DataFrame]) -> None:
    """Create requested PNG plots from summary tables."""
    by_asset = summaries["noaa_slr_hifld_exposure_summary_by_asset_type"]
    by_scenario = summaries["noaa_slr_hifld_exposure_summary_by_scenario"]

    if by_asset.empty:
        return

    # Number of exposed assets by scenario height.
    pivot = by_asset.pivot_table(
        index="slr_scenario_ft",
        columns="asset_type",
        values="exposed_assets",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()
    ax = pivot.plot(kind="bar", figsize=(10, 5.5))
    ax.set_xlabel("SLR scenario height above MHHW (ft)")
    ax.set_ylabel("Number of exposed assets")
    ax.set_title("NOAA SLR Exposed Florida Electricity Assets")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exposed_assets_by_slr_scenario.png", dpi=300)
    plt.close()

    # Percent of substations exposed.
    subs = by_asset[by_asset["asset_type"] == "substations"].sort_values("slr_scenario_ft")
    if not subs.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(subs["slr_scenario_ft"], subs["percent_exposed"], marker="o")
        ax.set_xlabel("SLR scenario height above MHHW (ft)")
        ax.set_ylabel("Substations exposed (%)")
        ax.set_title("Percent of Florida Substations Exposed")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "percent_substations_exposed_by_slr_scenario.png", dpi=300)
        plt.close()

    # Transmission line length.
    lines = by_asset[by_asset["asset_type"] == "transmission_lines"].sort_values("slr_scenario_ft")
    if not lines.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            lines["slr_scenario_ft"],
            lines["total_exposed_transmission_line_length_km"],
            marker="o",
        )
        ax.set_xlabel("SLR scenario height above MHHW (ft)")
        ax.set_ylabel("Total exposed transmission line length (km)")
        ax.set_title("NOAA SLR Exposed Florida Transmission Line Length")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "exposed_transmission_length_by_slr_scenario.png", dpi=300)
        plt.close()

    # Flooded power plant area.
    plants = by_asset[by_asset["asset_type"] == "powerplants"].sort_values("slr_scenario_ft")
    if not plants.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            plants["slr_scenario_ft"],
            plants["total_flooded_powerplant_area_km2"],
            marker="o",
        )
        ax.set_xlabel("SLR scenario height above MHHW (ft)")
        ax.set_ylabel("Flooded power plant area (km2)")
        ax.set_title("NOAA SLR Flooded Florida Power Plant Polygon Area")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "flooded_powerplant_area_by_slr_scenario.png", dpi=300)
        plt.close()

    # Exposure by NOAA region.
    for asset_type in by_scenario["asset_type"].dropna().unique():
        subset = by_scenario[by_scenario["asset_type"] == asset_type]
        wide = subset.pivot_table(
            index="slr_scenario_ft",
            columns="noaa_region",
            values="exposed_assets",
            aggfunc="sum",
            fill_value=0,
        ).sort_index()
        ax = wide.plot(kind="bar", stacked=True, figsize=(11, 5.5))
        ax.set_xlabel("SLR scenario height above MHHW (ft)")
        ax.set_ylabel("Exposed assets")
        ax.set_title(f"NOAA SLR Exposure by Region: {asset_type}")
        ax.legend(title="NOAA region", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"exposure_by_noaa_region_{asset_type}.png", dpi=300)
        plt.close()


def map_selected_scenarios(
    all_lines: list[gpd.GeoDataFrame],
    all_substations: list[gpd.GeoDataFrame],
    all_powerplants: list[gpd.GeoDataFrame],
) -> None:
    """Create simple exposed-asset maps for selected SLR scenario heights."""
    lines = pd.concat(all_lines, ignore_index=True) if all_lines else gpd.GeoDataFrame()
    subs = pd.concat(all_substations, ignore_index=True) if all_substations else gpd.GeoDataFrame()
    plants = pd.concat(all_powerplants, ignore_index=True) if all_powerplants else gpd.GeoDataFrame()

    for scenario_ft in sorted(SELECTED_MAP_SCENARIOS_FT):
        line_s = gpd.GeoDataFrame(
            lines[lines["slr_scenario_ft"].eq(scenario_ft)],
            geometry="geometry",
            crs=STORAGE_CRS,
        )
        sub_s = gpd.GeoDataFrame(
            subs[subs["slr_scenario_ft"].eq(scenario_ft)],
            geometry="geometry",
            crs=STORAGE_CRS,
        )
        plant_s = gpd.GeoDataFrame(
            plants[plants["slr_scenario_ft"].eq(scenario_ft)],
            geometry="geometry",
            crs=STORAGE_CRS,
        )
        if line_s.empty and sub_s.empty and plant_s.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        if not line_s.empty:
            line_s[line_s["is_exposed"]].plot(ax=ax, color="#2b8cbe", linewidth=0.7, label="Exposed lines")
        if not plant_s.empty:
            plant_s[plant_s["is_exposed"]].boundary.plot(ax=ax, color="#d73027", linewidth=0.8, label="Exposed power plants")
        if not sub_s.empty:
            sub_s[sub_s["is_exposed"]].plot(ax=ax, color="#000000", markersize=8, label="Exposed substations")
        ax.set_title(f"NOAA SLR Exposed Florida Electricity Assets: {scenario_ft:g} ft")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"map_exposed_assets_{safe_name(str(scenario_ft))}ft.png", dpi=300)
        plt.close()


def run_exposure_analysis(inventory: pd.DataFrame, assets: dict[str, gpd.GeoDataFrame]) -> None:
    """Loop through every NOAA raster and process all asset types."""
    all_lines = []
    all_substations = []
    all_powerplants = []

    for scenario in inventory.to_dict("records"):
        raster_path = Path(scenario["raster_path"])
        print(f"\nProcessing {raster_path.name}")
        with rasterio.open(raster_path) as src:
            print(f"  Raster CRS: {src.crs}")
            print(f"  Raster bounds: {src.bounds}")
            print(f"  Raster nodata: {src.nodata}")
            print(f"  Raster transform: {src.transform}")

            lines_inside = filter_to_raster_bounds(assets["transmission_lines"], src)
            substations_inside = filter_to_raster_bounds(assets["substations"], src)
            powerplants_inside = filter_to_raster_bounds(assets["powerplants"], src)
            print(f"  Transmission features inside raster: {len(lines_inside):,}")
            print(f"  Substations inside raster: {len(substations_inside):,}")
            print(f"  Power plant polygons inside raster: {len(powerplants_inside):,}")

            lines_exp = line_exposure(lines_inside, src, scenario, raster_path)
            subs_exp = point_exposure(substations_inside, src, scenario)
            plants_exp = polygon_exposure(powerplants_inside, src, scenario)

            for label, frame, depth_col in [
                ("lines", lines_exp, "flood_depth_m"),
                ("substations", subs_exp, "flood_depth_m"),
                ("powerplants", plants_exp, "max_depth_m"),
            ]:
                missing = int(frame[depth_col].isna().sum()) if depth_col in frame.columns else 0
                exposed = int(frame["is_exposed"].sum()) if "is_exposed" in frame.columns else 0
                min_depth = frame[depth_col].min() if depth_col in frame.columns and not frame.empty else np.nan
                mean_depth = frame[depth_col].mean() if depth_col in frame.columns and not frame.empty else np.nan
                max_depth = frame[depth_col].max() if depth_col in frame.columns and not frame.empty else np.nan
                print(
                    f"  {label}: exposed={exposed:,}, missing/nodata={missing:,}, "
                    f"min={min_depth}, mean={mean_depth}, max={max_depth}"
                )

            save_outputs_for_scenario(lines_exp, subs_exp, plants_exp, scenario)
            all_lines.append(lines_exp)
            all_substations.append(subs_exp)
            all_powerplants.append(plants_exp)

    summaries = create_summary_tables(all_lines, all_substations, all_powerplants)
    for name, summary in summaries.items():
        output = OUTPUT_DIR / f"{name}.csv"
        summary.to_csv(output, index=False)
        print(f"Saved summary: {output}")

    plot_results(summaries)
    map_selected_scenarios(all_lines, all_substations, all_powerplants)
    print(f"\nSaved NOAA SLR outputs in: {OUTPUT_DIR}")


def main() -> None:
    download_noaa_rasters()
    inventory = create_raster_inventory()
    if inventory.empty:
        raise FileNotFoundError(f"No NOAA SLR GeoTIFFs found in {RASTER_DIR}")
    assets = load_electricity_assets()
    print(f"\nSnail available: {HAS_SNAIL}")
    run_exposure_analysis(inventory, assets)


if __name__ == "__main__":
    main()
