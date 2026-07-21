"""
Download OSM power plant polygons for Florida and match them to cleaned plant points.

Inputs:
  - data/Electricity/florida_powerplants_cleaned.gpkg

Outputs:
  - data/Electricity/florida_osm_powerplant_polygons.gpkg
  - data/Electricity/florida_powerplants_cleaned_with_polygons.gpkg
  - data/Electricity/florida_powerplant_polygon_matches_review.csv

Matching priority:
  1. point within OSM power=plant polygon
  2. nearest OSM power=plant polygon within MAX_NEAREST_DISTANCE_M

Storage CRS is EPSG:4326. Distance and area calculations use EPSG:3086.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
BOUNDARY_DIR = PROJECT_DIR / "data" / "Boundaries"

CLEANED_POINTS_FILE = ELECTRICITY_DIR / "florida_powerplants_cleaned.gpkg"
OSM_POLYGONS_GPKG = ELECTRICITY_DIR / "florida_osm_powerplant_polygons.gpkg"
POINTS_WITH_POLYGONS_GPKG = (
    ELECTRICITY_DIR / "florida_powerplants_cleaned_with_polygons.gpkg"
)
MATCH_REVIEW_CSV = ELECTRICITY_DIR / "florida_powerplant_polygon_matches_review.csv"

CENSUS_STATES_ZIP_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
)
CENSUS_STATES_ZIP = BOUNDARY_DIR / "tl_2023_us_state.zip"

PROJECTED_CRS = "EPSG:3086"  # Florida Albers, meters
MAX_NEAREST_DISTANCE_M = 5_000


STOPWORDS = {
    "power",
    "plant",
    "station",
    "generating",
    "generation",
    "energy",
    "facility",
    "solar",
    "farm",
    "center",
    "electric",
    "project",
    "llc",
    "inc",
    "company",
    "co",
}


def clean_name(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [token for token in text.split() if token not in STOPWORDS]
    return " ".join(tokens)


def name_similarity(left, right) -> float:
    left_clean = clean_name(left)
    right_clean = clean_name(right)
    if not left_clean or not right_clean:
        return 0.0
    return SequenceMatcher(None, left_clean, right_clean).ratio()


def load_florida_boundary() -> gpd.GeoDataFrame:
    """Download/read Census state boundaries and return Florida in EPSG:4326."""
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    if not CENSUS_STATES_ZIP.exists():
        print("Downloading Census state boundaries:", CENSUS_STATES_ZIP_URL)
        urlretrieve(CENSUS_STATES_ZIP_URL, CENSUS_STATES_ZIP)

    states = gpd.read_file(CENSUS_STATES_ZIP).to_crs("EPSG:4326")
    if "STUSPS" in states.columns:
        florida = states[states["STUSPS"] == "FL"].copy()
    else:
        florida = states[states["NAME"].str.lower() == "florida"].copy()
    if florida.empty:
        raise ValueError("Florida boundary not found.")
    return florida


def download_or_load_osm_powerplant_polygons() -> gpd.GeoDataFrame:
    """Load cached OSM plant polygons or download them from Overpass."""
    ELECTRICITY_DIR.mkdir(parents=True, exist_ok=True)
    if OSM_POLYGONS_GPKG.exists():
        print("Loading cached OSM plant polygons:", OSM_POLYGONS_GPKG)
        return gpd.read_file(OSM_POLYGONS_GPKG).to_crs("EPSG:4326")

    florida = load_florida_boundary()
    florida_geometry = florida.geometry.iloc[0]

    print("Downloading OSM power=plant features for Florida using osmnx/Overpass...")
    ox.settings.timeout = 600
    ox.settings.use_cache = True
    ox.settings.log_console = True
    # The default area limit subdivides Florida into many requests. For this
    # narrow power=plant tag query, a larger query area avoids excessive
    # Overpass throttling while keeping the request size manageable.
    ox.settings.max_query_area_size = 500_000_000_000

    osm = ox.features_from_polygon(florida_geometry, tags={"power": "plant"})
    osm = osm.reset_index()
    osm = osm[osm.geometry.notna() & ~osm.geometry.is_empty].copy()
    osm = osm[osm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    osm = osm.to_crs("EPSG:4326")

    element_column = "element_type" if "element_type" in osm.columns else "element"
    osmid_column = "osmid" if "osmid" in osm.columns else "id"
    if element_column not in osm.columns:
        osm[element_column] = "unknown"
    if osmid_column not in osm.columns:
        osm[osmid_column] = osm.index.astype(str)
    osm["polygon_osm_id"] = (
        osm[element_column].astype(str) + "/" + osm[osmid_column].astype(str)
    )

    osm_area = osm.to_crs(PROJECTED_CRS)
    osm["polygon_area_m2"] = osm_area.geometry.area

    osm.to_file(
        OSM_POLYGONS_GPKG,
        layer="florida_osm_powerplant_polygons",
        driver="GPKG",
    )
    return osm


def load_cleaned_points() -> gpd.GeoDataFrame:
    if not CLEANED_POINTS_FILE.exists():
        raise FileNotFoundError(f"Missing cleaned point file: {CLEANED_POINTS_FILE}")
    points = gpd.read_file(CLEANED_POINTS_FILE).to_crs("EPSG:4326")
    points = points.reset_index(drop=True)
    points["point_id"] = points.index
    return points


def polygon_name_column(polygons: gpd.GeoDataFrame) -> str | None:
    for column in ["name", "plant:name", "operator", "owner"]:
        if column in polygons.columns:
            return column
    return None


def choose_best_containment_matches(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Find point-in-polygon matches and choose one polygon per point."""
    name_col = polygon_name_column(polygons)
    poly_cols = ["polygon_osm_id", "polygon_area_m2", "geometry"]
    if name_col:
        poly_cols.append(name_col)

    joined = gpd.sjoin(
        points,
        polygons[poly_cols],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return pd.DataFrame()

    records = []
    for _, row in joined.iterrows():
        polygon_name = row.get(name_col, np.nan) if name_col else np.nan
        point_name = row.get("name", np.nan)
        records.append(
            {
                "point_id": int(row.point_id),
                "polygon_osm_id": row.get("polygon_osm_id"),
                "polygon_name": polygon_name,
                "polygon_area_m2": row.get("polygon_area_m2"),
                "polygon_match_method": "contains",
                "polygon_match_distance_m": 0.0,
                "name_similarity": name_similarity(point_name, polygon_name),
            }
        )

    candidates = pd.DataFrame(records)
    candidates["sort_score"] = candidates["name_similarity"] + (
        candidates["polygon_area_m2"].rank(pct=True) * 0.05
    )
    best = (
        candidates.sort_values(["point_id", "sort_score"], ascending=[True, False])
        .drop_duplicates("point_id")
        .drop(columns="sort_score")
    )
    return best


def choose_nearest_matches_for_unmatched(
    unmatched_points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Nearest-polygon fallback for points without containment matches."""
    if unmatched_points.empty or polygons.empty:
        return pd.DataFrame()

    name_col = polygon_name_column(polygons)
    poly_cols = ["polygon_osm_id", "polygon_area_m2", "geometry"]
    if name_col:
        poly_cols.append(name_col)

    points_m = unmatched_points.to_crs(PROJECTED_CRS)
    polygons_m = polygons[poly_cols].to_crs(PROJECTED_CRS)

    nearest = gpd.sjoin_nearest(
        points_m,
        polygons_m,
        how="left",
        max_distance=MAX_NEAREST_DISTANCE_M,
        distance_col="polygon_match_distance_m",
    )
    nearest = nearest.dropna(subset=["polygon_osm_id"]).copy()
    if nearest.empty:
        return pd.DataFrame()

    records = []
    for _, row in nearest.iterrows():
        polygon_name = row.get(name_col, np.nan) if name_col else np.nan
        point_name = row.get("name", np.nan)
        records.append(
            {
                "point_id": int(row.point_id),
                "polygon_osm_id": row.get("polygon_osm_id"),
                "polygon_name": polygon_name,
                "polygon_area_m2": row.get("polygon_area_m2"),
                "polygon_match_method": "nearest",
                "polygon_match_distance_m": float(row.get("polygon_match_distance_m")),
                "name_similarity": name_similarity(point_name, polygon_name),
            }
        )
    return pd.DataFrame(records)


def attach_matches_to_points(
    points: gpd.GeoDataFrame,
    matches: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Attach polygon match fields to cleaned point records."""
    out = points.merge(matches, on="point_id", how="left")
    out["polygon_match_status"] = np.where(
        out["polygon_osm_id"].notna(),
        "matched",
        "unmatched",
    )
    out["polygon_match_method"] = out["polygon_match_method"].fillna("none")
    out["location_confidence"] = out.get("location_confidence", "unknown")
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


def write_review_csv(matches: pd.DataFrame, points_with_matches: gpd.GeoDataFrame) -> None:
    review_cols = [
        "point_id",
        "name",
        "capacity_mw",
        "primary_fuel",
        "matched_source",
        "location_source",
        "location_confidence",
        "polygon_match_status",
        "polygon_match_method",
        "polygon_match_distance_m",
        "polygon_osm_id",
        "polygon_name",
        "polygon_area_m2",
        "name_similarity",
    ]
    review = points_with_matches[[c for c in review_cols if c in points_with_matches.columns]].copy()
    review.to_csv(MATCH_REVIEW_CSV, index=False)


def print_summary(polygons: gpd.GeoDataFrame, points: gpd.GeoDataFrame, matched: gpd.GeoDataFrame) -> None:
    contains_count = int((matched["polygon_match_method"] == "contains").sum())
    nearest_count = int((matched["polygon_match_method"] == "nearest").sum())
    unmatched_count = int((matched["polygon_match_status"] == "unmatched").sum())

    print("\nSummary statistics")
    print("OSM plant polygons downloaded/loaded:", len(polygons))
    print("Cleaned plant points loaded:", len(points))
    print("Points matched by containment:", contains_count)
    print("Points matched by nearest polygon:", nearest_count)
    print("Unmatched points:", unmatched_count)
    print("Average polygon area (m2):", round(polygons["polygon_area_m2"].mean(), 1))

    largest = polygons.sort_values("polygon_area_m2", ascending=False).head(10).copy()
    name_col = polygon_name_column(largest)
    display_cols = ["polygon_osm_id", "polygon_area_m2"]
    if name_col:
        display_cols.insert(1, name_col)
    print("\nLargest OSM plant polygons by area:")
    print(largest[display_cols].to_string(index=False))


def main() -> None:
    ELECTRICITY_DIR.mkdir(parents=True, exist_ok=True)

    polygons = download_or_load_osm_powerplant_polygons()
    points = load_cleaned_points()

    containment_matches = choose_best_containment_matches(points, polygons)
    contained_point_ids = (
        set(containment_matches["point_id"]) if not containment_matches.empty else set()
    )
    unmatched_points = points[~points["point_id"].isin(contained_point_ids)].copy()
    nearest_matches = choose_nearest_matches_for_unmatched(unmatched_points, polygons)

    matches = pd.concat([containment_matches, nearest_matches], ignore_index=True)
    points_with_matches = attach_matches_to_points(points, matches)

    polygons.to_file(
        OSM_POLYGONS_GPKG,
        layer="florida_osm_powerplant_polygons",
        driver="GPKG",
    )
    points_with_matches.to_file(
        POINTS_WITH_POLYGONS_GPKG,
        layer="florida_powerplants_cleaned_with_polygons",
        driver="GPKG",
    )
    write_review_csv(matches, points_with_matches)

    print_summary(polygons, points, points_with_matches)

    print("\nSaved OSM plant polygons:", OSM_POLYGONS_GPKG)
    print("Saved cleaned points with polygon matches:", POINTS_WITH_POLYGONS_GPKG)
    print("Saved polygon match review CSV:", MATCH_REVIEW_CSV)


if __name__ == "__main__":
    main()
