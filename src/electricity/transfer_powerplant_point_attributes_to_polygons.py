"""
Transfer cleaned Florida power plant point attributes onto matched OSM polygons.

Inputs:
  - data/Electricity/florida_osm_powerplant_polygons.gpkg
  - data/Electricity/florida_powerplants_cleaned_with_polygons.gpkg

Output:
  - data/Electricity/florida_osm_powerplant_polygons_with_point_attributes.gpkg
  - data/Electricity/florida_osm_powerplant_polygons_with_point_attributes.csv

Only polygons matched to at least one cleaned point are included in the final layer.
If multiple cleaned points match the same polygon, capacity is summed and text
attributes are joined into semicolon-separated lists.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"

POLYGONS_FILE = ELECTRICITY_DIR / "florida_osm_powerplant_polygons.gpkg"
POINTS_WITH_POLYGONS_FILE = (
    ELECTRICITY_DIR / "florida_powerplants_cleaned_with_polygons.gpkg"
)

OUTPUT_GPKG = (
    ELECTRICITY_DIR / "florida_osm_powerplant_polygons_with_point_attributes.gpkg"
)
OUTPUT_CSV = (
    ELECTRICITY_DIR / "florida_osm_powerplant_polygons_with_point_attributes.csv"
)


def unique_join(values) -> str:
    """Join unique non-empty values as a readable string."""
    clean_values = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            clean_values.append(text)
    return "; ".join(sorted(set(clean_values)))


def first_nonempty(values):
    """Return the first non-empty value in a series-like object."""
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            return value
    return pd.NA


def aggregate_point_attributes(points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate matched point attributes to one row per polygon_osm_id."""
    matched = points[
        (points["polygon_match_status"] == "matched")
        & points["polygon_osm_id"].notna()
    ].copy()

    if matched.empty:
        raise ValueError("No matched point records found.")

    matched["capacity_mw"] = pd.to_numeric(matched["capacity_mw"], errors="coerce")

    grouped = (
        matched.groupby("polygon_osm_id", dropna=False)
        .agg(
            matched_point_count=("point_id", "count"),
            point_names=("name", unique_join),
            point_primary_fuels=("primary_fuel", unique_join),
            capacity_mw=("capacity_mw", "sum"),
            capacity_mw_max=("capacity_mw", "max"),
            gppd_ids=("gppd_idnr", unique_join),
            osm_project_ids=("osm_projectID", unique_join),
            matched_sources=("matched_source", unique_join),
            location_sources=("location_source", unique_join),
            attribute_sources=("attribute_source", unique_join),
            location_confidences=("location_confidence", unique_join),
            polygon_match_methods=("polygon_match_method", unique_join),
            mean_polygon_match_distance_m=("polygon_match_distance_m", "mean"),
            max_polygon_match_distance_m=("polygon_match_distance_m", "max"),
            mean_name_similarity=("name_similarity", "mean"),
            representative_point_name=("name", first_nonempty),
            representative_primary_fuel=("primary_fuel", first_nonempty),
        )
        .reset_index()
    )

    grouped["capacity_source_note"] = (
        "capacity_mw is summed from matched cleaned point records; "
        "capacity_mw_max is the largest single matched point capacity."
    )
    return grouped


def main() -> None:
    ELECTRICITY_DIR.mkdir(parents=True, exist_ok=True)

    if not POLYGONS_FILE.exists():
        raise FileNotFoundError(f"Missing polygon file: {POLYGONS_FILE}")
    if not POINTS_WITH_POLYGONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing cleaned point match file: {POINTS_WITH_POLYGONS_FILE}"
        )

    polygons = gpd.read_file(POLYGONS_FILE).to_crs("EPSG:4326")
    points = gpd.read_file(POINTS_WITH_POLYGONS_FILE).to_crs("EPSG:4326")

    point_attrs = aggregate_point_attributes(points)

    enriched = polygons.merge(point_attrs, on="polygon_osm_id", how="inner")
    enriched = gpd.GeoDataFrame(enriched, geometry="geometry", crs="EPSG:4326")

    # Friendly final columns. OSM polygon operator/source are kept as owner/source
    # where the cleaned point file does not have owner/source fields.
    enriched["name"] = enriched["representative_point_name"].fillna(
        enriched.get("name")
    )
    enriched["primary_fuel"] = enriched["representative_primary_fuel"].fillna(
        enriched["point_primary_fuels"]
    )
    enriched["owner"] = enriched.get("operator", pd.NA)
    enriched["source"] = enriched.get("source", pd.NA)
    enriched["polygon_attribute_status"] = "matched point attributes transferred"

    preferred_columns = [
        "polygon_osm_id",
        "name",
        "capacity_mw",
        "capacity_mw_max",
        "primary_fuel",
        "owner",
        "source",
        "polygon_area_m2",
        "matched_point_count",
        "point_names",
        "point_primary_fuels",
        "gppd_ids",
        "osm_project_ids",
        "matched_sources",
        "location_sources",
        "attribute_sources",
        "location_confidences",
        "polygon_match_methods",
        "mean_polygon_match_distance_m",
        "max_polygon_match_distance_m",
        "mean_name_similarity",
        "capacity_source_note",
        "polygon_attribute_status",
        "geometry",
    ]
    keep_columns = [column for column in preferred_columns if column in enriched.columns]
    enriched = enriched[keep_columns].copy()

    enriched.to_file(
        OUTPUT_GPKG,
        layer="florida_osm_powerplant_polygons_with_point_attributes",
        driver="GPKG",
    )
    enriched.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print("OSM polygons loaded:", len(polygons))
    print("Cleaned point records loaded:", len(points))
    print("Matched polygons with transferred point attributes:", len(enriched))
    print("\nCounts by primary fuel:")
    print(enriched["primary_fuel"].fillna("Unknown").value_counts().to_string())
    print("\nTotal transferred capacity by primary fuel (MW):")
    print(
        enriched.groupby("primary_fuel", dropna=False)["capacity_mw"]
        .sum()
        .sort_values(ascending=False)
        .round(2)
        .to_string()
    )
    print("\nSaved polygon output:", OUTPUT_GPKG)
    print("Saved CSV output:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
