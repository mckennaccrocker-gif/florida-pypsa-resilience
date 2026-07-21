"""
Extract OSM substation footprint polygons and match them to the reviewed grid.

The newest Florida grid keeps HIFLD-derived network buses/lines, but OSM can add
real substation footprint polygons for hazard exposure. This script reads a
cached Overpass response for `power=substation`, converts closed OSM ways to
polygons, computes polygon area, and links polygons to the nearest/containing
reviewed-grid buses.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point, Polygon


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
REVIEWED_NETWORK = ELECTRICITY_DIR / "pypsa_florida_network_island_reviewed_no_connectors"
RAW_OVERPASS = ELECTRICITY_DIR / "osm_florida_substations_overpass_raw.json"
OUTPUT_GPKG = REVIEWED_NETWORK / "qgis" / "florida_osm_substation_polygons.gpkg"
OUTPUT_MATCHES = REVIEWED_NETWORK / "osm_substation_polygon_bus_matches.csv"
OUTPUT_SUMMARY = REVIEWED_NETWORK / "osm_substation_polygon_summary.csv"
PROJECTED_CRS = "EPSG:3086"
MATCH_DISTANCE_M = 500.0


OVERPASS_QUERY = """[out:json][timeout:180];
(
  way["power"="substation"](24.396,-87.635,31.001,-79.974);
  relation["power"="substation"](24.396,-87.635,31.001,-79.974);
);
out tags geom;
"""


def download_overpass_if_needed() -> None:
    if RAW_OVERPASS.exists():
        return
    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=OVERPASS_QUERY.encode("utf-8"),
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "User-Agent": "oxford-tc-project/1.0",
        },
        timeout=240,
    )
    response.raise_for_status()
    RAW_OVERPASS.write_text(response.text, encoding="utf-8")


def tags_value(tags: dict, key: str) -> str:
    value = tags.get(key, "")
    if value is None:
        return ""
    return str(value)


def osm_substation_polygons() -> gpd.GeoDataFrame:
    download_overpass_if_needed()
    data = json.loads(RAW_OVERPASS.read_text(encoding="utf-8"))
    rows = []
    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue
        coords = [(node["lon"], node["lat"]) for node in element.get("geometry", [])]
        if len(coords) < 4 or coords[0] != coords[-1]:
            continue
        polygon = Polygon(coords)
        if polygon.is_empty or not polygon.is_valid or polygon.area == 0:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        tags = element.get("tags", {})
        rows.append(
            {
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
                "osm_name": tags_value(tags, "name"),
                "osm_operator": tags_value(tags, "operator"),
                "osm_substation": tags_value(tags, "substation"),
                "osm_voltage": tags_value(tags, "voltage"),
                "osm_location": tags_value(tags, "location"),
                "geometry": polygon,
            }
        )
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    projected = gdf.to_crs(PROJECTED_CRS)
    gdf["area_m2"] = projected.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000.0
    gdf["centroid_lon"] = projected.centroid.to_crs("EPSG:4326").x
    gdf["centroid_lat"] = projected.centroid.to_crs("EPSG:4326").y
    return gdf


def reviewed_buses() -> gpd.GeoDataFrame:
    buses = pd.read_csv(REVIEWED_NETWORK / "buses.csv")
    buses["name"] = buses["name"].astype(str)
    buses["node_id"] = buses["name"].str.replace("bus_", "", regex=False)
    return gpd.GeoDataFrame(
        buses,
        geometry=[Point(xy) for xy in zip(buses["x"], buses["y"])],
        crs="EPSG:4326",
    )


def match_buses_to_polygons(
    buses: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    buses_proj = buses.to_crs(PROJECTED_CRS)
    polygons_proj = polygons.to_crs(PROJECTED_CRS)

    inside = gpd.sjoin(
        buses_proj[["name", "node_id", "substation_name", "v_nom", "geometry"]],
        polygons_proj[["osm_id", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    inside["match_method"] = "within_polygon"
    inside["match_distance_m"] = 0.0

    unmatched = inside[inside["osm_id"].isna()][["name", "node_id", "substation_name", "v_nom", "geometry"]]
    nearest = gpd.sjoin_nearest(
        unmatched,
        polygons_proj[["osm_id", "geometry"]],
        how="left",
        max_distance=MATCH_DISTANCE_M,
        distance_col="match_distance_m",
    ).drop(columns=["index_right"], errors="ignore")
    nearest["match_method"] = "nearest_within_500m"

    matched = pd.concat(
        [
            inside[inside["osm_id"].notna()],
            nearest[nearest["osm_id"].notna()],
        ],
        ignore_index=True,
    )
    matched["osm_id"] = matched["osm_id"].astype("int64")
    matched = matched.sort_values(["name", "match_distance_m"]).drop_duplicates("name")

    polygon_attrs = polygons.drop(columns="geometry").copy()
    matched = matched.drop(columns="geometry").merge(polygon_attrs, on="osm_id", how="left")

    bus_groups = (
        matched.groupby("osm_id")
        .agg(
            matched_bus_count=("name", "count"),
            matched_buses=("name", lambda values: ";".join(sorted(map(str, values)))),
            matched_node_ids=("node_id", lambda values: ";".join(sorted(map(str, values)))),
            matched_bus_substation_names=("substation_name", lambda values: ";".join(sorted(set(map(str, values))))),
            min_bus_distance_m=("match_distance_m", "min"),
        )
        .reset_index()
    )
    polygons_out = polygons.merge(bus_groups, on="osm_id", how="left")
    polygons_out["matched_bus_count"] = polygons_out["matched_bus_count"].fillna(0).astype(int)
    for column in ["matched_buses", "matched_node_ids", "matched_bus_substation_names"]:
        polygons_out[column] = polygons_out[column].fillna("")
    polygons_out["min_bus_distance_m"] = polygons_out["min_bus_distance_m"].fillna(pd.NA)
    return polygons_out, matched


def main() -> None:
    polygons = osm_substation_polygons()
    buses = reviewed_buses()
    polygons_out, matches = match_buses_to_polygons(buses, polygons)

    OUTPUT_GPKG.parent.mkdir(exist_ok=True)
    for path in [OUTPUT_GPKG, OUTPUT_MATCHES, OUTPUT_SUMMARY]:
        path.unlink(missing_ok=True)

    polygons_out.to_file(OUTPUT_GPKG, layer="osm_substation_polygons", driver="GPKG")
    matches.to_csv(OUTPUT_MATCHES, index=False)

    summary = pd.DataFrame(
        [
            {
                "osm_substation_polygons": len(polygons_out),
                "polygons_matched_to_reviewed_buses": int((polygons_out["matched_bus_count"] > 0).sum()),
                "reviewed_buses": len(buses),
                "reviewed_buses_matched_to_osm_polygon": matches["name"].nunique(),
                "match_distance_m": MATCH_DISTANCE_M,
                "output_gpkg": str(OUTPUT_GPKG.resolve()),
                "output_matches": str(OUTPUT_MATCHES.resolve()),
            }
        ]
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
