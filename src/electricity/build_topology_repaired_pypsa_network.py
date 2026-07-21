"""
Build a topology-repaired copy of the Florida PyPSA CSV network.

The original PyPSA network uses endpoint-derived HIFLD topology. That is a
reasonable first pass, but it misses cases where two same-voltage line
geometries cross/touch without a generated node. This script creates a
diagnostic repaired network by:

1. Finding same-voltage line intersections away from existing endpoints.
2. Adding synthetic PyPSA buses at those crossing points.
3. Splitting affected PyPSA lines into same-capacity series segments.
4. Copying all other PyPSA CSV inputs unchanged into a new output folder.

This is intentionally non-destructive. It does not overwrite the calibrated
baseline network; it creates a separate folder for testing whether better
topology reduces the need for a global s_nom multiplier.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
BASE_NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network"
EDGE_GEOMETRY_FILE = ELECTRICITY_DIR / "florida_network_edges_transmission_lines.gpkg"
DEFAULT_OUTPUT_DIR = ELECTRICITY_DIR / "pypsa_florida_network_topology_repaired"
PROJECTED_CRS = "EPSG:3086"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create topology-repaired PyPSA network CSVs.")
    parser.add_argument("--base-network-dir", type=Path, default=BASE_NETWORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--edge-geometry-file", type=Path, default=EDGE_GEOMETRY_FILE)
    parser.add_argument(
        "--terminal-tolerance-m",
        type=float,
        default=50.0,
        help="Ignore intersections this close to either line endpoint.",
    )
    parser.add_argument(
        "--cluster-tolerance-m",
        type=float,
        default=5.0,
        help="Cluster crossing points this close into one synthetic bus.",
    )
    parser.add_argument(
        "--same-voltage-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only connect lines with the same v_nom. This avoids inventing transformers.",
    )
    return parser.parse_args()


def copy_network_folder(base_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in base_dir.glob("*.csv"):
        shutil.copy2(path, output_dir / path.name)


def load_line_geometries(network_dir: Path, edge_geometry_file: Path) -> gpd.GeoDataFrame:
    lines = pd.read_csv(network_dir / "lines.csv")
    if "name" not in lines.columns:
        raise ValueError("lines.csv must contain a name column.")
    edges = gpd.read_file(edge_geometry_file)[["edge_id", "geometry"]].to_crs("EPSG:4326")
    lines["source_edge_id"] = pd.to_numeric(lines["source_edge_id"], errors="coerce").astype("Int64")
    merged = lines.merge(
        edges,
        left_on="source_edge_id",
        right_on="edge_id",
        how="left",
        validate="many_to_one",
    )
    missing = int(merged["geometry"].isna().sum())
    if missing:
        raise ValueError(f"{missing} PyPSA lines could not be matched to source edge geometry.")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326").to_crs(PROJECTED_CRS)


def load_buses(network_dir: Path) -> gpd.GeoDataFrame:
    buses = pd.read_csv(network_dir / "buses.csv")
    return gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)


def iter_intersection_points(geometry) -> list[Point]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        points: list[Point] = []
        for line in geometry.geoms:
            coords = list(line.coords)
            points.extend([Point(coords[0]), Point(coords[-1])])
        return points
    if geometry.geom_type == "MultiPoint":
        return list(geometry.geoms)
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return [Point(coords[0]), Point(coords[-1])]
    if geometry.geom_type == "GeometryCollection":
        points = []
        for part in geometry.geoms:
            points.extend(iter_intersection_points(part))
        return points
    return []


def away_from_line_terminals(line: LineString, point: Point, tolerance_m: float) -> bool:
    length = float(line.length)
    distance = float(line.project(point))
    return tolerance_m < distance < (length - tolerance_m)


def cluster_point(
    point: Point,
    clusters: list[dict],
    cluster_tolerance_m: float,
) -> int:
    for idx, cluster in enumerate(clusters):
        if point.distance(cluster["point"]) <= cluster_tolerance_m:
            cluster["points"].append(point)
            xs = [p.x for p in cluster["points"]]
            ys = [p.y for p in cluster["points"]]
            cluster["point"] = Point(float(np.mean(xs)), float(np.mean(ys)))
            return idx
    clusters.append({"point": point, "points": [point], "line_names": set(), "v_nom": np.nan})
    return len(clusters) - 1


def find_same_voltage_crossings(
    lines_gdf: gpd.GeoDataFrame,
    terminal_tolerance_m: float,
    cluster_tolerance_m: float,
    same_voltage_only: bool,
) -> tuple[pd.DataFrame, dict[str, list[int]], list[dict]]:
    spatial_index = lines_gdf.sindex
    clusters: list[dict] = []
    line_to_clusters: dict[str, set[int]] = defaultdict(set)
    pair_records = []

    geoms = lines_gdf.geometry.reset_index(drop=True)
    names = lines_gdf["name"].reset_index(drop=True)
    voltages = pd.to_numeric(lines_gdf["v_nom"], errors="coerce").reset_index(drop=True)

    for left_idx, left_geom in enumerate(geoms):
        candidate_idx = spatial_index.query(left_geom, predicate="intersects")
        for right_idx in candidate_idx:
            right_idx = int(right_idx)
            if right_idx <= left_idx:
                continue
            if same_voltage_only and not math.isclose(
                float(voltages.iloc[left_idx]),
                float(voltages.iloc[right_idx]),
                rel_tol=0,
                abs_tol=1e-6,
            ):
                continue

            right_geom = geoms.iloc[right_idx]
            intersection = left_geom.intersection(right_geom)
            points = iter_intersection_points(intersection)
            for point in points:
                if not away_from_line_terminals(left_geom, point, terminal_tolerance_m):
                    continue
                if not away_from_line_terminals(right_geom, point, terminal_tolerance_m):
                    continue

                cluster_id = cluster_point(point, clusters, cluster_tolerance_m)
                left_name = str(names.iloc[left_idx])
                right_name = str(names.iloc[right_idx])
                clusters[cluster_id]["line_names"].update([left_name, right_name])
                clusters[cluster_id]["v_nom"] = float(voltages.iloc[left_idx])
                line_to_clusters[left_name].add(cluster_id)
                line_to_clusters[right_name].add(cluster_id)
                pair_records.append(
                    {
                        "crossing_cluster_id": cluster_id,
                        "line_a": left_name,
                        "line_b": right_name,
                        "v_nom": float(voltages.iloc[left_idx]),
                        "x_m": point.x,
                        "y_m": point.y,
                    }
                )

    return (
        pd.DataFrame(pair_records),
        {line: sorted(ids) for line, ids in line_to_clusters.items()},
        clusters,
    )


def interpolate_bus_row(bus_template: pd.Series, name: str, point: Point, v_nom: float) -> dict:
    row = bus_template.to_dict()
    row["name"] = name
    row["v_nom"] = v_nom
    row["carrier"] = row.get("carrier", "AC") or "AC"
    row["substation_name"] = "synthetic_topology_crossing"
    row["source_node_id"] = -1
    row["is_step_up_down"] = False
    row["connected_edge_count"] = 0
    point_4326 = gpd.GeoSeries([point], crs=PROJECTED_CRS).to_crs("EPSG:4326").iloc[0]
    row["x"] = point_4326.x
    row["y"] = point_4326.y
    return row


def scaled_line_row(
    original: pd.Series,
    name: str,
    bus0: str,
    bus1: str,
    segment: LineString,
    original_length_m: float,
    segment_index: int,
) -> dict:
    row = original.drop(labels=["geometry"], errors="ignore").to_dict()
    length_km = float(segment.length) / 1000.0
    original_length_km = max(float(original_length_m) / 1000.0, 1e-9)
    ratio = length_km / original_length_km

    row["name"] = name
    row["bus0"] = bus0
    row["bus1"] = bus1
    row["length"] = length_km
    row["topology_repair_source_line"] = original["name"]
    row["topology_repair_segment_index"] = segment_index

    for col in ["r", "x", "b", "g"]:
        if col in row and pd.notna(row[col]):
            row[col] = float(row[col]) * ratio
    return row


def split_lines_at_crossings(
    buses: pd.DataFrame,
    lines_gdf: gpd.GeoDataFrame,
    line_to_clusters: dict[str, list[int]],
    clusters: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bus_template = buses.iloc[0]
    synthetic_bus_rows = []
    cluster_to_bus = {}
    for cluster_id, cluster in enumerate(clusters):
        if not cluster["line_names"]:
            continue
        bus_name = f"bus_toporepair_crossing_{cluster_id}"
        cluster_to_bus[cluster_id] = bus_name
        synthetic_bus_rows.append(
            interpolate_bus_row(bus_template, bus_name, cluster["point"], float(cluster["v_nom"]))
        )

    line_rows = []
    split_records = []
    split_line_names = set(line_to_clusters)

    for _, line in lines_gdf.iterrows():
        original_name = str(line["name"])
        geom: LineString = line.geometry
        original_length_m = float(geom.length)
        cluster_ids = line_to_clusters.get(original_name, [])

        if not cluster_ids:
            row = line.drop(labels=["geometry"], errors="ignore").to_dict()
            row["topology_repair_source_line"] = original_name
            row["topology_repair_segment_index"] = 0
            line_rows.append(row)
            continue

        points = []
        for cluster_id in cluster_ids:
            point = clusters[cluster_id]["point"]
            dist = float(geom.project(point))
            if 0.0 < dist < original_length_m:
                points.append((dist, cluster_id, point))
        points = sorted(points, key=lambda item: item[0])
        nodes = [(0.0, str(line["bus0"]), Point(geom.coords[0]))]
        nodes.extend((dist, cluster_to_bus[cluster_id], point) for dist, cluster_id, point in points)
        nodes.append((original_length_m, str(line["bus1"]), Point(geom.coords[-1])))

        made_segments = 0
        for segment_index, (left, right) in enumerate(zip(nodes[:-1], nodes[1:])):
            left_dist, left_bus, _left_point = left
            right_dist, right_bus, _right_point = right
            if right_dist - left_dist <= 1.0:
                continue
            segment = substring(geom, left_dist, right_dist)
            if segment.is_empty or segment.length <= 1.0:
                continue
            line_rows.append(
                scaled_line_row(
                    line,
                    f"{original_name}_seg{segment_index}",
                    left_bus,
                    right_bus,
                    segment,
                    original_length_m,
                    segment_index,
                )
            )
            made_segments += 1

        split_records.append(
            {
                "source_line": original_name,
                "crossing_count": len(points),
                "segments_created": made_segments,
                "source_edge_id": line.get("source_edge_id"),
                "v_nom": line.get("v_nom"),
            }
        )

    repaired_buses = pd.concat([buses, pd.DataFrame(synthetic_bus_rows)], ignore_index=True)
    repaired_lines = pd.DataFrame(line_rows)
    repaired_lines = repaired_lines.drop(columns=["edge_id"], errors="ignore")

    return repaired_buses, repaired_lines, pd.DataFrame(split_records)


def component_summary(buses: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    graph = nx.Graph()
    graph.add_nodes_from(buses["name"].astype(str))
    graph.add_edges_from(zip(lines["bus0"].astype(str), lines["bus1"].astype(str)))
    rows = []
    for component_id, component in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True)):
        rows.append({"component_id": component_id, "bus_count": len(component)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    copy_network_folder(args.base_network_dir, args.output_dir)

    buses = pd.read_csv(args.base_network_dir / "buses.csv")
    lines_gdf = load_line_geometries(args.base_network_dir, args.edge_geometry_file)

    before_components = component_summary(buses, lines_gdf.drop(columns="geometry"))
    before_components.to_csv(args.output_dir / "topology_components_before_repair.csv", index=False)

    crossing_pairs, line_to_clusters, clusters = find_same_voltage_crossings(
        lines_gdf,
        terminal_tolerance_m=args.terminal_tolerance_m,
        cluster_tolerance_m=args.cluster_tolerance_m,
        same_voltage_only=args.same_voltage_only,
    )

    repaired_buses, repaired_lines, split_summary = split_lines_at_crossings(
        buses,
        lines_gdf,
        line_to_clusters,
        clusters,
    )

    repaired_buses.to_csv(args.output_dir / "buses.csv", index=False)
    repaired_lines.to_csv(args.output_dir / "lines.csv", index=False)
    crossing_pairs.to_csv(args.output_dir / "topology_repair_crossing_pairs.csv", index=False)
    split_summary.to_csv(args.output_dir / "topology_repair_split_lines.csv", index=False)
    after_components = component_summary(repaired_buses, repaired_lines)
    after_components.to_csv(args.output_dir / "topology_components_after_repair.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "base_buses": len(buses),
                "repaired_buses": len(repaired_buses),
                "synthetic_crossing_buses_added": len(repaired_buses) - len(buses),
                "base_lines": len(lines_gdf),
                "repaired_line_segments": len(repaired_lines),
                "lines_split": int(split_summary["source_line"].nunique()) if not split_summary.empty else 0,
                "crossing_pairs": len(crossing_pairs),
                "components_before": len(before_components),
                "components_after": len(after_components),
                "largest_component_before": int(before_components["bus_count"].max()),
                "largest_component_after": int(after_components["bus_count"].max()),
                "same_voltage_only": args.same_voltage_only,
                "terminal_tolerance_m": args.terminal_tolerance_m,
                "cluster_tolerance_m": args.cluster_tolerance_m,
            }
        ]
    )
    summary.to_csv(args.output_dir / "topology_repair_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("Saved topology-repaired network:", args.output_dir)


if __name__ == "__main__":
    main()
