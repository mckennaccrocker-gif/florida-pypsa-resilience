"""
Build a Florida transmission-line network and inferred substation nodes.

Input:
  - data/Electricity/TransmissionLines2.gpkg, if present
  - otherwise data/Electricty/TransmissionLines2.gpkg, matching the existing project folder

Outputs:
  - data/Electricity/florida_transmission_lines.gpkg
  - data/Electricity/florida_network_nodes_substations.gpkg
  - data/Electricity/florida_network_edges_transmission_lines.gpkg
  - data/Electricity/florida_substation_voltage_summary.csv

The generated nodes are inferred from transmission-line endpoints. Nearby duplicate
endpoints are snapped into a single node using SNAP_TOLERANCE_M in a projected CRS.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import LineString, MultiLineString, Point


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
LEGACY_ELECTRICTY_DIR = PROJECT_DIR / "data" / "Electricty"
BOUNDARY_DIR = PROJECT_DIR / "data" / "Boundaries"

REQUESTED_LINES_FILE = ELECTRICITY_DIR / "TransmissionLines2.gpkg"
LEGACY_LINES_FILE = LEGACY_ELECTRICTY_DIR / "TransmissionLines2.gpkg"

FLORIDA_LINES_OUTPUT = ELECTRICITY_DIR / "florida_transmission_lines.gpkg"
NODES_OUTPUT = ELECTRICITY_DIR / "florida_network_nodes_substations.gpkg"
EDGES_OUTPUT = ELECTRICITY_DIR / "florida_network_edges_transmission_lines.gpkg"
VOLTAGE_SUMMARY_OUTPUT = ELECTRICITY_DIR / "florida_substation_voltage_summary.csv"

CENSUS_STATES_ZIP_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
)
CENSUS_STATES_ZIP = BOUNDARY_DIR / "tl_2023_us_state.zip"

STORAGE_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3086"  # NAD83 / Florida GDL Albers, meters
SNAP_TOLERANCE_M = 100

EDGE_FIELDS = [
    "OBJECTID_1",
    "OBJECTID",
    "ID",
    "TYPE",
    "STATUS",
    "OWNER",
    "VOLTAGE",
    "VOLT_CLASS",
    "SUB_1",
    "SUB_2",
    "SOURCE",
    "SOURCEDATE",
    "VAL_METHOD",
    "VAL_DATE",
    "INFERRED",
    "Shape__Len",
]

MISSING_TEXT = {
    "",
    "nan",
    "none",
    "not available",
    "unknown",
    "-999999",
}


class UnionFind:
    """Small union-find helper for endpoint snapping clusters."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def clean_text(value) -> str | None:
    """Return a cleaned text value, or None for missing/placeholder values."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in MISSING_TEXT:
        return None
    return text


def clean_substation_name(value) -> str | None:
    """Clean SUB_1/SUB_2 names while preserving real names for review."""
    text = clean_text(value)
    if text is None:
        return None
    # HIFLD often stores placeholders like UNKNOWN120703. Treat them as unnamed.
    if re.fullmatch(r"UNKNOWN\d*", text.upper()):
        return None
    return text


def parse_voltage_values(voltage, volt_class=None) -> list[float]:
    """Extract numeric kV values from VOLTAGE and, if needed, VOLT_CLASS."""
    values: list[float] = []

    if not pd.isna(voltage):
        try:
            number = float(voltage)
            if number > 0 and number != -999999:
                values.append(number / 1000 if number >= 1000 else number)
        except (TypeError, ValueError):
            pass

    # Use VOLT_CLASS as a fallback if precise VOLTAGE is unavailable.
    if not values and clean_text(volt_class):
        text = str(volt_class).upper()
        if "UNDER" in text:
            numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
            if numbers:
                values.append(max(numbers))
        else:
            numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
            values.extend(number for number in numbers if number > 0)

    return sorted(set(values))


def select_transmission_layer(path: Path) -> str | None:
    """List GeoPackage layers and pick the likely transmission-line layer."""
    layers = pyogrio.list_layers(path)
    print("\nAvailable layers in", path)
    for layer_name, geometry_type in layers:
        print(f"  - {layer_name} ({geometry_type})")

    line_layers = [
        layer_name
        for layer_name, geometry_type in layers
        if "LineString" in str(geometry_type)
    ]
    if not line_layers:
        raise ValueError(f"No line layers found in {path}")

    for layer_name in line_layers:
        if "transmission" in layer_name.lower() or "electric" in layer_name.lower():
            return layer_name
    return line_layers[0]


def find_lines_file() -> Path:
    """Use the requested path if present, otherwise fall back to project legacy spelling."""
    if REQUESTED_LINES_FILE.exists():
        return REQUESTED_LINES_FILE
    if LEGACY_LINES_FILE.exists():
        print(
            "Requested file was not found, using existing project file:",
            LEGACY_LINES_FILE,
        )
        return LEGACY_LINES_FILE
    raise FileNotFoundError(
        f"Could not find {REQUESTED_LINES_FILE} or {LEGACY_LINES_FILE}"
    )


def load_transmission_lines() -> gpd.GeoDataFrame:
    """Load the transmission line layer and verify CRS/count."""
    path = find_lines_file()
    layer = select_transmission_layer(path)
    print("\nLoading layer:", layer)
    lines = gpd.read_file(path, layer=layer)
    if lines.crs is None:
        raise ValueError("Transmission line layer has no CRS.")

    print("Transmission lines loaded:", len(lines))
    print("Transmission line CRS:", lines.crs)
    return lines


def load_florida_boundary() -> gpd.GeoDataFrame:
    """Download/read Census state boundaries and return Florida in EPSG:4326."""
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    if not CENSUS_STATES_ZIP.exists():
        print("Downloading Census state boundaries:", CENSUS_STATES_ZIP_URL)
        urlretrieve(CENSUS_STATES_ZIP_URL, CENSUS_STATES_ZIP)

    states = gpd.read_file(CENSUS_STATES_ZIP).to_crs(STORAGE_CRS)
    if "STUSPS" in states.columns:
        florida = states[states["STUSPS"] == "FL"].copy()
    else:
        florida = states[states["NAME"].str.lower() == "florida"].copy()
    if florida.empty:
        raise ValueError("Florida boundary not found.")
    return florida


def filter_lines_to_florida(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip/filter lines to the Florida state polygon."""
    florida = load_florida_boundary()
    lines_4326 = lines.to_crs(STORAGE_CRS)

    # Bounding-box prefilter keeps the overlay faster.
    minx, miny, maxx, maxy = florida.total_bounds
    candidates = lines_4326.cx[minx:maxx, miny:maxy].copy()
    candidates = candidates[candidates.intersects(florida.geometry.iloc[0])].copy()

    florida_lines = gpd.overlay(
        candidates,
        florida[["geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    florida_lines = florida_lines.explode(index_parts=False, ignore_index=True)
    florida_lines = florida_lines[
        florida_lines.geometry.notna() & ~florida_lines.geometry.is_empty
    ].copy()
    florida_lines = florida_lines[
        florida_lines.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ].copy()

    # GeoPackage field names can become awkward after overlay; keep original fields.
    keep_fields = [field for field in EDGE_FIELDS if field in florida_lines.columns]
    florida_lines = florida_lines[keep_fields + ["geometry"]].copy()
    florida_lines["florida_line_id"] = np.arange(len(florida_lines))

    return florida_lines.to_crs(STORAGE_CRS)


def iter_linestring_parts(geometry):
    """Yield LineString pieces from LineString/MultiLineString geometries."""
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def prepare_edges(florida_lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode multiline geometries and keep one edge row per LineString."""
    rows = []
    for _, row in florida_lines.iterrows():
        for part_index, line in enumerate(iter_linestring_parts(row.geometry)):
            if line.is_empty or len(line.coords) < 2:
                continue
            new_row = row.drop(labels="geometry").to_dict()
            new_row["part_index"] = part_index
            new_row["edge_id"] = len(rows)
            new_row["geometry"] = line
            rows.append(new_row)

    edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=florida_lines.crs)
    edges_projected = edges.to_crs(PROJECTED_CRS)
    edges["length_m"] = edges_projected.geometry.length
    edges["voltage_values"] = [
        parse_voltage_values(row.get("VOLTAGE"), row.get("VOLT_CLASS"))
        for _, row in edges.iterrows()
    ]
    edges["voltage_clean"] = [
        values[0] if len(values) == 1 else np.nan
        for values in edges["voltage_values"]
    ]
    return edges


def build_endpoint_table(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create endpoint records for every edge start/end."""
    edges_projected = edges.to_crs(PROJECTED_CRS)
    records = []
    for _, row in edges_projected.iterrows():
        coords = list(row.geometry.coords)
        endpoint_specs = [
            ("from", coords[0], row.get("SUB_1")),
            ("to", coords[-1], row.get("SUB_2")),
        ]
        for endpoint_role, coord, sub_name in endpoint_specs:
            records.append(
                {
                    "endpoint_id": len(records),
                    "edge_id": int(row["edge_id"]),
                    "endpoint_role": endpoint_role,
                    "suggested_name": clean_substation_name(sub_name),
                    "voltage_values": row.get("voltage_values", []),
                    "geometry": Point(coord),
                }
            )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=PROJECTED_CRS)


def snap_endpoint_nodes(endpoints: gpd.GeoDataFrame) -> pd.Series:
    """Cluster endpoint points that fall within SNAP_TOLERANCE_M of one another."""
    if endpoints.empty:
        return pd.Series(dtype=int)

    union_find = UnionFind(len(endpoints))
    spatial_index = endpoints.sindex

    for idx, point in enumerate(endpoints.geometry):
        possible_matches = spatial_index.query(point.buffer(SNAP_TOLERANCE_M))
        for other_idx in possible_matches:
            if other_idx <= idx:
                continue
            if point.distance(endpoints.geometry.iloc[other_idx]) <= SNAP_TOLERANCE_M:
                union_find.union(idx, int(other_idx))

    roots = [union_find.find(i) for i in range(len(endpoints))]
    root_to_node_id = {root: node_id for node_id, root in enumerate(sorted(set(roots)))}
    return pd.Series([root_to_node_id[root] for root in roots], index=endpoints.index)


def build_nodes_and_edges(edges: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Infer substation nodes and assign from/to node IDs to edges."""
    endpoints = build_endpoint_table(edges)
    endpoints["node_id"] = snap_endpoint_nodes(endpoints)

    edge_from_to: dict[int, dict[str, int]] = defaultdict(dict)
    for row in endpoints.itertuples(index=False):
        edge_from_to[int(row.edge_id)][row.endpoint_role] = int(row.node_id)

    edges_out = edges.copy()
    edges_out["from_node_id"] = edges_out["edge_id"].map(
        lambda edge_id: edge_from_to[int(edge_id)].get("from")
    )
    edges_out["to_node_id"] = edges_out["edge_id"].map(
        lambda edge_id: edge_from_to[int(edge_id)].get("to")
    )

    node_records = []
    for node_id, group in endpoints.groupby("node_id"):
        names = [name for name in group["suggested_name"].dropna().tolist() if name]
        name_counts = Counter(names)
        substation_name = name_counts.most_common(1)[0][0] if name_counts else None
        all_names = "; ".join(sorted(set(names)))
        conflicting_names = len(set(names)) > 1

        voltages = sorted(
            {
                float(voltage)
                for values in group["voltage_values"]
                for voltage in (values if isinstance(values, list) else [])
                if pd.notna(voltage)
            }
        )
        connected_edges = sorted(set(group["edge_id"].astype(int).tolist()))
        centroid = group.unary_union.centroid

        node_records.append(
            {
                "node_id": int(node_id),
                "substation_name": substation_name,
                "substation_names_all": all_names,
                "conflicting_substation_names": conflicting_names,
                "voltages_connected": ";".join(str(int(v)) if v.is_integer() else str(v) for v in voltages),
                "min_voltage": min(voltages) if voltages else np.nan,
                "max_voltage": max(voltages) if voltages else np.nan,
                "voltage_count": len(voltages),
                "is_step_up_down": len(voltages) > 1,
                "connected_edge_count": len(connected_edges),
                "connected_edge_ids": ";".join(map(str, connected_edges)),
                "geometry": centroid,
            }
        )

    nodes = gpd.GeoDataFrame(node_records, geometry="geometry", crs=PROJECTED_CRS)
    return nodes.to_crs(STORAGE_CRS), edges_out.to_crs(STORAGE_CRS)


def save_outputs(
    florida_lines: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> None:
    """Save all requested GeoPackage and CSV outputs."""
    ELECTRICITY_DIR.mkdir(parents=True, exist_ok=True)

    florida_lines.to_file(
        FLORIDA_LINES_OUTPUT,
        layer="florida_transmission_lines",
        driver="GPKG",
    )
    nodes.to_file(
        NODES_OUTPUT,
        layer="florida_network_nodes_substations",
        driver="GPKG",
    )

    edge_keep = [
        "edge_id",
        "from_node_id",
        "to_node_id",
        "length_m",
        "voltage_clean",
        *[field for field in EDGE_FIELDS if field in edges.columns],
        "geometry",
    ]
    edges[edge_keep].to_file(
        EDGES_OUTPUT,
        layer="florida_network_edges_transmission_lines",
        driver="GPKG",
    )

    summary_cols = [
        "node_id",
        "substation_name",
        "substation_names_all",
        "conflicting_substation_names",
        "voltages_connected",
        "min_voltage",
        "max_voltage",
        "voltage_count",
        "is_step_up_down",
        "connected_edge_count",
        "connected_edge_ids",
    ]
    nodes[summary_cols].to_csv(VOLTAGE_SUMMARY_OUTPUT, index=False)


def print_summary(
    florida_lines: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> None:
    """Print requested summary statistics."""
    named_count = int(nodes["substation_name"].notna().sum())
    unnamed_count = int(nodes["substation_name"].isna().sum())
    step_count = int(nodes["is_step_up_down"].sum())

    all_voltages = []
    for values in edges["voltage_values"]:
        all_voltages.extend(values if isinstance(values, list) else [])
    voltage_counts = Counter(all_voltages)

    combo_counts = nodes["voltages_connected"].replace("", np.nan).dropna().value_counts()

    print("\nSummary statistics")
    print("Number of Florida transmission lines:", len(florida_lines))
    print("Number of Florida network edges:", len(edges))
    print("Number of generated substation nodes:", len(nodes))
    print("Number of named substations:", named_count)
    print("Number of unnamed substations:", unnamed_count)
    print("Number of substations connecting multiple voltage levels:", step_count)
    print(
        "Average connected transmission lines per substation:",
        round(nodes["connected_edge_count"].mean(), 2),
    )

    print("\nMost common voltage levels:")
    for voltage, count in voltage_counts.most_common(10):
        label = str(int(voltage)) if float(voltage).is_integer() else str(voltage)
        print(f"  {label} kV: {count}")

    print("\nMost common voltage combinations:")
    print(combo_counts.head(10).to_string())


def main() -> None:
    lines = load_transmission_lines()

    print("\nFiltering/clipping transmission lines to Florida...")
    florida_lines = filter_lines_to_florida(lines)
    print("Florida transmission lines after clipping:", len(florida_lines))

    print("\nCreating network topology from line endpoints...")
    edges = prepare_edges(florida_lines)
    nodes, edges = build_nodes_and_edges(edges)

    print("\nSaving outputs...")
    save_outputs(florida_lines, nodes, edges)
    print_summary(florida_lines, nodes, edges)

    print("\nSaved:", FLORIDA_LINES_OUTPUT)
    print("Saved:", NODES_OUTPUT)
    print("Saved:", EDGES_OUTPUT)
    print("Saved:", VOLTAGE_SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
