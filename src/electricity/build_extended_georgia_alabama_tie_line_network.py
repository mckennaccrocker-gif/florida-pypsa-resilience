"""
Extend clipped Florida boundary transmission lines to their real external ends.

The reviewed Florida grid intentionally keeps only real Florida lines and does
not add artificial island connectors. This script adds a second, experimental
network variant for dispatch checks: bulk HIFLD lines that continue into
Georgia/Alabama are extended from the clipped Florida boundary bus to their
full HIFLD endpoint, and that external endpoint is tagged as an import bus.

No loads are added outside Florida.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from pyproj import Geod
from shapely.geometry import LineString, Point
from shapely.ops import substring


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
INPUT_NETWORK = ELECTRICITY_DIR / "pypsa_florida_network_island_reviewed_no_connectors"
OUTPUT_NETWORK = ELECTRICITY_DIR / "pypsa_florida_network_extended_tie_lines"
FLORIDA_EDGES = ELECTRICITY_DIR / "florida_network_edges_transmission_lines.gpkg"
FULL_HIFLD_LINES = PROJECT_DIR / "data" / "Electricty" / "TransmissionLines2.gpkg"
PROJECTED_CRS = "EPSG:3086"
AC_FREQUENCY_HZ = 60.0
MIN_VOLTAGE_KV = 100.0

GEOD = Geod(ellps="WGS84")


LINE_PARAMETER_ASSUMPTIONS = {
    69: {"r_ohm_per_km": 0.249, "x_ohm_per_km": 0.400, "c_nf_per_km": 9.0},
    100: {"r_ohm_per_km": 0.200, "x_ohm_per_km": 0.390, "c_nf_per_km": 9.5},
    115: {"r_ohm_per_km": 0.153, "x_ohm_per_km": 0.380, "c_nf_per_km": 10.0},
    138: {"r_ohm_per_km": 0.120, "x_ohm_per_km": 0.370, "c_nf_per_km": 10.5},
    161: {"r_ohm_per_km": 0.100, "x_ohm_per_km": 0.360, "c_nf_per_km": 11.0},
    230: {"r_ohm_per_km": 0.080, "x_ohm_per_km": 0.330, "c_nf_per_km": 12.0},
    345: {"r_ohm_per_km": 0.050, "x_ohm_per_km": 0.300, "c_nf_per_km": 13.0},
    500: {"r_ohm_per_km": 0.030, "x_ohm_per_km": 0.280, "c_nf_per_km": 14.0},
    765: {"r_ohm_per_km": 0.020, "x_ohm_per_km": 0.270, "c_nf_per_km": 15.0},
}


def standard_voltage(voltage_kv: float) -> int:
    classes = np.array(sorted(LINE_PARAMETER_ASSUMPTIONS), dtype=float)
    return int(classes[np.argmin(np.abs(classes - float(voltage_kv)))])


def line_parameters(voltage_kv: float, length_km: float) -> dict[str, float | int | str]:
    voltage = standard_voltage(voltage_kv)
    params = LINE_PARAMETER_ASSUMPTIONS[voltage]
    capacitance_f = params["c_nf_per_km"] * 1e-9 * length_km
    return {
        "r": params["r_ohm_per_km"] * length_km,
        "x": params["x_ohm_per_km"] * length_km,
        "b": 2 * math.pi * AC_FREQUENCY_HZ * capacitance_f,
        "g": 0.0,
        "num_parallel": 1,
        "r_ohm_per_km_assumed": params["r_ohm_per_km"],
        "x_ohm_per_km_assumed": params["x_ohm_per_km"],
        "c_nf_per_km_assumed": params["c_nf_per_km"],
        "line_parameter_source": "external_tie_voltage_class_overhead_assumption",
    }


def geometry_endpoints(geometry) -> list[Point]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        coords = list(geometry.coords)
        return [Point(coords[0]), Point(coords[-1])]
    if geometry.geom_type == "MultiLineString":
        points = []
        for part in geometry.geoms:
            coords = list(part.coords)
            if len(coords) >= 2:
                points.extend([Point(coords[0]), Point(coords[-1])])
        return points
    return []


def is_georgia_alabama_endpoint(point: Point) -> bool:
    # Florida's clipped source bounds end at about y=31.0008 and x=-87.6034.
    # This catches northern Georgia/Alabama continuations and westward Alabama
    # continuations from the panhandle while avoiding unrelated national lines.
    return bool((point.y > 31.0009 or point.x < -87.6035) and 24.0 <= point.y <= 35.0 and -90.0 <= point.x <= -79.0)


def geodesic_distance_m(a: Point, b: Point) -> float:
    return float(GEOD.inv(a.x, a.y, b.x, b.y)[2])


def extension_geometry(full_geometry, boundary_point: Point, external_point: Point):
    """Return the original-HIFLD geometry segment from boundary bus to outside endpoint."""
    full_projected = gpd.GeoSeries([full_geometry], crs="EPSG:4326").to_crs(PROJECTED_CRS).iloc[0]
    boundary_projected = gpd.GeoSeries([boundary_point], crs="EPSG:4326").to_crs(PROJECTED_CRS).iloc[0]
    external_projected = gpd.GeoSeries([external_point], crs="EPSG:4326").to_crs(PROJECTED_CRS).iloc[0]

    parts = list(full_projected.geoms) if full_projected.geom_type == "MultiLineString" else [full_projected]
    best = None
    for part in parts:
        start = Point(part.coords[0])
        end = Point(part.coords[-1])
        endpoint_distance = min(start.distance(external_projected), end.distance(external_projected))
        boundary_distance = part.distance(boundary_projected)
        score = endpoint_distance + boundary_distance
        if best is None or score < best[0]:
            best = (score, part)
    if best is None:
        return LineString([boundary_point, external_point]), geodesic_distance_m(boundary_point, external_point) / 1000.0

    part = best[1]
    boundary_measure = part.project(boundary_projected)
    external_measure = part.project(external_projected)
    segment = substring(part, boundary_measure, external_measure)
    if segment.is_empty or segment.length <= 0:
        segment = LineString([boundary_projected, external_projected])
    segment_4326 = gpd.GeoSeries([segment], crs=PROJECTED_CRS).to_crs("EPSG:4326").iloc[0]
    return segment_4326, float(segment.length / 1000.0)


def safe_copytree(src: Path, dst: Path) -> None:
    project_abs = PROJECT_DIR.resolve()
    dst_abs = dst.resolve()
    if project_abs not in dst_abs.parents:
        raise ValueError(f"Refusing to write outside project: {dst_abs}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def export_checked_netcdf(network_dir: Path) -> None:
    path = network_dir / "florida_network.nc"
    path.unlink(missing_ok=True)
    network = pypsa.Network()
    network.import_from_csv_folder(str(network_dir))
    for df in [network.buses, network.lines, network.generators, network.loads, network.carriers]:
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].where(pd.notna(df[column]), "").astype(str)
    network.export_to_netcdf(path)
    check = pypsa.Network(str(path))
    if len(check.buses) != len(network.buses) or len(check.lines) != len(network.lines):
        raise RuntimeError(f"NetCDF export check failed for {path}")


def main() -> None:
    safe_copytree(INPUT_NETWORK, OUTPUT_NETWORK)

    buses = pd.read_csv(OUTPUT_NETWORK / "buses.csv")
    lines = pd.read_csv(OUTPUT_NETWORK / "lines.csv")
    florida_edges = gpd.read_file(FLORIDA_EDGES, layer="edge_component")
    full_hifld = gpd.read_file(FULL_HIFLD_LINES).to_crs("EPSG:4326")

    buses["name"] = buses["name"].astype(str)
    lines["name"] = lines["name"].astype(str)

    edge_to_hifld = florida_edges.set_index("edge_id")["ID"].astype(int).to_dict()
    full_by_id = {int(row.ID): row.geometry for row in full_hifld.itertuples() if pd.notna(row.ID)}
    bus_lookup = buses.set_index("name")

    buses["is_external_tie_import"] = False
    buses["external_tie_hifld_id"] = pd.NA
    buses["external_tie_source"] = ""

    new_bus_rows = []
    new_line_rows = []
    qgis_rows = []
    used_external_keys = set()

    for line in lines.itertuples():
        voltage = float(line.v_nom)
        if voltage < MIN_VOLTAGE_KV:
            continue
        source_edge_id = int(float(line.source_edge_id))
        hifld_id = edge_to_hifld.get(source_edge_id)
        full_geometry = full_by_id.get(hifld_id)
        if full_geometry is None:
            continue

        external_endpoints = [point for point in geometry_endpoints(full_geometry) if is_georgia_alabama_endpoint(point)]
        if not external_endpoints:
            continue

        bus0 = bus_lookup.loc[str(line.bus0)]
        bus1 = bus_lookup.loc[str(line.bus1)]
        bus0_point = Point(float(bus0["x"]), float(bus0["y"]))
        bus1_point = Point(float(bus1["x"]), float(bus1["y"]))

        # Choose the external endpoint and clipped Florida bus endpoint that are
        # closest to each other.
        options = []
        for external_point in external_endpoints:
            options.append((geodesic_distance_m(bus0_point, external_point), str(line.bus0), bus0_point, external_point))
            options.append((geodesic_distance_m(bus1_point, external_point), str(line.bus1), bus1_point, external_point))
        distance_m, boundary_bus, boundary_point, external_point = sorted(options, key=lambda item: item[0])[0]
        if distance_m < 500:
            continue

        external_key = (hifld_id, round(external_point.x, 6), round(external_point.y, 6), boundary_bus)
        if external_key in used_external_keys:
            continue
        used_external_keys.add(external_key)

        extension_geom, length_km = extension_geometry(full_geometry, boundary_point, external_point)
        external_bus = f"external_tie_bus_{hifld_id}_{len(new_bus_rows) + 1}"
        new_line = f"external_tie_extension_{line.name}"
        params = line_parameters(voltage, length_km)

        new_bus_rows.append(
            {
                **{column: pd.NA for column in buses.columns},
                "name": external_bus,
                "v_nom": voltage,
                "carrier": "AC",
                "x": external_point.x,
                "y": external_point.y,
                "substation_name": f"EXTERNAL_TIE_{hifld_id}",
                "source_node_id": f"external_{hifld_id}",
                "is_step_up_down": False,
                "connected_edge_count": 1,
                "is_external_tie_import": True,
                "external_tie_hifld_id": hifld_id,
                "external_tie_source": "full_HIFLD_TransmissionLines2_endpoint",
            }
        )
        new_line_rows.append(
            {
                **{column: pd.NA for column in lines.columns},
                "name": new_line,
                "bus0": boundary_bus,
                "bus1": external_bus,
                "carrier": "AC",
                "length": length_km,
                "v_nom": voltage,
                "s_nom": float(line.s_nom),
                "source_edge_id": -300000 - source_edge_id,
                "slr_amps": getattr(line, "slr_amps", pd.NA),
                "capacity_source": "external_tie_copied_from_boundary_line",
                "owner": getattr(line, "owner", ""),
                "status": "external_tie_extension",
                **params,
            }
        )
        qgis_rows.append(
            {
                "line": new_line,
                "original_boundary_line": line.name,
                "boundary_bus": boundary_bus,
                "external_bus": external_bus,
                "hifld_id": hifld_id,
                "source_edge_id": source_edge_id,
                "v_nom": voltage,
                "s_nom": float(line.s_nom),
                "length_km": length_km,
                "owner": getattr(line, "owner", ""),
                "geometry": extension_geom,
            }
        )

    buses_out = pd.concat([buses, pd.DataFrame(new_bus_rows)], ignore_index=True)
    lines_out = pd.concat([lines, pd.DataFrame(new_line_rows)], ignore_index=True)
    buses_out.to_csv(OUTPUT_NETWORK / "buses.csv", index=False)
    lines_out.to_csv(OUTPUT_NETWORK / "lines.csv", index=False)

    external_buses = pd.DataFrame(new_bus_rows)[
        ["name", "v_nom", "x", "y", "substation_name", "external_tie_hifld_id"]
    ]
    external_buses.to_csv(OUTPUT_NETWORK / "external_tie_import_buses.csv", index=False)
    extensions = gpd.GeoDataFrame(qgis_rows, crs="EPSG:4326")
    qgis_dir = OUTPUT_NETWORK / "qgis"
    qgis_dir.mkdir(exist_ok=True)
    extensions.to_file(qgis_dir / "external_tie_line_extensions.gpkg", layer="external_tie_line_extensions", driver="GPKG")

    summary = pd.DataFrame(
        [
            {
                "input_network": str(INPUT_NETWORK.resolve()),
                "output_network": str(OUTPUT_NETWORK.resolve()),
                "external_tie_buses_added": len(new_bus_rows),
                "external_tie_lines_added": len(new_line_rows),
                "minimum_voltage_kv": MIN_VOLTAGE_KV,
            }
        ]
    )
    summary.to_csv(OUTPUT_NETWORK / "external_tie_extension_summary.csv", index=False)

    readme = f"""# Florida Grid With Extended Georgia/Alabama Tie Lines

Created by `data/Electricity/build_extended_georgia_alabama_tie_line_network.py`.

Input network:
`{INPUT_NETWORK.resolve()}`

Output network:
`{OUTPUT_NETWORK.resolve()}`

Method:
- matched current Florida line `source_edge_id` values back to HIFLD `ID`;
- used the full U.S. HIFLD geometry in `data/Electricty/TransmissionLines2.gpkg`;
- identified bulk lines with `v_nom >= {MIN_VOLTAGE_KV}` kV whose full geometry continues north/west outside Florida;
- added an external terminal bus at the real full-geometry endpoint;
- added one extension line from the clipped Florida boundary bus to that external endpoint;
- tagged each new external bus with `is_external_tie_import=True`.

No loads or generators were added outside Florida in the static network. Dispatch
scripts can place import slack at the tagged external tie buses.
"""
    (OUTPUT_NETWORK / "EXTENDED_TIE_LINES_README.md").write_text(readme, encoding="utf-8")

    export_checked_netcdf(OUTPUT_NETWORK)

    print(summary.to_string(index=False))
    print("External import buses:", OUTPUT_NETWORK / "external_tie_import_buses.csv")
    print("QGIS extension lines:", qgis_dir / "external_tie_line_extensions.gpkg")


if __name__ == "__main__":
    main()
