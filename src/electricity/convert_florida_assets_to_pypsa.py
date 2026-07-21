"""
Convert the cleaned Florida electricity GIS layers into PyPSA-ready tables.

Inputs:
  - data/Electricity/florida_network_nodes_substations.gpkg
  - data/Electricity/florida_network_edges_transmission_lines.gpkg
  - data/Electricity/florida_lines_with_s_nom.gpkg
  - data/Electricity/florida_powerplants_osm_matched_plus_unmatched.gpkg

Outputs:
  - data/Electricity/pypsa_florida_network/buses.csv
  - data/Electricity/pypsa_florida_network/lines.csv
  - data/Electricity/pypsa_florida_network/generators.csv
  - data/Electricity/pypsa_florida_network/loads.csv
  - data/Electricity/pypsa_florida_network/carriers.csv
  - data/Electricity/pypsa_florida_network/*_review.csv

The script creates a first-pass PyPSA network representation for vulnerability
analysis. It uses the existing endpoint-derived topology rather than rebuilding
topology with snkit, because the Florida line edge table already has from/to
node IDs.

If PyPSA is installed, the script also tries to export:
  - data/Electricity/pypsa_florida_network/florida_network.nc
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import networkx

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"

NODES_FILE = ELECTRICITY_DIR / "florida_network_nodes_substations.gpkg"
EDGES_FILE = ELECTRICITY_DIR / "florida_network_edges_transmission_lines.gpkg"
LINES_WITH_S_NOM_FILE = ELECTRICITY_DIR / "florida_lines_with_s_nom.gpkg"
DEFAULT_PLANTS_FILE = ELECTRICITY_DIR / "florida_powerplants_osm_matched_plus_unmatched.gpkg"
ALL_PLANTS_FILE = ELECTRICITY_DIR / "florida_powerplants_cleaned.gpkg"

OUTPUT_DIR = ELECTRICITY_DIR / "pypsa_florida_network"
PROJECTED_CRS = "EPSG:3086"
DEFAULT_VOLTAGE_KV = 230.0
AC_FREQUENCY_HZ = 60.0

# Approximate thermal capacities for a first-pass planning/vulnerability model.
# These are placeholders for lines that do not yet have DOE/NREL SLR ratings.
S_NOM_BY_VOLTAGE_MVA = {
    69: 100,
    100: 150,
    115: 200,
    138: 250,
    161: 300,
    230: 600,
    345: 1200,
    500: 2500,
    765: 4000,
}

# First-pass overhead AC line assumptions by voltage class. Values are positive
# sequence per-km approximations for regional vulnerability screening; replace
# with utility/FERC 715 branch data when available.
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

GENERATOR_MARGINAL_COST_USD_PER_MWH = {
    "Solar": 0.0,
    "Wind": 0.0,
    "Hydro": 5.0,
    "Nuclear": 10.0,
    "Gas": 45.0,
    "Coal": 55.0,
    "Oil": 120.0,
    "Biomass": 35.0,
    "Waste": 30.0,
    "Storage": 0.0,
    "Geothermal": 20.0,
    "Unknown": 50.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PyPSA-ready Florida network tables from GIS assets."
    )
    parser.add_argument(
        "--include-gppd-only",
        action="store_true",
        help=(
            "Use florida_powerplants_cleaned.gpkg, including GPPD-only plants. "
            "Default uses OSM-trusted matched plus unmatched OSM plants."
        ),
    )
    parser.add_argument(
        "--total-load-mw",
        type=float,
        default=0.0,
        help=(
            "Optional static Florida load to distribute across buses. Default is "
            "0, creating placeholder loads for later replacement with demand data."
        ),
    )
    parser.add_argument(
        "--load-weight",
        choices=["equal", "connected_edges"],
        default="connected_edges",
        help="How to distribute --total-load-mw across buses.",
    )
    return parser.parse_args()


def standard_voltage(voltage_kv: float) -> int:
    if pd.isna(voltage_kv) or voltage_kv <= 0:
        return int(DEFAULT_VOLTAGE_KV)
    classes = np.array(sorted(S_NOM_BY_VOLTAGE_MVA.keys()), dtype=float)
    return int(classes[np.argmin(np.abs(classes - float(voltage_kv)))])


def line_electrical_parameters(voltage: int, length_km: float, num_parallel: int = 1) -> dict:
    """Estimate PyPSA line r/x/b from per-km overhead line assumptions."""
    params = LINE_PARAMETER_ASSUMPTIONS[voltage]
    parallel = max(int(num_parallel), 1)
    capacitance_f = params["c_nf_per_km"] * 1e-9 * length_km
    return {
        "r": params["r_ohm_per_km"] * length_km / parallel,
        "x": params["x_ohm_per_km"] * length_km / parallel,
        "b": 2 * math.pi * AC_FREQUENCY_HZ * capacitance_f * parallel,
        "g": 0.0,
        "num_parallel": parallel,
        "r_ohm_per_km_assumed": params["r_ohm_per_km"],
        "x_ohm_per_km_assumed": params["x_ohm_per_km"],
        "c_nf_per_km_assumed": params["c_nf_per_km"],
        "line_parameter_source": "voltage_class_overhead_assumption",
    }


def print_columns(label: str, frame: pd.DataFrame) -> None:
    print(f"\n{label} columns:")
    print(", ".join(frame.columns.astype(str)))


def load_inputs(include_gppd_only: bool) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    plants_file = ALL_PLANTS_FILE if include_gppd_only else DEFAULT_PLANTS_FILE
    for path in [NODES_FILE, EDGES_FILE, plants_file]:
        if not path.exists():
            raise FileNotFoundError(path)

    nodes = gpd.read_file(NODES_FILE).to_crs("EPSG:4326")
    edges = gpd.read_file(EDGES_FILE).to_crs("EPSG:4326")
    edges, nodes = components(
        edges,
        nodes,
        node_id_column="node_id",
        edge_id_column="edge_id",
        from_node_column="from_node_id",
        to_node_column="to_node_id",
    )
    edges.to_file(EDGES_FILE, driver="GPKG", layer="edge_component")
    nodes.to_file(NODES_FILE, driver="GPKG", layer="node_component")

    plants = gpd.read_file(plants_file).to_crs("EPSG:4326")

    print_columns("Substation nodes", nodes)
    print_columns("Transmission edges", edges)
    print_columns("Power plants", plants)

    if LINES_WITH_S_NOM_FILE.exists():
        ratings = gpd.read_file(LINES_WITH_S_NOM_FILE).drop(columns="geometry", errors="ignore")
        print_columns("SLR-rated transmission lines", ratings)
        required = {"florida_line_id", "slr_amps", "s_nom_mva"}
        missing = required.difference(ratings.columns)
        if missing:
            print("Skipping SLR capacity merge; missing columns:", sorted(missing))
        elif "edge_id" not in edges.columns:
            print("Skipping SLR capacity merge; transmission edges are missing edge_id.")
        else:
            merge_cols = ["florida_line_id", "slr_amps", "s_nom_mva"]
            if "hifld_id_for_slr" in ratings.columns:
                merge_cols.append("hifld_id_for_slr")
            before_cols = set(edges.columns)
            edges = edges.merge(
                ratings[merge_cols],
                left_on="edge_id",
                right_on="florida_line_id",
                how="left",
                validate="one_to_one",
            )
            added = sorted(set(edges.columns).difference(before_cols))
            print("Merged SLR capacities into edges using edge_id -> florida_line_id.")
            print("Added columns:", added)
            print("Lines with SLR amps:", int(edges["slr_amps"].notna().sum()), "/", len(edges))
            print("Lines with s_nom_mva:", int(edges["s_nom_mva"].notna().sum()), "/", len(edges))
    else:
        print("\nNo SLR capacity file found; using voltage-class placeholder s_nom values.")

    return nodes, edges, plants


def make_buses(nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    buses = pd.DataFrame(
        {
            "name": "bus_" + nodes["node_id"].astype(int).astype(str),
            "v_nom": pd.to_numeric(nodes["max_voltage"], errors="coerce").fillna(
                DEFAULT_VOLTAGE_KV
            ),
            "carrier": "AC",
            "x": nodes.geometry.x,
            "y": nodes.geometry.y,
            "substation_name": nodes["substation_name"].fillna(""),
            "source_node_id": nodes["node_id"].astype(int),
            "is_step_up_down": nodes["is_step_up_down"].astype(bool),
            "connected_edge_count": nodes["connected_edge_count"].fillna(0).astype(int),
        }
    )
    buses = buses.set_index("name")
    buses.index.name = "name"
    return buses


def make_lines(edges: gpd.GeoDataFrame, valid_bus_names: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    dropped = []
    used_slr_capacity = 0
    used_placeholder_capacity = 0

    for row in edges.itertuples(index=False):
        bus0 = f"bus_{int(row.from_node_id)}" if pd.notna(row.from_node_id) else ""
        bus1 = f"bus_{int(row.to_node_id)}" if pd.notna(row.to_node_id) else ""
        voltage = standard_voltage(getattr(row, "voltage_clean", np.nan))
        s_nom_mva = pd.to_numeric(getattr(row, "s_nom_mva", np.nan), errors="coerce")
        if pd.notna(s_nom_mva) and s_nom_mva > 0:
            s_nom = float(s_nom_mva)
            capacity_source = "DOE_NREL_SLR_A_100C"
            used_slr_capacity += 1
        else:
            s_nom = float(S_NOM_BY_VOLTAGE_MVA[voltage])
            capacity_source = "voltage_class_placeholder"
            used_placeholder_capacity += 1

        reason = None
        if bus0 not in valid_bus_names or bus1 not in valid_bus_names:
            reason = "missing_bus"
        elif bus0 == bus1:
            reason = "self_loop"

        if reason:
            dropped.append(
                {
                    "edge_id": int(row.edge_id),
                    "bus0": bus0,
                    "bus1": bus1,
                    "drop_reason": reason,
                }
            )
            continue

        records.append(
            {
                "name": f"line_{int(row.edge_id)}",
                "bus0": bus0,
                "bus1": bus1,
                "carrier": "AC",
                "length": float(row.length_m) / 1000,
                "v_nom": voltage,
                "s_nom": s_nom,
                "source_edge_id": int(row.edge_id),
                "slr_amps": pd.to_numeric(getattr(row, "slr_amps", np.nan), errors="coerce"),
                "capacity_source": capacity_source,
                "owner": getattr(row, "OWNER", ""),
                "status": getattr(row, "STATUS", ""),
                **line_electrical_parameters(voltage, float(row.length_m) / 1000),
            }
        )

    lines = pd.DataFrame(records).set_index("name")
    lines.index.name = "name"
    dropped_lines = pd.DataFrame(dropped)
    print("\nLine capacity assignment:")
    print("DOE/NREL SLR-based s_nom:", used_slr_capacity)
    print("Voltage-class placeholder s_nom:", used_placeholder_capacity)
    print("Line electrical parameters: voltage-class overhead assumptions")
    return lines, dropped_lines


def nearest_bus_matches(
    plants: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    plants_m = plants.to_crs(PROJECTED_CRS).copy()
    nodes_m = nodes[["node_id", "substation_name", "geometry"]].to_crs(PROJECTED_CRS)

    matched = gpd.sjoin_nearest(
        plants_m,
        nodes_m.rename(
            columns={
                "node_id": "nearest_node_id",
                "substation_name": "nearest_substation_name",
            }
        ),
        how="left",
        distance_col="nearest_bus_distance_m",
    )
    return matched.to_crs("EPSG:4326")


def components(edges,nodes,
                node_id_column="node_id",edge_id_column="edge_id",
                from_node_column="from_node_id",to_node_column="to_edge_id"):
    G = networkx.Graph()
    G.add_nodes_from(
        (getattr(n, node_id_column), {"geometry": n.geometry}) for n in nodes.itertuples()
    )
    G.add_edges_from(
        (getattr(e,from_node_column), getattr(e,to_node_column), 
            {edge_id_column: getattr(e,edge_id_column), "geometry": e.geometry})
        for e in edges.itertuples()
    )
    components = networkx.connected_components(G)
    for num, c in enumerate(components):
        print(f"Component {num} has {len(c)} nodes")
        edges.loc[(edges[from_node_column].isin(c) | edges[to_node_column].isin(c)), "component"] = num
        nodes.loc[nodes[node_id_column].isin(c), "component"] = num
 
    return edges, nodes

def make_generators(
    plants: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    valid_bus_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched = nearest_bus_matches(plants, nodes)

    records = []
    review_records = []
    for idx, row in matched.reset_index(drop=True).iterrows():
        if pd.isna(row.nearest_node_id):
            continue

        bus = f"bus_{int(row.nearest_node_id)}"
        if bus not in valid_bus_names:
            continue

        carrier = str(row.primary_fuel).strip() if pd.notna(row.primary_fuel) else "Unknown"
        if not carrier:
            carrier = "Unknown"

        p_nom = pd.to_numeric(row.capacity_mw, errors="coerce")
        if pd.isna(p_nom) or p_nom < 0:
            p_nom = 0.0

        generator_name = f"gen_{idx}_{str(row.get('name', 'unnamed')).strip()[:60]}"
        records.append(
            {
                "name": generator_name,
                "bus": bus,
                "carrier": carrier,
                "p_nom": float(p_nom),
                "marginal_cost": GENERATOR_MARGINAL_COST_USD_PER_MWH.get(
                    carrier,
                    GENERATOR_MARGINAL_COST_USD_PER_MWH["Unknown"],
                ),
                "source_name": row.get("name", ""),
                "matched_source": row.get("matched_source", ""),
                "location_confidence": row.get("location_confidence", ""),
            }
        )
        review_records.append(
            {
                "generator": generator_name,
                "plant_name": row.get("name", ""),
                "carrier": carrier,
                "capacity_mw": float(p_nom),
                "bus": bus,
                "nearest_node_id": int(row.nearest_node_id),
                "nearest_substation_name": row.get("nearest_substation_name", ""),
                "nearest_bus_distance_m": float(row.nearest_bus_distance_m),
                "matched_source": row.get("matched_source", ""),
                "location_confidence": row.get("location_confidence", ""),
                "longitude": row.geometry.x,
                "latitude": row.geometry.y,
            }
        )

    generators = pd.DataFrame(records).set_index("name")
    generators.index.name = "name"
    generator_review = pd.DataFrame(review_records)
    return generators, generator_review


def make_loads(buses: pd.DataFrame, total_load_mw: float, load_weight: str) -> pd.DataFrame:
    if total_load_mw <= 0:
        p_set = pd.Series(0.0, index=buses.index)
    elif load_weight == "equal":
        p_set = pd.Series(total_load_mw / len(buses), index=buses.index)
    else:
        weights = buses["connected_edge_count"].clip(lower=1).astype(float)
        p_set = weights / weights.sum() * total_load_mw

    loads = pd.DataFrame(
        {
            "name": ["load_" + bus.replace("bus_", "") for bus in buses.index],
            "bus": buses.index,
            "p_set": p_set.to_numpy(),
            "carrier": "electricity",
        }
    ).set_index("name")
    loads.index.name = "name"
    return loads


def make_carriers(generators: pd.DataFrame) -> pd.DataFrame:
    carrier_names = sorted(set(generators["carrier"].dropna().astype(str).tolist()))
    carrier_names.extend(["AC", "electricity"])
    carriers = pd.DataFrame(index=sorted(set(carrier_names)))
    carriers.index.name = "name"
    carriers["co2_emissions"] = 0.0
    return carriers


def write_assumptions(
    output_dir: Path,
    include_gppd_only: bool,
    total_load_mw: float,
    load_weight: str,
) -> None:
    plant_source = ALL_PLANTS_FILE if include_gppd_only else DEFAULT_PLANTS_FILE
    text = f"""# Florida PyPSA Conversion Notes

This folder contains a first-pass PyPSA-ready representation of the cleaned
Florida electricity network.

## Component mapping

- `florida_network_nodes_substations.gpkg` -> `buses.csv`
- `florida_network_edges_transmission_lines.gpkg` plus
  `florida_lines_with_s_nom.gpkg` -> `lines.csv`
- `{plant_source.name}` -> `generators.csv`
- placeholder bus loads -> `loads.csv`

## Current assumptions

- Substations inferred from HIFLD line endpoints are treated as PyPSA buses.
- HIFLD endpoint topology was already created by snapping endpoints within 100 m.
- Transmission lines with missing buses or identical from/to buses are excluded
  from `lines.csv` and written to `dropped_lines_review.csv`.
- Line thermal capacities (`s_nom`) use DOE/NREL `SLR_A-100C.h5` static line
  ratings where available via `florida_lines_with_s_nom.gpkg`. Remaining lines
  fall back to voltage-class placeholders:
  {S_NOM_BY_VOLTAGE_MVA}
- Line electrical parameters (`r`, `x`, `b`) are estimated from voltage-class
  overhead AC assumptions at {AC_FREQUENCY_HZ} Hz. The per-km lookup is:
  {LINE_PARAMETER_ASSUMPTIONS}
  PyPSA `r` and `x` are total line ohms; `b` is total shunt susceptance in
  siemens. `num_parallel` is set to 1 because circuit/conductor bundle counts
  are not yet available.
- Power plants are connected to the nearest inferred substation bus.
- Generator marginal costs are rough placeholders by fuel type:
  {GENERATOR_MARGINAL_COST_USD_PER_MWH}
- Loads are placeholders. `total_load_mw={total_load_mw}` was distributed using
  `{load_weight}` weights. Replace `loads.csv` or add time series before dispatch
  or capacity-expansion analysis.

## Missing information needed for a stronger energy-grid vulnerability model

- Hourly or representative Florida demand by county, balancing authority, or bus.
- Generator operating parameters: heat rates, marginal costs, min stable output,
  ramp rates, outage rates, and renewable time series.
- Transmission electrical parameters or line types: resistance, reactance,
  susceptance from utility/FERC 715 branch data.
- Transformer representation for buses connecting multiple voltage levels.
- A formal decision on whether to keep or drop GPPD-only plants.
- A hazard-to-component failure model: e.g. how TC wind, flood depth, or surge
  changes line/generator/bus availability or capacity.
- Restoration assumptions: repair times, replacement costs, and cascading or
  redispatch constraints.
"""
    (output_dir / "README.md").write_text(text)


def try_export_pypsa_network(output_dir: Path) -> None:
    if importlib.util.find_spec("pypsa") is None:
        print("PyPSA is not installed; wrote CSV tables only.")
        return

    import pypsa

    network = pypsa.Network()
    network.import_from_csv_folder(str(output_dir), skip_time=True)

    loads_p_set_path = output_dir / "loads-p_set.csv"
    if loads_p_set_path.exists():
        loads_p_set = pd.read_csv(loads_p_set_path)
        if "snapshot" not in loads_p_set.columns:
            raise ValueError(f"{loads_p_set_path} must contain a snapshot column.")
        snapshots = pd.to_datetime(loads_p_set.pop("snapshot"), errors="raise")
        missing_loads = sorted(set(loads_p_set.columns).difference(network.loads.index))
        if missing_loads:
            raise ValueError(
                "loads-p_set.csv contains loads that are not in loads.csv: "
                f"{missing_loads[:10]}"
            )
        network.set_snapshots(snapshots)
        loads_p_set.index = snapshots
        network.loads_t.p_set = loads_p_set.reindex(columns=network.loads.index, fill_value=0.0)

    generators_p_max_pu_path = output_dir / "generators-p_max_pu.csv"
    if generators_p_max_pu_path.exists():
        generators_p_max_pu = pd.read_csv(generators_p_max_pu_path)
        if "snapshot" not in generators_p_max_pu.columns:
            raise ValueError(f"{generators_p_max_pu_path} must contain a snapshot column.")
        snapshots = pd.to_datetime(generators_p_max_pu.pop("snapshot"), errors="raise")
        missing_generators = sorted(set(generators_p_max_pu.columns).difference(network.generators.index))
        if missing_generators:
            raise ValueError(
                "generators-p_max_pu.csv contains generators that are not in generators.csv: "
                f"{missing_generators[:10]}"
            )
        if len(network.snapshots) == 1 and str(network.snapshots[0]) == "now":
            network.set_snapshots(snapshots)
        elif not network.snapshots.equals(pd.DatetimeIndex(snapshots)):
            raise ValueError("generators-p_max_pu.csv snapshots do not match the network snapshots.")
        generators_p_max_pu.index = pd.DatetimeIndex(snapshots)
        network.generators_t.p_max_pu = generators_p_max_pu

    network.export_to_netcdf(output_dir / "florida_network.nc")
    print("Saved PyPSA NetCDF:", output_dir / "florida_network.nc")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nodes, edges, plants = load_inputs(args.include_gppd_only)

    buses = make_buses(nodes)
    valid_bus_names = set(buses.index)
    lines, dropped_lines = make_lines(edges, valid_bus_names)
    generators, generator_review = make_generators(plants, nodes, valid_bus_names)
    loads = make_loads(buses, args.total_load_mw, args.load_weight)
    carriers = make_carriers(generators)

    buses.to_csv(OUTPUT_DIR / "buses.csv")
    lines.to_csv(OUTPUT_DIR / "lines.csv")
    generators.to_csv(OUTPUT_DIR / "generators.csv")
    loads.to_csv(OUTPUT_DIR / "loads.csv")
    carriers.to_csv(OUTPUT_DIR / "carriers.csv")

    dropped_lines.to_csv(OUTPUT_DIR / "dropped_lines_review.csv", index=False)
    generator_review.to_csv(OUTPUT_DIR / "generator_bus_matches_review.csv", index=False)

    write_assumptions(OUTPUT_DIR, args.include_gppd_only, args.total_load_mw, args.load_weight)
    try_export_pypsa_network(OUTPUT_DIR)

    print("\nFlorida PyPSA-ready tables saved in:", OUTPUT_DIR)
    print("Buses:", len(buses))
    print("Lines:", len(lines), "| dropped:", len(dropped_lines))
    print("Lines using SLR s_nom:", int(lines["capacity_source"].eq("DOE_NREL_SLR_A_100C").sum()))
    print("Lines using placeholder s_nom:", int(lines["capacity_source"].eq("voltage_class_placeholder").sum()))
    print("Generators:", len(generators), "| total p_nom MW:", round(generators["p_nom"].sum(), 2))
    print("Loads:", len(loads), "| total p_set MW:", round(loads["p_set"].sum(), 2))
    print("Generator-bus review:", OUTPUT_DIR / "generator_bus_matches_review.csv")
    print("Assumptions:", OUTPUT_DIR / "README.md")


if __name__ == "__main__":
    main()


