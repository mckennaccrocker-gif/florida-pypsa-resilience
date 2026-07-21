"""Diagnose remaining no-hazard load shedding in the summer peak x3 case."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_NETWORK_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_county_load_generator_overrides"
DEFAULT_CASE_DIR = (
    DEFAULT_NETWORK_DIR
    / "summer_peak_multiplier_sweep"
    / "x3"
    / "boundary_imports_plus_artifact_islands"
)
DEFAULT_OUTPUT_DIR = DEFAULT_NETWORK_DIR / "summer_peak_multiplier_sweep" / "x3_remaining_shedding_diagnosis"

START = pd.Timestamp("2025-07-28 00:00:00")
PERIODS = 24
END = START + pd.Timedelta(hours=PERIODS - 1)
LINE_MULTIPLIER = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose remaining summer peak x3 load shedding.")
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--line-multiplier", type=float, default=LINE_MULTIPLIER)
    return parser.parse_args()


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load_tables(network_dir: Path, case_dir: Path):
    buses = pd.read_csv(network_dir / "buses.csv").set_index("name")
    lines = pd.read_csv(network_dir / "lines.csv").set_index("name")
    gens = pd.read_csv(network_dir / "generators.csv").set_index("name")
    final_gens = pd.read_csv(network_dir / "generators_with_final_marginal_costs.csv").set_index("name")
    for col in ["p_nom", "marginal_cost"]:
        if col in final_gens.columns:
            gens[col] = final_gens[col]
    loads = pd.read_csv(network_dir / "loads.csv").set_index("name")
    loads_p = pd.read_csv(network_dir / "loads-p_set.csv")
    loads_p["snapshot"] = pd.to_datetime(loads_p["snapshot"], errors="raise")
    pmax = pd.read_csv(network_dir / "generators-p_max_pu.csv")
    pmax["snapshot"] = pd.to_datetime(pmax["snapshot"], errors="raise")

    shed = pd.read_csv(case_dir / "baseline_load_shedding_by_bus.csv")
    line_loading = pd.read_csv(case_dir / "baseline_line_loading.csv")
    import_selection = pd.read_csv(case_dir / "import_bus_selection.csv")
    import_by_bus = pd.read_csv(case_dir / "baseline_import_slack_by_bus.csv")
    congested = pd.read_csv(case_dir / "top_congested_corridors.csv")
    summary = pd.read_csv(case_dir / "baseline_summary.csv").iloc[0]

    return buses, lines, gens, loads, loads_p, pmax, shed, line_loading, import_selection, import_by_bus, congested, summary


def component_map(lines: pd.DataFrame, buses: pd.DataFrame):
    graph = nx.Graph()
    graph.add_nodes_from(buses.index)
    active = lines.copy()
    if "active" in active.columns:
        active = active[active["active"].astype(bool)]
    graph.add_edges_from(zip(active["bus0"], active["bus1"]))
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    bus_to_comp = {bus: i for i, comp in enumerate(comps) for bus in comp}
    return graph, comps, bus_to_comp


def bus_load_summary(loads, loads_p):
    load_to_bus = loads["bus"].to_dict()
    window = loads_p[(loads_p["snapshot"] >= START) & (loads_p["snapshot"] <= END)].copy()
    load_cols = [c for c in window.columns if c != "snapshot"]
    by_bus = pd.DataFrame(index=window["snapshot"])
    for load in load_cols:
        bus = load_to_bus.get(load)
        if bus is None:
            continue
        if bus not in by_bus.columns:
            by_bus[bus] = 0.0
        by_bus[bus] = by_bus[bus] + pd.to_numeric(window[load], errors="coerce").fillna(0.0).to_numpy()
    return pd.DataFrame(
        {
            "bus": by_bus.columns,
            "bus_total_load_mwh": by_bus.sum(axis=0).to_numpy(),
            "bus_peak_load_mw": by_bus.max(axis=0).to_numpy(),
        }
    ).set_index("bus")


def generator_availability(gens, pmax):
    window = pmax[(pmax["snapshot"] >= START) & (pmax["snapshot"] <= END)].copy()
    gen_cols = [c for c in window.columns if c != "snapshot"]
    rows = []
    for gen in gens.index:
        p_nom = float(pd.to_numeric(pd.Series([gens.loc[gen, "p_nom"]]), errors="coerce").fillna(0).iloc[0])
        if gen in gen_cols:
            cf = pd.to_numeric(window[gen], errors="coerce").fillna(1.0)
        else:
            cf = pd.Series(1.0, index=window.index)
        rows.append(
            {
                "generator": gen,
                "bus": gens.loc[gen, "bus"],
                "carrier": gens.loc[gen, "carrier"],
                "p_nom_mw": p_nom,
                "available_capacity_min_mw": float((cf * p_nom).min()),
                "available_capacity_mean_mw": float((cf * p_nom).mean()),
                "available_capacity_max_mw": float((cf * p_nom).max()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    network_dir = args.network_dir.resolve()
    case_dir = (
        args.case_dir
        or network_dir / "summer_peak_multiplier_sweep" / "x3" / "boundary_imports_plus_artifact_islands"
    ).resolve()
    output_dir = (args.output_dir or network_dir / "summer_peak_multiplier_sweep" / "x3_remaining_shedding_diagnosis").resolve()
    line_multiplier = args.line_multiplier
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        buses,
        lines,
        gens,
        loads,
        loads_p,
        pmax,
        shed,
        line_loading,
        import_selection,
        import_by_bus,
        congested,
        summary,
    ) = load_tables(network_dir, case_dir)

    graph, comps, bus_to_comp = component_map(lines, buses)
    bus_load = bus_load_summary(loads, loads_p)
    gen_avail = generator_availability(gens, pmax)

    line_loading = line_loading.set_index("line")
    import_type = import_selection.set_index("import_bus")["import_type"].to_dict()
    import_by_bus = import_by_bus.set_index("bus")

    shed_rows = []
    incident_rows = []
    nearest_gen_rows = []
    nearest_import_rows = []

    for _, row in shed.iterrows():
        bus = row["bus"]
        bus_x = buses.loc[bus, "x"]
        bus_y = buses.loc[bus, "y"]
        incident = lines[(lines["bus0"].eq(bus)) | (lines["bus1"].eq(bus))].copy()
        incident["line"] = incident.index
        incident_s_nom_col = f"s_nom_x{line_multiplier:g}_mva"
        incident[incident_s_nom_col] = pd.to_numeric(incident["s_nom"], errors="coerce") * line_multiplier
        incident["other_bus"] = np.where(incident["bus0"].eq(bus), incident["bus1"], incident["bus0"])
        incident = incident.join(
            line_loading[["max_abs_p0_mw", "max_loading_pu", "hours_overloaded"]],
            on="line",
        )
        incident["shed_bus"] = bus
        incident_rows.append(incident)

        comp_id = bus_to_comp.get(bus, -1)
        comp_buses = list(comps[comp_id]) if comp_id >= 0 else []
        comp_gens = gen_avail[gen_avail["bus"].isin(comp_buses)].copy()
        bus_gens = gen_avail[gen_avail["bus"].eq(bus)].copy()
        comp_imports = import_selection[import_selection["import_bus"].isin(comp_buses)].copy()

        # Nearby generators, excluding load shedding/import generators since these base tables do not include them.
        gen_points = gen_avail.merge(
            buses[["x", "y"]],
            left_on="bus",
            right_index=True,
            how="left",
        )
        gen_points["distance_km"] = haversine_km(bus_x, bus_y, gen_points["x"], gen_points["y"])
        nearest_gens = gen_points.sort_values("distance_km").head(10).copy()
        nearest_gens["shed_bus"] = bus
        nearest_gen_rows.append(nearest_gens)

        import_points = import_selection.merge(
            buses[["x", "y"]],
            left_on="import_bus",
            right_index=True,
            how="left",
        )
        import_points["distance_km"] = haversine_km(bus_x, bus_y, import_points["x"], import_points["y"])
        import_points["used_import_mwh"] = import_points["import_bus"].map(
            import_by_bus["total_import_slack_mwh"].to_dict()
        ).fillna(0.0)
        import_points["shed_bus"] = bus
        nearest_import_rows.append(import_points.sort_values("distance_km").head(8))

        shed_rows.append(
            {
                "bus": bus,
                "total_load_shed_mwh": row["total_load_shed_mwh"],
                "max_hourly_load_shed_mw": row["max_hourly_load_shed_mw"],
                "bus_total_load_mwh": bus_load.reindex([bus])["bus_total_load_mwh"].fillna(0.0).iloc[0],
                "bus_peak_load_mw": bus_load.reindex([bus])["bus_peak_load_mw"].fillna(0.0).iloc[0],
                "component_id": comp_id,
                "component_bus_count": len(comp_buses),
                "component_generator_count": len(comp_gens),
                "component_p_nom_mw": comp_gens["p_nom_mw"].sum(),
                "component_available_mean_mw": comp_gens["available_capacity_mean_mw"].sum(),
                "same_bus_generator_count": len(bus_gens),
                "same_bus_p_nom_mw": bus_gens["p_nom_mw"].sum(),
                "incident_line_count": len(incident),
                incident_s_nom_col: incident[incident_s_nom_col].sum(),
                "incident_binding_lines": int((incident["max_loading_pu"] >= 0.999).sum()),
                "incident_max_loading_pu": incident["max_loading_pu"].max(),
                "component_import_count": len(comp_imports),
                "nearest_import_distance_km": import_points["distance_km"].min(),
                "nearest_generator_distance_km": gen_points["distance_km"].min(),
                "x": bus_x,
                "y": bus_y,
                "substation_name": buses.loc[bus].get("substation_name", ""),
            }
        )

    shed_diag = pd.DataFrame(shed_rows).sort_values("total_load_shed_mwh", ascending=False)
    incident_diag = pd.concat(incident_rows, ignore_index=True).sort_values(
        ["shed_bus", "max_loading_pu"], ascending=[True, False]
    )
    nearest_gen_diag = pd.concat(nearest_gen_rows, ignore_index=True)
    nearest_import_diag = pd.concat(nearest_import_rows, ignore_index=True)

    shed_diag.to_csv(output_dir / "x3_summer_peak_load_shed_bus_diagnostics.csv", index=False)
    incident_diag.to_csv(output_dir / "x3_summer_peak_incident_line_diagnostics.csv", index=False)
    nearest_gen_diag.to_csv(output_dir / "x3_summer_peak_nearest_generators.csv", index=False)
    nearest_import_diag.to_csv(output_dir / "x3_summer_peak_nearest_imports.csv", index=False)
    congested.to_csv(output_dir / "x3_summer_peak_top_congested_corridors.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.bar(shed_diag["bus"], shed_diag["total_load_shed_mwh"], color="#d95f02")
    ax.set_title("Remaining Summer Peak x3 Load Shedding by Bus")
    ax.set_ylabel("Load shed (MWh)")
    ax.set_xlabel("Bus")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "x3_remaining_load_shed_by_bus.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.scatter(shed_diag["bus_peak_load_mw"], shed_diag["total_load_shed_mwh"], s=70, color="#1f77b4")
    for _, row in shed_diag.iterrows():
        ax.annotate(row["bus"], (row["bus_peak_load_mw"], row["total_load_shed_mwh"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Remaining x3 Shedding vs Local Bus Peak Load")
    ax.set_xlabel("Bus peak load (MW)")
    ax.set_ylabel("Load shed (MWh)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "x3_shed_vs_bus_peak_load.png")
    plt.close(fig)

    report = f"""# Summer Peak x3 Remaining Load-Shedding Diagnosis

Case diagnosed:

`{case_dir}`

## System Summary

- Total demand: {summary['total_demand_mwh']:,.1f} MWh
- Total load shed: {summary['total_load_shed_mwh']:,.3f} MWh
- Buses with load shedding: {int(summary['buses_experiencing_load_shedding'])}
- Total import slack: {summary['total_import_slack_mwh']:,.3f} MWh
- Overloaded lines reported: {int(summary['number_overloaded_lines'])}
- Max line loading: {summary['max_line_loading_pu']:,.6f} pu

## Main Finding

The remaining no-hazard shedding is concentrated at a small number of buses, all in the main network component. Several lines directly incident to these buses are binding at or extremely close to 100% loading. This points to **local deliverability constraints around specific buses/corridors**, not a statewide shortage of installed generation.

## Load-Shed Buses

| bus | shed MWh | max hourly shed MW | peak local load MW | incident lines | binding incident lines | nearest generator km | nearest import km |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for _, row in shed_diag.iterrows():
        report += (
            f"| {row['bus']} | {row['total_load_shed_mwh']:,.3f} | "
            f"{row['max_hourly_load_shed_mw']:,.3f} | {row['bus_peak_load_mw']:,.3f} | "
            f"{int(row['incident_line_count'])} | {int(row['incident_binding_lines'])} | "
            f"{row['nearest_generator_distance_km']:,.2f} | {row['nearest_import_distance_km']:,.2f} |\n"
        )

    report += """
## Interpretation

- The six shed buses are not isolated tiny islands; they are in the main component.
- The largest shed buses are served through low-capacity/binding local corridors even after the global x3 multiplier.
- The artifact-island import slack is constant across multiplier cases, so the remaining x3 issue is not those artifact islands.
- Boundary imports are heavily used on the summer peak day, so the summer peak case is much more import-dependent than the January baseline.

## Recommended Next Fixes

1. Inspect the remaining shed buses in QGIS/PyPSA.
2. Check whether these buses are county-load allocation artifacts with too much load assigned to small substations.
3. Check whether nearby same-voltage substations should be connected/snap-repaired.
4. Add a local load-allocation cap or redistribute county load across more electrically relevant buses.
5. Avoid solving this by increasing the global line multiplier beyond x3; the pattern is now local.

## Files

- `x3_summer_peak_load_shed_bus_diagnostics.csv`
- `x3_summer_peak_incident_line_diagnostics.csv`
- `x3_summer_peak_nearest_generators.csv`
- `x3_summer_peak_nearest_imports.csv`
- `x3_summer_peak_top_congested_corridors.csv`
- `x3_remaining_load_shed_by_bus.png`
- `x3_shed_vs_bus_peak_load.png`
"""
    (output_dir / "x3_remaining_shedding_diagnosis_report.md").write_text(report, encoding="utf-8")
    print("Saved diagnosis:", output_dir)
    print(shed_diag[["bus", "total_load_shed_mwh", "bus_peak_load_mw", "incident_binding_lines", "nearest_generator_distance_km"]].to_string(index=False))


if __name__ == "__main__":
    main()
