"""Diagnose remaining summer peak load shedding in the transformer network.

This reads the already-solved x1.75 summer peak case for the voltage-layer
transformer network and summarizes whether remaining load shedding appears to
come from local bottlenecks, import caps, or generator availability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


NETWORK_DIR = Path("data/Electricity/pypsa_florida_network_voltage_transformers")
CASE_DIR = (
    NETWORK_DIR
    / "summer_peak_x1p75_diagnosis_rerun"
    / "boundary_imports_plus_artifact_islands"
)
OUTPUT_DIR = NETWORK_DIR / "summer_peak_x1p75_diagnosis"
START = pd.Timestamp("2025-07-28 00:00:00")
END = pd.Timestamp("2025-07-28 23:00:00")


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def active_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "active" not in df.columns:
        return df.copy()
    return df[df["active"].astype(bool)].copy()


def build_graph(lines: pd.DataFrame, transformers: pd.DataFrame, buses: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(buses["name"])
    graph.add_edges_from(zip(lines["bus0"], lines["bus1"]))
    if not transformers.empty:
        graph.add_edges_from(zip(transformers["bus0"], transformers["bus1"]))
    return graph


def assign_components(graph: nx.Graph) -> dict[str, int]:
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    return {bus: i for i, component in enumerate(components) for bus in component}


def load_for_buses(loads: pd.DataFrame, load_buses: list[str]) -> pd.DataFrame:
    loads_for_buses = loads[loads["bus"].isin(load_buses)].copy()
    if loads_for_buses.empty:
        return pd.DataFrame(columns=["bus", "summer_peak_total_load_mwh", "summer_peak_max_load_mw"])

    usecols = ["snapshot"] + loads_for_buses["name"].tolist()
    load_ts = read_csv(NETWORK_DIR / "loads-p_set.csv", usecols=usecols)
    load_ts["snapshot"] = pd.to_datetime(load_ts["snapshot"])
    load_ts = load_ts[(load_ts["snapshot"] >= START) & (load_ts["snapshot"] <= END)]
    melted = load_ts.melt(id_vars="snapshot", var_name="load", value_name="load_mw")
    merged = melted.merge(loads_for_buses[["name", "bus"]], left_on="load", right_on="name")
    return (
        merged.groupby("bus")["load_mw"]
        .agg(summer_peak_total_load_mwh="sum", summer_peak_max_load_mw="max")
        .reset_index()
    )


def nearest_points(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_id: str,
    target_id: str,
    prefix: str,
    n: int = 3,
) -> pd.DataFrame:
    rows = []
    target_xy = target[[target_id, "x", "y"]].dropna().copy()
    for _, src in source.dropna(subset=["x", "y"]).iterrows():
        dx = (target_xy["x"] - src["x"]) * 111.0 * np.cos(np.deg2rad(src["y"]))
        dy = (target_xy["y"] - src["y"]) * 111.0
        distances = np.sqrt(dx**2 + dy**2)
        for rank, idx in enumerate(distances.nsmallest(n).index, start=1):
            rows.append(
                {
                    source_id: src[source_id],
                    "rank": rank,
                    f"{prefix}_id": target_xy.loc[idx, target_id],
                    f"{prefix}_distance_km": distances.loc[idx],
                }
            )
    return pd.DataFrame(rows)


def incident_asset_summary(
    shed_buses: pd.DataFrame,
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    line_loading: pd.DataFrame,
    transformers: pd.DataFrame,
    transformer_loading: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bus_set = set(shed_buses["bus"])
    incident_lines = lines[lines["bus0"].isin(bus_set) | lines["bus1"].isin(bus_set)].copy()
    incident_lines = incident_lines.merge(
        line_loading,
        left_on="name",
        right_on="line",
        how="left",
        suffixes=("", "_loading"),
    )
    if not incident_lines.empty:
        incident_lines["shed_bus"] = incident_lines.apply(
            lambda r: r["bus0"] if r["bus0"] in bus_set else r["bus1"], axis=1
        )

    incident_transformers = transformers[
        transformers["bus0"].isin(bus_set) | transformers["bus1"].isin(bus_set)
    ].copy()
    incident_transformers = incident_transformers.merge(
        transformer_loading,
        left_on="name",
        right_on="transformer",
        how="left",
        suffixes=("", "_loading"),
    )
    if not incident_transformers.empty:
        incident_transformers["shed_bus"] = incident_transformers.apply(
            lambda r: r["bus0"] if r["bus0"] in bus_set else r["bus1"], axis=1
        )

    # Also look one voltage-transformer hop away, because load buses may be
    # connected to the line voltage layer through an auxiliary bus.
    aux_neighbors = set()
    for _, row in incident_transformers.iterrows():
        aux_neighbors.add(row["bus1"] if row["bus0"] in bus_set else row["bus0"])
    one_hop_lines = lines[lines["bus0"].isin(aux_neighbors) | lines["bus1"].isin(aux_neighbors)].copy()
    one_hop_lines = one_hop_lines.merge(
        line_loading,
        left_on="name",
        right_on="line",
        how="left",
        suffixes=("", "_loading"),
    )
    if not one_hop_lines.empty:
        one_hop_lines["shed_bus"] = one_hop_lines.apply(
            lambda r: next(
                (
                    tr["bus0"]
                    for _, tr in incident_transformers.iterrows()
                    if tr["bus0"] in bus_set and (r["bus0"] == tr["bus1"] or r["bus1"] == tr["bus1"])
                ),
                "",
            ),
            axis=1,
        )
        incident_lines = pd.concat([incident_lines, one_hop_lines], ignore_index=True)
        incident_lines = incident_lines.drop_duplicates(subset=["name", "shed_bus"])

    return incident_lines, incident_transformers


def generator_availability(generators: pd.DataFrame) -> pd.DataFrame:
    pmax_path = NETWORK_DIR / "generators-p_max_pu.csv"
    if not pmax_path.exists():
        generators = generators.copy()
        generators["summer_peak_available_mwh"] = generators["p_nom"] * 24
        generators["summer_peak_max_available_mw"] = generators["p_nom"]
    else:
        pmax = read_csv(pmax_path)
        pmax["snapshot"] = pd.to_datetime(pmax["snapshot"])
        pmax = pmax[(pmax["snapshot"] >= START) & (pmax["snapshot"] <= END)]
        generator_cols = [c for c in pmax.columns if c != "snapshot"]
        generator_names = generators["name"].tolist()
        existing_cols = [g for g in generator_names if g in generator_cols]
        missing_cols = [g for g in generator_names if g not in generator_cols]
        p_nom = generators.set_index("name")["p_nom"]
        available_mwh = pd.Series(index=generator_names, dtype=float)
        max_available_mw = pd.Series(index=generator_names, dtype=float)
        if existing_cols:
            available_existing = pmax[existing_cols].multiply(p_nom.reindex(existing_cols), axis=1)
            available_mwh.loc[existing_cols] = available_existing.sum(axis=0)
            max_available_mw.loc[existing_cols] = available_existing.max(axis=0)
        if missing_cols:
            available_mwh.loc[missing_cols] = p_nom.reindex(missing_cols) * len(pmax)
            max_available_mw.loc[missing_cols] = p_nom.reindex(missing_cols)
        generators = generators.copy()
        generators["summer_peak_available_mwh"] = available_mwh.reindex(generators["name"]).values
        generators["summer_peak_max_available_mw"] = max_available_mw.reindex(generators["name"]).values

    return (
        generators.groupby("carrier")
        .agg(
            generators=("name", "count"),
            p_nom_mw=("p_nom", "sum"),
            summer_peak_available_mwh=("summer_peak_available_mwh", "sum"),
            summer_peak_max_available_mw=("summer_peak_max_available_mw", "sum"),
            average_marginal_cost=("marginal_cost", "mean"),
        )
        .reset_index()
        .sort_values("p_nom_mw", ascending=False)
    )


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path, color: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df[x].astype(str), df[y], color=color)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    buses = read_csv(NETWORK_DIR / "buses.csv")
    lines = active_frame(read_csv(NETWORK_DIR / "lines.csv"))
    transformers = active_frame(read_csv(NETWORK_DIR / "transformers.csv"))
    generators = read_csv(NETWORK_DIR / "generators.csv")
    loads = read_csv(NETWORK_DIR / "loads.csv")

    summary = read_csv(CASE_DIR / "baseline_summary.csv")
    shed = read_csv(CASE_DIR / "baseline_load_shedding_by_bus.csv")
    line_loading = read_csv(CASE_DIR / "baseline_line_loading.csv")
    transformer_loading = read_csv(CASE_DIR / "baseline_transformer_loading.csv")
    import_by_bus = read_csv(CASE_DIR / "baseline_import_slack_by_bus.csv")
    import_selection = read_csv(CASE_DIR / "import_bus_selection.csv")

    graph = build_graph(lines, transformers, buses)
    component_by_bus = assign_components(graph)
    buses = buses.copy()
    buses["component_id"] = buses["name"].map(component_by_bus)

    shed = shed.merge(buses[["name", "x", "y", "v_nom", "substation_name", "component_id"]], left_on="bus", right_on="name", how="left")
    shed = shed.drop(columns=["name"])
    shed = shed.merge(load_for_buses(loads, shed["bus"].tolist()), on="bus", how="left")
    shed["shed_to_local_load_ratio"] = np.where(
        shed["summer_peak_total_load_mwh"] > 0,
        shed["total_load_shed_mwh"] / shed["summer_peak_total_load_mwh"],
        np.nan,
    )
    shed = shed.sort_values("total_load_shed_mwh", ascending=False)
    shed.to_csv(OUTPUT_DIR / "summer_peak_load_shed_bus_diagnosis.csv", index=False)

    incident_lines, incident_transformers = incident_asset_summary(
        shed, buses, lines, line_loading, transformers, transformer_loading
    )
    incident_lines.to_csv(OUTPUT_DIR / "summer_peak_incident_lines_near_shed_buses.csv", index=False)
    incident_transformers.to_csv(OUTPUT_DIR / "summer_peak_incident_transformers_near_shed_buses.csv", index=False)

    top_lines = line_loading.sort_values("max_loading_pu", ascending=False).head(40)
    top_transformers = transformer_loading.sort_values("max_loading_pu", ascending=False).head(40)
    top_lines.to_csv(OUTPUT_DIR / "summer_peak_top_line_loading.csv", index=False)
    top_transformers.to_csv(OUTPUT_DIR / "summer_peak_top_transformer_loading.csv", index=False)

    import_diag = import_selection.merge(
        import_by_bus, left_on="import_bus", right_on="bus", how="left"
    )
    import_diag["total_import_slack_mwh"] = import_diag["total_import_slack_mwh"].fillna(0.0)
    import_diag["max_hourly_import_slack_mw"] = import_diag["max_hourly_import_slack_mw"].fillna(0.0)
    import_diag["max_import_cap_utilization"] = np.where(
        import_diag["import_p_nom_mw"] > 0,
        import_diag["max_hourly_import_slack_mw"] / import_diag["import_p_nom_mw"],
        np.nan,
    )
    import_diag = import_diag.sort_values("max_import_cap_utilization", ascending=False)
    import_diag.to_csv(OUTPUT_DIR / "summer_peak_import_cap_utilization.csv", index=False)

    buses_for_nearest = buses.rename(columns={"name": "bus"})
    gen_points = generators.merge(buses_for_nearest[["bus", "x", "y"]], on="bus", how="left")
    nearest_gens = nearest_points(
        shed[["bus", "x", "y"]],
        gen_points.rename(columns={"name": "generator"}),
        "bus",
        "generator",
        "nearest_generator",
        n=5,
    )
    nearest_gens = nearest_gens.merge(
        generators.rename(columns={"name": "nearest_generator_id"})[
            ["nearest_generator_id", "carrier", "p_nom", "marginal_cost"]
        ],
        on="nearest_generator_id",
        how="left",
    )
    nearest_gens.to_csv(OUTPUT_DIR / "summer_peak_nearest_generators_to_shed_buses.csv", index=False)

    import_points = import_selection.rename(columns={"import_bus": "bus"}).merge(
        buses_for_nearest[["bus", "x", "y"]], on="bus", how="left"
    )
    nearest_imports = nearest_points(
        shed[["bus", "x", "y"]],
        import_points[["bus", "x", "y"]],
        "bus",
        "bus",
        "nearest_import",
        n=5,
    )
    nearest_imports = nearest_imports.merge(
        import_diag.rename(columns={"import_bus": "nearest_import_id"})[
            [
                "nearest_import_id",
                "import_type",
                "import_p_nom_mw",
                "max_hourly_import_slack_mw",
                "max_import_cap_utilization",
            ]
        ],
        on="nearest_import_id",
        how="left",
    )
    nearest_imports.to_csv(OUTPUT_DIR / "summer_peak_nearest_imports_to_shed_buses.csv", index=False)

    gen_availability = generator_availability(generators)
    gen_availability.to_csv(OUTPUT_DIR / "summer_peak_generator_availability_by_carrier.csv", index=False)

    top_shed = shed.head(15)
    plot_bar(
        top_shed,
        "bus",
        "total_load_shed_mwh",
        "Summer peak load shedding by bus",
        "Load shed (MWh)",
        OUTPUT_DIR / "summer_peak_load_shed_by_bus.png",
        "#d95f02",
    )
    plot_bar(
        top_lines.head(20),
        "line",
        "max_loading_pu",
        "Top line loading during summer peak",
        "Maximum loading (p.u.)",
        OUTPUT_DIR / "summer_peak_top_line_loading.png",
        "#1b9e77",
    )
    plot_bar(
        top_transformers.head(20),
        "transformer",
        "max_loading_pu",
        "Top transformer loading during summer peak",
        "Maximum loading (p.u.)",
        OUTPUT_DIR / "summer_peak_top_transformer_loading.png",
        "#7570b3",
    )
    plot_bar(
        import_diag.head(20),
        "import_bus",
        "max_import_cap_utilization",
        "Import cap utilization during summer peak",
        "Maximum import utilization (p.u.)",
        OUTPUT_DIR / "summer_peak_import_cap_utilization.png",
        "#e7298a",
    )

    total_import_cap = import_selection["import_p_nom_mw"].sum()
    used_boundary = import_diag[
        import_diag["import_type"].astype(str).str.contains("boundary", case=False, na=False)
    ]["total_import_slack_mwh"].sum()
    used_artifact = import_diag[
        import_diag["import_type"].astype(str).str.contains("artifact", case=False, na=False)
    ]["total_import_slack_mwh"].sum()
    max_import_util = import_diag["max_import_cap_utilization"].max()
    binding_imports = int((import_diag["max_import_cap_utilization"] >= 0.999).sum())
    lines_at_limit = int((line_loading["max_loading_pu"] >= 0.999).sum())
    transformers_at_limit = int((transformer_loading["max_loading_pu"] >= 0.999).sum())
    shed_exceeds_local_load = shed[
        shed["total_load_shed_mwh"] > shed["summer_peak_total_load_mwh"].fillna(0.0) + 1e-6
    ].copy()

    report = [
        "# Summer Peak x1.75 Transformer Network Diagnosis",
        "",
        "Case: voltage-layer transformer network, boundary imports plus artifact-island imports, July 28 2025, 24 hours.",
        "",
        "## System Result",
        f"- Total demand: {summary.loc[0, 'total_demand_mwh']:,.1f} MWh",
        f"- Demand served: {summary.loc[0, 'total_demand_served_mwh']:,.1f} MWh",
        f"- Load shed: {summary.loc[0, 'total_load_shed_mwh']:,.1f} MWh",
        f"- Import slack: {summary.loc[0, 'total_import_slack_mwh']:,.1f} MWh",
        f"- Boundary import used: {used_boundary:,.1f} MWh",
        f"- Artifact-island import used: {used_artifact:,.1f} MWh",
        f"- Total installed import cap represented: {total_import_cap:,.1f} MW",
        "",
        "## Bottleneck Checks",
        f"- Lines at or near their limit: {lines_at_limit}",
        f"- Transformers at or near their limit: {transformers_at_limit}",
        f"- Import buses at or near their cap: {binding_imports}",
        f"- Highest import cap utilization: {max_import_util:.3f} p.u.",
        "",
        "## Main Finding",
    ]
    if binding_imports > 0:
        report.append(
            "- The remaining summer peak shedding is strongly tied to capped imports and local deliverability during the highest-load day."
        )
    else:
        report.append(
            "- The remaining shedding is not explained by explicit import caps binding; inspect local generation availability and bus-level deliverability."
        )
    if transformers_at_limit == 0:
        report.append(
            "- The new voltage-layer transformers are not acting as artificial bottlenecks in this run."
        )
    if lines_at_limit > 0:
        report.append(
            "- Several lines are exactly at their limits, so line deliverability is still part of the peak constraint."
        )
    if not shed_exceeds_local_load.empty:
        report.append(
            "- Important modelling check: some load-shedding generators dispatch more energy than the local load assigned to their bus. This means load shedding can act like high-cost local emergency generation unless it is capped by each bus load profile."
        )
    report.extend(
        [
            "",
            "## Top Load-Shed Buses",
            shed[["bus", "total_load_shed_mwh", "max_hourly_load_shed_mw", "summer_peak_total_load_mwh", "shed_to_local_load_ratio", "substation_name", "component_id"]]
            .head(10)
            .to_string(index=False),
            "",
            "## Load-Shedding Generator Cap Check",
            f"- Buses where reported load shedding exceeds assigned local 24-hour load: {len(shed_exceeds_local_load)}",
            "- Recommended next model fix: cap each load-shedding generator by its bus load profile so it cannot export artificial high-cost power to neighboring buses.",
            "",
            "## Files Written",
            "- `summer_peak_load_shed_bus_diagnosis.csv`",
            "- `summer_peak_import_cap_utilization.csv`",
            "- `summer_peak_top_line_loading.csv`",
            "- `summer_peak_top_transformer_loading.csv`",
            "- `summer_peak_incident_lines_near_shed_buses.csv`",
            "- `summer_peak_incident_transformers_near_shed_buses.csv`",
            "- `summer_peak_generator_availability_by_carrier.csv`",
            "- PNG diagnostic plots in this folder",
        ]
    )
    (OUTPUT_DIR / "SUMMER_PEAK_X1P75_DIAGNOSIS_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote diagnosis to {OUTPUT_DIR}")
    print(f"Load shed: {summary.loc[0, 'total_load_shed_mwh']:,.1f} MWh")
    print(f"Lines near limit: {lines_at_limit}")
    print(f"Transformers near limit: {transformers_at_limit}")
    print(f"Import buses near cap: {binding_imports}")
    print(f"Max import utilization: {max_import_util:.3f} p.u.")


if __name__ == "__main__":
    main()
