"""
Interpret gradual return-period PyPSA hazard results and calculate annualized risk.

This post-processor does not rerun PyPSA. It reads the gradual return-period
suite outputs, ranks repeatedly affected assets, summarizes load-shed locations,
creates maps, and computes expected annual load shed/cost from the available
return-period curves.

Criticality notes:
  - Line local load-shed association is approximated as load shed at the line's
    endpoint buses in the same scenario.
  - Generator local load-shed association is approximated as load shed at the
    generator bus in the same scenario.
  - These are screening metrics, not individual counterfactual outage impacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

from run_florida_pypsa_calibrated_hazard_scenarios import (
    PROJECT_DIR,
    load_calibrated_network,
)


PYPSA_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network"
SUITE_DIR = PYPSA_DIR / "calibrated_hazard_scenarios" / "gradual_return_period_suite"
SUMMARY_FILE = SUITE_DIR / "gradual_return_period_suite_summary.csv"
OUTPUT_DIR = SUITE_DIR / "interpretation_and_annualized_risk"
NETWORK_DIR = PYPSA_DIR
LINE_CAPACITY_MULTIPLIER = 1.75

LINE_GEOMETRY = PROJECT_DIR / "data" / "Electricity" / "florida_lines_with_s_nom.gpkg"
FLORIDA_CRS = "EPSG:4326"
PLOT_CRS = "EPSG:3086"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_suite_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_FILE)
    summary["annual_exceedance_probability"] = 1.0 / summary["return_period"].astype(float)
    return summary


def load_bus_component_lookup(network) -> pd.DataFrame:
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index.astype(str))
    active_lines = network.lines[network.lines.get("active", True).fillna(True)].copy()
    graph.add_edges_from(zip(active_lines["bus0"].astype(str), active_lines["bus1"].astype(str)))

    rows = []
    for component_id, buses in enumerate(nx.connected_components(graph)):
        for bus in buses:
            rows.append({"bus": bus, "base_component_id": component_id})
    return pd.DataFrame(rows)


def load_network_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    network = load_calibrated_network(NETWORK_DIR, LINE_CAPACITY_MULTIPLIER)
    buses = network.buses.reset_index(names="bus")[["bus", "x", "y"]]
    buses = buses.merge(load_bus_component_lookup(network), on="bus", how="left")

    lines = network.lines.reset_index(names="line")[
        ["line", "bus0", "bus1", "v_nom", "s_nom", "source_edge_id"]
    ].copy()
    lines["source_edge_id"] = pd.to_numeric(lines["source_edge_id"], errors="coerce")

    generators = network.generators[
        ~network.generators["carrier"].isin(["import_slack", "load_shedding"])
    ].reset_index(names="generator")
    generators = generators[["generator", "bus", "carrier", "p_nom", "source_name"]]
    generators = generators.merge(buses, on="bus", how="left")
    return buses, lines, generators


def load_line_geometries(lines: pd.DataFrame) -> gpd.GeoDataFrame:
    source = gpd.read_file(LINE_GEOMETRY)
    if "florida_line_id" not in source.columns:
        raise ValueError(f"{LINE_GEOMETRY} must contain florida_line_id.")
    source["florida_line_id"] = pd.to_numeric(source["florida_line_id"], errors="coerce")
    geo = lines.merge(
        source[["florida_line_id", "geometry"]],
        left_on="source_edge_id",
        right_on="florida_line_id",
        how="left",
    )
    return gpd.GeoDataFrame(geo, geometry="geometry", crs=source.crs or FLORIDA_CRS)


def scenario_dirs(summary: pd.DataFrame) -> list[Path]:
    return [SUITE_DIR / scenario_id for scenario_id in summary["scenario_id"]]


def load_deratings(summary: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    frames = []
    for _, row in summary.iterrows():
        scenario_id = row["scenario_id"]
        path = SUITE_DIR / scenario_id / f"{asset_type}_capacity_deratings.csv"
        df = safe_read_csv(path)
        if df.empty:
            continue
        df["scenario_id"] = scenario_id
        df["hazard"] = row["hazard"]
        df["return_period"] = int(row["return_period"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_load_shedding(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in summary.iterrows():
        scenario_id = row["scenario_id"]
        path = SUITE_DIR / scenario_id / "load_shedding_by_bus.csv"
        df = safe_read_csv(path)
        if df.empty:
            continue
        df["scenario_id"] = scenario_id
        df["hazard"] = row["hazard"]
        df["return_period"] = int(row["return_period"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "bus",
                "total_load_shed_mwh",
                "max_hourly_load_shed_mw",
                "scenario_id",
                "hazard",
                "return_period",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def summarize_buses(load_shed: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
    if load_shed.empty:
        return pd.DataFrame()
    bus_summary = (
        load_shed.groupby(["hazard", "bus"], as_index=False)
        .agg(
            affected_return_periods=("return_period", "nunique"),
            first_affected_return_period=("return_period", "min"),
            total_load_shed_mwh=("total_load_shed_mwh", "sum"),
            max_scenario_load_shed_mwh=("total_load_shed_mwh", "max"),
            max_hourly_load_shed_mw=("max_hourly_load_shed_mw", "max"),
        )
        .merge(buses, on="bus", how="left")
        .sort_values(["hazard", "total_load_shed_mwh"], ascending=[True, False])
    )
    bus_summary.to_csv(OUTPUT_DIR / "top_load_shed_buses_by_hazard.csv", index=False)

    island_summary = (
        bus_summary.groupby(["hazard", "base_component_id"], as_index=False)
        .agg(
            buses_with_load_shed=("bus", "nunique"),
            total_load_shed_mwh=("total_load_shed_mwh", "sum"),
            max_bus_load_shed_mwh=("max_scenario_load_shed_mwh", "max"),
            mean_x=("x", "mean"),
            mean_y=("y", "mean"),
        )
        .sort_values(["hazard", "total_load_shed_mwh"], ascending=[True, False])
    )
    island_summary.to_csv(OUTPUT_DIR / "load_shed_by_base_network_component.csv", index=False)
    return bus_summary


def endpoint_load_shed_lookup(load_shed: pd.DataFrame) -> pd.DataFrame:
    if load_shed.empty:
        return pd.DataFrame(columns=["scenario_id", "bus", "total_load_shed_mwh"])
    return load_shed[["scenario_id", "bus", "total_load_shed_mwh"]].copy()


def summarize_lines(
    line_deratings: pd.DataFrame,
    lines: pd.DataFrame,
    load_shed: pd.DataFrame,
) -> pd.DataFrame:
    if line_deratings.empty:
        return pd.DataFrame()
    line_df = line_deratings.copy()
    line_df["capacity_loss_mva"] = pd.to_numeric(line_df["capacity_loss_mva"], errors="coerce").fillna(0.0)
    line_df["damage_ratio"] = pd.to_numeric(line_df["damage_ratio"], errors="coerce").fillna(0.0)
    merge_cols = ["line", "v_nom", "source_edge_id"]
    if "bus0" not in line_df.columns:
        merge_cols.append("bus0")
    if "bus1" not in line_df.columns:
        merge_cols.append("bus1")
    line_df = line_df.merge(lines[merge_cols], on="line", how="left")

    bus_loss = endpoint_load_shed_lookup(load_shed)
    bus0_loss = bus_loss.rename(
        columns={"bus": "bus0", "total_load_shed_mwh": "bus0_load_shed_mwh"}
    )
    bus1_loss = bus_loss.rename(
        columns={"bus": "bus1", "total_load_shed_mwh": "bus1_load_shed_mwh"}
    )
    line_df = line_df.merge(bus0_loss, on=["scenario_id", "bus0"], how="left")
    line_df = line_df.merge(bus1_loss, on=["scenario_id", "bus1"], how="left")
    line_df["endpoint_load_shed_mwh"] = (
        line_df["bus0_load_shed_mwh"].fillna(0.0) + line_df["bus1_load_shed_mwh"].fillna(0.0)
    )

    summary = (
        line_df.groupby(["hazard", "line"], as_index=False)
        .agg(
            affected_return_periods=("return_period", "nunique"),
            first_affected_return_period=("return_period", "min"),
            fully_failed_return_periods=("damage_ratio", lambda s: int((s >= 1.0).sum())),
            max_damage_ratio=("damage_ratio", "max"),
            total_capacity_loss_mva=("capacity_loss_mva", "sum"),
            max_capacity_loss_mva=("capacity_loss_mva", "max"),
            associated_endpoint_load_shed_mwh=("endpoint_load_shed_mwh", "sum"),
        )
        .merge(lines, on="line", how="left")
        .sort_values(
            ["hazard", "total_capacity_loss_mva", "associated_endpoint_load_shed_mwh"],
            ascending=[True, False, False],
        )
    )
    summary.to_csv(OUTPUT_DIR / "top_critical_lines_by_hazard.csv", index=False)
    return summary


def summarize_generators(
    generator_deratings: pd.DataFrame,
    generators: pd.DataFrame,
    load_shed: pd.DataFrame,
) -> pd.DataFrame:
    if generator_deratings.empty:
        return pd.DataFrame()
    gen_df = generator_deratings.copy()
    gen_df["capacity_loss_mw"] = pd.to_numeric(gen_df["capacity_loss_mw"], errors="coerce").fillna(0.0)
    gen_df["damage_ratio"] = pd.to_numeric(gen_df["damage_ratio"], errors="coerce").fillna(0.0)
    generator_merge_cols = ["generator", "x", "y"]
    for column in ["bus", "carrier", "source_name", "base_component_id"]:
        if column in generators.columns and column not in gen_df.columns:
            generator_merge_cols.append(column)
    gen_df = gen_df.merge(generators[generator_merge_cols], on="generator", how="left")
    bus_loss = endpoint_load_shed_lookup(load_shed).rename(
        columns={"total_load_shed_mwh": "generator_bus_load_shed_mwh"}
    )
    gen_df = gen_df.merge(bus_loss, on=["scenario_id", "bus"], how="left")
    gen_df["generator_bus_load_shed_mwh"] = gen_df["generator_bus_load_shed_mwh"].fillna(0.0)

    summary = (
        gen_df.groupby(["hazard", "generator"], as_index=False)
        .agg(
            carrier=("carrier", "first"),
            source_name=("source_name", "first"),
            bus=("bus", "first"),
            affected_return_periods=("return_period", "nunique"),
            first_affected_return_period=("return_period", "min"),
            fully_failed_return_periods=("damage_ratio", lambda s: int((s >= 1.0).sum())),
            max_damage_ratio=("damage_ratio", "max"),
            total_capacity_loss_mw=("capacity_loss_mw", "sum"),
            max_capacity_loss_mw=("capacity_loss_mw", "max"),
            associated_bus_load_shed_mwh=("generator_bus_load_shed_mwh", "sum"),
            x=("x", "first"),
            y=("y", "first"),
        )
        .sort_values(
            ["hazard", "total_capacity_loss_mw", "associated_bus_load_shed_mwh"],
            ascending=[True, False, False],
        )
    )
    summary.to_csv(OUTPUT_DIR / "top_critical_generators_by_hazard.csv", index=False)
    return summary


def analyze_return_period_shape(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hazard, group in summary.groupby("hazard"):
        group = group.sort_values("return_period")
        rp10 = group[group["return_period"].eq(10)]
        rp100 = group[group["return_period"].eq(100)]
        rp500 = group[group["return_period"].eq(500)]
        rows.append(
            {
                "hazard": hazard,
                "rp10_load_shed_mwh": float(rp10["load_shed_mwh"].iloc[0]) if not rp10.empty else np.nan,
                "rp100_load_shed_mwh": float(rp100["load_shed_mwh"].iloc[0]) if not rp100.empty else np.nan,
                "rp500_load_shed_mwh": float(rp500["load_shed_mwh"].iloc[0]) if not rp500.empty else np.nan,
                "rp100_to_rp500_increment_mwh": float(rp500["load_shed_mwh"].iloc[0] - rp100["load_shed_mwh"].iloc[0])
                if not rp100.empty and not rp500.empty
                else np.nan,
                "interpretation": interpretation_text(hazard),
            }
        )
    shape = pd.DataFrame(rows)
    shape.to_csv(OUTPUT_DIR / "return_period_curve_interpretation.csv", index=False)
    return shape


def interpretation_text(hazard: str) -> str:
    if hazard == "flood":
        return (
            "With F6.2 replacing the previous generalized F6.3 line curve, the flood "
            "return-period scenarios derate exposed lines, generators, and substations "
            "but do not shed load in the calibrated 24-hour dispatch window. The "
            "RP100-to-RP500 increase is therefore expressed mainly through additional "
            "capacity derating and incremental system cost, not unserved energy."
        )
    return (
        "TC RP10 and RP20 report line derating but no generator or bus derating because "
        "the relevant W1.* and W2.3 curves remain at zero damage for those wind speeds. "
        "Higher return periods activate W2.3 substation derating, and RP500 also "
        "activates generator derating and load shedding."
    )


def annualized_metric(
    hazard_df: pd.DataFrame,
    value_column: str,
    metric_name: str,
) -> dict:
    observed = hazard_df.sort_values("annual_exceedance_probability").copy()
    points = observed[["annual_exceedance_probability", value_column, "return_period"]].copy()
    points = points.rename(columns={value_column: "metric_value"})

    max_rp_row = observed.loc[observed["return_period"].idxmax()]
    endpoint_rare = {
        "annual_exceedance_probability": 0.0,
        "metric_value": float(max_rp_row[value_column]),
        "return_period": np.inf,
        "point_type": "rare_tail_constant_at_largest_available_rp",
    }
    endpoint_frequent = {
        "annual_exceedance_probability": 1.0,
        "metric_value": 0.0,
        "return_period": 1.0,
        "point_type": "frequent_endpoint_zero_damage",
    }
    points["point_type"] = "available_return_period"
    curve = pd.concat([pd.DataFrame([endpoint_rare]), points, pd.DataFrame([endpoint_frequent])], ignore_index=True)
    curve = curve.sort_values("annual_exceedance_probability")
    expected = float(np.trapezoid(curve["metric_value"], curve["annual_exceedance_probability"]))
    return {
        "metric": metric_name,
        "expected_annual_value": expected,
        "curve": curve,
    }


def calculate_annualized_risk(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    curve_rows = []
    metrics = [
        ("load_shed_mwh", "expected_annual_load_shed_mwh_per_year"),
        ("system_cost_usd", "expected_annual_system_cost_usd_per_year"),
        ("incremental_system_cost_usd", "expected_annual_incremental_system_cost_usd_per_year"),
    ]
    for hazard, group in summary.groupby("hazard"):
        row = {
            "hazard": hazard,
            "integration_method": "trapezoidal_area_under_exceedance_curve",
            "endpoints": "p=0 uses largest available RP value; p=1 uses zero damage/cost",
            "missing_return_period_treatment": "No missing RP scenarios were invented; trapezoidal interpolation uses available adjacent return periods.",
        }
        for value_column, metric_name in metrics:
            result = annualized_metric(group, value_column, metric_name)
            row[metric_name] = result["expected_annual_value"]
            curve = result["curve"].copy()
            curve["hazard"] = hazard
            curve["metric"] = metric_name
            curve_rows.append(curve)
        rows.append(row)

    risk = pd.DataFrame(rows)
    risk.to_csv(OUTPUT_DIR / "annualized_risk_summary.csv", index=False)
    curves = pd.concat(curve_rows, ignore_index=True)
    curves.to_csv(OUTPUT_DIR / "annualized_risk_exceedance_curve_points.csv", index=False)
    return risk


def plot_top_assets_map(
    lines_geo: gpd.GeoDataFrame,
    generators: pd.DataFrame,
    buses: pd.DataFrame,
    line_summary: pd.DataFrame,
    generator_summary: pd.DataFrame,
    bus_summary: pd.DataFrame,
    hazard: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 9))

    line_base = lines_geo.dropna(subset=["geometry"]).to_crs(PLOT_CRS)
    line_base.plot(ax=ax, color="#d8d8d8", linewidth=0.25, alpha=0.55)

    top_lines = line_summary[line_summary["hazard"].eq(hazard)].copy()
    top_lines = top_lines.sort_values(
        ["affected_return_periods", "total_capacity_loss_mva", "associated_endpoint_load_shed_mwh"],
        ascending=False,
    ).head(50)
    if not top_lines.empty:
        top_lines_geo = line_base.merge(top_lines[["line", "affected_return_periods", "total_capacity_loss_mva"]], on="line")
        top_lines_geo.plot(
            ax=ax,
            column="affected_return_periods",
            cmap="magma",
            linewidth=1.6,
            legend=True,
            legend_kwds={"label": "Affected RPs"},
        )

    top_gens = generator_summary[generator_summary["hazard"].eq(hazard)].copy()
    top_gens = top_gens.sort_values(
        ["affected_return_periods", "total_capacity_loss_mw", "associated_bus_load_shed_mwh"],
        ascending=False,
    ).head(30)
    if not top_gens.empty:
        gen_geo = gpd.GeoDataFrame(
            top_gens,
            geometry=[Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None for xy in zip(top_gens["x"], top_gens["y"])],
            crs=FLORIDA_CRS,
        ).dropna(subset=["geometry"]).to_crs(PLOT_CRS)
        gen_geo.plot(
            ax=ax,
            markersize=np.clip(gen_geo["total_capacity_loss_mw"] / 15.0, 20, 260),
            color="#ff7f2a",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label="Top generators",
        )

    top_buses = bus_summary[bus_summary["hazard"].eq(hazard)].head(50).copy()
    if not top_buses.empty:
        bus_geo = gpd.GeoDataFrame(
            top_buses,
            geometry=[Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None for xy in zip(top_buses["x"], top_buses["y"])],
            crs=FLORIDA_CRS,
        ).dropna(subset=["geometry"]).to_crs(PLOT_CRS)
        bus_geo.plot(
            ax=ax,
            markersize=np.clip(bus_geo["total_load_shed_mwh"] / 400.0, 12, 180),
            color="#1464f4",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.75,
            label="Top load-shed buses",
        )

    ax.set_axis_off()
    ax.set_title(f"{hazard.replace('_', ' ').title()} repeatedly affected / operationally important assets")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{hazard}_critical_assets_map.png", dpi=250)
    plt.close(fig)


def create_maps(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    generators: pd.DataFrame,
    line_summary: pd.DataFrame,
    generator_summary: pd.DataFrame,
    bus_summary: pd.DataFrame,
) -> None:
    lines_geo = load_line_geometries(lines)
    for hazard in ["flood", "tropical_cyclone"]:
        plot_top_assets_map(lines_geo, generators, buses, line_summary, generator_summary, bus_summary, hazard)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze gradual return-period results and calculate annualized risk."
    )
    parser.add_argument("--suite-dir", type=Path, default=SUITE_DIR)
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--line-capacity-multiplier", type=float, default=LINE_CAPACITY_MULTIPLIER)
    return parser.parse_args()


def main() -> None:
    global SUITE_DIR, SUMMARY_FILE, OUTPUT_DIR, NETWORK_DIR, LINE_CAPACITY_MULTIPLIER
    args = parse_args()
    SUITE_DIR = args.suite_dir
    SUMMARY_FILE = SUITE_DIR / "gradual_return_period_suite_summary.csv"
    OUTPUT_DIR = SUITE_DIR / "interpretation_and_annualized_risk"
    NETWORK_DIR = args.network_dir
    LINE_CAPACITY_MULTIPLIER = args.line_capacity_multiplier

    ensure_output_dir()
    summary = load_suite_summary()
    buses, lines, generators = load_network_tables()
    generators = generators.merge(
        buses[["bus", "base_component_id"]],
        on="bus",
        how="left",
    )

    line_deratings = load_deratings(summary, "line")
    generator_deratings = load_deratings(summary, "generator")
    load_shed = load_load_shedding(summary)

    bus_summary = summarize_buses(load_shed, buses)
    line_summary = summarize_lines(line_deratings, lines, load_shed)
    generator_summary = summarize_generators(generator_deratings, generators, load_shed)
    shape = analyze_return_period_shape(summary)
    risk = calculate_annualized_risk(summary)
    create_maps(buses, lines, generators, line_summary, generator_summary, bus_summary)

    print("\nTop lines by capacity loss:")
    print(line_summary.groupby("hazard").head(10).to_string(index=False, max_cols=12))
    print("\nTop generators by capacity loss:")
    print(generator_summary.groupby("hazard").head(10).to_string(index=False, max_cols=12))
    print("\nTop load-shed buses:")
    print(bus_summary.groupby("hazard").head(10).to_string(index=False))
    print("\nReturn-period curve interpretation:")
    print(shape.to_string(index=False))
    print("\nAnnualized risk summary:")
    print(risk.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print("\nSaved interpretation outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
