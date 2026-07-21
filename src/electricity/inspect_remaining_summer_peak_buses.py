"""Inspect the remaining summer-peak load-shedding buses.

This diagnostic focuses on the local-adjusted network after the first load
redistribution pass. It summarizes whether the remaining x3 no-hazard shedding
looks like load allocation, topology/voltage-layer, or genuine deliverability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
NETWORK_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_county_load_local_adjusted"
DIAG_DIR = NETWORK_DIR / "summer_peak_multiplier_sweep" / "x3_remaining_shedding_diagnosis"
CASE_DIR = NETWORK_DIR / "summer_peak_multiplier_sweep" / "x3" / "boundary_imports_plus_artifact_islands"
OUTPUT_DIR = NETWORK_DIR / "remaining_summer_peak_bus_inspection"

TARGET_BUSES = ["bus_751", "bus_918", "bus_311"]
START = pd.Timestamp("2025-07-28 00:00:00")
END = pd.Timestamp("2025-07-28 23:00:00")


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def classify_issue(row: pd.Series) -> str:
    if row["v_nom"] <= 69 and row["incident_binding_lines"] >= 1:
        return "likely low-voltage tap / load-allocation artifact"
    if row["mixed_voltage_incident_lines"] > 0 and row["parallel_low_voltage_binding_lines"] > 0:
        return "likely voltage-layer topology artifact"
    if row["incident_binding_lines"] >= 1:
        return "local deliverability bottleneck"
    return "needs manual review"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    buses = pd.read_csv(NETWORK_DIR / "buses.csv").set_index("name")
    lines = pd.read_csv(NETWORK_DIR / "lines.csv").set_index("name")
    loads = pd.read_csv(NETWORK_DIR / "loads.csv").set_index("name")
    weights = pd.read_csv(NETWORK_DIR / "bus_county_population_load_weights.csv")
    shed = pd.read_csv(DIAG_DIR / "x3_summer_peak_load_shed_bus_diagnostics.csv").set_index("bus")
    incident = pd.read_csv(DIAG_DIR / "x3_summer_peak_incident_line_diagnostics.csv")
    nearest_gens = pd.read_csv(DIAG_DIR / "x3_summer_peak_nearest_generators.csv")
    line_loading = pd.read_csv(CASE_DIR / "baseline_line_loading.csv").set_index("line")

    target_loads = loads[loads["bus"].isin(TARGET_BUSES)].index.tolist()
    load_p = pd.read_csv(
        NETWORK_DIR / "loads-p_set.csv",
        usecols=["snapshot", *target_loads],
    )
    load_p["snapshot"] = pd.to_datetime(load_p["snapshot"], errors="raise")
    load_p = load_p[(load_p["snapshot"] >= START) & (load_p["snapshot"] <= END)]
    peak_by_load = load_p.drop(columns=["snapshot"]).max().to_dict()
    total_by_load = load_p.drop(columns=["snapshot"]).sum().to_dict()

    rows = []
    candidates = []
    for bus in TARGET_BUSES:
        bus_info = buses.loc[bus]
        bus_incident = lines[(lines["bus0"].eq(bus)) | (lines["bus1"].eq(bus))].copy()
        bus_incident["line"] = bus_incident.index
        bus_incident = bus_incident.join(line_loading[["max_loading_pu", "max_abs_p0_mw"]], on="line")
        bus_incident["is_binding"] = bus_incident["max_loading_pu"].fillna(0) >= 0.999
        bus_incident["is_mixed_voltage"] = pd.to_numeric(bus_incident["v_nom"], errors="coerce") != float(bus_info["v_nom"])
        bus_incident["parallel_low_voltage_binding"] = bus_incident["is_binding"] & (
            pd.to_numeric(bus_incident["v_nom"], errors="coerce") < float(bus_info["v_nom"])
        )

        bus_loads = loads[loads["bus"].eq(bus)].copy()
        load_names = bus_loads.index.tolist()
        peak_load = sum(float(peak_by_load.get(load, 0.0)) for load in load_names)
        total_load = sum(float(total_by_load.get(load, 0.0)) for load in load_names)

        nearest = nearest_gens[nearest_gens["shed_bus"].eq(bus)].sort_values("distance_km").head(3)
        nearest_desc = "; ".join(
            f"{r.generator} ({r.carrier}, {r.p_nom_mw:.1f} MW, {r.distance_km:.1f} km)"
            for r in nearest.itertuples()
        )

        weight_row = weights[weights["name"].eq(bus)]
        county = weight_row["county_name"].iloc[0] if not weight_row.empty else ""

        target = {
            "bus": bus,
            "county_name": county,
            "v_nom": float(bus_info["v_nom"]),
            "connected_edge_count": int(bus_info["connected_edge_count"]),
            "substation_name": bus_info.get("substation_name", ""),
            "x": float(bus_info["x"]),
            "y": float(bus_info["y"]),
            "summer_peak_load_mw": peak_load,
            "summer_day_load_mwh": total_load,
            "x3_load_shed_mwh": float(shed.loc[bus, "total_load_shed_mwh"]),
            "x3_max_hourly_shed_mw": float(shed.loc[bus, "max_hourly_load_shed_mw"]),
            "incident_line_count": len(bus_incident),
            "incident_binding_lines": int(bus_incident["is_binding"].sum()),
            "mixed_voltage_incident_lines": int(bus_incident["is_mixed_voltage"].sum()),
            "parallel_low_voltage_binding_lines": int(bus_incident["parallel_low_voltage_binding"].sum()),
            "binding_line_ids": "; ".join(bus_incident.loc[bus_incident["is_binding"], "line"].astype(str)),
            "nearest_generators": nearest_desc,
        }
        target["interpretation"] = classify_issue(pd.Series(target))
        rows.append(target)

        if county:
            county_buses = weights[weights["county_name"].eq(county)].merge(
                buses[["v_nom", "connected_edge_count", "x", "y", "substation_name"]],
                left_on="name",
                right_index=True,
                how="left",
                suffixes=("", "_current"),
            )
        else:
            county_buses = weights.merge(
                buses[["v_nom", "connected_edge_count", "x", "y", "substation_name"]],
                left_on="name",
                right_index=True,
                how="left",
                suffixes=("", "_current"),
            )

        county_buses = county_buses[county_buses["name"].ne(bus)].copy()
        county_buses["distance_km"] = haversine_km(
            float(bus_info["x"]),
            float(bus_info["y"]),
            county_buses["x_current"],
            county_buses["y_current"],
        )
        county_buses["voltage_score"] = pd.to_numeric(county_buses["v_nom_current"], errors="coerce").fillna(0)
        county_buses["candidate_score"] = (
            county_buses["voltage_score"] / 500.0
            + pd.to_numeric(county_buses["connected_edge_count_current"], errors="coerce").fillna(0) / 20.0
            - county_buses["distance_km"] / 100.0
        )
        county_buses = county_buses[
            (county_buses["distance_km"] <= 40.0)
            & (county_buses["voltage_score"] >= min(float(bus_info["v_nom"]), 230.0))
        ].sort_values("candidate_score", ascending=False)
        county_buses["source_bus"] = bus
        candidates.append(county_buses.head(12))

        bus_incident.to_csv(OUTPUT_DIR / f"{bus}_incident_lines.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "remaining_shed_bus_inspection_summary.csv", index=False)
    if candidates:
        pd.concat(candidates, ignore_index=True).to_csv(
            OUTPUT_DIR / "remaining_shed_bus_candidate_recipient_buses.csv",
            index=False,
        )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.bar(summary["bus"], summary["x3_load_shed_mwh"], color="#d95f02")
    for _, row in summary.iterrows():
        ax.text(row["bus"], row["x3_load_shed_mwh"], f"{row['x3_load_shed_mwh']:.1f}", ha="center", va="bottom")
    ax.set_title("Remaining Summer Peak x3 Load Shed After Local Load Adjustment")
    ax.set_ylabel("Load shed (MWh)")
    ax.set_xlabel("Bus")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "remaining_shed_bus_inspection_summary.png")
    plt.close(fig)

    report = "# Remaining Summer Peak Bus Inspection\n\n"
    report += "Network inspected:\n\n"
    report += f"`{NETWORK_DIR}`\n\n"
    report += "## Findings\n\n"
    for row in summary.itertuples(index=False):
        report += f"### {row.bus}\n\n"
        report += f"- County: {row.county_name}\n"
        report += f"- Voltage: {row.v_nom:.0f} kV\n"
        report += f"- Connected lines: {row.connected_edge_count}\n"
        report += f"- Summer peak local load: {row.summer_peak_load_mw:.3f} MW\n"
        report += f"- x3 summer peak load shed: {row.x3_load_shed_mwh:.3f} MWh\n"
        report += f"- Binding incident lines: {row.binding_line_ids}\n"
        report += f"- Mixed-voltage incident lines: {row.mixed_voltage_incident_lines}\n"
        report += f"- Nearest generators: {row.nearest_generators}\n"
        report += f"- Interpretation: **{row.interpretation}**\n\n"

    report += "## Recommended Treatment\n\n"
    report += (
        "- `bus_311` is the clearest load-allocation/topology-artifact candidate because it is a 69 kV tap with "
        "very small load and binding 69 kV incident lines.\n"
    )
    report += (
        "- `bus_751` is a stronger 230 kV bus, but its remaining shedding is tied to the 69 kV path through `bus_311`; "
        "inspect the Marion County voltage-layer topology before moving much more load.\n"
    )
    report += (
        "- `bus_918` is physically close to major generation at `bus_9` but still binds low-voltage lines between the same "
        "area buses, which suggests a voltage-layer/topology representation issue rather than missing generation.\n"
    )
    report += (
        "- Do not use this as evidence for a blanket `s_nom x3` official baseline yet. The next better fix is targeted "
        "voltage-layer cleanup or second-pass local load redistribution for these three pockets.\n"
    )
    (OUTPUT_DIR / "remaining_shed_bus_inspection_report.md").write_text(report, encoding="utf-8")

    print("Saved inspection:", OUTPUT_DIR)
    print(summary[["bus", "x3_load_shed_mwh", "v_nom", "binding_line_ids", "interpretation"]].to_string(index=False))


if __name__ == "__main__":
    main()
