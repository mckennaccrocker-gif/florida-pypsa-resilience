from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
NETWORK_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_extended_tie_lines"
BASELINE_DIR = NETWORK_DIR / "baseline_calibrated_no_hazard"
SCENARIO_DIR = NETWORK_DIR / "ibtracs_cascading_top5_direct_events"
OUT_DIR = SCENARIO_DIR / "summary_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def event_id_from_scenario(scenario_id: str) -> str:
    parts = str(scenario_id).split("_")
    return parts[1] if len(parts) > 1 else str(scenario_id)


def main() -> None:
    baseline = pd.read_csv(BASELINE_DIR / "baseline_summary.csv").iloc[0]
    scenarios = pd.read_csv(SCENARIO_DIR / "all_scenario_summary.csv")
    scenarios["event_id"] = scenarios["scenario_id"].map(event_id_from_scenario)
    scenarios["baseline_system_cost_usd"] = float(baseline["total_system_cost_usd"])
    scenarios["incremental_system_cost_usd"] = (
        pd.to_numeric(scenarios["total_system_cost_usd"], errors="coerce")
        - float(baseline["total_system_cost_usd"])
    )
    scenarios["baseline_load_shed_mwh"] = float(baseline["total_load_shed_mwh"])
    scenarios["incremental_load_shed_mwh"] = (
        pd.to_numeric(scenarios["total_load_shed_mwh"], errors="coerce")
        - float(baseline["total_load_shed_mwh"])
    )
    scenarios["baseline_import_slack_mwh"] = float(baseline["total_import_slack_mwh"])
    scenarios["incremental_import_slack_mwh"] = (
        pd.to_numeric(scenarios["total_import_slack_mwh"], errors="coerce")
        - float(baseline["total_import_slack_mwh"])
    )
    scenarios = scenarios.sort_values(
        ["incremental_load_shed_mwh", "incremental_system_cost_usd"],
        ascending=False,
    )
    scenarios.to_csv(OUT_DIR / "storm_level_cascading_operational_summary.csv", index=False)

    baseline_ls = pd.read_csv(BASELINE_DIR / "baseline_load_shedding_by_bus.csv")
    baseline_ls = baseline_ls.set_index("bus")["total_load_shed_mwh"] if not baseline_ls.empty else pd.Series(dtype=float)

    load_shed_rows = []
    corridor_rows = []
    for scenario_id in scenarios["scenario_id"]:
        folder = SCENARIO_DIR / scenario_id
        if not folder.exists():
            folder = SCENARIO_DIR / scenario_id.replace(":", "_")
        event_id = event_id_from_scenario(scenario_id)

        ls_path = folder / "load_shedding_by_bus.csv"
        if ls_path.exists():
            ls = pd.read_csv(ls_path)
            if not ls.empty:
                ls["event_id"] = event_id
                ls["scenario_id"] = scenario_id
                ls["baseline_total_load_shed_mwh"] = ls["bus"].map(baseline_ls).fillna(0.0)
                ls["incremental_load_shed_mwh"] = (
                    pd.to_numeric(ls["total_load_shed_mwh"], errors="coerce")
                    - ls["baseline_total_load_shed_mwh"]
                )
                load_shed_rows.append(ls)

        line_path = folder / "line_loading.csv"
        if line_path.exists():
            line_loading = pd.read_csv(line_path)
            if not line_loading.empty:
                line_loading["event_id"] = event_id
                line_loading["scenario_id"] = scenario_id
                corridor_rows.append(line_loading.sort_values("max_loading_pu", ascending=False).head(25))

    if load_shed_rows:
        load_shed = pd.concat(load_shed_rows, ignore_index=True)
        load_shed.sort_values(
            ["incremental_load_shed_mwh", "total_load_shed_mwh"],
            ascending=False,
        ).to_csv(OUT_DIR / "cascade_load_shedding_buses_by_event.csv", index=False)
        (
            load_shed.groupby("bus", dropna=False)
            .agg(
                total_load_shed_mwh_across_scenarios=("total_load_shed_mwh", "sum"),
                max_event_load_shed_mwh=("total_load_shed_mwh", "max"),
                total_incremental_load_shed_mwh=("incremental_load_shed_mwh", "sum"),
                events_with_load_shedding=("event_id", "nunique"),
            )
            .reset_index()
            .sort_values(
                ["total_incremental_load_shed_mwh", "total_load_shed_mwh_across_scenarios"],
                ascending=False,
            )
            .to_csv(OUT_DIR / "top_cascade_load_shedding_buses_across_scenarios.csv", index=False)
        )

    if corridor_rows:
        corridors = pd.concat(corridor_rows, ignore_index=True)
        corridors.to_csv(OUT_DIR / "top_loaded_corridors_by_event.csv", index=False)

    print("Saved cascade summaries:", OUT_DIR)
    print(scenarios[[
        "event_id",
        "damaged_lines",
        "damaged_generators",
        "damaged_buses",
        "incremental_load_shed_mwh",
        "incremental_import_slack_mwh",
        "incremental_system_cost_usd",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
