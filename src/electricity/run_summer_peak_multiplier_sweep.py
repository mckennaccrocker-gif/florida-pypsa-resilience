"""Sweep line-capacity multipliers on the summer peak no-hazard day.

This follows up on the seasonal validation result showing no-hazard load
shedding during the summer peak window. The goal is to identify whether a
larger ``s_nom`` multiplier alone removes that shedding, or whether other
constraints such as generator availability, import placement, or load
allocation need attention.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_NETWORK_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_county_load_generator_overrides"
BOUNDARY_BUSES = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network" / "boundary_import_buses" / "selected_boundary_import_buses.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_NETWORK_DIR / "summer_peak_multiplier_sweep"

SUMMER_PEAK_START = "2025-07-28 00:00:00"
PERIODS = 24
CHUNK_SIZE = 24
MULTIPLIERS = [1.75, 2.0, 2.25, 2.5, 3.0]
BOUNDARY_IMPORT_CAP_FRACTION = 1.0
ARTIFACT_IMPORT_PEAK_LOAD_MARGIN = 1.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run summer peak line multiplier sweep.")
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--multipliers", nargs="+", type=float, default=MULTIPLIERS)
    return parser.parse_args()


def run_case(network_dir: Path, output_dir: Path, multiplier: float) -> pd.Series:
    case_id = f"x{multiplier:g}".replace(".", "p")
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "run.log"

    cmd = [
        sys.executable,
        "data/Electricity/run_boundary_import_baseline.py",
        "--network-dir",
        str(network_dir),
        "--boundary-buses",
        str(BOUNDARY_BUSES),
        "--output-dir",
        str(case_dir),
        "--line-limit-multiplier",
        str(multiplier),
        "--boundary-import-cap-fraction",
        str(BOUNDARY_IMPORT_CAP_FRACTION),
        "--artifact-import-peak-load-margin",
        str(ARTIFACT_IMPORT_PEAK_LOAD_MARGIN),
        "--add-artifact-island-imports",
        "--start",
        SUMMER_PEAK_START,
        "--periods",
        str(PERIODS),
        "--chunk-size",
        str(CHUNK_SIZE),
        "--solver",
        "highs",
        "--highs-method",
        "ipm",
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    solved_dir = case_dir / "boundary_imports_plus_artifact_islands"
    summary_path = solved_dir / "baseline_summary.csv"
    if not summary_path.exists():
        return pd.Series(
            {
                "case_id": case_id,
                "line_capacity_multiplier": multiplier,
                "return_code": completed.returncode,
                "solver_status": "missing_summary",
                "solver_condition": "missing_summary",
                "case_output_dir": str(case_dir),
            }
        )

    summary = pd.read_csv(summary_path).iloc[0]
    import_by_bus_path = solved_dir / "baseline_import_slack_by_bus.csv"
    import_by_bus = pd.read_csv(import_by_bus_path) if import_by_bus_path.exists() else pd.DataFrame()

    boundary_used_mwh = 0.0
    artifact_used_mwh = 0.0
    import_selection_path = solved_dir / "import_bus_selection.csv"
    if import_selection_path.exists() and not import_by_bus.empty:
        import_selection = pd.read_csv(import_selection_path)
        type_by_bus = import_selection.set_index("import_bus")["import_type"].to_dict()
        import_by_bus["import_type"] = import_by_bus["bus"].map(type_by_bus).fillna("unknown")
        boundary_used_mwh = float(
            import_by_bus.loc[
                import_by_bus["import_type"].eq("northern_boundary_intertie_import"),
                "total_import_slack_mwh",
            ].sum()
        )
        artifact_used_mwh = float(
            import_by_bus.loc[
                import_by_bus["import_type"].eq("topology_artifact_island_import"),
                "total_import_slack_mwh",
            ].sum()
        )

    return pd.Series(
        {
            "case_id": case_id,
            "line_capacity_multiplier": multiplier,
            "return_code": completed.returncode,
            "solver_status": summary.get("solver_status"),
            "solver_condition": summary.get("solver_condition"),
            "total_demand_mwh": summary.get("total_demand_mwh"),
            "demand_served_mwh": summary.get("total_demand_served_mwh"),
            "load_shed_mwh": summary.get("total_load_shed_mwh"),
            "maximum_hourly_load_shed_mw": summary.get("maximum_hourly_load_shed_mw"),
            "buses_experiencing_load_shedding": summary.get("buses_experiencing_load_shedding"),
            "import_slack_mwh": summary.get("total_import_slack_mwh"),
            "boundary_import_slack_mwh": boundary_used_mwh,
            "artifact_import_slack_mwh": artifact_used_mwh,
            "number_overloaded_lines": summary.get("number_overloaded_lines"),
            "max_line_loading_pu": summary.get("max_line_loading_pu"),
            "system_cost_usd": summary.get("total_system_cost_usd"),
            "case_output_dir": str(case_dir),
        }
    )


def plot_results(summary: pd.DataFrame) -> None:
    summary = summary.sort_values("line_capacity_multiplier")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(
        summary["line_capacity_multiplier"],
        summary["load_shed_mwh"],
        marker="o",
        linewidth=2.5,
        color="#d95f02",
    )
    ax.set_title("Summer Peak No-Hazard Load Shedding vs Line-Capacity Multiplier")
    ax.set_xlabel("Line capacity multiplier applied to s_nom")
    ax.set_ylabel("Load shed (MWh, 24-hour summer peak)")
    ax.grid(True, alpha=0.25)
    for _, row in summary.iterrows():
        ax.annotate(
            f"{row['load_shed_mwh']:,.0f}",
            (row["line_capacity_multiplier"], row["load_shed_mwh"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summer_peak_load_shed_multiplier_sweep.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(
        summary["line_capacity_multiplier"],
        summary["import_slack_mwh"],
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="Total import slack",
    )
    ax.plot(
        summary["line_capacity_multiplier"],
        summary["boundary_import_slack_mwh"],
        marker="s",
        linewidth=2,
        color="#2ca02c",
        label="Boundary imports",
    )
    ax.plot(
        summary["line_capacity_multiplier"],
        summary["artifact_import_slack_mwh"],
        marker="^",
        linewidth=2,
        color="#9467bd",
        label="Artifact-island imports",
    )
    ax.set_title("Summer Peak Import Slack vs Line-Capacity Multiplier")
    ax.set_xlabel("Line capacity multiplier applied to s_nom")
    ax.set_ylabel("Import slack (MWh, 24-hour summer peak)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summer_peak_import_slack_multiplier_sweep.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame, network_dir: Path, multipliers: list[float]) -> None:
    summary = summary.sort_values("line_capacity_multiplier")
    first_zero = summary[pd.to_numeric(summary["load_shed_mwh"], errors="coerce") <= 1e-6]

    if first_zero.empty:
        conclusion = (
            "No tested multiplier eliminated summer peak no-hazard load shedding. "
            "This indicates that line-capacity relaxation alone is not enough, or "
            "that the required multiplier would be too large to defend without "
            "additional topology/generator/import validation."
        )
    else:
        multiplier = float(first_zero.iloc[0]["line_capacity_multiplier"])
        conclusion = (
            f"The first tested multiplier that eliminates summer peak no-hazard "
            f"load shedding is `s_nom x {multiplier:g}`."
        )

    report = f"""# Summer Peak Line-Capacity Multiplier Sweep

This sweep tests whether summer peak no-hazard load shedding can be removed by increasing the global line-capacity multiplier.

## Setup

- Network: `{network_dir.name}`
- Summer peak window: `{SUMMER_PEAK_START}` for `{PERIODS}` hours
- Boundary import cap fraction: `{BOUNDARY_IMPORT_CAP_FRACTION}`
- Artifact-island import margin: `{ARTIFACT_IMPORT_PEAK_LOAD_MARGIN} x peak island load`
- Multipliers tested: `{', '.join(str(x) for x in multipliers)}`

## Conclusion

{conclusion}

## Summary

| multiplier | load shed MWh | max hourly shed MW | shed buses | import slack MWh | boundary import MWh | artifact import MWh | max line loading pu | solver |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
"""
    for _, row in summary.iterrows():
        report += (
            f"| {row['line_capacity_multiplier']:.2f} | "
            f"{row['load_shed_mwh']:,.3f} | "
            f"{row['maximum_hourly_load_shed_mw']:,.3f} | "
            f"{int(row['buses_experiencing_load_shedding'])} | "
            f"{row['import_slack_mwh']:,.3f} | "
            f"{row['boundary_import_slack_mwh']:,.3f} | "
            f"{row['artifact_import_slack_mwh']:,.3f} | "
            f"{row['max_line_loading_pu']:,.3f} | "
            f"{row['solver_status']}/{row['solver_condition']} |\n"
        )

    report += """
## Files

- `summer_peak_multiplier_sweep_summary.csv`
- `summer_peak_load_shed_multiplier_sweep.png`
- `summer_peak_import_slack_multiplier_sweep.png`
"""
    (OUTPUT_DIR / "summer_peak_multiplier_sweep_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    global OUTPUT_DIR
    args = parse_args()
    network_dir = args.network_dir
    OUTPUT_DIR = args.output_dir or (network_dir / "summer_peak_multiplier_sweep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for multiplier in args.multipliers:
        print(f"Running summer peak multiplier x{multiplier:g}")
        rows.append(run_case(network_dir, OUTPUT_DIR, multiplier))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summer_peak_multiplier_sweep_summary.csv", index=False)
    plot_results(summary)
    write_report(summary, network_dir, args.multipliers)
    print("Saved summer peak multiplier sweep:", OUTPUT_DIR)
    print(
        summary[
            [
                "line_capacity_multiplier",
                "load_shed_mwh",
                "import_slack_mwh",
                "boundary_import_slack_mwh",
                "artifact_import_slack_mwh",
                "solver_status",
                "solver_condition",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
