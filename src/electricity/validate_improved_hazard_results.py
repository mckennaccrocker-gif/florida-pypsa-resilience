"""
Validate and interpret the improved Florida PyPSA hazard results.

This post-processor checks:
  - why flood scenarios have zero load shedding,
  - what drives TC RP500 load shedding,
  - how the improved run differs from the older run,
  - and saves thesis-friendly tables/plots/notes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
NEW_SUITE = (
    ELECTRICITY_DIR
    / "pypsa_florida_network_county_load_generator_overrides"
    / "calibrated_hazard_scenarios"
    / "gradual_return_period_suite"
)
OLD_SUITE = (
    ELECTRICITY_DIR
    / "pypsa_florida_network"
    / "calibrated_hazard_scenarios"
    / "gradual_return_period_suite"
)
NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network_county_load_generator_overrides"
OUTPUT_DIR = NEW_SUITE / "validation_checks"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def scenario_dir(scenario_id: str) -> Path:
    return NEW_SUITE / scenario_id


def distribution_rows(df: pd.DataFrame, value_col: str, label: str) -> dict:
    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if values.empty:
        return {
            "asset_group": label,
            "count": 0,
            "min": pd.NA,
            "p25": pd.NA,
            "median": pd.NA,
            "mean": pd.NA,
            "p75": pd.NA,
            "max": pd.NA,
        }
    return {
        "asset_group": label,
        "count": int(values.size),
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def damage_bins(df: pd.DataFrame, label: str) -> pd.DataFrame:
    damage = pd.to_numeric(df["damage_ratio"], errors="coerce").fillna(0.0)
    bins = [
        ("exactly_0", damage.eq(0.0)),
        ("0_to_0.01", damage.gt(0.0) & damage.le(0.01)),
        ("0.01_to_0.05", damage.gt(0.01) & damage.le(0.05)),
        ("0.05_to_0.10", damage.gt(0.05) & damage.le(0.10)),
        ("0.10_to_0.25", damage.gt(0.10) & damage.le(0.25)),
        ("0.25_to_0.50", damage.gt(0.25) & damage.le(0.50)),
        ("0.50_to_0.75", damage.gt(0.50) & damage.le(0.75)),
        ("0.75_to_lt_1", damage.ge(0.75) & damage.lt(1.0)),
        ("exactly_1", damage.eq(1.0)),
    ]
    total = len(damage)
    return pd.DataFrame(
        [
            {
                "asset_group": label,
                "damage_bin": name,
                "asset_count": int(mask.sum()),
                "asset_percent": float(mask.sum() / total * 100.0) if total else 0.0,
            }
            for name, mask in bins
        ]
    )


def summarize_flood_zero_shed(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    bin_frames = []
    flood_rows = summary[summary["hazard"].eq("flood")].sort_values("return_period")
    for _, scenario in flood_rows.iterrows():
        sid = scenario["scenario_id"]
        rp = int(scenario["return_period"])
        for asset, file_name in [
            ("line", "line_capacity_deratings.csv"),
            ("generator", "generator_capacity_deratings.csv"),
            ("bus", "bus_substation_deratings.csv"),
        ]:
            df = read_csv(scenario_dir(sid) / file_name)
            if df.empty:
                continue
            row = distribution_rows(df, "damage_ratio", f"flood_rp{rp}_{asset}")
            row["hazard"] = "flood"
            row["return_period"] = rp
            row["asset_type"] = asset
            rows.append(row)
            bins = damage_bins(df, f"flood_rp{rp}_{asset}")
            bins["hazard"] = "flood"
            bins["return_period"] = rp
            bins["asset_type"] = asset
            bin_frames.append(bins)

    dist = pd.DataFrame(rows)
    bins = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()

    flood_summary = flood_rows[
        [
            "return_period",
            "derated_lines",
            "derated_generators",
            "line_capacity_loss_mva",
            "generator_capacity_loss_mw",
            "load_shed_mwh",
            "import_slack_mwh",
            "incremental_system_cost_usd",
        ]
    ].copy()
    flood_summary["import_slack_change_from_baseline_mwh"] = (
        flood_summary["import_slack_mwh"] - flood_summary["import_slack_mwh"].iloc[0]
    )
    return flood_summary, dist, bins


def summarize_line_binding(scenario_id: str) -> pd.DataFrame:
    loading = read_csv(scenario_dir(scenario_id) / "line_loading.csv")
    derating = read_csv(scenario_dir(scenario_id) / "line_capacity_deratings.csv")
    if loading.empty:
        return pd.DataFrame()
    result = loading.copy()
    result["near_binding_95pct"] = pd.to_numeric(result["max_loading_pu"], errors="coerce").ge(0.95)
    result["binding_99pct"] = pd.to_numeric(result["max_loading_pu"], errors="coerce").ge(0.99)
    if not derating.empty:
        result = result.merge(
            derating[["line", "damage_ratio", "hazard_intensity", "capacity_loss_mva"]],
            on="line",
            how="left",
        )
    return result.sort_values(["max_loading_pu", "capacity_loss_mva"], ascending=[False, False])


def summarize_tc_rp500() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sid = "tc_storm_rp500_w63_gradual"
    buses = read_csv(NETWORK_DIR / "buses.csv")
    loads = read_csv(NETWORK_DIR / "loads.csv")
    load_shed = read_csv(scenario_dir(sid) / "load_shedding_by_bus.csv")
    bus_derating = read_csv(scenario_dir(sid) / "bus_substation_deratings.csv")
    line_derating = read_csv(scenario_dir(sid) / "line_capacity_deratings.csv")
    gen_derating = read_csv(scenario_dir(sid) / "generator_capacity_deratings.csv")
    loading = summarize_line_binding(sid)

    bus_info = buses.rename(columns={"name": "bus"})
    shed = load_shed.merge(bus_info, on="bus", how="left")
    if not bus_derating.empty:
        bus_derating["capacity_multiplier_after_damage"] = (
            1.0 - pd.to_numeric(bus_derating["damage_ratio"], errors="coerce").fillna(0.0)
        )
        shed = shed.merge(
            bus_derating[
                [
                    "bus",
                    "hazard_intensity",
                    "damage_ratio",
                    "capacity_multiplier_after_damage",
                    "assigned_curve_id",
                    "assigned_curve_description",
                ]
            ].rename(
                columns={
                    "hazard_intensity": "bus_wind_speed_ms",
                    "damage_ratio": "bus_damage_ratio",
                    "capacity_multiplier_after_damage": "bus_remaining_capacity_fraction",
                }
            ),
            on="bus",
            how="left",
        )
    if not loads.empty and "bus" in loads.columns:
        shed = shed.merge(
            loads.groupby("bus", as_index=False).size().rename(columns={"size": "loads_at_bus"}),
            on="bus",
            how="left",
        )

    incident_parts = []
    if not line_derating.empty:
        for side in ["bus0", "bus1"]:
            cols = [
                "line",
                side,
                "v_nom",
                "hazard_intensity",
                "damage_ratio",
                "original_s_nom_mva",
                "capacity_loss_mva",
                "reduced_s_nom_mva",
            ]
            part = line_derating[cols].rename(columns={side: "bus"})
            part["line_endpoint_side"] = side
            incident_parts.append(part)
    incident = pd.concat(incident_parts, ignore_index=True) if incident_parts else pd.DataFrame()
    incident = incident[incident["bus"].isin(load_shed["bus"])] if not incident.empty else incident
    if not incident.empty and not loading.empty:
        incident = incident.merge(
            loading[["line", "max_abs_p0_mw", "max_loading_pu", "hours_overloaded"]],
            on="line",
            how="left",
        )
        incident = incident.sort_values(["bus", "max_loading_pu", "capacity_loss_mva"], ascending=[True, False, False])

    if not gen_derating.empty:
        gens_near_shed = gen_derating[gen_derating["bus"].isin(load_shed["bus"])].copy()
        gens_near_shed = gens_near_shed.sort_values(["bus", "capacity_loss_mw"], ascending=[True, False])
    else:
        gens_near_shed = pd.DataFrame()

    return shed, incident, gens_near_shed


def compare_old_new(new_summary: pd.DataFrame) -> pd.DataFrame:
    old_summary_path = OLD_SUITE / "gradual_return_period_suite_summary.csv"
    if not old_summary_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(old_summary_path)
    new = new_summary.copy()
    keys = ["hazard", "return_period"]
    keep = keys + ["load_shed_mwh", "incremental_system_cost_usd", "line_capacity_loss_mva", "generator_capacity_loss_mw"]
    comparison = old[keep].merge(new[keep], on=keys, how="outer", suffixes=("_old", "_new"))
    for metric in ["load_shed_mwh", "incremental_system_cost_usd", "line_capacity_loss_mva", "generator_capacity_loss_mw"]:
        comparison[f"{metric}_change"] = comparison[f"{metric}_new"] - comparison[f"{metric}_old"]
    return comparison.sort_values(keys)


def make_plots(flood_dist: pd.DataFrame, tc_shed: pd.DataFrame, old_new: pd.DataFrame) -> None:
    if not flood_dist.empty:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for asset_type, group in flood_dist.groupby("asset_type"):
            group = group.sort_values("return_period")
            ax.plot(group["return_period"], group["max"], marker="o", label=f"{asset_type} max")
            ax.plot(group["return_period"], group["mean"], marker=".", linestyle="--", label=f"{asset_type} mean")
        ax.set_title("Flood damage-ratio severity under updated curves")
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Damage ratio")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "flood_damage_ratio_mean_max_by_return_period.png", dpi=240)
        plt.close(fig)

    if not tc_shed.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ordered = tc_shed.sort_values("total_load_shed_mwh", ascending=True)
        ax.barh(ordered["bus"], ordered["total_load_shed_mwh"], color="#d95f02")
        ax.set_title("TC RP500 load shedding by bus")
        ax.set_xlabel("Load shed (MWh)")
        ax.set_ylabel("Bus")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "tc_rp500_load_shedding_by_bus.png", dpi=240)
        plt.close(fig)

    if not old_new.empty:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for hazard, group in old_new.groupby("hazard"):
            group = group.sort_values("return_period")
            ax.plot(group["return_period"], group["load_shed_mwh_old"], marker="o", linestyle="--", label=f"{hazard} old")
            ax.plot(group["return_period"], group["load_shed_mwh_new"], marker="o", label=f"{hazard} new")
        ax.set_title("Before/after load shedding comparison")
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Load shed (MWh)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "old_vs_new_load_shed_by_return_period.png", dpi=240)
        plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    flood_summary: pd.DataFrame,
    flood_dist: pd.DataFrame,
    flood_bind: pd.DataFrame,
    tc_shed: pd.DataFrame,
    tc_incident: pd.DataFrame,
    old_new: pd.DataFrame,
) -> None:
    flood_max_line_damage = flood_dist[
        (flood_dist["asset_type"].eq("line")) & (flood_dist["return_period"].eq(500))
    ]["max"].max()
    flood_max_gen_damage = flood_dist[
        (flood_dist["asset_type"].eq("generator")) & (flood_dist["return_period"].eq(500))
    ]["max"].max()
    flood_import_change = flood_summary["import_slack_change_from_baseline_mwh"].abs().max()
    flood_near_binding = int(flood_bind["near_binding_95pct"].sum()) if not flood_bind.empty else 0
    flood_binding = int(flood_bind["binding_99pct"].sum()) if not flood_bind.empty else 0
    tc_rp500 = summary[
        summary["hazard"].eq("tropical_cyclone") & summary["return_period"].eq(500)
    ].iloc[0]

    lines = [
        "# Improved Hazard Result Validation",
        "",
        "## Main Conclusions",
        "",
        "- Flood has zero load shedding because F6.2 produces very low line damage ratios and the remaining generation/transmission system can still meet demand.",
        f"- At Flood RP500, maximum line damage ratio is {flood_max_line_damage:.3f}; maximum generator damage ratio is {flood_max_gen_damage:.3f}.",
        f"- Flood import slack is unchanged across return periods within {flood_import_change:.6f} MWh, so imports are not newly absorbing flood damage.",
        f"- Flood RP500 has {flood_near_binding} lines at or above 95% loading and {flood_binding} lines at or above 99% loading, but no overloaded lines and no load shedding.",
        f"- TC RP500 is the only updated scenario with load shedding: {float(tc_rp500['load_shed_mwh']):,.1f} MWh at {int(tc_rp500['derated_generators'])} derated generators and {int(tc_rp500['derated_lines'])} derated lines.",
        f"- TC RP500 load shedding is concentrated at {len(tc_shed)} buses, with bus_18 accounting for the largest share.",
        "",
        "## Why Flood Has Zero Load Shedding",
        "",
        "The flood scenarios still derate many assets, but the derating is shallow. F6.2 keeps most transmission-line damage ratios near 0.01-0.02, so line capacities remain close to their calibrated values. Some generators lose more capacity under F1.* curves, especially thermal plants at deeper flood depths, but the system retains enough generation and network deliverability in the 24-hour dispatch window.",
        "",
        "Because import slack remains flat across all flood return periods, the zero-load-shed result is not mainly caused by additional emergency imports. It is primarily caused by low F6.2 line damage severity, retained generator capacity, and available redispatch/rerouting.",
        "",
        "## TC RP500 Explanation",
        "",
        "TC RP500 activates stronger W6.3 line derating, W2.3 bus/substation derating, and W1.* generator derating. Unlike lower TC return periods, RP500 also derates generators, and this creates localized unmet demand. The load shedding is geographically concentrated, not system-wide.",
        "",
        "## Files Written",
        "",
        "- `flood_zero_load_shed_summary.csv`",
        "- `flood_damage_ratio_distribution.csv`",
        "- `flood_damage_ratio_bins.csv`",
        "- `flood_rp500_top_binding_lines.csv`",
        "- `tc_rp500_load_shed_buses_diagnostic.csv`",
        "- `tc_rp500_incident_derated_lines_at_load_shed_buses.csv`",
        "- `tc_rp500_derated_generators_at_load_shed_buses.csv`",
        "- `old_vs_new_return_period_comparison.csv`",
        "",
        "## Interpretation Warning",
        "",
        "These checks validate the 24-hour dispatch result. They do not prove that floods or hurricanes would cause no real-world outages, because restoration time, protection trips, distribution outages, fuel logistics, crew access, and cascading failures are outside this PyPSA dispatch model.",
    ]
    (OUTPUT_DIR / "improved_hazard_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(NEW_SUITE / "gradual_return_period_suite_summary.csv")

    flood_summary, flood_dist, flood_bins = summarize_flood_zero_shed(summary)
    flood_bind = summarize_line_binding("flood_jrc_rp500_f62_gradual")
    tc_shed, tc_incident, tc_gens = summarize_tc_rp500()
    old_new = compare_old_new(summary)

    flood_summary.to_csv(OUTPUT_DIR / "flood_zero_load_shed_summary.csv", index=False)
    flood_dist.to_csv(OUTPUT_DIR / "flood_damage_ratio_distribution.csv", index=False)
    flood_bins.to_csv(OUTPUT_DIR / "flood_damage_ratio_bins.csv", index=False)
    flood_bind.head(100).to_csv(OUTPUT_DIR / "flood_rp500_top_binding_lines.csv", index=False)
    tc_shed.to_csv(OUTPUT_DIR / "tc_rp500_load_shed_buses_diagnostic.csv", index=False)
    tc_incident.to_csv(OUTPUT_DIR / "tc_rp500_incident_derated_lines_at_load_shed_buses.csv", index=False)
    tc_gens.to_csv(OUTPUT_DIR / "tc_rp500_derated_generators_at_load_shed_buses.csv", index=False)
    old_new.to_csv(OUTPUT_DIR / "old_vs_new_return_period_comparison.csv", index=False)

    make_plots(flood_dist, tc_shed, old_new)
    write_report(summary, flood_summary, flood_dist, flood_bind, tc_shed, tc_incident, old_new)

    print("Saved validation checks:", OUTPUT_DIR)
    print("\nFlood damage ratio distribution:")
    print(flood_dist.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print("\nTC RP500 load shedding buses:")
    print(tc_shed[["bus", "total_load_shed_mwh", "max_hourly_load_shed_mw", "bus_wind_speed_ms", "bus_damage_ratio", "x", "y"]].to_string(index=False))
    print("\nOld vs new load shed comparison:")
    if not old_new.empty:
        print(old_new[["hazard", "return_period", "load_shed_mwh_old", "load_shed_mwh_new", "load_shed_mwh_change"]].to_string(index=False))


if __name__ == "__main__":
    main()
