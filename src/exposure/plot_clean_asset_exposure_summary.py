"""
Create combined graphics for the cleaned Florida asset exposure workflow.

This script intentionally uses the cleaned outputs generated in the current
workflow and does not use the incomplete NOAA flood data.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
ELECTRICITY_SCENARIO_DIR = (
    PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network" / "hazard_scenarios"
)
OUTPUT_DIR = EXPOSURE_DIR / "clean_asset_exposure_visual_summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOOD_DIR = EXPOSURE_DIR / "powerplant_polygon_flood_exposure"
TC_POLYGON_DIR = EXPOSURE_DIR / "powerplant_polygon_tc_exposure"
SNAIL_STORM_DIR = EXPOSURE_DIR / "florida_clean_assets_tc_snail_intersection"
SNAIL_EVENT_DIR = EXPOSURE_DIR / "florida_clean_assets_tc_event_damage_snail"
PYPSA_FRAGILITY_DIR = ELECTRICITY_SCENARIO_DIR / "tc_wind_voltage_fragility"
PYPSA_THRESHOLD_DIR = ELECTRICITY_SCENARIO_DIR / "tc_wind_threshold_comparison"

FLOOD_SUMMARY = FLOOD_DIR / "powerplant_polygon_flood_summary_by_return_period.csv"
FLOOD_FUEL = FLOOD_DIR / "powerplant_polygon_flood_summary_by_fuel.csv"
FLOOD_GPKG = FLOOD_DIR / "powerplant_polygon_flood_exposure.gpkg"
TC_SUMMARY = TC_POLYGON_DIR / "powerplant_polygon_tc_summary_by_dataset.csv"
TC_FUEL = TC_POLYGON_DIR / "powerplant_polygon_tc_summary_by_fuel.csv"
TC_GPKG = TC_POLYGON_DIR / "powerplant_polygon_tc_exposure.gpkg"
TRANSMISSION_LINES_GPKG = PROJECT_DIR / "data" / "Electricity" / "florida_lines_with_s_nom.gpkg"
SNAIL_DAMAGE = SNAIL_STORM_DIR / "snail_storm_return_period_damage_vs_fred.csv"
SNAIL_LINE_CLASS = SNAIL_STORM_DIR / "snail_storm_return_period_line_length_by_wind_class.csv"
EVENT_DAMAGE = SNAIL_EVENT_DIR / "snail_ibtracs_event_network_damage.csv"
PYPSA_FRAGILITY_IMPACT = PYPSA_FRAGILITY_DIR / "incremental_impact_summary.csv"
PYPSA_FRAGILITY_CURVES = PYPSA_FRAGILITY_DIR / "fragility_curve_summary.csv"
PYPSA_FRAGILITY_DISPATCH = PYPSA_FRAGILITY_DIR / "dispatch_by_snapshot.csv"
PYPSA_THRESHOLD_COMPARISON = PYPSA_THRESHOLD_DIR / "tc_wind_threshold_comparison.csv"

FUEL_COLORS = {
    "Gas": "#6b6b6b",
    "Solar": "#f2c84b",
    "Coal": "#333333",
    "Oil": "#9e9e9e",
    "Nuclear": "#36a65f",
    "Hydro": "#2b8cbe",
    "Wind": "#41ab5d",
    "Biomass": "#a1d99b",
    "Waste": "#8c6bb1",
    "Storage": "#fb6a4a",
    "Unknown": "#bdbdbd",
}
FLOOD_COLORS = {
    "0 m": "#eeeeee",
    "0-0.5 m": "#bfe3ef",
    "0.5-1 m": "#7fb9d6",
    "1-2 m": "#4779b3",
    "2-5 m": "#f6a05d",
    ">5 m": "#c43b3b",
}
TC_COLORS = {
    "<25": "#4169a8",
    "25-30": "#98d4df",
    "30-35": "#f7e98b",
    "35-40": "#f5a65b",
    ">40": "#c63d3d",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing optional input: {path}")
        return pd.DataFrame()
    print(f"Reading {path}")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def billions(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return values / 1e9


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def dataset_label(row: pd.Series) -> str:
    if row["dataset"] == "open_gira_historical_max":
        return "OpenGIRA historical max"
    if pd.notna(row.get("return_period")):
        return f"STORM RP{int(row['return_period'])}"
    return str(row["dataset"])


def plot_dashboard() -> Path:
    flood = read_csv(FLOOD_SUMMARY)
    tc = read_csv(TC_SUMMARY)
    snail = read_csv(SNAIL_DAMAGE)
    events = read_csv(EVENT_DAMAGE)
    pypsa_threshold = read_csv(PYPSA_THRESHOLD_COMPARISON)
    pypsa_fragility = read_csv(PYPSA_FRAGILITY_IMPACT)

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(
        "Florida Clean Asset Hazard Exposure and PyPSA Network Impacts",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )

    ax = axes[0, 0]
    if not flood.empty:
        ax.plot(
            flood["return_period"],
            flood["exposed_capacity_mw"] / 1000,
            marker="o",
            color="#276fbf",
            linewidth=2.5,
        )
        ax.fill_between(
            flood["return_period"],
            flood["exposed_capacity_mw"] / 1000,
            color="#9bd1e5",
            alpha=0.35,
        )
        ax.set_xscale("log")
        ax.set_xticks(flood["return_period"])
        ax.set_xticklabels(flood["return_period"].astype(int))
        ax.set_ylabel("Exposed capacity (GW)")
        ax.set_xlabel("JRC flood return period")
    add_panel_label(ax, "A. Power plant polygon flood exposure")
    style_axis(ax)

    ax = axes[0, 1]
    if not tc.empty:
        tc = tc.copy()
        tc["label"] = tc.apply(dataset_label, axis=1)
        colors = ["#855c75" if "OpenGIRA" in label else "#d95f02" for label in tc["label"]]
        ax.bar(tc["label"], tc["capacity_ge_25ms"] / 1000, color=colors)
        ax.set_ylabel("Capacity >= 25 m/s (GW)")
        ax.tick_params(axis="x", rotation=35, labelsize=9)
    add_panel_label(ax, "B. Power plant polygon TC wind exposure")
    style_axis(ax)

    ax = axes[1, 0]
    if not snail.empty:
        ax.plot(
            snail["return_period"],
            billions(snail["storm_damage_usd"]),
            marker="o",
            linewidth=2.5,
            color="#b23a48",
            label="STORM return period",
        )
        fred = snail["fred_open_gira_historical_max_damage_usd"].dropna()
        if not fred.empty:
            ax.axhline(
                billions(fred.iloc[0]),
                color="#333333",
                linestyle="--",
                linewidth=2,
                label="OpenGIRA historical max",
            )
        ax.set_xscale("log")
        ax.set_xticks(snail["return_period"])
        ax.set_xticklabels(snail["return_period"].astype(int))
        ax.set_ylabel("Estimated line damage (billion USD)")
        ax.set_xlabel("STORM return period")
        ax.legend(frameon=False)
    add_panel_label(ax, "C. Transmission line damage by wind return period")
    style_axis(ax)

    ax = axes[1, 1]
    if not events.empty:
        top = events.nlargest(8, "total_network_damage_usd").sort_values(
            "total_network_damage_usd"
        )
        labels = top["event_id"].astype(str) + " (" + top["event_year"].astype(str) + ")"
        ax.barh(labels, billions(top["total_network_damage_usd"]), color="#5c4d7d")
        ax.set_xlabel("Damage (billion USD)")
    add_panel_label(ax, "D. Top OpenGIRA/IBTrACS event damages")
    style_axis(ax)

    ax = axes[2, 0]
    if not pypsa_threshold.empty:
        ax.plot(
            pypsa_threshold["threshold"],
            pypsa_threshold["incremental_load_shed_mwh"] / 1000,
            marker="o",
            linewidth=2.5,
            color="#d1495b",
            label="Load shed",
        )
        ax2 = ax.twinx()
        ax2.plot(
            pypsa_threshold["threshold"],
            pypsa_threshold["outaged_lines"],
            marker="s",
            linewidth=2.0,
            color="#00798c",
            label="Outaged lines",
        )
        ax.set_xlabel("Deterministic wind outage threshold (m/s)")
        ax.set_ylabel("Incremental load shed (GWh)")
        ax2.set_ylabel("Outaged lines")
        ax2.spines["top"].set_visible(False)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], frameon=False)
    add_panel_label(ax, "E. PyPSA deterministic line-outage sensitivity")
    style_axis(ax)

    ax = axes[2, 1]
    if not pypsa_fragility.empty:
        values = pypsa_fragility.iloc[0]
        labels = ["Load shed", "Import slack", "Capacity loss"]
        vals = [
            values["incremental_load_shed_mwh"] / 1000,
            values["incremental_import_slack_mwh"] / 1000,
            values["expected_capacity_loss_mva"] / 1000,
        ]
        ax.bar(labels, vals, color=["#d1495b", "#edae49", "#00798c"])
        ax.set_ylabel("GWh or GVA")
        for idx, val in enumerate(vals):
            ax.text(idx, val, f"{val:,.1f}", ha="center", va="bottom", fontsize=10)
    add_panel_label(ax, "F. PyPSA voltage-fragility impact")
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return save_figure(fig, "01_clean_asset_exposure_and_network_dashboard.png")


def plot_polygon_exposure_detail() -> Path:
    flood = read_csv(FLOOD_SUMMARY)
    flood_fuel = read_csv(FLOOD_FUEL)
    tc = read_csv(TC_SUMMARY)
    tc_fuel = read_csv(TC_FUEL)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Power Plant Polygon Exposure Details", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    if not flood.empty:
        ax.bar(
            flood["return_period"].astype(str),
            flood["exposed_polygons"],
            color="#276fbf",
            label="Exposed polygons",
        )
        ax2 = ax.twinx()
        ax2.plot(
            flood["return_period"].astype(str),
            flood["exposed_polygon_percent"],
            color="#333333",
            marker="o",
            label="Percent exposed",
        )
        ax.set_ylabel("Plant polygons")
        ax2.set_ylabel("Percent exposed")
        ax2.spines["top"].set_visible(False)
    add_panel_label(ax, "A. Flood-exposed plant polygons")
    style_axis(ax)

    ax = axes[0, 1]
    if not tc.empty:
        tc = tc.copy()
        tc["label"] = tc.apply(dataset_label, axis=1)
        ax.bar(tc["label"], tc["polygons_ge_25ms"], color="#d95f02")
        ax.set_ylabel("Plant polygons >= 25 m/s")
        ax.tick_params(axis="x", rotation=35, labelsize=9)
    add_panel_label(ax, "B. TC-exposed plant polygons")
    style_axis(ax)

    ax = axes[1, 0]
    if not flood_fuel.empty:
        focus = flood_fuel[flood_fuel["return_period"] == 100].copy()
        focus = focus.sort_values("exposed_capacity_mw", ascending=False).head(8)
        ax.barh(
            focus["primary_fuel"],
            focus["exposed_capacity_mw"] / 1000,
            color=[FUEL_COLORS.get(fuel, "#bdbdbd") for fuel in focus["primary_fuel"]],
        )
        ax.invert_yaxis()
        ax.set_xlabel("Exposed capacity (GW)")
    add_panel_label(ax, "C. Flood RP100 exposed capacity by fuel")
    style_axis(ax)

    ax = axes[1, 1]
    if not tc_fuel.empty:
        focus = tc_fuel[tc_fuel["dataset"] == "open_gira_historical_max"].copy()
        focus = focus.sort_values("exposed_capacity_mw", ascending=False).head(8)
        ax.barh(
            focus["primary_fuel"],
            focus["exposed_capacity_mw"] / 1000,
            color=[FUEL_COLORS.get(fuel, "#bdbdbd") for fuel in focus["primary_fuel"]],
        )
        ax.invert_yaxis()
        ax.set_xlabel("Exposed capacity (GW)")
    add_panel_label(ax, "D. OpenGIRA historical max TC exposure by fuel")
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, "02_powerplant_polygon_exposure_details.png")


def plot_transmission_damage_detail() -> Path:
    snail = read_csv(SNAIL_DAMAGE)
    line_class = read_csv(SNAIL_LINE_CLASS)
    events = read_csv(EVENT_DAMAGE)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Transmission Line Wind Exposure and Damage", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    if not snail.empty:
        ax.plot(snail["return_period"], billions(snail["storm_damage_usd"]), marker="o")
        ax.set_xscale("log")
        ax.set_xticks(snail["return_period"])
        ax.set_xticklabels(snail["return_period"].astype(int))
        ax.set_ylabel("Damage (billion USD)")
        ax.set_xlabel("STORM return period")
    add_panel_label(ax, "A. Return-period damage curve")
    style_axis(ax)

    ax = axes[0, 1]
    if not snail.empty:
        ax.scatter(
            snail["max_wind_speed_ms"],
            billions(snail["storm_damage_usd"]),
            s=80,
            color="#b23a48",
        )
        for _, row in snail.iterrows():
            ax.annotate(
                f"RP{int(row['return_period'])}",
                (row["max_wind_speed_ms"], row["storm_damage_usd"] / 1e9),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
        ax.set_xlabel("Max line wind speed (m/s)")
        ax.set_ylabel("Damage (billion USD)")
    add_panel_label(ax, "B. Damage vs maximum wind")
    style_axis(ax)

    ax = axes[1, 0]
    if not line_class.empty:
        class_candidates = ["tc_wind_class", "wind_class", "wind_speed_class"]
        class_col = next((col for col in class_candidates if col in line_class.columns), None)
        value_col = "line_length_km" if "line_length_km" in line_class.columns else "length_km"
        if class_col is not None and value_col in line_class.columns:
            pivot = line_class.pivot_table(
                index="return_period",
                columns=class_col,
                values=value_col,
                aggfunc="sum",
                fill_value=0,
            )
            ordered_cols = [col for col in ["<25", "25-30", "30-35", "35-40", ">40"] if col in pivot]
            if ordered_cols:
                pivot[ordered_cols].plot(
                    kind="bar",
                    stacked=True,
                    ax=ax,
                    color=[TC_COLORS[col] for col in ordered_cols],
                    width=0.8,
                )
                ax.legend(title="Wind class (m/s)", frameon=False, ncol=3, fontsize=8)
                ax.set_xlabel("STORM return period")
                ax.set_ylabel("Line length (km)")
                ax.tick_params(axis="x", rotation=0)
    add_panel_label(ax, "C. Line length by wind class")
    style_axis(ax)

    ax = axes[1, 1]
    if not events.empty:
        top = events.nlargest(10, "total_network_damage_usd").copy()
        ax.scatter(
            top["max_line_wind_ms"],
            billions(top["total_network_damage_usd"]),
            s=np.clip(top["lines_with_wind_ge_25ms"] / 2, 30, 260),
            color="#5c4d7d",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
        )
        for _, row in top.head(5).iterrows():
            ax.annotate(
                str(row["event_year"]),
                (row["max_line_wind_ms"], row["total_network_damage_usd"] / 1e9),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
            )
        ax.set_xlabel("Max line wind in event (m/s)")
        ax.set_ylabel("Event damage (billion USD)")
    add_panel_label(ax, "D. Top events: wind, footprint, damage")
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, "03_transmission_line_wind_damage_details.png")


def plot_pypsa_network_detail() -> Path:
    threshold = read_csv(PYPSA_THRESHOLD_COMPARISON)
    fragility = read_csv(PYPSA_FRAGILITY_IMPACT)
    curves = read_csv(PYPSA_FRAGILITY_CURVES)
    dispatch = read_csv(PYPSA_FRAGILITY_DISPATCH)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("PyPSA Hazard Scenario Results", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    if not threshold.empty:
        ax.bar(
            threshold["threshold"].astype(str),
            threshold["outaged_line_capacity_mva"] / 1000,
            color="#00798c",
        )
        ax.set_xlabel("Wind threshold (m/s)")
        ax.set_ylabel("Outaged capacity (GVA)")
    add_panel_label(ax, "A. Deterministic outage capacity")
    style_axis(ax)

    ax = axes[0, 1]
    if not threshold.empty:
        ax.plot(
            threshold["threshold"],
            threshold["incremental_load_shed_mwh"] / 1000,
            marker="o",
            linewidth=2.5,
            color="#d1495b",
            label="Load shed",
        )
        ax.plot(
            threshold["threshold"],
            threshold["incremental_import_slack_mwh"] / 1000,
            marker="s",
            linewidth=2.5,
            color="#edae49",
            label="Emergency import",
        )
        ax.set_xlabel("Wind threshold (m/s)")
        ax.set_ylabel("Incremental energy (GWh)")
        ax.legend(frameon=False)
    add_panel_label(ax, "B. Dispatch impact by threshold")
    style_axis(ax)

    ax = axes[1, 0]
    if not curves.empty:
        curves = curves.sort_values("expected_capacity_loss_mva", ascending=False)
        ax.bar(
            curves["fragility_curve_id"],
            curves["expected_capacity_loss_mva"] / 1000,
            color=["#d1495b", "#00798c", "#edae49"][: len(curves)],
        )
        ax.set_xlabel("Fragility curve")
        ax.set_ylabel("Expected capacity loss (GVA)")
    add_panel_label(ax, "C. Voltage fragility capacity loss")
    style_axis(ax)

    ax = axes[1, 1]
    if not dispatch.empty:
        dispatch = dispatch.copy()
        dispatch["snapshot"] = pd.to_datetime(dispatch["snapshot"])
        daily = dispatch.set_index("snapshot")[["import_slack_mwh", "load_shed_mwh"]].resample(
            "D"
        ).sum()
        ax.plot(daily.index, daily["import_slack_mwh"] / 1000, color="#edae49", label="Import")
        ax.plot(daily.index, daily["load_shed_mwh"] / 1000, color="#d1495b", label="Load shed")
        ax.set_ylabel("Daily energy (GWh)")
        ax.legend(frameon=False)
    elif not fragility.empty:
        vals = fragility.iloc[0]
        ax.bar(
            ["Load shed", "Import"],
            [vals["incremental_load_shed_mwh"] / 1000, vals["incremental_import_slack_mwh"] / 1000],
            color=["#d1495b", "#edae49"],
        )
        ax.set_ylabel("Incremental energy (GWh)")
    add_panel_label(ax, "D. Voltage-fragility dispatch profile")
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, "04_pypsa_hazard_network_results.png")


def plot_polygon_maps() -> Path:
    flood = gpd.read_file(FLOOD_GPKG) if FLOOD_GPKG.exists() else gpd.GeoDataFrame()
    tc = gpd.read_file(TC_GPKG) if TC_GPKG.exists() else gpd.GeoDataFrame()
    lines = (
        gpd.read_file(TRANSMISSION_LINES_GPKG)
        if TRANSMISSION_LINES_GPKG.exists()
        else gpd.GeoDataFrame()
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 9))
    fig.suptitle("Power Plant Polygon Hazard Maps", fontsize=18, fontweight="bold")

    ax = axes[0]
    if not flood.empty:
        focus = flood[flood["return_period"] == 100].copy()
        if not lines.empty:
            lines.to_crs(focus.crs).plot(ax=ax, color="#d0d0d0", linewidth=0.25, alpha=0.65)
        focus["flood_class"] = pd.Categorical(
            focus["flood_class"],
            categories=["0 m", "0-0.5 m", "0.5-1 m", "1-2 m", "2-5 m", ">5 m"],
            ordered=True,
        )
        focus.plot(
            ax=ax,
            color=focus["flood_class"].astype(object).map(FLOOD_COLORS).fillna("#eeeeee"),
            edgecolor="#303030",
            linewidth=0.25,
            alpha=0.9,
        )
        point_focus = focus.copy()
        point_focus["geometry"] = point_focus.geometry.representative_point()
        point_focus["marker_size"] = np.clip(np.sqrt(point_focus["capacity_mw"].fillna(0)) * 2.5, 8, 125)
        point_focus.plot(
            ax=ax,
            color=point_focus["flood_class"].astype(object).map(FLOOD_COLORS).fillna("#eeeeee"),
            edgecolor="#111111",
            linewidth=0.35,
            markersize=point_focus["marker_size"],
            alpha=0.95,
        )
        handles = [
            Line2D([0], [0], marker="s", color="none", markerfacecolor=color, label=label, markersize=8)
            for label, color in FLOOD_COLORS.items()
        ]
        ax.legend(handles=handles, title="Max depth", frameon=True, loc="lower left", fontsize=8)
    ax.set_title("JRC RP100 flood depth by plant footprint")
    ax.set_axis_off()

    ax = axes[1]
    if not tc.empty:
        focus = tc[tc["dataset"] == "open_gira_historical_max"].copy()
        if not lines.empty:
            lines.to_crs(focus.crs).plot(ax=ax, color="#d0d0d0", linewidth=0.25, alpha=0.65)
        focus["tc_wind_class"] = pd.Categorical(
            focus["tc_wind_class"],
            categories=["<25", "25-30", "30-35", "35-40", ">40"],
            ordered=True,
        )
        focus.plot(
            ax=ax,
            color=focus["tc_wind_class"].astype(object).map(TC_COLORS).fillna("#eeeeee"),
            edgecolor="#303030",
            linewidth=0.25,
            alpha=0.9,
        )
        point_focus = focus.copy()
        point_focus["geometry"] = point_focus.geometry.representative_point()
        point_focus["marker_size"] = np.clip(np.sqrt(point_focus["capacity_mw"].fillna(0)) * 2.5, 8, 125)
        point_focus.plot(
            ax=ax,
            color=point_focus["tc_wind_class"].astype(object).map(TC_COLORS).fillna("#eeeeee"),
            edgecolor="#111111",
            linewidth=0.35,
            markersize=point_focus["marker_size"],
            alpha=0.95,
        )
        handles = [
            Line2D([0], [0], marker="s", color="none", markerfacecolor=color, label=label, markersize=8)
            for label, color in TC_COLORS.items()
        ]
        ax.legend(handles=handles, title="Wind class (m/s)", frameon=True, loc="lower left", fontsize=8)
    ax.set_title("OpenGIRA historical max TC wind by plant footprint")
    ax.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, "05_powerplant_polygon_hazard_maps.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 120,
        }
    )

    outputs = [
        plot_dashboard(),
        plot_polygon_exposure_detail(),
        plot_transmission_damage_detail(),
        plot_pypsa_network_detail(),
        plot_polygon_maps(),
    ]

    print("\nCreated visual summary pack:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
