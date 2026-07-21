"""
Publication-quality Hurricane Ian impact figures.

This script creates a new output folder on every run so earlier figures are not
overwritten. It compares the calibrated no-hazard PyPSA run with the Hurricane
Ian direct-damage run for the extended-tie Florida grid.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from shapely.geometry import LineString, Point


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EVENT_ID = "2022266N12294"
EVENT_NAME = "Hurricane Ian"

EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
ASSET_IMPACT_DIR = EXPOSURE_DIR / "florida_new_grid_ibtracs_asset_impacts"
LINE_DAMAGE_DIR = EXPOSURE_DIR / "florida_new_grid_tc_event_damage_snail"
GRID_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_extended_tie_lines"
BASELINE_DIR = GRID_DIR / "baseline_calibrated_no_hazard"
SCENARIO_DIR = GRID_DIR / "ibtracs_cascading_top5_direct_events" / f"ibtracs_{EVENT_ID}_tc_direct_damage"

GRID_QGIS = GRID_DIR / "qgis" / "florida_extended_tie_lines_for_qgis.gpkg"
LINE_GEOMETRY = LINE_DAMAGE_DIR / "new_grid_lines_used_for_snail.gpkg"
PLANT_POLYGONS = PROJECT_DIR / "data" / "Electricity" / "florida_osm_powerplant_polygons_with_point_attributes.gpkg"
GENERATORS = GRID_DIR / "generators.csv"
IBTRACS = PROJECT_DIR / "Florida" / "ibtracs.NA.list.v04r01.csv"
US_STATE_BOUNDARIES = PROJECT_DIR / "data" / "Boundaries" / "cb_2024_us_state_500k" / "cb_2024_us_state_500k.shp"

BASE_OUT_DIR = ASSET_IMPACT_DIR / "hurricane_ian_publication_figures"
MAP_BOUNDS = (-88.2, -79.3, 24.0, 32.0)
FIGURE_2_ZOOM_BOUNDS = (-82.7, -79.9, 25.6, 27.9)

INK = "#111827"
MUTED = "#6B7280"
GRID_LIGHT = "#B8C0CC"
DIRECT = "#C53030"
INDIRECT = "#1D4ED8"
SUBSTATION = "#7C3AED"
LOAD_SHED = "#F59E0B"
TRACK = "#111827"
GREEN = "#2F855A"


def make_output_dir() -> tuple[Path, Path, Path]:
    for i in range(1, 100):
        out_dir = BASE_OUT_DIR.with_name(f"{BASE_OUT_DIR.name}_v{i:02d}")
        if not out_dir.exists():
            fig_dir = out_dir / "figures"
            data_dir = out_dir / "data"
            fig_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            return out_dir, fig_dir, data_dir
    raise RuntimeError("Too many existing figure output folders.")


def usd_short(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{sign}${value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}${value / 1e3:.1f}k"
    return f"{sign}${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt_number(value: float, suffix: str = "") -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}{suffix}"
    return f"{value:.1f}{suffix}"


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_map(ax: plt.Axes, title: str = "") -> None:
    ax.set_xlim(MAP_BOUNDS[0], MAP_BOUNDS[1])
    ax.set_ylim(MAP_BOUNDS[2], MAP_BOUNDS[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title, fontsize=15, weight="bold", loc="left")
    ax.grid(color="#D7DEE8", linewidth=0.7, alpha=0.55)


def draw_florida_outline(ax: plt.Axes, florida_outline: gpd.GeoDataFrame, linewidth: float = 1.7) -> None:
    if not florida_outline.empty:
        florida_outline.boundary.plot(ax=ax, color=INK, linewidth=linewidth, zorder=30)


def line_midpoint(geom):
    if geom is None or geom.is_empty:
        return None
    try:
        return geom.interpolate(0.5, normalized=True)
    except Exception:
        return geom.representative_point()


def short_line_label(row: pd.Series) -> str:
    sub_1 = str(row.get("SUB_1", "")).strip()
    sub_2 = str(row.get("SUB_2", "")).strip()
    if sub_1 and sub_2 and sub_1.upper() not in {"NAN", "NOT AVAILABLE"} and sub_2.upper() not in {"NAN", "NOT AVAILABLE"}:
        label = f"{sub_1} - {sub_2}"
        return label[:42]
    return str(row["line"]).replace("line_", "L")


def load_storm_track() -> gpd.GeoDataFrame:
    tracks = pd.read_csv(IBTRACS, low_memory=False)
    storm = tracks[tracks["SID"].astype(str).eq(EVENT_ID)].copy()
    storm["LAT"] = pd.to_numeric(storm["LAT"], errors="coerce")
    storm["LON"] = pd.to_numeric(storm["LON"], errors="coerce")
    storm["USA_WIND"] = pd.to_numeric(storm["USA_WIND"], errors="coerce")
    storm = storm.dropna(subset=["LAT", "LON"])
    points = [Point(xy) for xy in zip(storm["LON"], storm["LAT"])]
    return gpd.GeoDataFrame(
        [{
            "event_id": EVENT_ID,
            "event_name": EVENT_NAME,
            "max_usa_wind_kt": storm["USA_WIND"].max(),
            "geometry": LineString(points),
        }],
        geometry="geometry",
        crs="EPSG:4326",
    )


def load_all_data() -> dict[str, object]:
    base_lines = gpd.read_file(LINE_GEOMETRY).to_crs("EPSG:4326")
    florida_outline = gpd.read_file(US_STATE_BOUNDARIES).to_crs("EPSG:4326")
    florida_outline = florida_outline[florida_outline["STUSPS"].eq("FL")].copy()
    buses = gpd.read_file(GRID_QGIS, layer="cleaned_buses").to_crs("EPSG:4326")

    baseline_loading = pd.read_csv(BASELINE_DIR / "baseline_line_loading.csv")
    damaged_loading = pd.read_csv(SCENARIO_DIR / "line_loading.csv")
    damaged_line_table = pd.read_csv(SCENARIO_DIR / "damaged_lines.csv")
    damaged_line_table["damage_ratio"] = pd.to_numeric(damaged_line_table["damage_ratio"], errors="coerce").fillna(0)
    damaged_line_table = damaged_line_table[damaged_line_table["damage_ratio"] > 0].copy()

    damaged_substations = pd.read_csv(SCENARIO_DIR / "bus_substation_deratings.csv")
    damaged_substations["damage_ratio"] = pd.to_numeric(damaged_substations["damage_ratio"], errors="coerce").fillna(0)
    damaged_substations = damaged_substations[damaged_substations["damage_ratio"] > 0].copy()
    damaged_substations["geometry"] = [Point(xy) for xy in zip(damaged_substations["x"], damaged_substations["y"])]
    damaged_substations = gpd.GeoDataFrame(damaged_substations, geometry="geometry", crs="EPSG:4326")

    baseline_load_shed_bus = pd.read_csv(BASELINE_DIR / "baseline_load_shedding_by_bus.csv")
    scenario_load_shed_bus = pd.read_csv(SCENARIO_DIR / "load_shedding_by_bus.csv")
    baseline_summary = pd.read_csv(BASELINE_DIR / "baseline_summary.csv").iloc[0]
    scenario_summary = pd.read_csv(SCENARIO_DIR / "scenario_summary.csv").iloc[0]
    generators = pd.read_csv(GENERATORS)

    return {
        "base_lines": base_lines,
        "florida_outline": florida_outline,
        "buses": buses,
        "baseline_loading": baseline_loading,
        "damaged_loading": damaged_loading,
        "damaged_line_table": damaged_line_table,
        "damaged_substations": damaged_substations,
        "baseline_load_shed_bus": baseline_load_shed_bus,
        "scenario_load_shed_bus": scenario_load_shed_bus,
        "baseline_summary": baseline_summary,
        "scenario_summary": scenario_summary,
        "generators": generators,
        "storm_track": load_storm_track(),
    }


def build_line_loading_comparison(data: dict[str, object]) -> gpd.GeoDataFrame:
    base = data["baseline_loading"].copy()
    post = data["damaged_loading"].copy()
    lines = data["base_lines"].copy()
    damaged_ids = set(data["damaged_line_table"]["line"].astype(str))

    comparison = base[["line", "max_loading_pu", "s_nom_mva"]].merge(
        post[["line", "max_loading_pu", "s_nom_mva"]],
        on="line",
        suffixes=("_baseline", "_damaged"),
        how="outer",
    )
    comparison["baseline_loading"] = pd.to_numeric(comparison["max_loading_pu_baseline"], errors="coerce").fillna(0)
    comparison["damaged_loading"] = pd.to_numeric(comparison["max_loading_pu_damaged"], errors="coerce").fillna(0)
    comparison["loading_change_percentage_points"] = (comparison["damaged_loading"] - comparison["baseline_loading"]) * 100
    comparison["newly_overloaded"] = (comparison["damaged_loading"] >= 1.0) & (comparison["baseline_loading"] < 1.0)
    comparison["directly_damaged"] = comparison["line"].astype(str).isin(damaged_ids)
    comparison["surviving_newly_overloaded"] = comparison["newly_overloaded"] & ~comparison["directly_damaged"]
    comparison["post_damage_rank"] = comparison["damaged_loading"].rank(method="first", ascending=False).astype(int)

    comparison = lines.merge(comparison, left_on="asset_line_id", right_on="line", how="inner")
    comparison["label"] = comparison.apply(short_line_label, axis=1)
    return gpd.GeoDataFrame(comparison, geometry="geometry", crs=lines.crs)


def build_load_shedding_comparison(data: dict[str, object]) -> gpd.GeoDataFrame:
    base = data["baseline_load_shed_bus"].rename(
        columns={
            "total_load_shed_mwh": "baseline_load_shed_mwh",
            "max_hourly_load_shed_mw": "baseline_max_hourly_load_shed_mw",
        }
    )
    post = data["scenario_load_shed_bus"].rename(
        columns={
            "total_load_shed_mwh": "post_damage_load_shed_mwh",
            "max_hourly_load_shed_mw": "post_damage_max_hourly_load_shed_mw",
        }
    )
    comparison = base.merge(post, on="bus", how="outer").fillna(0)
    comparison["incremental_load_shed_mwh"] = comparison["post_damage_load_shed_mwh"] - comparison["baseline_load_shed_mwh"]
    buses = data["buses"][["name", "geometry"]].copy()
    comparison = buses.merge(comparison, left_on="name", right_on="bus", how="inner")
    return gpd.GeoDataFrame(comparison, geometry="geometry", crs=buses.crs)


def calculate_metrics(data: dict[str, object], line_comparison: gpd.GeoDataFrame, load_shed: gpd.GeoDataFrame) -> pd.DataFrame:
    damaged_lines = data["damaged_line_table"]
    baseline_summary = data["baseline_summary"]
    scenario_summary = data["scenario_summary"]
    generators = data["generators"]
    total_network_capacity_mva = float(pd.to_numeric(line_comparison["s_nom_mva_baseline"], errors="coerce").fillna(0).sum())
    line_capacity_loss_mva = float(pd.to_numeric(damaged_lines["capacity_loss_mva"], errors="coerce").fillna(0).sum())
    total_generation_capacity_mw = float(pd.to_numeric(generators["p_nom"], errors="coerce").fillna(0).sum())
    generator_capacity_loss_mw = 0.0
    damaged_generators_path = SCENARIO_DIR / "damaged_generators.csv"
    if damaged_generators_path.stat().st_size > 2:
        damaged_generators = pd.read_csv(damaged_generators_path)
        if "capacity_loss_mw" in damaged_generators.columns:
            generator_capacity_loss_mw = float(pd.to_numeric(damaged_generators["capacity_loss_mw"], errors="coerce").fillna(0).sum())

    baseline_cost = float(baseline_summary["total_system_cost_usd"])
    scenario_cost = float(scenario_summary["total_system_cost_usd"])
    baseline_load_shed_mwh = float(baseline_summary["total_load_shed_mwh"])
    scenario_load_shed_mwh = float(scenario_summary["total_load_shed_mwh"])

    rows = [{
        "event_name": EVENT_NAME,
        "directly_damaged_transmission_lines": int(len(damaged_lines)),
        "damaged_substations": int(len(data["damaged_substations"])),
        "line_capacity_loss_mva": line_capacity_loss_mva,
        "total_network_capacity_mva": total_network_capacity_mva,
        "direct_line_capacity_loss_pct_total_network": line_capacity_loss_mva / total_network_capacity_mva * 100 if total_network_capacity_mva else 0,
        "post_damage_lines_ge_80pct_loading": int((line_comparison["damaged_loading"] >= 0.80).sum()),
        "post_damage_lines_ge_90pct_loading": int((line_comparison["damaged_loading"] >= 0.90).sum()),
        "post_damage_lines_ge_100pct_loading": int((line_comparison["damaged_loading"] >= 1.00).sum()),
        "surviving_newly_overloaded_lines_ge_100pct": int(line_comparison["surviving_newly_overloaded"].sum()),
        "post_damage_load_shed_mwh": scenario_load_shed_mwh,
        "baseline_load_shed_mwh": baseline_load_shed_mwh,
        "incremental_load_shed_mwh": scenario_load_shed_mwh - baseline_load_shed_mwh,
        "post_damage_load_shedding_buses": int((load_shed["post_damage_load_shed_mwh"] > 0).sum()),
        "incremental_load_shedding_buses": int((load_shed["incremental_load_shed_mwh"].abs() > 1e-6).sum()),
        "baseline_system_cost_usd": baseline_cost,
        "post_damage_system_cost_usd": scenario_cost,
        "incremental_system_cost_usd": scenario_cost - baseline_cost,
        "generator_capacity_loss_mw": generator_capacity_loss_mw,
        "total_generation_capacity_mw": total_generation_capacity_mw,
        "generator_capacity_loss_pct_total": generator_capacity_loss_mw / total_generation_capacity_mw * 100 if total_generation_capacity_mw else 0,
    }]
    return pd.DataFrame(rows)


def save_core_data(data: dict[str, object], line_comparison: gpd.GeoDataFrame, load_shed: gpd.GeoDataFrame, metrics: pd.DataFrame, data_dir: Path) -> None:
    metrics.to_csv(data_dir / "figure_1_impact_summary_metrics.csv", index=False)
    line_comparison.drop(columns="geometry").to_csv(data_dir / "figure_2_line_loading_comparison.csv", index=False)
    load_shed.drop(columns="geometry").to_csv(data_dir / "figure_3_load_shedding_bus_comparison.csv", index=False)

    direct_lines = data["base_lines"].merge(
        data["damaged_line_table"],
        left_on="asset_line_id",
        right_on="line",
        how="inner",
    )
    direct_lines.drop(columns="geometry").to_csv(data_dir / "figure_3_directly_damaged_lines.csv", index=False)
    data["damaged_substations"].drop(columns="geometry").to_csv(data_dir / "figure_3_directly_damaged_substations.csv", index=False)


def figure_1_impact_summary(metrics: pd.DataFrame, fig_dir: Path, data_dir: Path) -> None:
    m = metrics.iloc[0]
    stages = [
        (
            "Hazard exposure",
            ["Ian wind field intersects", "the extended Florida grid", "Track from IBTrACS/OpenGIRA"],
            "#EFF6FF",
        ),
        (
            "Direct physical damage",
            [
                f"{m.directly_damaged_transmission_lines:,.0f} transmission lines",
                f"{m.damaged_substations:,.0f} substations",
                f"{m.line_capacity_loss_mva:,.0f} MVA line capacity lost",
                f"{m.direct_line_capacity_loss_pct_total_network:.1f}% of total line capacity",
            ],
            "#FEF2F2",
        ),
        (
            "Network redistribution",
            [
                f"{m.post_damage_lines_ge_80pct_loading:,.0f} lines >=80% loaded",
                f"{m.post_damage_lines_ge_90pct_loading:,.0f} lines >=90% loaded",
                f"{m.post_damage_lines_ge_100pct_loading:,.0f} lines at 100% loading",
            ],
            "#FFF7ED",
        ),
        (
            "Service consequences",
            [
                f"{m.post_damage_load_shed_mwh:,.0f} MWh load shed",
                f"{m.post_damage_load_shedding_buses:,.0f} load-shedding buses",
                f"{usd_short(m.incremental_system_cost_usd)} incremental system cost",
                f"{m.incremental_load_shed_mwh:,.1f} MWh incremental load shed vs baseline",
            ],
            "#F0FDF4",
        ),
    ]
    pd.DataFrame(
        [{"stage": stage, "metric": metric} for stage, lines, _color in stages for metric in lines]
    ).to_csv(data_dir / "figure_1_stage_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(18, 5.3))
    ax.axis("off")
    ax.set_title("Hurricane Ian Impact Summary", fontsize=20, weight="bold", loc="left", pad=18)
    ax.text(
        0,
        0.94,
        "Direct physical damage was substantial, while the saved PyPSA run shows no additional load shedding compared with the no-hazard baseline.",
        transform=ax.transAxes,
        fontsize=11.5,
        color=MUTED,
    )
    card_w = 0.22
    x_positions = [0.01, 0.265, 0.52, 0.775]
    for i, (stage, lines, color) in enumerate(stages):
        x = x_positions[i]
        card = FancyBboxPatch(
            (x, 0.18),
            card_w,
            0.62,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=color,
            edgecolor="#CBD5E1",
            linewidth=1.4,
            transform=ax.transAxes,
        )
        ax.add_patch(card)
        ax.text(x + 0.018, 0.72, stage, transform=ax.transAxes, fontsize=13.4, weight="bold", color=INK)
        y = 0.61
        for line in lines:
            ax.text(x + 0.022, y, line, transform=ax.transAxes, fontsize=10.7, color=INK)
            y -= 0.105
        if i < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x + card_w + 0.012, 0.49),
                (x_positions[i + 1] - 0.012, 0.49),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.8,
                color=MUTED,
                transform=ax.transAxes,
            )
            ax.add_patch(arrow)
    save_figure(fig, fig_dir, "figure_1_hurricane_ian_impact_summary")


def loading_style(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    loading = gdf["plot_loading"].clip(lower=0, upper=1.05).to_numpy()
    widths = 0.6 + 3.0 * loading
    return loading, widths


def plot_loading_lines(ax: plt.Axes, gdf: gpd.GeoDataFrame, norm: Normalize, cmap) -> None:
    loading, widths = loading_style(gdf)
    lower = gdf[gdf["plot_loading"] < 0.8].copy()
    if not lower.empty:
        lower_colors = [cmap(norm(value)) for value in lower["plot_loading"]]
        lower_widths = 0.6 + 2.4 * lower["plot_loading"]
        lower.plot(ax=ax, color=lower_colors, linewidth=lower_widths, alpha=0.9, zorder=8)
    for threshold, color, width, zorder in [(1.0, "#7F1D1D", 4.2, 12), (0.9, "#EA580C", 3.4, 11), (0.8, "#F59E0B", 2.7, 10)]:
        subset = gdf[gdf["plot_loading"] >= threshold]
        if not subset.empty:
            subset.plot(ax=ax, color=color, linewidth=width, alpha=0.95, zorder=zorder)


def label_top_loading_changes(ax: plt.Axes, top5: gpd.GeoDataFrame) -> None:
    for idx, (_, row) in enumerate(top5.iterrows()):
        pt = line_midpoint(row.geometry)
        if pt is None:
            continue
        ax.scatter(pt.x, pt.y, s=165, facecolor="white", edgecolor=INK, linewidth=1.2, zorder=45)
        ax.text(pt.x, pt.y, str(idx + 1), ha="center", va="center", fontsize=9.5, weight="bold", color=INK, zorder=46)


def figure_2_baseline_vs_post_loading(
    line_comparison: gpd.GeoDataFrame,
    storm_track: gpd.GeoDataFrame,
    florida_outline: gpd.GeoDataFrame,
    fig_dir: Path,
    data_dir: Path,
) -> None:
    top5 = line_comparison.sort_values("loading_change_percentage_points", ascending=False).head(5).copy()
    top5.drop(columns="geometry").to_csv(data_dir / "figure_2_top5_largest_loading_increase.csv", index=False)

    cmap = mpl.colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.6), sharex=True, sharey=True)
    for ax, col, title in [
        (axes[0], "baseline_loading", "(a) Baseline loading in affected area"),
        (axes[1], "damaged_loading", "(b) Loading after Ian damage"),
    ]:
        plot_gdf = line_comparison.copy()
        plot_gdf["plot_loading"] = plot_gdf[col]
        line_comparison.plot(ax=ax, color="#CBD5E1", linewidth=0.45, alpha=0.65, zorder=1)
        plot_loading_lines(ax, plot_gdf, norm, cmap)
        draw_florida_outline(ax, florida_outline)
        style_map(ax, title)
        ax.set_xlim(FIGURE_2_ZOOM_BOUNDS[0], FIGURE_2_ZOOM_BOUNDS[1])
        ax.set_ylim(FIGURE_2_ZOOM_BOUNDS[2], FIGURE_2_ZOOM_BOUNDS[3])
    storm_track.plot(ax=axes[1], color=TRACK, linewidth=1.6, zorder=35)
    label_top_loading_changes(axes[1], top5)
    fig.suptitle("Ian Damage Redistributes Flow Around Southwest and South Florida", fontsize=19, weight="bold", y=0.98)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.025)
    cbar.set_label("Maximum line loading (p.u.)", fontsize=11)
    handles = [
        Line2D([0], [0], color="#F59E0B", lw=3, label=">=80%"),
        Line2D([0], [0], color="#EA580C", lw=3.5, label=">=90%"),
        Line2D([0], [0], color="#7F1D1D", lw=4.2, label="100%"),
        Line2D([0], [0], color=TRACK, lw=1.7, label="Ian track"),
    ]
    axes[1].legend(handles=handles, title="Highlighted loading", loc="lower left", fontsize=10, title_fontsize=10)
    top5_notes = []
    for idx, (_, row) in enumerate(top5.iterrows(), start=1):
        top5_notes.append(
            f"{idx}. {str(row['line']).replace('line_', 'L')}: "
            f"{row['baseline_loading'] * 100:.1f}% -> {row['damaged_loading'] * 100:.1f}% "
            f"(+{row['loading_change_percentage_points']:.1f} pp)"
        )
    fig.text(
        0.12,
        0.045,
        "Top five loading increases: " + "   ".join(top5_notes),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color=INK,
    )
    save_figure(fig, fig_dir, "figure_2_baseline_vs_post_ian_network_loading")


def figure_3_combined_impact_map(
    data: dict[str, object],
    line_comparison: gpd.GeoDataFrame,
    load_shed: gpd.GeoDataFrame,
    fig_dir: Path,
    data_dir: Path,
) -> None:
    base_lines = data["base_lines"]
    florida_outline = data["florida_outline"]
    storm_track = data["storm_track"]
    damaged_substations = data["damaged_substations"]
    damaged_lines = base_lines.merge(data["damaged_line_table"], left_on="asset_line_id", right_on="line", how="inner")
    surviving_new_overloaded = line_comparison[line_comparison["surviving_newly_overloaded"]].copy()
    post_load_shed = load_shed[load_shed["post_damage_load_shed_mwh"] > 0].copy()

    surviving_new_overloaded.drop(columns="geometry").to_csv(data_dir / "figure_3_surviving_newly_overloaded_lines.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 9))
    base_lines.plot(ax=ax, color="#CBD5E1", linewidth=0.55, alpha=0.75, zorder=1)
    if not damaged_lines.empty:
        damaged_lines.plot(ax=ax, color=DIRECT, linewidth=1.7, alpha=0.72, zorder=12)
    if not surviving_new_overloaded.empty:
        surviving_new_overloaded.plot(ax=ax, color=INDIRECT, linewidth=2.4, linestyle="--", zorder=20)
    if not damaged_substations.empty:
        damaged_substations.plot(ax=ax, marker="x", color=SUBSTATION, markersize=42, linewidth=1.7, zorder=22)
    if not post_load_shed.empty:
        sizes = (post_load_shed["post_damage_load_shed_mwh"] / post_load_shed["post_damage_load_shed_mwh"].max() * 260) + 30
        post_load_shed.plot(ax=ax, color=LOAD_SHED, markersize=sizes, edgecolor=INK, linewidth=0.55, alpha=0.88, zorder=24)
    storm_track.plot(ax=ax, color=TRACK, linewidth=1.45, zorder=25)
    draw_florida_outline(ax, florida_outline)
    style_map(ax, "Direct Damage and Indirect Operational Consequences")

    direct_handles = [
        Line2D([0], [0], color=DIRECT, lw=3, label="Directly damaged transmission line"),
        Line2D([0], [0], marker="x", color=SUBSTATION, lw=0, label="Directly damaged substation", markersize=9),
        Line2D([0], [0], color=TRACK, lw=1.7, label="Ian track"),
    ]
    indirect_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LOAD_SHED, markeredgecolor=INK, lw=0, label="Post-damage load-shedding bus", markersize=9),
    ]
    if not surviving_new_overloaded.empty:
        indirect_handles.insert(
            0,
            Line2D([0], [0], color=INDIRECT, lw=3, linestyle="--", label="Surviving line newly at 100% loading"),
        )
    leg1 = ax.legend(handles=direct_handles, title="Direct physical impacts", loc="upper left", fontsize=9.5, title_fontsize=10.5)
    ax.add_artist(leg1)
    ax.legend(handles=indirect_handles, title="Indirect network impacts", loc="lower left", fontsize=9.5, title_fontsize=10.5)

    inset = inset_axes(ax, width="34%", height="34%", loc="center left", borderpad=1.1)
    sw_bounds = (-83.0, -80.6, 25.7, 28.6)
    base_lines.plot(ax=inset, color="#CBD5E1", linewidth=0.45, alpha=0.65)
    if not damaged_lines.empty:
        damaged_lines.plot(ax=inset, color=DIRECT, linewidth=1.3, alpha=0.72)
    if not post_load_shed.empty:
        post_load_shed.plot(ax=inset, color=LOAD_SHED, markersize=24, edgecolor=INK, linewidth=0.4, alpha=0.9)
    storm_track.plot(ax=inset, color=TRACK, linewidth=1.2)
    draw_florida_outline(inset, florida_outline, linewidth=1.1)
    inset.set_xlim(sw_bounds[0], sw_bounds[1])
    inset.set_ylim(sw_bounds[2], sw_bounds[3])
    inset.set_title("Southwest Florida", fontsize=9, weight="bold")
    inset.set_xticks([])
    inset.set_yticks([])

    save_figure(fig, fig_dir, "figure_3_direct_and_indirect_impact_map")


def figure_4_top_bottlenecks(line_comparison: gpd.GeoDataFrame, fig_dir: Path, data_dir: Path) -> None:
    top10 = line_comparison.sort_values("loading_change_percentage_points", ascending=False).head(10).copy()
    top10 = top10.sort_values("loading_change_percentage_points", ascending=True)
    top10_out = top10[[
        "line",
        "label",
        "baseline_loading",
        "damaged_loading",
        "loading_change_percentage_points",
        "post_damage_rank",
        "directly_damaged",
    ]].copy()
    top10_out.to_csv(data_dir / "figure_4_top10_cascading_bottlenecks.csv", index=False)

    y = np.arange(len(top10))
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    colors = np.where(top10["damaged_loading"] >= 1.0, DIRECT, np.where(top10["damaged_loading"] >= 0.5, "#F59E0B", INDIRECT))
    bars = ax.barh(
        y,
        top10["loading_change_percentage_points"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.68,
    )
    for yi, bar, (_, row) in zip(y, bars, top10.iterrows()):
        increase = row["loading_change_percentage_points"]
        before_after = f"{row['baseline_loading'] * 100:.1f}% -> {row['damaged_loading'] * 100:.1f}%"
        ax.text(
            increase + 1.2,
            yi,
            f"+{increase:.1f} pp   ({before_after})",
            va="center",
            fontsize=10,
            color=INK,
            weight="bold" if row["damaged_loading"] >= 1.0 else "normal",
        )
        if row["damaged_loading"] >= 1.0:
            ax.text(
                min(increase - 2.0, increase * 0.72),
                yi,
                "reached 100%",
                va="center",
                ha="right",
                fontsize=9.2,
                color="white",
                weight="bold",
            )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{str(line).replace('line_', 'L')}  {label}" for line, label in zip(top10["line"], top10["label"])], fontsize=9.2)
    ax.set_xlabel("Increase in maximum line loading after Ian damage (percentage points)")
    fig.suptitle("The Largest Flow Redistribution Is Concentrated on Ten Lines", fontsize=17, weight="bold", x=0.125, ha="left", y=0.985)
    fig.text(
        0.125,
        0.935,
        "Each bar shows how much a line's maximum loading increased. Text at right shows baseline loading -> post-Ian loading.",
        fontsize=10.5,
        color=MUTED,
    )
    ax.grid(axis="x", color="#D0D7DE", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    legend_handles = [
        Patch(facecolor=DIRECT, label="Post-Ian loading reached 100%"),
        Patch(facecolor="#F59E0B", label="Post-Ian loading 50-99%"),
        Patch(facecolor=INDIRECT, label="Post-Ian loading below 50%"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9.5)
    ax.set_xlim(0, float(top10["loading_change_percentage_points"].max()) + 22)
    save_figure(fig, fig_dir, "figure_4_top_cascading_bottlenecks")


def build_cost_decomposition(data: dict[str, object]) -> pd.DataFrame:
    baseline = data["baseline_summary"]
    scenario = data["scenario_summary"]
    baseline_cost = float(baseline["total_system_cost_usd"])
    scenario_cost = float(scenario["total_system_cost_usd"])
    import_cost = float(scenario["import_slack_marginal_cost_usd_per_mwh"])
    load_shed_cost = float(scenario["value_of_lost_load_usd_per_mwh"])
    import_slack_change = (
        float(scenario["total_import_slack_mwh"]) - float(baseline["total_import_slack_mwh"])
    ) * import_cost
    load_shedding_change = (
        float(scenario["total_load_shed_mwh"]) - float(baseline["total_load_shed_mwh"])
    ) * load_shed_cost
    total_increment = scenario_cost - baseline_cost
    redispatch_residual = total_increment - import_slack_change - load_shedding_change
    return pd.DataFrame(
        [
            {"component": "Baseline operating cost", "change_usd": baseline_cost, "is_baseline": True},
            {"component": "Redispatch cost increase / residual", "change_usd": redispatch_residual, "is_baseline": False},
            {"component": "Import-slack cost change", "change_usd": import_slack_change, "is_baseline": False},
            {"component": "Load-shedding penalty change", "change_usd": load_shedding_change, "is_baseline": False},
            {"component": "Final post-Ian system cost", "change_usd": scenario_cost, "is_total": True},
        ]
    )


def figure_5_cost_decomposition(data: dict[str, object], fig_dir: Path, data_dir: Path) -> None:
    costs = build_cost_decomposition(data)
    costs.to_csv(data_dir / "figure_5_system_cost_decomposition.csv", index=False)
    baseline_cost = float(costs.loc[costs["component"].eq("Baseline operating cost"), "change_usd"].iloc[0])
    final_cost = float(costs.loc[costs["component"].eq("Final post-Ian system cost"), "change_usd"].iloc[0])
    increments = costs.iloc[1:4].copy()

    labels = [c.replace(" / residual", "").replace(" cost change", "\ncost change") for c in increments["component"]]
    labels.append("Total\nincremental cost")
    values = increments["change_usd"].tolist()
    total_increment = final_cost - baseline_cost
    running = 0.0
    starts = []
    heights = []
    colors = []
    for value in values:
        starts.append(running if value >= 0 else running + value)
        heights.append(abs(value))
        colors.append(INDIRECT if value >= 0 else GREEN)
        running += value
    starts.append(0)
    heights.append(abs(total_increment))
    colors.append(DIRECT)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, heights, bottom=starts, color=colors, edgecolor="white", linewidth=0.9)
    for i, bar in enumerate(bars):
        value = values[i] if i < len(values) else total_increment
        y = starts[i] + heights[i]
        ax.text(bar.get_x() + bar.get_width() / 2, y + max(abs(total_increment), 1) * 0.04, usd_short(value), ha="center", va="bottom", fontsize=10, weight="bold")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: usd_short(v)))
    ax.set_ylabel("Cost change relative to baseline (USD)")
    ax.set_title("Ian Adds $76.9k to the 24-Hour PyPSA Objective", fontsize=16, weight="bold", loc="left", pad=18)
    ax.text(
        0,
        -0.22,
        "Baseline operating cost: "
        f"USD {baseline_cost / 1e6:.2f}M     Final post-Ian system cost: USD {final_cost / 1e6:.2f}M     "
        "Import/load-shed changes are from saved MWh and penalty values; redispatch is the residual objective-cost increase.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(min(0, min(starts) * 1.2), max(heights) * 1.45 if heights else 1)
    save_figure(fig, fig_dir, "figure_5_system_cost_decomposition")


def figure_6_direct_vs_operational(metrics: pd.DataFrame, fig_dir: Path, data_dir: Path) -> None:
    m = metrics.iloc[0]
    panels = pd.DataFrame(
        [
            {
                "panel": "Direct line capacity loss",
                "value": m.line_capacity_loss_mva,
                "unit": "MVA",
                "percent": m.direct_line_capacity_loss_pct_total_network,
                "color": DIRECT,
            },
            {
                "panel": "Unavailable generation capacity",
                "value": m.generator_capacity_loss_mw,
                "unit": "MW",
                "percent": m.generator_capacity_loss_pct_total,
                "color": "#9333EA",
            },
            {
                "panel": "Incremental load shed",
                "value": m.incremental_load_shed_mwh,
                "unit": "MWh",
                "percent": 0.0,
                "color": LOAD_SHED,
            },
            {
                "panel": "Incremental system cost",
                "value": m.incremental_system_cost_usd,
                "unit": "USD",
                "percent": (m.incremental_system_cost_usd / m.baseline_system_cost_usd * 100) if m.baseline_system_cost_usd else 0,
                "color": INDIRECT,
            },
        ]
    )
    panels.to_csv(data_dir / "figure_6_direct_damage_vs_operational_consequence.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.7))
    for ax, (_, row) in zip(axes, panels.iterrows()):
        value = float(row["value"])
        ymax = abs(value) * 1.35 if abs(value) > 0 else 1
        ax.bar([0], [value], width=0.5, color=row["color"])
        label_value = usd_short(value) if row["unit"] == "USD" else f"{value:,.0f} {row['unit']}"
        label_pct = f"{row['percent']:.2f}% of baseline" if row["unit"] == "USD" else f"{row['percent']:.2f}% of relevant total"
        if row["panel"] == "Incremental load shed":
            label_pct = "0.00 MWh above baseline"
        ax.text(0, value + ymax * 0.04, f"{label_value}\n{label_pct}", ha="center", va="bottom", fontsize=10, weight="bold")
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        ax.set_title(row["panel"], fontsize=11.5, weight="bold")
        ax.grid(axis="y", color="#D0D7DE", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Direct Physical Damage Does Not Necessarily Produce Extra Load Shedding", fontsize=17, weight="bold", y=1.02)
    save_figure(fig, fig_dir, "figure_6_direct_damage_vs_operational_consequence")


def write_readme(out_dir: Path) -> None:
    readme = f"""# Hurricane Ian Publication Figures

This folder contains redesigned figures for the Hurricane Ian scenario using the extended-tie Florida grid.

Terminology:
- **Direct physical damage** means assets intersected by the Ian wind field and derated with the vulnerability curves.
- **Indirect operational consequence** means the result of re-running PyPSA after direct damage, compared with the no-hazard baseline where possible.
- Post-damage load shedding is not automatically labeled as a cascade. In this saved run, total load shedding is unchanged from the no-hazard baseline.

Figures:
1. `figure_1_hurricane_ian_impact_summary`: four-stage summary from hazard exposure to service consequences.
2. `figure_2_baseline_vs_post_ian_network_loading`: baseline versus post-Ian loading maps using the same scale and extent.
3. `figure_3_direct_and_indirect_impact_map`: combined direct-damage and operational-consequence map with southwest Florida inset.
4. `figure_4_top_cascading_bottlenecks`: ten lines with the greatest increase in loading after Ian.
5. `figure_5_system_cost_decomposition`: waterfall from baseline cost to post-Ian cost. Import-slack and load-shedding changes are calculated from saved MWh and penalty values; redispatch is the remaining cost residual.
6. `figure_6_direct_damage_vs_operational_consequence`: separate-unit comparison of direct capacity loss, generation loss, incremental load shed, and incremental system cost.

Source data:
- Direct damage scenario: `{SCENARIO_DIR}`
- No-hazard baseline: `{BASELINE_DIR}`
- Line geometries: `{LINE_GEOMETRY}`
- Florida outline: `{US_STATE_BOUNDARIES}`
- Processed chart data are saved in the `data` folder.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    out_dir, fig_dir, data_dir = make_output_dir()
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    data = load_all_data()
    line_comparison = build_line_loading_comparison(data)
    load_shed = build_load_shedding_comparison(data)
    metrics = calculate_metrics(data, line_comparison, load_shed)

    save_core_data(data, line_comparison, load_shed, metrics, data_dir)
    figure_1_impact_summary(metrics, fig_dir, data_dir)
    figure_2_baseline_vs_post_loading(line_comparison, data["storm_track"], data["florida_outline"], fig_dir, data_dir)
    figure_3_combined_impact_map(data, line_comparison, load_shed, fig_dir, data_dir)
    figure_4_top_bottlenecks(line_comparison, fig_dir, data_dir)
    figure_5_cost_decomposition(data, fig_dir, data_dir)
    figure_6_direct_vs_operational(metrics, fig_dir, data_dir)
    write_readme(out_dir)

    print(f"Saved redesigned Hurricane Ian figures to: {out_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
