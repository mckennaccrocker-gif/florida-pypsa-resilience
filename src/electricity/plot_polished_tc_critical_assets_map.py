"""Create polished map-only critical-asset figures for flood and TC hazards.

The figures use the latest capped-load-shedding return-period suite:
SNAIL exposure, updated NHESS curve assignments, voltage-layer transformers,
and the improved PyPSA network. They intentionally avoid inset line charts so
the Florida map stays the visual focus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from shapely.geometry import Point

PROJECT_DIR = Path(r"C:\oxford_tc_project")
NETWORK_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_voltage_transformers"
SUITE_DIR = (
    NETWORK_DIR
    / "calibrated_hazard_scenarios"
    / "gradual_return_period_suite_capped_load_shedding"
)
RISK_DIR = SUITE_DIR / "interpretation_and_annualized_risk"
LINE_GEOMETRY = PROJECT_DIR / "data" / "Electricity" / "florida_lines_with_s_nom.gpkg"
PLOT_CRS = "EPSG:3086"
FLORIDA_CRS = "EPSG:4326"


@dataclass(frozen=True)
class HazardStyle:
    hazard: str
    title: str
    subtitle: str
    output_name: str
    line_label: str
    capacity_column_label: str
    cmap: str
    generator_color: str
    generator_edge: str
    summary_label: str


HAZARDS = [
    HazardStyle(
        hazard="tropical_cyclone",
        title="Tropical Cyclone Critical Electricity Assets",
        subtitle=(
            "SNAIL + STORM return periods + updated NHESS TC curves, "
            "improved network topology, voltage-layer transformers, capped load shedding"
        ),
        output_name="tropical_cyclone_critical_assets_map_polished.png",
        line_label="Critical TC derated corridors",
        capacity_column_label="Total TC transmission capacity lost",
        cmap="inferno",
        generator_color="#ff8c32",
        generator_edge="#3b2111",
        summary_label="TC operational + annualized impact",
    ),
    HazardStyle(
        hazard="flood",
        title="Flood Critical Electricity Assets",
        subtitle=(
            "SNAIL + JRC return periods + F6.2/F1/F2 NHESS flood curves, "
            "improved network topology, voltage-layer transformers, capped load shedding"
        ),
        output_name="flood_critical_assets_map_polished.png",
        line_label="Critical flood derated corridors",
        capacity_column_label="Total flood transmission capacity lost",
        cmap="viridis",
        generator_color="#2d9cdb",
        generator_edge="#0e3d59",
        summary_label="Flood operational + annualized impact",
    ),
]


def money(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.0f}k"
    return f"${value:.0f}"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    suite = pd.read_csv(SUITE_DIR / "gradual_return_period_suite_summary.csv")
    lines = pd.read_csv(RISK_DIR / "top_critical_lines_by_hazard.csv")
    generators = pd.read_csv(RISK_DIR / "top_critical_generators_by_hazard.csv")
    buses = pd.read_csv(RISK_DIR / "top_load_shed_buses_by_hazard.csv")
    risk = pd.read_csv(RISK_DIR / "annualized_risk_summary.csv")
    return suite, lines, generators, buses, risk


def load_line_geometries(lines: pd.DataFrame) -> gpd.GeoDataFrame:
    geo = gpd.read_file(LINE_GEOMETRY)
    if "source_edge_id" not in geo.columns and "florida_line_id" in geo.columns:
        geo = geo.rename(columns={"florida_line_id": "source_edge_id"})
    geo["source_edge_id"] = pd.to_numeric(geo["source_edge_id"], errors="coerce")
    if "line" not in lines.columns and "name" in lines.columns:
        lines = lines.rename(columns={"name": "line"})
    merged = lines[["line", "source_edge_id"]].merge(geo, on="source_edge_id", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=geo.crs).dropna(subset=["geometry"])


def geodataframe_from_xy(df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(x, y) if pd.notna(x) and pd.notna(y) else None for x, y in zip(df["x"], df["y"])],
        crs=FLORIDA_CRS,
    )
    return gdf.dropna(subset=["geometry"]).to_crs(PLOT_CRS)


def add_text_box(ax, text: str, x: float, y: float, size: float = 9.2) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        ha="left",
        va="top",
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.42,rounding_size=0.12",
            "facecolor": "white",
            "edgecolor": "#cfcfcf",
            "alpha": 0.94,
        },
        zorder=20,
    )


def annotate_top_generators(ax, gen_geo: gpd.GeoDataFrame, count: int = 8) -> None:
    if gen_geo.empty:
        return
    for _, row in gen_geo.sort_values("total_capacity_loss_mw", ascending=False).head(count).iterrows():
        label = str(row.get("source_name", row["generator"]))
        label = fill(label, width=20)
        ax.annotate(
            label,
            xy=(row.geometry.x, row.geometry.y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=6.8,
            color="#333333",
            path_effects=[pe.withStroke(linewidth=2.7, foreground="white")],
            zorder=12,
        )


def build_summary_text(style: HazardStyle, suite: pd.DataFrame, risk: pd.DataFrame) -> str:
    hazard_suite = suite[suite["hazard"].eq(style.hazard)].sort_values("return_period")
    hazard_risk = risk[risk["hazard"].eq(style.hazard)].iloc[0]
    max_rp = hazard_suite["return_period"].max()
    max_rp_row = hazard_suite[hazard_suite["return_period"].eq(max_rp)].iloc[0]
    return (
        f"{style.summary_label}\n"
        f"Highest RP shown: RP{int(max_rp)}\n"
        f"Load shed: {max_rp_row['load_shed_mwh']:.1f} MWh\n"
        f"Demand served: {max_rp_row['demand_served_percent']:.3f}%\n"
        f"Line capacity lost: {max_rp_row['line_capacity_loss_mva']:,.0f} MVA\n"
        f"Generator capacity lost: {max_rp_row['generator_capacity_loss_mw']:,.0f} MW\n"
        f"EALS: {hazard_risk['expected_annual_load_shed_mwh_per_year']:.2f} MWh/year\n"
        f"Incremental EAC: {money(hazard_risk['expected_annual_incremental_system_cost_usd_per_year'])}/year"
    )


def plot_hazard(
    style: HazardStyle,
    all_lines_geo: gpd.GeoDataFrame,
    suite: pd.DataFrame,
    line_summary: pd.DataFrame,
    gen_summary: pd.DataFrame,
    bus_summary: pd.DataFrame,
    risk: pd.DataFrame,
) -> Path:
    hazard_lines = line_summary[line_summary["hazard"].eq(style.hazard)].copy()
    hazard_gens = gen_summary[gen_summary["hazard"].eq(style.hazard)].copy()
    hazard_buses = bus_summary[bus_summary["hazard"].eq(style.hazard)].copy()

    top_lines = hazard_lines.sort_values(
        ["total_capacity_loss_mva", "associated_endpoint_load_shed_mwh", "affected_return_periods"],
        ascending=False,
    ).head(120)
    top_lines_geo = all_lines_geo.merge(
        top_lines[
            [
                "line",
                "total_capacity_loss_mva",
                "max_damage_ratio",
                "affected_return_periods",
                "associated_endpoint_load_shed_mwh",
            ]
        ],
        on="line",
        how="inner",
    )

    top_gens = hazard_gens.sort_values(
        ["total_capacity_loss_mw", "max_damage_ratio", "affected_return_periods"],
        ascending=False,
    ).head(45)
    gen_geo = geodataframe_from_xy(top_gens)
    bus_geo = geodataframe_from_xy(hazard_buses) if not hazard_buses.empty else gpd.GeoDataFrame(geometry=[], crs=PLOT_CRS)
    minx, miny, maxx, maxy = all_lines_geo.total_bounds
    midx = (minx + maxx) / 2.0

    fig = plt.figure(figsize=(9.2, 11.2), facecolor="#fbfbf8")
    ax = fig.add_axes([0.035, 0.075, 0.79, 0.82])
    cax = fig.add_axes([0.86, 0.23, 0.028, 0.50])
    ax.set_facecolor("#fbfbf8")

    all_lines_geo.plot(ax=ax, color="#d7d7d7", linewidth=0.24, alpha=0.43, zorder=1)

    if not top_lines_geo.empty:
        top_lines_geo.plot(ax=ax, color="white", linewidth=5.4, alpha=0.78, zorder=2)
        widths = np.clip(top_lines_geo["max_damage_ratio"] * 9.0, 1.25, 5.2)
        top_lines_geo.plot(
            ax=ax,
            column="total_capacity_loss_mva",
            cmap=style.cmap,
            linewidth=widths,
            alpha=0.96,
            zorder=4,
        )
        norm = Normalize(
            vmin=float(top_lines_geo["total_capacity_loss_mva"].min()),
            vmax=float(top_lines_geo["total_capacity_loss_mva"].max()),
        )
        sm = ScalarMappable(norm=norm, cmap=style.cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(f"{style.capacity_column_label} (MVA across RPs)", fontsize=8.8)
        cbar.ax.tick_params(labelsize=8)
    else:
        cax.set_visible(False)

    if not gen_geo.empty:
        sizes = np.clip(gen_geo["total_capacity_loss_mw"] * 7.0, 42, 430)
        gen_geo.plot(ax=ax, markersize=sizes * 1.6, color="white", edgecolor="none", alpha=0.78, zorder=5)
        gen_geo.plot(
            ax=ax,
            markersize=sizes,
            color=style.generator_color,
            edgecolor=style.generator_edge,
            linewidth=0.68,
            alpha=0.92,
            zorder=6,
        )
        annotate_top_generators(ax, gen_geo, count=8)

    if not bus_geo.empty:
        bus_sizes = np.clip(bus_geo["total_load_shed_mwh"] * 6.0, 170, 560)
        bus_geo.plot(
            ax=ax,
            markersize=bus_sizes,
            color="#1f63ff",
            edgecolor="white",
            linewidth=1.5,
            alpha=0.98,
            zorder=8,
        )
        for _, row in bus_geo.iterrows():
            label_to_right = row.geometry.x <= midx
            ax.annotate(
                f"{row['bus']}\n{row['total_load_shed_mwh']:.1f} MWh shed",
                xy=(row.geometry.x, row.geometry.y),
                xytext=(18, 16) if label_to_right else (-18, 16),
                textcoords="offset points",
                fontsize=8.3,
                fontweight="bold",
                ha="left" if label_to_right else "right",
                color="#1f3f91",
                arrowprops={"arrowstyle": "-", "color": "#1f63ff", "lw": 1.15},
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
                zorder=13,
            )

    xpad = (maxx - minx) * 0.03
    ypad = (maxy - miny) * 0.035
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.text(0.035, 0.955, style.title, fontsize=20, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.928, style.subtitle, fontsize=9.1, color="#555555", ha="left", va="top")

    summary_text = build_summary_text(style, suite, risk)
    add_text_box(ax, summary_text, 0.02, 0.16, size=8.8)

    handles = [
        Line2D([0], [0], color="#6b1f7a" if style.hazard == "tropical_cyclone" else "#226a72", lw=3.2, label=style.line_label),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=style.generator_color,
            markeredgecolor=style.generator_edge,
            markersize=8.5,
            label="Top derated generators",
        ),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f63ff", markeredgecolor="white", markersize=9, label="Load-shed bus"),
        Line2D([0], [0], color="#d7d7d7", lw=1.15, label="Full Florida network"),
    ]
    if bus_geo.empty:
        handles.pop(2)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.33), frameon=True, framealpha=0.94, fontsize=8.7)

    footer = (
        "Map-only critical asset view. Corridor color = cumulative transmission capacity loss; "
        "line width = maximum damage ratio; generator size = cumulative MW lost across available return periods."
    )
    fig.text(0.035, 0.035, fill(footer, width=128), ha="left", fontsize=8.0, color="#555555")

    output = RISK_DIR / style.output_name
    fig.savefig(output, dpi=340)
    plt.close(fig)
    return output


def main() -> None:
    suite, line_summary, gen_summary, bus_summary, risk = read_inputs()
    all_lines = pd.read_csv(NETWORK_DIR / "lines.csv")
    all_lines_geo = load_line_geometries(all_lines).to_crs(PLOT_CRS)

    outputs = []
    for style in HAZARDS:
        outputs.append(plot_hazard(style, all_lines_geo, suite, line_summary, gen_summary, bus_summary, risk))

    for output in outputs:
        print("Saved:", output)


if __name__ == "__main__":
    main()
