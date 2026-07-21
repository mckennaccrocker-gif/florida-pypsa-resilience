"""
Review large generator-to-bus assignments in the Florida PyPSA network.

Outputs:
  - large_generator_bus_assignment_review.csv
  - large_generator_bus_override_template.csv
  - large_generator_bus_assignment_map.png

The review flags large plants assigned to weak/low-voltage/radial/distant buses
and lists nearby high-voltage candidate buses. It does not change the network.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network_county_population_load"
BASE_NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network"
OUTPUT_DIR = NETWORK_DIR / "generator_bus_assignment_review"
PROJECTED_CRS = "EPSG:3086"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_path = NETWORK_DIR / "generator_bus_matches_review.csv"
    if not review_path.exists():
        review_path = BASE_NETWORK_DIR / "generator_bus_matches_review.csv"
    review = pd.read_csv(review_path)
    generators = pd.read_csv(NETWORK_DIR / "generators_with_final_marginal_costs.csv")
    buses = pd.read_csv(NETWORK_DIR / "buses.csv")
    return review, generators, buses


def candidate_buses_for_generators(
    generators_review: pd.DataFrame,
    buses: pd.DataFrame,
    max_distance_km: float = 20.0,
    high_voltage_kv: float = 230.0,
    top_n: int = 5,
) -> pd.DataFrame:
    gen_geo = gpd.GeoDataFrame(
        generators_review.copy(),
        geometry=gpd.points_from_xy(generators_review["longitude"], generators_review["latitude"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    bus_geo = gpd.GeoDataFrame(
        buses.copy(),
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    bus_geo["v_nom"] = pd.to_numeric(bus_geo["v_nom"], errors="coerce")
    bus_geo["connected_edge_count"] = pd.to_numeric(
        bus_geo["connected_edge_count"], errors="coerce"
    ).fillna(0)

    candidate_bus_geo = bus_geo[bus_geo["v_nom"] >= high_voltage_kv].copy()
    tree = cKDTree(np.c_[candidate_bus_geo.geometry.x, candidate_bus_geo.geometry.y])

    rows = []
    for gen in gen_geo.itertuples(index=False):
        distances, indices = tree.query([[gen.geometry.x, gen.geometry.y]], k=min(top_n, len(candidate_bus_geo)))
        for rank, (distance_m, candidate_idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if float(distance_m) > max_distance_km * 1000:
                continue
            bus = candidate_bus_geo.iloc[int(candidate_idx)]
            rows.append(
                {
                    "generator": gen.generator,
                    "candidate_rank": rank,
                    "candidate_bus": bus["name"],
                    "candidate_bus_v_nom": bus["v_nom"],
                    "candidate_bus_connected_edge_count": bus["connected_edge_count"],
                    "candidate_bus_substation_name": bus.get("substation_name", ""),
                    "candidate_distance_m": float(distance_m),
                }
            )
    return pd.DataFrame(rows)


def build_review_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    review, generators, buses = load_inputs()
    generators = generators[["name", "bus", "carrier", "p_nom", "source_name", "marginal_cost"]].copy()
    review = review.merge(
        generators.rename(columns={"name": "generator"})[
            ["generator", "bus", "p_nom", "marginal_cost"]
        ],
        on=["generator", "bus"],
        how="left",
    )
    review["capacity_mw"] = pd.to_numeric(review["capacity_mw"], errors="coerce")
    review["p_nom"] = pd.to_numeric(review["p_nom"], errors="coerce").fillna(review["capacity_mw"])

    buses_light = buses[
        ["name", "v_nom", "connected_edge_count", "substation_name", "x", "y"]
    ].rename(
        columns={
            "name": "bus",
            "v_nom": "assigned_bus_v_nom",
            "connected_edge_count": "assigned_bus_connected_edge_count",
            "substation_name": "assigned_bus_substation_name",
            "x": "assigned_bus_x",
            "y": "assigned_bus_y",
        }
    )
    review = review.merge(buses_light, on="bus", how="left", validate="many_to_one")
    review["assigned_bus_v_nom"] = pd.to_numeric(review["assigned_bus_v_nom"], errors="coerce")
    review["assigned_bus_connected_edge_count"] = pd.to_numeric(
        review["assigned_bus_connected_edge_count"], errors="coerce"
    )

    candidates = candidate_buses_for_generators(review, buses)
    nearest_candidate = (
        candidates.sort_values(["generator", "candidate_rank"])
        .groupby("generator", as_index=False)
        .first()
    )
    nearest_candidate = nearest_candidate.rename(
        columns={
            "candidate_bus": "nearest_high_voltage_candidate_bus",
            "candidate_bus_v_nom": "nearest_high_voltage_candidate_v_nom",
            "candidate_bus_connected_edge_count": "nearest_high_voltage_candidate_connected_edges",
            "candidate_bus_substation_name": "nearest_high_voltage_candidate_substation_name",
            "candidate_distance_m": "nearest_high_voltage_candidate_distance_m",
        }
    )
    review = review.merge(
        nearest_candidate.drop(columns=["candidate_rank"], errors="ignore"),
        on="generator",
        how="left",
    )

    review["large_generator"] = review["p_nom"] >= 100.0
    review["very_large_generator"] = review["p_nom"] >= 500.0
    review["assigned_low_voltage_for_large_gen"] = (
        review["large_generator"] & (review["assigned_bus_v_nom"] < 115.0)
    )
    review["assigned_below_230_for_very_large_gen"] = (
        review["very_large_generator"] & (review["assigned_bus_v_nom"] < 230.0)
    )
    review["assigned_radial_or_weak_bus"] = (
        review["large_generator"] & (review["assigned_bus_connected_edge_count"] <= 1)
    )
    review["assigned_far_from_bus"] = (
        review["large_generator"] & (pd.to_numeric(review["nearest_bus_distance_m"], errors="coerce") > 5000)
    )
    review["nearby_hv_candidate_better_than_assigned"] = (
        review["large_generator"]
        & review["nearest_high_voltage_candidate_bus"].notna()
        & (review["nearest_high_voltage_candidate_bus"] != review["bus"])
        & (
            review["nearest_high_voltage_candidate_v_nom"].fillna(0)
            > review["assigned_bus_v_nom"].fillna(0)
        )
    )
    flag_cols = [
        "assigned_low_voltage_for_large_gen",
        "assigned_below_230_for_very_large_gen",
        "assigned_radial_or_weak_bus",
        "assigned_far_from_bus",
        "nearby_hv_candidate_better_than_assigned",
    ]
    review["suspicious_assignment_flag_count"] = review[flag_cols].sum(axis=1)
    review["review_priority_score"] = review["p_nom"].fillna(0) * (
        1 + review["suspicious_assignment_flag_count"]
    )

    candidate_strings = (
        candidates.assign(
            candidate_summary=lambda df: (
                df["candidate_rank"].astype(str)
                + ":"
                + df["candidate_bus"].astype(str)
                + " "
                + df["candidate_bus_v_nom"].round(0).astype("Int64").astype(str)
                + "kV "
                + (df["candidate_distance_m"] / 1000).round(2).astype(str)
                + "km"
            )
        )
        .groupby("generator")["candidate_summary"]
        .apply("; ".join)
        .rename("nearby_high_voltage_candidates")
        .reset_index()
    )
    review = review.merge(candidate_strings, on="generator", how="left")

    review = review.sort_values(["review_priority_score", "p_nom"], ascending=False)
    return review, candidates


def write_override_template(review: pd.DataFrame) -> pd.DataFrame:
    flagged = review[
        (review["large_generator"]) & (review["suspicious_assignment_flag_count"] > 0)
    ].copy()
    template = flagged[
        [
            "generator",
            "plant_name",
            "carrier",
            "p_nom",
            "bus",
            "assigned_bus_v_nom",
            "assigned_bus_connected_edge_count",
            "nearest_high_voltage_candidate_bus",
            "nearest_high_voltage_candidate_v_nom",
            "nearest_high_voltage_candidate_distance_m",
            "nearby_high_voltage_candidates",
            "suspicious_assignment_flag_count",
        ]
    ].copy()
    template = template.rename(columns={"bus": "current_bus"})
    template.insert(
        template.columns.get_loc("current_bus") + 1,
        "proposed_bus_override",
        template["nearest_high_voltage_candidate_bus"],
    )
    template["apply_override"] = False
    template["override_reason"] = ""
    return template


def plot_review_map(review: pd.DataFrame) -> None:
    top = review.head(50).copy()
    buses = pd.read_csv(NETWORK_DIR / "buses.csv")
    lines = pd.read_csv(NETWORK_DIR / "lines.csv")

    bus_geo = gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    )
    line_bus0 = lines.merge(bus_geo[["name", "geometry"]], left_on="bus0", right_on="name", how="left")
    line_bus0 = line_bus0.rename(columns={"geometry": "geom0"}).drop(columns=["name_y"], errors="ignore")
    line_bus = line_bus0.merge(bus_geo[["name", "geometry"]], left_on="bus1", right_on="name", how="left")
    line_bus = line_bus.rename(columns={"geometry": "geom1"}).drop(columns=["name"], errors="ignore")

    from shapely.geometry import LineString

    line_geoms = [
        LineString([a, b]) if a is not None and b is not None else None
        for a, b in zip(line_bus["geom0"], line_bus["geom1"])
    ]
    line_geo = gpd.GeoDataFrame(line_bus, geometry=line_geoms, crs="EPSG:4326")
    gen_geo = gpd.GeoDataFrame(
        top,
        geometry=gpd.points_from_xy(top["longitude"], top["latitude"]),
        crs="EPSG:4326",
    )
    assigned_bus_geo = bus_geo[bus_geo["name"].isin(top["bus"])]

    fig, ax = plt.subplots(figsize=(11, 10))
    line_geo.plot(ax=ax, color="#cfd8dc", linewidth=0.35, alpha=0.45)
    assigned_bus_geo.plot(ax=ax, color="#1976d2", markersize=12, alpha=0.7, label="Assigned buses")
    sizes = np.sqrt(top["p_nom"].clip(lower=1)) * 10
    colors = np.where(top["suspicious_assignment_flag_count"] > 0, "#d84315", "#2e7d32")
    gen_geo.plot(ax=ax, color=colors, markersize=sizes, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.set_title("Top 50 Generator-to-Bus Assignments by Review Priority")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "large_generator_bus_assignment_map.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    review, candidates = build_review_table()
    review.to_csv(OUTPUT_DIR / "large_generator_bus_assignment_review.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "large_generator_nearby_bus_candidates.csv", index=False)
    template = write_override_template(review)
    template.to_csv(OUTPUT_DIR / "large_generator_bus_override_template.csv", index=False)
    plot_review_map(review)

    summary = pd.DataFrame(
        [
            {
                "generators_reviewed": len(review),
                "large_generators_100mw_plus": int((review["p_nom"] >= 100).sum()),
                "very_large_generators_500mw_plus": int((review["p_nom"] >= 500).sum()),
                "flagged_large_generators": int(
                    ((review["p_nom"] >= 100) & (review["suspicious_assignment_flag_count"] > 0)).sum()
                ),
                "override_template_rows": len(template),
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "large_generator_bus_assignment_review_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nTop flagged generators:")
    cols = [
        "generator",
        "plant_name",
        "carrier",
        "p_nom",
        "bus",
        "assigned_bus_v_nom",
        "assigned_bus_connected_edge_count",
        "nearest_bus_distance_m",
        "nearest_high_voltage_candidate_bus",
        "nearest_high_voltage_candidate_v_nom",
        "nearest_high_voltage_candidate_distance_m",
        "suspicious_assignment_flag_count",
    ]
    print(
        review[(review["p_nom"] >= 100) & (review["suspicious_assignment_flag_count"] > 0)][cols]
        .head(20)
        .to_string(index=False)
    )
    print("\nSaved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
