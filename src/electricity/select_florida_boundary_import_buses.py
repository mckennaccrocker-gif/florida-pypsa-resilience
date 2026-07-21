"""
Select physically motivated emergency import buses for the Florida PyPSA model.

The model previously placed import slack in every load-bearing island. This is
useful as a calibration device, but real imports into Florida should primarily
enter through northern interties with Georgia and Alabama.

This script infers import buses from the existing PyPSA bus table by:

1. Reading Census state boundaries for Florida, Georgia, and Alabama.
2. Building the shared/near northern state-boundary geometry.
3. Selecting Florida buses close to that boundary.
4. Prioritising high-voltage and highly connected buses.
5. Saving selected buses plus all candidates for review.

It does not overwrite the network. The selected buses are used by calibration
and scenario scripts as explicit import-slack locations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
PYPSA_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network"
BOUNDARY_ZIP = PROJECT_DIR / "data" / "Boundaries" / "tl_2023_us_state.zip"
DEFAULT_OUTPUT_DIR = PYPSA_DIR / "boundary_import_buses"
PROJECTED_CRS = "EPSG:3086"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Florida boundary import buses.")
    parser.add_argument("--network-dir", type=Path, default=PYPSA_DIR)
    parser.add_argument("--state-boundary", type=Path, default=BOUNDARY_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-distance-km", type=float, default=35.0)
    parser.add_argument("--min-voltage-kv", type=float, default=230.0)
    parser.add_argument("--count", type=int, default=8)
    return parser.parse_args()


def load_boundary_geometry(path: Path):
    states = gpd.read_file(path).to_crs(PROJECTED_CRS)
    subset = states[states["STUSPS"].isin(["FL", "GA", "AL"])].copy()
    if set(subset["STUSPS"]) != {"FL", "GA", "AL"}:
        raise ValueError("Could not find FL, GA, and AL in state boundary file.")

    florida = subset[subset["STUSPS"].eq("FL")].geometry.iloc[0]
    neighbors = subset[subset["STUSPS"].isin(["GA", "AL"])].unary_union

    # Census state polygons may not share exact linework after projection, so use
    # the portion of Florida's boundary that is close to GA/AL rather than exact
    # line intersection only.
    northern_boundary = florida.boundary.intersection(neighbors.buffer(5000))
    if northern_boundary.is_empty:
        northern_boundary = florida.boundary
    return northern_boundary


def select_import_buses(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    buses = pd.read_csv(args.network_dir / "buses.csv")
    buses_geo = gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    boundary = load_boundary_geometry(args.state_boundary)

    buses_geo["distance_to_ga_al_boundary_km"] = buses_geo.geometry.distance(boundary) / 1000.0
    buses_geo["v_nom"] = pd.to_numeric(buses_geo["v_nom"], errors="coerce")
    buses_geo["connected_edge_count"] = pd.to_numeric(
        buses_geo.get("connected_edge_count", 0), errors="coerce"
    ).fillna(0)

    candidates = buses_geo[
        (buses_geo["distance_to_ga_al_boundary_km"] <= args.max_distance_km)
        & (buses_geo["v_nom"] >= args.min_voltage_kv)
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No boundary import candidates found. Try increasing --max-distance-km "
            "or lowering --min-voltage-kv."
        )

    # Spread selected buses across the border by longitude while still preferring
    # high voltage and strong connectivity.
    candidates["longitude_bin"] = pd.qcut(
        candidates["x"].rank(method="first"),
        q=min(args.count, len(candidates)),
        duplicates="drop",
    )
    ranked = candidates.sort_values(
        ["v_nom", "connected_edge_count", "distance_to_ga_al_boundary_km"],
        ascending=[False, False, True],
    )
    selected_rows = []
    used_bins = set()
    for _, row in ranked.iterrows():
        bin_label = str(row["longitude_bin"])
        if bin_label in used_bins and len(selected_rows) < args.count:
            continue
        selected_rows.append(row)
        used_bins.add(bin_label)
        if len(selected_rows) >= args.count:
            break
    if len(selected_rows) < args.count:
        selected_names = {row["name"] for row in selected_rows}
        for _, row in ranked.iterrows():
            if row["name"] in selected_names:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= args.count:
                break

    selected = gpd.GeoDataFrame(selected_rows, crs=buses_geo.crs).to_crs("EPSG:4326")
    candidates = candidates.to_crs("EPSG:4326")
    keep = [
        "name",
        "substation_name",
        "v_nom",
        "connected_edge_count",
        "x",
        "y",
        "distance_to_ga_al_boundary_km",
    ]
    return selected[keep].copy(), candidates[keep].sort_values(
        ["distance_to_ga_al_boundary_km", "v_nom"], ascending=[True, False]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, candidates = select_import_buses(args)
    selected.to_csv(args.output_dir / "selected_boundary_import_buses.csv", index=False)
    candidates.to_csv(args.output_dir / "boundary_import_bus_candidates.csv", index=False)
    print("Selected boundary import buses:")
    print(selected.to_string(index=False))
    print("\nSaved:", args.output_dir)


if __name__ == "__main__":
    main()
