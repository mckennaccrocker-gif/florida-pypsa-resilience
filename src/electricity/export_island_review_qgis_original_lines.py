"""
Export the island-reviewed network for QGIS using the original HIFLD line shapes.

This differs from a PyPSA topology plot: PyPSA lines are bus-to-bus model edges,
but this file preserves the original detailed transmission-line geometry from
`florida_network_edges_transmission_lines.gpkg` wherever a line came from HIFLD.
Manual island-review connector lines are exported separately because they are
new topology connectors, not original HIFLD geometries.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
REVIEWED_NETWORK = ELECTRICITY_DIR / "pypsa_florida_network_island_reviewed_no_connectors"
INPUT_NETWORK = ELECTRICITY_DIR / "pypsa_florida_network_bus311_load_only_cleanup"
SOURCE_LINES = ELECTRICITY_DIR / "florida_network_edges_transmission_lines.gpkg"
OUTPUT = REVIEWED_NETWORK / "qgis" / "florida_island_reviewed_original_transmission_geometry.gpkg"
REMOVED_OUTPUT = REVIEWED_NETWORK / "qgis" / "removed_disregarded_original_transmission_lines.gpkg"


def clean_for_gpkg(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    for column in gdf.columns:
        if column == gdf.geometry.name:
            continue
        if gdf[column].dtype == object:
            gdf[column] = gdf[column].where(pd.notna(gdf[column]), "").astype(str)
    return gdf


def make_bus_layer(buses: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        buses.copy(),
        geometry=[Point(xy) for xy in zip(buses["x"], buses["y"])],
        crs="EPSG:4326",
    )


def make_connector_layer(lines: pd.DataFrame, buses: pd.DataFrame) -> gpd.GeoDataFrame:
    bus_lookup = buses.set_index("name")
    connectors = lines[lines["source_edge_id"].astype(float).lt(0)].copy()
    rows = []
    for _, line in connectors.iterrows():
        bus0 = bus_lookup.loc[str(line["bus0"])]
        bus1 = bus_lookup.loc[str(line["bus1"])]
        row = line.to_dict()
        row["geometry"] = LineString(
            [
                (float(bus0["x"]), float(bus0["y"])),
                (float(bus1["x"]), float(bus1["y"])),
            ]
        )
        rows.append(row)
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    if REMOVED_OUTPUT.exists():
        REMOVED_OUTPUT.unlink()

    buses = pd.read_csv(REVIEWED_NETWORK / "buses.csv")
    lines = pd.read_csv(REVIEWED_NETWORK / "lines.csv")
    input_lines = pd.read_csv(INPUT_NETWORK / "lines.csv")
    source_edges = gpd.read_file(SOURCE_LINES, layer="edge_component")

    buses["name"] = buses["name"].astype(str)
    lines["name"] = lines["name"].astype(str)
    input_lines["name"] = input_lines["name"].astype(str)

    reviewed_source_ids = set(
        pd.to_numeric(lines["source_edge_id"], errors="coerce").dropna().astype(int)
    )
    reviewed_source_ids = {edge_id for edge_id in reviewed_source_ids if edge_id >= 0}

    original_source_ids = set(
        pd.to_numeric(input_lines["source_edge_id"], errors="coerce").dropna().astype(int)
    )
    original_source_ids = {edge_id for edge_id in original_source_ids if edge_id >= 0}
    removed_source_ids = original_source_ids - reviewed_source_ids

    pypsa_attrs = lines[lines["source_edge_id"].isin(reviewed_source_ids)].copy()
    pypsa_attrs["edge_id"] = pypsa_attrs["source_edge_id"].astype(int)
    for column in ["review_component_id", "review_decision_note"]:
        if column not in pypsa_attrs.columns:
            pypsa_attrs[column] = ""
    pypsa_attrs = pypsa_attrs[
        [
            "edge_id",
            "name",
            "bus0",
            "bus1",
            "v_nom",
            "s_nom",
            "r",
            "x",
            "capacity_source",
            "review_component_id",
            "review_decision_note",
        ]
    ].rename(
        columns={
            "name": "pypsa_line",
            "bus0": "pypsa_bus0",
            "bus1": "pypsa_bus1",
            "v_nom": "pypsa_v_nom",
            "s_nom": "pypsa_s_nom",
        }
    )

    kept_original = source_edges[source_edges["edge_id"].isin(reviewed_source_ids)].merge(
        pypsa_attrs,
        on="edge_id",
        how="left",
    )
    removed_original = source_edges[source_edges["edge_id"].isin(removed_source_ids)].copy()

    # The one-bus self-loop with HIFLD ID 129742 was already dropped before PyPSA
    # line import, so also expose it in the removed layer by its original HIFLD ID.
    dropped_self_loop = source_edges[source_edges["ID"].astype(str).eq("129742")].copy()
    if not dropped_self_loop.empty:
        removed_original = pd.concat([removed_original, dropped_self_loop], ignore_index=True)
        removed_original = removed_original.drop_duplicates(subset=["edge_id"])

    bus_layer = make_bus_layer(buses)
    clean_for_gpkg(bus_layer).to_file(OUTPUT, layer="cleaned_buses", driver="GPKG")
    clean_for_gpkg(kept_original).to_file(
        OUTPUT,
        layer="cleaned_original_transmission_lines",
        driver="GPKG",
    )
    clean_for_gpkg(removed_original).to_file(
        REMOVED_OUTPUT,
        layer="removed_disregarded_original_lines",
        driver="GPKG",
    )

    print(OUTPUT.resolve())
    print("cleaned buses:", len(bus_layer))
    print("cleaned original transmission lines:", len(kept_original))
    print("added review connectors:", 0)
    print("removed/disregarded original lines, QA file:", len(removed_original), REMOVED_OUTPUT.resolve())


if __name__ == "__main__":
    main()
