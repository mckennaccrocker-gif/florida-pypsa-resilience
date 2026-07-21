"""Create a locally adjusted load-allocation copy of the improved network.

The summer peak diagnostics showed that remaining no-hazard load shedding at
high global line multipliers is concentrated at six buses with binding local
incident corridors. This script creates a documented network variant that
redistributes part of those buses' county-population load to nearby, same-county
recipient buses with equal-or-higher voltage and stronger connectivity.

This is a targeted test of whether the residual problem is caused by overly
concentrated load allocation at specific buses rather than statewide shortage.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
BASE_NETWORK = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_county_load_generator_overrides"
OUTPUT_NETWORK = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_county_load_local_adjusted"
DIAG_DIR = BASE_NETWORK / "summer_peak_multiplier_sweep" / "x3_remaining_shedding_diagnosis"

SOURCE_BUSES = ["bus_918", "bus_769", "bus_751", "bus_311", "bus_723", "bus_327"]
TRANSFER_FRACTION = 0.50
RECIPIENT_COUNT = 3


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def copy_network() -> None:
    if OUTPUT_NETWORK.exists():
        shutil.rmtree(OUTPUT_NETWORK)
    OUTPUT_NETWORK.mkdir(parents=True)
    for path in BASE_NETWORK.iterdir():
        if path.is_file():
            shutil.copy2(path, OUTPUT_NETWORK / path.name)


def choose_recipients(buses: pd.DataFrame, source_bus: str) -> pd.DataFrame:
    source = buses.loc[source_bus]
    same_county = buses[
        (buses["county_name"].eq(source["county_name"]))
        & (buses.index != source_bus)
    ].copy()

    preferred = same_county[
        (pd.to_numeric(same_county["v_nom"], errors="coerce") >= float(source["v_nom"]))
        & (
            pd.to_numeric(same_county["connected_edge_count"], errors="coerce")
            >= max(3, float(source["connected_edge_count"]) - 1)
        )
    ].copy()
    if len(preferred) < RECIPIENT_COUNT:
        preferred = same_county[
            pd.to_numeric(same_county["v_nom"], errors="coerce") >= 115
        ].copy()
    if len(preferred) < RECIPIENT_COUNT:
        preferred = same_county.copy()

    preferred["distance_km"] = haversine_km(
        source["x"],
        source["y"],
        preferred["x"],
        preferred["y"],
    )
    preferred["recipient_score"] = (
        pd.to_numeric(preferred["v_nom"], errors="coerce").fillna(0.0) / 500.0
        + pd.to_numeric(preferred["connected_edge_count"], errors="coerce").fillna(0.0) / 20.0
        - preferred["distance_km"] / 100.0
    )
    return preferred.sort_values(
        ["distance_km", "recipient_score"], ascending=[True, False]
    ).head(RECIPIENT_COUNT)


def main() -> None:
    copy_network()

    buses = pd.read_csv(BASE_NETWORK / "bus_county_population_load_weights.csv").set_index("name")
    loads = pd.read_csv(OUTPUT_NETWORK / "loads.csv")
    loads_p = pd.read_csv(OUTPUT_NETWORK / "loads-p_set.csv")
    load_by_bus = loads.set_index("bus")["name"].to_dict()

    adjustment_rows = []
    for source_bus in SOURCE_BUSES:
        if source_bus not in load_by_bus:
            raise ValueError(f"No load found at source bus {source_bus}")
        source_load = load_by_bus[source_bus]
        recipients = choose_recipients(buses, source_bus)
        if recipients.empty:
            raise ValueError(f"No recipients found for {source_bus}")

        transfer = loads_p[source_load] * TRANSFER_FRACTION
        loads_p[source_load] = loads_p[source_load] - transfer

        weights = pd.to_numeric(recipients["connected_edge_count"], errors="coerce").fillna(1.0)
        weights = weights / weights.sum()
        for recipient_bus, weight in weights.items():
            recipient_load = load_by_bus.get(recipient_bus)
            if recipient_load is None:
                continue
            loads_p[recipient_load] = loads_p[recipient_load] + transfer * float(weight)
            adjustment_rows.append(
                {
                    "source_bus": source_bus,
                    "source_load": source_load,
                    "source_county": buses.loc[source_bus, "county_name"],
                    "source_v_nom": buses.loc[source_bus, "v_nom"],
                    "source_connected_edges": buses.loc[source_bus, "connected_edge_count"],
                    "recipient_bus": recipient_bus,
                    "recipient_load": recipient_load,
                    "recipient_v_nom": recipients.loc[recipient_bus, "v_nom"],
                    "recipient_connected_edges": recipients.loc[recipient_bus, "connected_edge_count"],
                    "recipient_distance_km": recipients.loc[recipient_bus, "distance_km"],
                    "transfer_fraction_of_source_load": TRANSFER_FRACTION,
                    "recipient_weight": float(weight),
                    "transferred_peak_mw": float((transfer * float(weight)).max()),
                    "transferred_total_mwh_2025": float((transfer * float(weight)).sum()),
                }
            )

    loads_p.to_csv(OUTPUT_NETWORK / "loads-p_set.csv", index=False)
    adjustments = pd.DataFrame(adjustment_rows)
    adjustments.to_csv(OUTPUT_NETWORK / "local_load_redistribution_adjustments.csv", index=False)

    readme = f"""# Local Load-Adjusted Florida PyPSA Network

This network is copied from:

`{BASE_NETWORK}`

and modifies only `loads-p_set.csv`.

The adjustment redistributes `{TRANSFER_FRACTION:.0%}` of the load assigned to
the six buses that still shed load in the summer-peak `s_nom x3` diagnostic:

{', '.join(SOURCE_BUSES)}

Recipient buses are selected within the same county, prioritizing equal-or-higher
voltage, strong connectivity, and proximity. This tests whether the residual
summer peak no-hazard shedding is caused by overly concentrated county-load
allocation at a few local substations.

Files:

- `local_load_redistribution_adjustments.csv`
- updated `loads-p_set.csv`
"""
    (OUTPUT_NETWORK / "LOCAL_LOAD_ADJUSTMENT_README.md").write_text(readme, encoding="utf-8")

    print("Saved local load-adjusted network:", OUTPUT_NETWORK)
    print(adjustments.to_string(index=False))


if __name__ == "__main__":
    main()
