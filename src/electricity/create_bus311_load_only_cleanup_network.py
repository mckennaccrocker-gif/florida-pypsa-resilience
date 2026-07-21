"""Create a conservative cleanup network that only moves bus_311 load.

The targeted topology cleanup that deactivated low-voltage lines worsened the
summer peak sweep. This control network keeps all lines active and only removes
remaining load from the 69 kV bus_311 tap, moving it to stronger nearby Marion
County buses.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
BASE_NETWORK = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_county_load_local_adjusted"
OUTPUT_NETWORK = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_bus311_load_only_cleanup"

SOURCE_BUS = "bus_311"
RECIPIENTS = {
    "bus_672": 0.40,
    "bus_2097": 0.25,
    "bus_557": 0.20,
    "bus_612": 0.15,
}


def copy_network() -> None:
    if OUTPUT_NETWORK.exists():
        shutil.rmtree(OUTPUT_NETWORK)
    OUTPUT_NETWORK.mkdir(parents=True)
    for path in BASE_NETWORK.iterdir():
        if path.is_file():
            shutil.copy2(path, OUTPUT_NETWORK / path.name)


def main() -> None:
    copy_network()
    loads = pd.read_csv(OUTPUT_NETWORK / "loads.csv")
    loads_p = pd.read_csv(OUTPUT_NETWORK / "loads-p_set.csv")
    load_by_bus = loads.set_index("bus")["name"].to_dict()

    source_load = load_by_bus[SOURCE_BUS]
    source_profile = loads_p[source_load].copy()
    loads_p[source_load] = 0.0

    rows = []
    for recipient_bus, weight in RECIPIENTS.items():
        recipient_load = load_by_bus[recipient_bus]
        transfer = source_profile * weight
        loads_p[recipient_load] = loads_p[recipient_load] + transfer
        rows.append(
            {
                "source_bus": SOURCE_BUS,
                "source_load": source_load,
                "recipient_bus": recipient_bus,
                "recipient_load": recipient_load,
                "recipient_weight": weight,
                "transferred_peak_mw": float(transfer.max()),
                "transferred_total_mwh_2025": float(transfer.sum()),
            }
        )

    loads_p.to_csv(OUTPUT_NETWORK / "loads-p_set.csv", index=False)
    pd.DataFrame(rows).to_csv(OUTPUT_NETWORK / "bus311_load_only_move.csv", index=False)

    readme = f"""# Bus 311 Load-Only Cleanup Network

This network is copied from:

`{BASE_NETWORK}`

It keeps all transmission lines active and only moves the remaining load from
the 69 kV tap `{SOURCE_BUS}` to stronger nearby Marion County buses:

{', '.join(f'{bus} ({weight:.0%})' for bus, weight in RECIPIENTS.items())}

This is a conservative control case after the line-deactivation cleanup made
the summer peak sweep worse.
"""
    (OUTPUT_NETWORK / "BUS311_LOAD_ONLY_CLEANUP_README.md").write_text(readme, encoding="utf-8")
    print("Saved bus_311 load-only cleanup network:", OUTPUT_NETWORK)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
