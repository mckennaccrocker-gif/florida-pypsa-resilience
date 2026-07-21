"""
Download hourly Florida electricity demand from the EIA v2 API.

Source:
  EIA API v2 electricity/rto/region-data
  Respondent: FLA (Florida)

Outputs:
  - data/Electricity/eia_florida_hourly_demand/eia_florida_region_data_2025_raw.csv
  - data/Electricity/eia_florida_hourly_demand/eia_florida_hourly_demand_2025.csv
  - data/Electricity/eia_florida_hourly_demand/eia_florida_hourly_demand_2025_timeseries.csv

Optional PyPSA output:
  - data/Electricity/pypsa_florida_network/loads-p_set.csv

Usage:
  1. Set your EIA API key:
       PowerShell:  $env:EIA_API_KEY = "your_key_here"
       bash:        export EIA_API_KEY="your_key_here"

  2. Download and save Florida hourly demand:
       python data/Electricity/download_eia_florida_hourly_demand.py

  3. Also create PyPSA hourly load time series:
       python data/Electricity/download_eia_florida_hourly_demand.py --write-pypsa-loads

Notes:
  EIA values for hourly demand are reported as megawatthours for each hour.
  For an hourly interval, MWh over the hour is numerically equivalent to average
  MW during that hour, which is what PyPSA's load p_set expects.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
OUTPUT_DIR = ELECTRICITY_DIR / "eia_florida_hourly_demand"
PYPSA_DIR = ELECTRICITY_DIR / "pypsa_florida_network"

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
DEFAULT_YEAR = 2025
DEFAULT_RESPONDENT = "FLA"
PAGE_LENGTH = 5_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download EIA hourly Florida demand and optionally write PyPSA loads."
    )
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--respondent", default=DEFAULT_RESPONDENT)
    parser.add_argument(
        "--api-key",
        default=os.getenv("EIA_API_KEY"),
        help="EIA API key. Defaults to EIA_API_KEY environment variable.",
    )
    parser.add_argument(
        "--write-pypsa-loads",
        action="store_true",
        help="Write data/Electricity/pypsa_florida_network/loads-p_set.csv.",
    )
    parser.add_argument(
        "--load-weight",
        choices=["connected_edges", "equal"],
        default="connected_edges",
        help="How to distribute Florida-wide demand across PyPSA bus loads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload even if cached raw data exists.",
    )
    return parser.parse_args()


def eia_params(
    year: int,
    respondent: str,
    api_key: str | None,
    offset: int,
    length: int,
) -> dict:
    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "start": f"{year}-01-01T00",
        "end": f"{year}-12-31T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": length,
    }
    if api_key:
        params["api_key"] = api_key
    return params


def download_eia_region_data(
    year: int,
    respondent: str,
    api_key: str | None,
) -> pd.DataFrame:
    records = []
    offset = 0
    total = None

    while total is None or offset < total:
        params = eia_params(year, respondent, api_key, offset, PAGE_LENGTH)
        response = requests.get(EIA_URL, params=params, timeout=60)
        if response.status_code == 403 and not api_key:
            raise RuntimeError(
                "EIA API returned 403. Set EIA_API_KEY or pass --api-key."
            )
        if not response.ok:
            message = response.text[:500].replace(api_key or "", "[redacted]")
            raise RuntimeError(
                f"EIA API returned HTTP {response.status_code}: {message}"
            )

        payload = response.json()
        response_data = payload.get("response", {})
        page = response_data.get("data", [])
        total = int(response_data.get("total", len(page)))

        records.extend(page)
        print(f"Downloaded {len(records):,}/{total:,} EIA rows")

        if not page:
            break
        offset += PAGE_LENGTH
        time.sleep(0.2)

    return pd.DataFrame(records)


def clean_hourly_demand(raw: pd.DataFrame, year: int, respondent: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("EIA response was empty.")

    demand = raw.loc[raw["type"].eq("D")].copy()
    demand["period"] = pd.to_datetime(demand["period"], errors="coerce")
    demand["value"] = pd.to_numeric(demand["value"], errors="coerce")
    demand = demand.dropna(subset=["period", "value"]).sort_values("period")
    demand = demand.drop_duplicates(subset=["period"], keep="last")
    demand["demand_mwh"] = demand["value"]
    demand["demand_mw"] = demand["value"]
    demand["respondent"] = respondent
    demand["year"] = year

    keep_cols = [
        "period",
        "respondent",
        "respondent-name",
        "type",
        "type-name",
        "demand_mwh",
        "demand_mw",
        "value-units",
        "year",
    ]
    keep_cols = [col for col in keep_cols if col in demand.columns]
    return demand[keep_cols].reset_index(drop=True)


def write_demand_timeseries(demand: pd.DataFrame, path: Path) -> None:
    series = demand[["period", "demand_mw"]].copy()
    series = series.rename(columns={"period": "snapshot", "demand_mw": "FLA_demand_mw"})
    series.to_csv(path, index=False)


def load_pypsa_load_weights(load_weight: str) -> pd.Series:
    loads_path = PYPSA_DIR / "loads.csv"
    buses_path = PYPSA_DIR / "buses.csv"
    if not loads_path.exists() or not buses_path.exists():
        raise FileNotFoundError(
            "Missing PyPSA buses.csv or loads.csv. Run "
            "convert_florida_assets_to_pypsa.py first."
        )

    loads = pd.read_csv(loads_path)
    buses = pd.read_csv(buses_path)

    if load_weight == "equal":
        weights = pd.Series(1.0, index=loads["name"])
    else:
        bus_weights = buses.set_index("name")["connected_edge_count"].clip(lower=1)
        weights = loads.set_index("name")["bus"].map(bus_weights).astype(float)
        weights = weights.fillna(1.0)

    return weights / weights.sum()


def write_pypsa_load_timeseries(demand: pd.DataFrame, load_weight: str) -> Path:
    weights = load_pypsa_load_weights(load_weight)
    snapshots = demand["period"].dt.strftime("%Y-%m-%d %H:%M:%S")
    values = np.outer(demand["demand_mw"].to_numpy(dtype=float), weights.to_numpy())
    out = pd.DataFrame(values, columns=weights.index)
    out.insert(0, "snapshot", snapshots)

    output_path = PYPSA_DIR / "loads-p_set.csv"
    out.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = OUTPUT_DIR / f"eia_florida_region_data_{args.year}_raw.csv"
    demand_path = OUTPUT_DIR / f"eia_florida_hourly_demand_{args.year}.csv"
    timeseries_path = OUTPUT_DIR / f"eia_florida_hourly_demand_{args.year}_timeseries.csv"

    if raw_path.exists() and not args.overwrite:
        print("Using cached raw EIA data:", raw_path)
        raw = pd.read_csv(raw_path)
    else:
        raw = download_eia_region_data(args.year, args.respondent, args.api_key)
        raw.to_csv(raw_path, index=False)

    demand = clean_hourly_demand(raw, args.year, args.respondent)
    demand.to_csv(demand_path, index=False)
    write_demand_timeseries(demand, timeseries_path)

    print("\nSaved raw EIA data:", raw_path)
    print("Saved clean hourly demand:", demand_path)
    print("Saved one-column demand timeseries:", timeseries_path)
    print("Demand rows:", len(demand))
    print("Demand range MW:", round(demand["demand_mw"].min(), 2), "-", round(demand["demand_mw"].max(), 2))
    print("Mean demand MW:", round(demand["demand_mw"].mean(), 2))

    if args.write_pypsa_loads:
        output_path = write_pypsa_load_timeseries(demand, args.load_weight)
        print("Saved PyPSA load time series:", output_path)


if __name__ == "__main__":
    main()
