"""
Add DOE/NREL static line ratings to Florida HIFLD transmission lines.

This script:
  1. Loads the Florida HIFLD transmission lines GeoPackage.
  2. Loads SLR_A-100C.h5 and prints its HDF5 structure.
  3. Extracts static line rating values in amps.
  4. Matches ratings to Florida lines using a HIFLD ID column.
  5. Converts amps to PyPSA apparent-power capacity:

       s_nom_mva = sqrt(3) * voltage_kv * slr_amps / 1000

  6. Saves:
       data/Electricity/florida_lines_with_s_nom.gpkg
       data/Electricity/florida_lines_with_s_nom.csv

Notes:
  - Requires h5py: pip install h5py
  - If auto-detection cannot identify the HDF5 ID/rating datasets, rerun with:
       --h5-id-dataset /path/in/h5
       --h5-rating-dataset /path/in/h5
  - If the Florida HIFLD ID column is not auto-detected, rerun with:
       --hifld-id-column ID
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"

DEFAULT_LINES_GPKG = ELECTRICITY_DIR / "florida_transmission_lines.gpkg"
DEFAULT_SLR_H5 = ELECTRICITY_DIR / "SLR_A-100C.h5"
OUTPUT_GPKG = ELECTRICITY_DIR / "florida_lines_with_s_nom.gpkg"
OUTPUT_CSV = ELECTRICITY_DIR / "florida_lines_with_s_nom.csv"

LIKELY_HIFLD_ID_COLUMNS = [
    "ID",
    "OBJECTID",
    "OBJECTID_1",
    "hifld_id",
    "HIFLD_ID",
    "line_id",
    "LINE_ID",
]

LIKELY_RATING_NAME_PATTERNS = [
    "slr",
    "rating",
    "amp",
    "amps",
    "static",
    "value",
    "data",
]

LIKELY_ID_NAME_PATTERNS = [
    "index",
    "hifld",
    "objectid",
    "object_id",
    "line_id",
    "transmission",
    "id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match DOE/NREL SLR amps to Florida HIFLD lines and compute s_nom."
    )
    parser.add_argument("--lines-gpkg", type=Path, default=DEFAULT_LINES_GPKG)
    parser.add_argument("--slr-h5", type=Path, default=DEFAULT_SLR_H5)
    parser.add_argument("--hifld-id-column", default=None)
    parser.add_argument("--h5-id-dataset", default=None)
    parser.add_argument("--h5-rating-dataset", default=None)
    parser.add_argument(
        "--max-unmatched-print",
        type=int,
        default=100,
        help="Maximum number of unmatched HIFLD IDs to print.",
    )
    return parser.parse_args()


def require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "This script requires h5py to read .h5 files. Install it with: "
            "pip install h5py"
        ) from exc
    return h5py


def decode_if_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def normalize_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = decode_if_bytes(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def parse_voltage_kv(value: Any, volt_class: Any = None) -> float:
    if pd.isna(value):
        primary_values = []
    else:
        text = str(value).upper()
        primary_values = [
            float(match)
            for match in re.findall(r"-?\d+(?:\.\d+)?", text)
        ]

    voltages = []
    for number in primary_values:
        if number <= 0 or number == -999999:
            continue
        voltages.append(number / 1000 if number >= 1000 else number)

    if voltages:
        return max(voltages)

    if pd.isna(volt_class):
        return np.nan

    class_text = str(volt_class).upper()
    class_numbers = [
        float(match)
        for match in re.findall(r"\d+(?:\.\d+)?", class_text)
    ]
    class_numbers = [number for number in class_numbers if number > 0]
    if not class_numbers:
        return np.nan
    return max(class_numbers)


def list_hdf5_structure(h5_path: Path) -> list[dict[str, Any]]:
    h5py = require_h5py()
    dataset_info: list[dict[str, Any]] = []

    print("\nHDF5 structure:", h5_path)
    with h5py.File(h5_path, "r") as h5:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                attrs = {
                    key: decode_if_bytes(value)
                    for key, value in obj.attrs.items()
                    if np.asarray(value).size <= 5
                }
                info = {
                    "path": "/" + name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "attrs": attrs,
                }
                dataset_info.append(info)
                print(
                    f"  DATASET /{name} shape={obj.shape} dtype={obj.dtype} attrs={attrs}"
                )
            else:
                print(f"  GROUP   /{name}")

        h5.visititems(visitor)

    return dataset_info


def read_h5_dataset(h5_path: Path, dataset_path: str) -> np.ndarray:
    h5py = require_h5py()
    clean_path = dataset_path[1:] if dataset_path.startswith("/") else dataset_path
    with h5py.File(h5_path, "r") as h5:
        data = h5[clean_path][()]
    data = np.asarray(data)
    if data.ndim > 1 and 1 in data.shape:
        data = data.reshape(-1)
    return data


def score_dataset_path(path: str, patterns: list[str]) -> int:
    lower = path.lower()
    return sum(1 for pattern in patterns if pattern in lower)


def candidate_one_dimensional_datasets(
    dataset_info: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for info in dataset_info:
        shape = tuple(info["shape"])
        if len(shape) == 1 or (len(shape) == 2 and 1 in shape):
            candidates.append(info)
    return candidates


def auto_pick_rating_dataset(dataset_info: list[dict[str, Any]]) -> str:
    candidates = candidate_one_dimensional_datasets(dataset_info)
    numeric_candidates = [
        info
        for info in candidates
        if any(token in info["dtype"].lower() for token in ["int", "float"])
    ]
    if not numeric_candidates:
        raise ValueError("No one-dimensional numeric HDF5 datasets found for ratings.")

    scored = sorted(
        numeric_candidates,
        key=lambda info: (
            score_dataset_path(info["path"], LIKELY_RATING_NAME_PATTERNS),
            np.prod(info["shape"]),
        ),
        reverse=True,
    )
    picked = scored[0]["path"]
    print("Auto-selected HDF5 rating dataset:", picked)
    return picked


def auto_pick_id_dataset(
    dataset_info: list[dict[str, Any]],
    rating_length: int,
) -> str:
    candidates = [
        info
        for info in candidate_one_dimensional_datasets(dataset_info)
        if int(np.prod(info["shape"])) == rating_length
    ]
    if not candidates:
        raise ValueError(
            "No one-dimensional HDF5 ID dataset with the same length as ratings."
        )

    scored = sorted(
        candidates,
        key=lambda info: (
            score_dataset_path(info["path"], LIKELY_ID_NAME_PATTERNS),
            "int" in info["dtype"].lower(),
        ),
        reverse=True,
    )
    picked = scored[0]["path"]
    print("Auto-selected HDF5 ID dataset:", picked)
    return picked


def extract_slr_table(
    h5_path: Path,
    dataset_info: list[dict[str, Any]],
    h5_id_dataset: str | None,
    h5_rating_dataset: str | None,
) -> pd.DataFrame:
    rating_dataset = h5_rating_dataset or auto_pick_rating_dataset(dataset_info)
    ratings = read_h5_dataset(h5_path, rating_dataset)

    if ratings.ndim != 1:
        raise ValueError(
            f"Rating dataset {rating_dataset} has shape {ratings.shape}; expected 1D."
        )

    id_dataset = h5_id_dataset or auto_pick_id_dataset(dataset_info, len(ratings))
    ids = read_h5_dataset(h5_path, id_dataset)
    if ids.ndim != 1:
        raise ValueError(f"ID dataset {id_dataset} has shape {ids.shape}; expected 1D.")
    if len(ids) != len(ratings):
        raise ValueError(
            f"ID dataset length {len(ids)} does not match rating length {len(ratings)}."
        )

    table = pd.DataFrame(
        {
            "hifld_id_for_slr": [normalize_id(value) for value in ids],
            "slr_amps": pd.to_numeric(ratings, errors="coerce"),
        }
    )
    table = table.dropna(subset=["hifld_id_for_slr", "slr_amps"])
    table = table.drop_duplicates(subset=["hifld_id_for_slr"], keep="first")

    print("\nSLR table columns before merge:", list(table.columns))
    print("SLR table rows:", len(table))
    print(table.head().to_string(index=False))
    return table


def pick_hifld_id_column(lines: gpd.GeoDataFrame, requested: str | None) -> str:
    print("\nFlorida line columns before merge:")
    print(list(lines.columns))

    if requested:
        if requested not in lines.columns:
            raise ValueError(f"Requested HIFLD ID column not found: {requested}")
        return requested

    for column in LIKELY_HIFLD_ID_COLUMNS:
        if column in lines.columns:
            print("Auto-selected Florida HIFLD ID column:", column)
            return column

    raise ValueError(
        "Could not auto-detect HIFLD ID column. Rerun with --hifld-id-column."
    )


def add_s_nom(lines: gpd.GeoDataFrame, slr_table: pd.DataFrame, id_column: str) -> gpd.GeoDataFrame:
    out = lines.copy()
    out["hifld_id_for_slr"] = out[id_column].apply(normalize_id)

    voltage_source = "voltage_clean" if "voltage_clean" in out.columns else "VOLTAGE"
    if voltage_source not in out.columns:
        raise ValueError("No voltage column found. Expected voltage_clean or VOLTAGE.")

    volt_class_values = out["VOLT_CLASS"] if "VOLT_CLASS" in out.columns else pd.Series(np.nan, index=out.index)
    out["voltage_kv"] = [
        parse_voltage_kv(voltage, volt_class)
        for voltage, volt_class in zip(out[voltage_source], volt_class_values)
    ]
    out = out.merge(slr_table, on="hifld_id_for_slr", how="left")
    out["s_nom_mva"] = (
        np.sqrt(3) * out["voltage_kv"] * out["slr_amps"] / 1000
    )
    return gpd.GeoDataFrame(out, geometry="geometry", crs=lines.crs)


def print_summary(
    lines: gpd.GeoDataFrame,
    id_column: str,
    max_unmatched_print: int,
) -> None:
    matched = lines["slr_amps"].notna()
    unmatched_ids = (
        lines.loc[~matched, id_column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print("\nSummary")
    print("Florida lines:", len(lines))
    print("Successfully matched to SLR ratings:", int(matched.sum()))
    print("Missing SLR ratings:", int((~matched).sum()))
    print("\ns_nom_mva summary statistics:")
    print(lines["s_nom_mva"].describe().to_string())
    print(f"\nUnmatched IDs from {id_column} (first {max_unmatched_print}):")
    print(unmatched_ids[:max_unmatched_print])
    if len(unmatched_ids) > max_unmatched_print:
        print(f"... {len(unmatched_ids) - max_unmatched_print} more unmatched IDs not shown")


def main() -> None:
    args = parse_args()
    if not args.lines_gpkg.exists():
        raise FileNotFoundError(args.lines_gpkg)
    if not args.slr_h5.exists():
        raise FileNotFoundError(
            f"{args.slr_h5} not found. Put SLR_A-100C.h5 there or pass --slr-h5."
        )

    lines = gpd.read_file(args.lines_gpkg)
    id_column = pick_hifld_id_column(lines, args.hifld_id_column)

    dataset_info = list_hdf5_structure(args.slr_h5)
    slr_table = extract_slr_table(
        args.slr_h5,
        dataset_info,
        args.h5_id_dataset,
        args.h5_rating_dataset,
    )

    enriched = add_s_nom(lines, slr_table, id_column)

    enriched.to_file(
        OUTPUT_GPKG,
        layer="florida_lines_with_s_nom",
        driver="GPKG",
    )
    enriched.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print_summary(enriched, id_column, args.max_unmatched_print)
    print("\nSaved GeoPackage:", OUTPUT_GPKG)
    print("Saved CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
