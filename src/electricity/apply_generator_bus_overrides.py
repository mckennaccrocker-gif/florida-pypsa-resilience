"""
Apply reviewed generator bus overrides to a copied PyPSA network folder.

The override CSV should contain:
  - generator
  - proposed_bus_override
  - apply_override
  - override_reason

Only rows with apply_override == true are applied. This keeps generator
assignment changes auditable and reversible.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
DEFAULT_INPUT_DIR = ELECTRICITY_DIR / "pypsa_florida_network_county_population_load"
DEFAULT_REVIEW_DIR = DEFAULT_INPUT_DIR / "generator_bus_assignment_review"
DEFAULT_OVERRIDE_FILE = DEFAULT_REVIEW_DIR / "large_generator_bus_override_template.csv"
DEFAULT_OUTPUT_DIR = ELECTRICITY_DIR / "pypsa_florida_network_county_load_generator_overrides"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply PyPSA generator bus overrides.")
    parser.add_argument("--input-network-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--override-file", type=Path, default=DEFAULT_OVERRIDE_FILE)
    parser.add_argument("--output-network-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def copy_network(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_dir.glob("*.csv"):
        shutil.copy2(path, output_dir / path.name)


def apply_overrides(args: argparse.Namespace) -> pd.DataFrame:
    if not args.override_file.exists():
        raise FileNotFoundError(args.override_file)
    copy_network(args.input_network_dir, args.output_network_dir)

    overrides = pd.read_csv(args.override_file)
    overrides = overrides[overrides["apply_override"].map(truthy)].copy()
    if overrides.empty:
        pd.DataFrame().to_csv(args.output_network_dir / "applied_generator_bus_overrides.csv", index=False)
        return overrides

    buses = pd.read_csv(args.output_network_dir / "buses.csv")
    bus_set = set(buses["name"].astype(str))
    generators_path = args.output_network_dir / "generators.csv"
    final_generators_path = args.output_network_dir / "generators_with_final_marginal_costs.csv"
    generators = pd.read_csv(generators_path)
    final_generators = pd.read_csv(final_generators_path)

    gen_set = set(generators["name"].astype(str))
    records = []
    for row in overrides.itertuples(index=False):
        generator = str(row.generator)
        new_bus = str(row.proposed_bus_override)
        if generator not in gen_set:
            raise ValueError(f"Override generator not found: {generator}")
        if new_bus not in bus_set:
            raise ValueError(f"Override bus not found: {new_bus}")
        old_bus = generators.loc[generators["name"].eq(generator), "bus"].iloc[0]
        generators.loc[generators["name"].eq(generator), "bus"] = new_bus
        final_generators.loc[final_generators["name"].eq(generator), "bus"] = new_bus
        records.append(
            {
                "generator": generator,
                "old_bus": old_bus,
                "new_bus": new_bus,
                "override_reason": getattr(row, "override_reason", ""),
            }
        )

    generators.to_csv(generators_path, index=False)
    final_generators.to_csv(final_generators_path, index=False)
    applied = pd.DataFrame(records)
    applied.to_csv(args.output_network_dir / "applied_generator_bus_overrides.csv", index=False)
    return applied


def main() -> None:
    args = parse_args()
    applied = apply_overrides(args)
    print(f"Applied {len(applied)} generator bus overrides.")
    if not applied.empty:
        print(applied.to_string(index=False))
    print("Saved network:", args.output_network_dir)


if __name__ == "__main__":
    main()
