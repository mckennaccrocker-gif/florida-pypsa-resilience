# Florida PyPSA Resilience

A Florida PyPSA electric-grid resilience model for studying hurricane wind and flood impacts. The workflow builds the grid from transmission, substation, power plant, demand, and cost data, then tests hazard impacts and adaptation strategies.

## What This Repository Contains

This repository is a cleaned, documented version of the Florida resilience workflow. It includes code and small summary outputs, but it intentionally does not include large raw datasets, PyPSA NetCDF networks, raster hazard files, or proprietary/local downloads.

The workflow covers:

- Florida transmission line preprocessing
- endpoint snapping and topology repair
- substation-derived buses
- voltage-layer buses and transformer/transfer components
- transmission line capacities from static line ratings
- power plant matching between OSM and GPPD
- generator-to-bus assignment
- generator marginal-cost construction
- hourly Florida electricity demand
- solar availability profiles
- emergency import slack
- load-shedding generators
- no-hazard baseline dispatch
- Georgia and Alabama tie-line extensions
- tropical-cyclone wind exposure with OpenGIRA/SNAIL-style workflows
- flood exposure, flood-damage scenarios, and adaptation analysis

## Repository Layout

```text
src/
  electricity/   PyPSA network construction, dispatch, and hazard scenario scripts
  exposure/      Tropical cyclone and flood exposure scripts
  cost/          Fragility, direct-damage, and EAD scripts
  adaptation/    Flood adaptation, cost-benefit, and final table scripts
docs/
  data_inputs.md
  pypsa_network_workflow.md
  hazard_workflow.md
  flood_adaptation_workflow.md
config/
  paths.example.yml
outputs/
  summary_tables/ small final CSV/Markdown summary outputs
```

## Quick Start

1. Create a local data folder matching `config/paths.example.yml`.
2. Download or place the required source datasets described in `docs/data_inputs.md`.
3. Build cleaned grid assets using the scripts in `src/electricity/`.
4. Convert the cleaned assets into a PyPSA network.
5. Run the no-hazard baseline.
6. Run wind or flood hazard scenarios.
7. Run adaptation or cost-benefit scripts if needed.

The scripts preserve the original project path structure used during development. For a new machine, update path constants or adapt them to read from `config/paths.example.yml`.

## Important Modeling Notes

The PyPSA model represents transmission-scale operational behavior. It is not a full distribution-grid outage model, so local distribution outages observed in datasets such as EAGLE-I will not be fully represented. Load shedding in this model reflects transmission/generation/import constraints under modeled assumptions.

Economic results use illustrative Value of Lost Load assumptions. These are societal avoided outage values, not direct utility revenue or verified Florida-specific cash savings.

## Final Flood Adaptation Result

The final five-asset RP100 flood adaptation package was evaluated across RP10, RP20, RP50, RP75, RP100, RP200, and RP500.

Main saved result:

- Baseline EENS: 190,229.9 MWh/year
- Adapted EENS: 186,129.4 MWh/year
- Avoided EENS: 4,100.5 MWh/year
- Risk reduction: 2.16%
- Central BCR: 19.76
- Central NPV: $763.1M

See `outputs/summary_tables/` for final summary CSVs.

## Status

This repository is research code for thesis analysis. It is designed for transparency and reproducibility of the workflow, not as a general-purpose package.
