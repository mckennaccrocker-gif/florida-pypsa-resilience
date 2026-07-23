# Florida PyPSA Resilience

This repository documents the Florida PyPSA resilience workflow I used to build a transmission-scale electricity model and then test hurricane wind, flood, N-1, validation, and adaptation questions on top of it.

The project builds a Florida grid from transmission lines, buses/substations, power plants, load, costs, and generator profiles. It also extends selected tie-lines into Georgia and Alabama so the model is not artificially cut off at the state border. From there, it can run baseline PyPSA dispatch with emergency import slack and load-shedding generators, apply tropical cyclone or flood damage to the grid, compare Hurricane Ian results with observed EAGLE-I outage patterns, and use the flood scenarios to test hardening options and cost-benefit results.

This is research code, so it is not a one-click package. But the goal of this repo is to make the whole workflow understandable and to show where each major piece lives.

## What This Repository Contains

This repository is a cleaned, documented version of the Florida resilience workflow. It includes code and small summary outputs, but it intentionally does not include large raw datasets, PyPSA NetCDF networks, raster hazard files, or proprietary/local downloads.

The workflow covers:

- HIFLD transmission-line preprocessing, endpoint/bus creation, and topology repair
- OSM substation polygons and OSM/GPPD power plant matching
- generator-to-bus assignment and review for large plants
- extension of selected Florida lines into Georgia and Alabama boundary connections
- line-capacity assignment using the static line rating equation
- hourly demand assignment using county population weights
- emergency import slack generators and load-shedding generators
- generator marginal costs and solar availability profiles
- fragility/vulnerability curves for TC wind and flood damage
- SNAIL/OpenGIRA-style asset intersections for IBTrACS storms, synthetic storm return periods, and JRC flood depths
- expected annual electricity-cost/load-shedding calculations and exceedance curves
- N-1 bulk-line contingency screening
- single-storm Hurricane Ian direct and indirect damage figures
- EAGLE-I comparison figures for Hurricane Ian county outage patterns
- flood hardening and cost-benefit analysis

## Repository Layout

```text
src/
  electricity/   PyPSA network construction, dispatch, and hazard scenario scripts
  exposure/      Tropical cyclone and flood exposure scripts
  cost/          Fragility, direct-damage, and EAD scripts
  adaptation/    Flood adaptation, cost-benefit, and final table scripts
  validation/     EAGLE-I Hurricane Ian validation/comparison scripts
docs/
  01_data_availability.md
  02_data_inputs.md
  03_full_workflow_story.md
  04_pypsa_network_workflow.md
  05_hazard_workflow.md
  06_flood_adaptation_workflow.md
  07_validation_and_diagnostics.md
config/
  paths.example.yml
outputs/
  summary_tables/ small final CSV/Markdown summary outputs
  figures/ selected thesis-ready figure PNGs
```

## Quick Start

Start with the main narrative:

```text
docs/03_full_workflow_story.md
```

At a high level:

1. Create a local data folder matching `config/paths.example.yml`.
2. Download or place the required source datasets described in `docs/02_data_inputs.md`.
3. Build cleaned grid assets using the scripts in `src/electricity/`.
4. Convert the cleaned assets into a PyPSA network.
5. Run the no-hazard baseline.
6. Run wind, flood, N-1, or Hurricane Ian scenarios.
7. Run validation/adaptation/cost-benefit scripts depending on the question.

Most scripts assume the local project folder structure used during this analysis. To rerun the workflow on another machine, update the paths in `config/paths.example.yml` or edit the path constants in the relevant scripts.

## Important Modeling Notes

The PyPSA model represents transmission-scale operational behavior. It is not a full distribution-grid outage model, so local distribution outages observed in datasets such as EAGLE-I will not be fully represented. Load shedding in this model reflects transmission/generation/import constraints under modeled assumptions.

The baseline model originally showed load shedding even before adding a hazard, which should not really be happening in a no-hazard case. To keep the hazard results from being driven by that baseline capacity issue, the line capacities were adjusted with a multiplier during calibration, including a 2x line-capacity case used to remove unrealistic baseline load shedding.

The fragility/vulnerability curves are also important assumptions. The current selected curves, especially some of the flood curves, often produce relatively small damage ratios, so the modeled hazard impacts can look muted. These curves are saved and applied transparently in the scripts, and they can be swapped or stress-tested if a more conservative damage assumption is needed.

Economic results use illustrative Value of Lost Load assumptions. These are societal avoided outage values, not direct utility revenue or verified Florida-specific cash savings.

Raw restricted or very large data files are not included here. For security and reproducibility, the repo gives source links and workflow notes instead of committing local data extracts.

## Status

This repository is research code for thesis analysis. It is designed for transparency and reproducibility of the workflow, not as a general-purpose package.
