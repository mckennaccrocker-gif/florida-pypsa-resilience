# Data Inputs

This project uses several public and locally processed datasets. Large source files are not committed to the repository.

## Transmission Lines

The Florida transmission network was built from transmission-line geometries and attributes, including line owner, voltage class, status, endpoint substation names, and geometry. The workflow preserves original line geometry where possible.

Key processing steps:

- filter usable overhead transmission lines
- retain line voltage attributes
- extract endpoints from each line geometry
- snap nearby endpoints to create network nodes
- preserve `SUB_1` and `SUB_2` where available
- infer connected voltage levels from line attributes
- review small islands manually in QGIS
- extend selected northern lines into Georgia and Alabama to real external endpoints

## Substations

Substation-like buses were created from transmission-line endpoints and reviewed topology. OSM substation polygons can also be extracted to estimate physical substation footprint areas for exposure analysis.

Substation polygon extraction is handled by:

```text
src/electricity/extract_osm_substation_polygons_for_reviewed_grid.py
```

## Power Plants

Power plant data came from OSM and GPPD-style point records. OSM provided polygon footprints where available, while GPPD provided additional plant attributes such as capacity, fuel/technology, and plant-level identifiers.

Matching logic is in:

```text
src/electricity/merge_florida_osm_gppd_powerplants.py
src/electricity/match_florida_powerplants_to_osm_polygons.py
```

## Generator Costs

Generator marginal costs were built using available fuel and heat-rate information where possible, with documented fallback assumptions. These costs are used by PyPSA dispatch to decide which generators operate before more expensive import slack or load shedding.

Relevant scripts:

```text
src/electricity/populate_florida_generator_marginal_costs.py
src/electricity/finalize_florida_generator_marginal_costs.py
```

## Hourly Electricity Demand

Florida hourly demand was downloaded and converted into load profiles. County population weighting was used to distribute statewide demand to model buses.

Relevant scripts:

```text
src/electricity/download_eia_florida_hourly_demand.py
src/electricity/build_county_population_load_profiles.py
```

## Solar Availability

Solar availability profiles were created for the Florida generator portfolio and used as time-varying generator availability in the PyPSA model.

Relevant script:

```text
src/electricity/create_florida_renewable_profiles.py
```

## Static Line Ratings

Transmission capacities were assigned using static line-rating current assumptions when available. PyPSA line capacity is represented as apparent power:

```text
s_nom = sqrt(3) * voltage_kV * amps / 1000
```

Relevant script:

```text
src/electricity/add_slr_s_nom_to_florida_lines.py
```

## Hazard Data

Tropical cyclone wind exposure uses OpenGIRA/SNAIL-style intersection outputs. Flood exposure uses return-period flood depths intersected with lines, generators, and buses.

Relevant folders:

```text
src/exposure/
src/cost/
```
