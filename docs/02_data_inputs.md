# Data Inputs

This project uses several public and locally processed datasets. Large source files are not committed to the repository. For security/reproducibility, this repo gives links to the original sources instead of uploading local data extracts.

## Data Source Links

Core source links:

- HIFLD electric transmission lines: https://catalog.data.gov/dataset/electric-power-transmission-lines
- HIFLD ArcGIS item: https://www.arcgis.com/home/item.html?id=7759b0df07274f30a422e86dc11d4761
- OpenStreetMap / Geofabrik U.S. state extracts: https://download.geofabrik.de/north-america/us.html
- WRI Global Power Plant Database: https://datasets.wri.org/datasets/global-power-plant-database
- EIA Open Data API: https://www.eia.gov/opendata/
- NOAA IBTrACS: https://www.ncei.noaa.gov/products/international-best-track-archive
- STORM fixed-return-period tropical cyclone wind data: https://zenodo.org/records/7438145
- JRC river flood hazard map collection: https://data.jrc.ec.europa.eu/collection/id-0054
- JRC flood maps in Google Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/JRC_CEMS_GLOFAS_FloodHazard_v2_1
- EAGLE-I portal: https://eagle-i.doe.gov/
- EAGLE-I historical outage dataset reference: https://openenergyhub.ornl.gov/explore/dataset/eaglei_outages_2014/
- snkit documentation: https://snkit.readthedocs.io/

## Transmission Lines

The Florida transmission network was built from HIFLD-style transmission-line geometries and attributes, including line owner, voltage class, status, endpoint substation names, and geometry. The workflow preserves original line geometry where possible.

Key processing steps:

- filter usable overhead transmission lines
- retain line voltage attributes
- extract endpoints from each line geometry
- snap nearby endpoints to create network nodes
- preserve `SUB_1` and `SUB_2` where available
- infer connected voltage levels from line attributes
- review small islands manually in QGIS
- extend selected northern lines into Georgia and Alabama to real external endpoints

The Georgia/Alabama extension is important because the Florida grid is not electrically isolated at the state boundary. The extension is not a full Southeast model; it is a boundary representation that lets selected external ties behave more realistically in dispatch.

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

## Emergency Slack and Load Shedding

Emergency import slack generators represent costly external support that can enter at selected boundary buses. Load-shedding generators are very expensive artificial generators at demand buses; they let PyPSA solve even when not all demand can be served. The reported load shedding is therefore modeled unserved demand, not observed customer outages.

Relevant scripts:

```text
src/electricity/select_florida_boundary_import_buses.py
src/electricity/run_florida_pypsa_load_shedding_dispatch.py
src/electricity/run_boundary_import_baseline.py
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

## Fragility and Vulnerability Curves

The final hazard scenarios use local curve CSVs derived from the selected fragility/vulnerability sources. They are applied through linear interpolation between saved curve points.

Main curve set:

- Flood lines: F6.2.
- Flood power plants: F1.1, F1.2, F1.3, F1.4.
- Flood substations/buses: F2.1, F2.2, F2.3.
- TC wind lines: W6.3.
- TC wind generators: W1.10, W1.11, W1.12, W1.13, W1.6.
- TC wind substations/buses: W2.3.

Relevant scripts:

```text
src/electricity/run_florida_pypsa_calibrated_hazard_scenarios.py
src/electricity/run_florida_pypsa_gradual_return_period_suite.py
src/electricity/validate_gradual_damage_scenarios.py
```

## Hazard Data

Tropical cyclone wind exposure uses OpenGIRA/SNAIL-style intersection outputs. Flood exposure uses return-period flood depths intersected with lines, generators, and buses.

Relevant folders:

```text
src/exposure/
src/cost/
```

The tropical cyclone workflow supports both historical IBTrACS events and synthetic/return-period storm wind data. The flood workflow uses JRC return-period flood depths. See `docs/03_full_workflow_story.md` for the full connection between source data, asset intersections, PyPSA scenarios, exceedance curves, EAGLE-I validation, and flood cost-benefit analysis.
