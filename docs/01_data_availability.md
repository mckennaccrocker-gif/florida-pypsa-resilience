# Data Availability and Reproducibility Notes

This repository is designed to make the Florida PyPSA resilience workflow understandable and reusable without committing very large geospatial or model-output files. The code, documentation, small summary tables, and selected thesis-ready figures are included. Large raw datasets and generated network/scenario artifacts are intentionally excluded.

## Included in This Repository

The repository includes:

- Python scripts used to build the Florida transmission-scale PyPSA model
- scripts for power plant matching, bus creation, line capacities, demand assignment, solar profiles, import slack, and load shedding
- tropical cyclone wind, flood exposure, and adaptation analysis scripts
- EAGLE-I Hurricane Ian validation/comparison scripts
- diagnostic and plotting scripts used during model development
- final flood-adaptation summary CSV and Markdown outputs
- selected PNG figures from the final flood-adaptation analysis
- documentation describing the workflow and modeling assumptions

These files are small enough to keep GitHub readable and are intended to document how the analysis was performed.

## Not Included

The following files are not committed because they are large, generated, machine-specific, or better obtained from their original providers:

- raw transmission, substation, power plant, demand, and hazard datasets
- raster flood-depth and tropical-cyclone wind files
- GeoPackage and QGIS review layers
- PyPSA NetCDF network files
- full dispatch/scenario run folders
- complete EAGLE-I outage extracts
- intermediate SNAIL/OpenGIRA intersection outputs

These exclusions are controlled through `.gitignore`.

## Expected Local Data Structure

The original working project used this local root:

```text
C:/oxford_tc_project/
```

The expected local folders are shown in:

```text
config/paths.example.yml
```

To reproduce the workflow on another machine, copy that file to:

```text
config/paths.yml
```

and update the paths to match the local data location.

## Main External or Locally Prepared Inputs

The workflow assumes access to locally prepared versions of:

- HIFLD-style transmission line data with geometry, voltage, owner, status, and endpoint attributes: https://catalog.data.gov/dataset/electric-power-transmission-lines
- OSM power plant polygons and OSM substation polygons: https://download.geofabrik.de/north-america/us.html
- GPPD-style power plant point records: https://datasets.wri.org/datasets/global-power-plant-database
- EIA hourly Florida electricity demand: https://www.eia.gov/opendata/
- EIA fuel, heat-rate, or generator cost data used for marginal-cost assumptions: https://www.eia.gov/opendata/
- static line rating/current data used to estimate PyPSA line capacities
- NOAA IBTrACS historical tropical cyclone tracks: https://www.ncei.noaa.gov/products/international-best-track-archive
- STORM synthetic/return-period tropical cyclone wind data: https://zenodo.org/records/7438145
- JRC flood return-period raster or exposure files: https://data.jrc.ec.europa.eu/collection/id-0054
- county population and county boundary data used for demand weighting and validation maps
- EAGLE-I county outage data used for Hurricane Ian validation: https://eagle-i.doe.gov/

More detail on the role of each dataset is in:

```text
docs/02_data_inputs.md
```

## Reproducibility Level

This repository supports transparent workflow review and partial reproduction from local data. It is not a self-contained archive of all raw data and generated outputs.

The intended reproduction path is:

1. prepare or download the source datasets described in `docs/02_data_inputs.md`;
2. update local paths using `config/paths.example.yml`;
3. follow the workflow described in `docs/03_full_workflow_story.md`;
4. regenerate the large network, exposure, and scenario outputs locally.
