# Data access

## Observational records

Shimen discharge, rain-gauge precipitation and Sanjiangkou reservoir observations were provided by the Hunan Hydrology and Water Resources Survey Center. These records are not publicly available through this repository. Access and any redistribution require permission from the data provider. The repository does not promise that the authors can supply these records on request.

Merged precipitation and model-ready feature arrays depend in part on these observations. Those inputs, daily prediction exports and neural checkpoints are not distributed here. Model-output files generated locally may contain observations and must be handled under the provider's conditions.

## Public source products

- **ERA5-Land:** [Copernicus Climate Data Store dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview), [dataset DOI](https://doi.org/10.24381/cds.e2161bac). Retrieve the meteorological variables, spatial domain and study period described in the manuscript, including the preceding state-warmup period. Follow the platform's account, access and attribution requirements.
- **Copernicus GLO-30 DEM:** [Copernicus Data Space product and access instructions](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM), [dataset DOI](https://doi.org/10.5270/ESA-c5d3d65). Use the GLO-30 product covering the study basin and retain its attribution and release information.
- **SoilGrids 2.0:** [SoilGrids portal](https://soilgrids.org/), [ISRIC download and service documentation](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs.html). Obtain the soil properties and depth intervals used in the manuscript; follow the documented download route and product terms.

The included upstream grid fractions are derived using Copernicus GLO-30 geometry. Attribution: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. These fractions are spatial configuration values, not hydrological observations.

## Preparing inputs

Downloading the public products does not by itself reproduce the study's model-ready inputs. Gauge merging, hydrological-day alignment, feature construction and auxiliary-state preparation also require authorized observations and the preprocessing described in the manuscript. This repository provides training and evaluation from prepared inputs, rather than a complete raw-product preprocessing pipeline.

The required file names, dimensions and fields are described in [Input format](docs/INPUT_FORMAT.md). Supply the files from an authorized local source and run:

```bash
python run.py --action check-data --data /path/to/authorized_inputs
```

Reference hashes identify the study inputs. Different public-product releases or preprocessing will generally not match those hashes and cannot be claimed as an exact reproduction of the reported experiment. The source-code MIT license does not change any data provider's access or redistribution terms.
