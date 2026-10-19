# Input format

Place the required files together in a local input directory. This document describes their schema and contains no observational values.

## Shared daily and spatial inputs

- `dates.csv`: a `date` column with 4,933 unique consecutive dates from 2011-07-01 to 2024-12-31.
- `grid_features_full.npy`: numeric array with shape `(4933, 634, 29)`, aligned with `dates.csv` and the grid order in `static_grid_features.csv`.
- `input_metadata.json`: an ordered `columns` array. Required order: `rain`, `pet`, `P_30d`, `P_60d`, `sof`, `if`, `bf`, `sm`, `rain_intensity`, `rel_sat`, `soil_deficit`, `tc_sof`, `elev`, `slope`, `ksat`, `poro`, `tc_if`, `P_90d`, `flow_len`, `cn`, `fc`, `wp`, `gw_slow`, `gw_fast`, `reinf`, `lon`, `lat`, `is_raining`, `is_saturated`. The `if` value is a field name, not Python syntax.
- `static_grid_features.csv`: one row per grid, in the same grid order as the feature array. Required fields include `area_km2`, `elev`, `slope`, `ksat`, `poro`, `cn`, `fc`, and `wp`; retain the complete prepared study file when reproducing the input hashes.
- `nodes.csv`: 15 ordered node rows. Node IDs are 0–14; fields used in network construction include `node_id`, `downstream_node`, `dist_to_outlet`, `local_area_km2`, and `upstream_area_km2`.
- `topology.npz`: `node_edges` with shape `(2, 14)`, `grid_edges` with shape `(2, 2795)`, and grid-to-node `mapping` with shape `(15, 634)`. Grid membership must sum to one before applying the upstream fractions.
- `basin_daily.csv`: columns `date`, `q_obs_m3_s`, `rain`, and `pet`, aligned with `dates.csv`. Discharge is nonnegative and measured in m³/s. This file contains restricted observations.
- `reservoir_causal_daily.csv`: historical reservoir fields `res_level`, `delta_level`, `res_outflow`, `res_in_final`, and `res_outflow_lag1`, with date alignment matching `dates.csv`. `inflow_coverage` is required for the inflow supervision mask; valid targets require coverage of at least 0.95. Missing historical values are allowed and are filled using training-segment means. This file contains restricted observations.

## Continuous HBV forcing

- `forcing_dates.csv`: a consecutive `date` column covering the feature dates plus the preceding 365 days, for 5,298 days in total.
- `forcing_rain_mm.npy`: nonnegative precipitation array, shape `(5298, 634)`, in mm per hydrological day.
- `forcing_pet_mm.npy`: nonnegative reference evapotranspiration array with the same shape, in mm per hydrological day.

The study forcing incorporates gauge-derived information and is not distributed in the repository. The additional 365 days are used for standard-HBV state initialization. Differentiable HBV uses historical feature forcing and its own 365-day context.

## Time alignment and evaluation

A hydrological day runs from 08:00 Beijing time to 08:00 the following day and is labeled by its starting date. Inputs must follow the manuscript's timestamp alignment; file names alone do not establish causal alignment.

The chronological split indices are `TRAIN_END = int(0.7*N)` and `VAL_END = int(0.85*N)`. Training ends on 2020-12-12; the validation segment ends on 2022-12-22. For split bounds `[beg, end)` and lead `L`, evaluation target indices run from `beg + 14 + L` to `end - 1`. All models are compared on the same dates at each lead; context and warmup requirements can reduce their usable training targets differently.

Historical input windows end at `target_index - lead`. Future forcing uses only the specified forecast-horizon precipitation and, for HBV models, PET; future observed hydrological states are not forecast inputs. Scaling, imputation and discharge reference quantiles use the training segment. Metrics are calculated on physical-unit discharge and summarized over five seeds using sample standard deviation (`ddof=1`).

Exact input fingerprints and schema checks are stored in each model group's `expected_data.json`. Do not replace missing observations with invented values or bypass fingerprints when reporting an exact reproduction of the study.
