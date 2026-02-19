# Changelog

## Version 0.2.2 (22/1/2026)

- Updated protobuf to >=5.29.5, tensorboard to >=2.15 and tensorboardX to >=2.6
- Made azureml-core and mlflow optional dependencies (install with `pip install marian-tensorboard[azureml]` or `[mlflow]`)
- Updated minimum Python version to 3.8
- Suppressed deprecation warnings from TensorBoard's vendored dependencies

## Version 0.2.1 (3/3/2023)

- Fixed a bug when parsing logs without batch or total labels (GitHub issue #12)

## Version 0.2.0 (5/10/2022)

- Added `--step` option for choosing which stat to use for TensorBoard step
- Parsing additional statistics: WPS, gradient norm, label counts

## Version 0.1.1 (15/9/2022)

- Updated logic for `--tool` in cases where AzureML is not detected

## Version 0.1.0 (25/8/2022)

- Added `--tool` option for choosing logging into TensorBoard or AzureML or both
- Removed `--azureml` option
- Setting `--port 0` will skip starting a local TensorBoard server
- Fixed a bug with loading version number when calling the script directly

## Version 0.0.5 (19/8/2022)

- Trying to set working directory on Azure ML to `/tb_logs`
- Creating working directory if it does not exist
- Added `--update-freq` option

## Version 0.0.4 (28/7/2022)

- Added parsing learning rates
- Added parsing log files with logical epochs
- Exit with error code if log file does not exist
- Set `--azureml` automatically when running on Azure ML
- Created package: marian-tensorboard
- Added integration for TensorBoard and Azure ML Metrics
- Started the project at MTMA 2022
