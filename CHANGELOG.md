# Changelog

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
