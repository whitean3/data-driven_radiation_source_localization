# table-top radiation source detection and localization

this repo contains the data and code to reproduce the results and visualizations in:
> A. White, C. Simon, H. Yang. "Environment-specific radiation source detection and localization via a sensor network and ensemble of randomized decision trees." (2026)

## data
42x32 grid: contains infomation on experimental setup and experiments.
forward model data: Contains raw csv's of the detector array outputs for each experiment.
Background_for_localization_model.csv: background data used for source presence classifier
## code
delta_learning.py has the most uptodat data processing and figures.
see the marimo notebook to be run in Python.
