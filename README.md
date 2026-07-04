# table-top radiation source detection and localization

this repo contains the data and code to reproduce the results and visualizations in:
> A. White, C. Simon, H. Yang. "Environment-specific radiation source detection and localization via a sensor network and ensemble of randomized decision trees." (2026)

## data
42x32 grid: contains infomation on experimental setup and experiments. 

forward model data: Contains raw csv's of the detector array outputs for each experiment.

Background_for_localization_model.csv: background data used for source presence classifier

k_optimization_10-30.csv, k_optimization_17-22.csv, and k_optimization_19-22.csv contain the errors for varying k-sigma values used for finding the optimal threshold for multileteration sensor selection.

## code
model_analysis.py has the most up-to-date data processing and figures.
see the marimo notebook to be run in Python.
