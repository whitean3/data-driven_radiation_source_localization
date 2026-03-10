import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import csv
    import os
    import math
    import pandas as pd
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    import matplotlib.font_manager as fm
    from scipy.optimize import root_scalar, minimize, differential_evolution
    from sklearn.metrics import ConfusionMatrixDisplay
    import scipy.optimize as optimize
    import matplotlib as mpl
    import seaborn as sns
    import ptitprince as pt # for raincloud
    plt.style.use('rose-pine-dawn.mplstyle') # https://github.com/h4pZ/rose-pine-matplotlib/tree/main/themes
    fm.fontManager.addfont("SourceCodePro-Regular.ttf") # https://fonts.google.com/specimen/Source+Code+Pro
    plt.rcParams["font.family"] = "Source Code Pro"

    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest
    from sklearn.model_selection import LeaveOneOut, train_test_split
    return (
        ConfusionMatrixDisplay,
        Ellipse,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        LeaveOneOut,
        csv,
        differential_evolution,
        mo,
        np,
        optimize,
        os,
        patches,
        pd,
        plt,
        pt,
        root_scalar,
        sns,
        transforms,
    )


@app.cell
def _(sns):
    # plot settings
    theme_colors = sns.color_palette("Set2")
    thing_to_color = {
        'true source loc': theme_colors[3], 
        'pred source loc': theme_colors[4], 
        'sensor': theme_colors[2],
        'lead': theme_colors[7],
        'cardboard': theme_colors[6],
        'MDF': theme_colors[0],
        'pine': theme_colors[1],
        'ML': theme_colors[0],
        'trad': theme_colors[1]
    }
    sensor_colormap = "Purples"

    # theme_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # sns.color_palette(theme_colors)
    sns.color_palette("Set2")
    return sensor_colormap, theme_colors, thing_to_color


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ::icon-park:data:: read in sensor network response data

    * each data set is the sensor network response to a source in a particular location.
    * `SN` gives sensor network ID
    * `ICR` gives count rate
    * the list of source locations are below.

    ## locations of detectors in the environment
    """
    )
    return


@app.cell
def _():
    box_dims = (42.0, 33.0) # box size, units: in
    return (box_dims,)


@app.cell
def _():
    n_sensors = 8
    return (n_sensors,)


@app.cell
def _(box_dims, n_sensors, np):
    # maps sensor ID to coordinates i.e. (x, y) positions. 
    #   the origin is the top left (we rectify below.)
    sensor_to_loc = {
        "16518": [2, 3],
        "16520": [2, 18],
        "16519": [2, 27], 
        "16513": [14, 10],
        "16516": [21, 25],
        "16521": [26, 1],
        "16517": [38, 8],
        "16512": [40, 24]
    }
    assert len(sensor_to_loc) == n_sensors

    for (sensor, loc) in sensor_to_loc.items():
        # convert to numpy array
        sensor_to_loc[sensor] = np.array(loc)

        # translate so that origin in bottom left
        sensor_to_loc[sensor][1] = box_dims[1] - sensor_to_loc[sensor][1]
    return (sensor_to_loc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## locations of the radioactive source placed in the environment

    the first 25 rows come from Latin Hypercube sampling.

    the next 25 rows are manually-selected on the corners/boundary.

    the remaining 25 are from another round of Latin Hypercube sampling.
    """
    )
    return


@app.cell
def _(box_dims, pd):
    df_source_locs = pd.DataFrame(
        {
            'x_s': [28.0, 38.0, 4.0, 2.0, 10.0, 41.0,32.0, 5.0, 40.0, 24.0, 16.0, 7.0, 18.0, 35.0, 33.0, 30.0, 23.0, 26.0, 19.0, 12.0, 0.0, 9.0, 21.0, 37.0, 14.0, 2, 5,  7,  7, 10, 10, 12, 16, 40, 41, 31, 31, 34, 41, 41, 41, 36, 28, 17, 12,  0,  7,  7,  7, 10, 13.0, 38.0, 41.0, 19.0,  7.0, 17.0, 39.0, 36.0,  9.0, 25.0, 32.0, 16.0,  6.0, 42.0,  0.0,  2.0, 26.0, 20.0, 28.0, 30.0,  3.0, 29.0, 14.0, 23.0, 22.0, 2.0, 3.0, 4.0, 4.0, 7.0, 8.0, 11.0, 12.0, 13.0, 16.0, 17.0, 19.0, 21.0, 22.0, 22.0, 26.0, 26.0, 28.0, 31.0, 31.0, 31.0, 35.0, 37.0, 37.0, 39.0],
            'y_s': [14.0,30.0,28.0,21.0,16.0,10.0,4.0, 6.0, 18.0,32.0,19.0,22.0,26.0,12.0,8.0, 6.0, 11.0,29.0,15.0,3.0, 0.0, 1.0, 23.0,23.0,32.0,0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 4.0, 0.0, 4.0, 4.0, 0.0, 0.0, 15.0,20.0,28.0,32.0,32.0,32.0,32.0,32.0,32.0,29.0,25.0,25.0, 20.0, 10.0, 24.0, 11.0, 15.0, 30.0, 14.0, 32.0, 0.0 , 32.0, 9.0 , 13.0, 6.0 , 27.0, 29.0, 23.0, 10.0, 2.0 , 18.0, 27.0, 16.0, 1.0 , 17.0, 19.0, 8.0, 12.0, 10.0, 32.0, 3.0, 9.0, 29.0, 24.0, 6.0, 21.0, 7.0, 0.0, 13.0, 19.0, 28.0, 1.0, 15.0, 25.0, 22.0, 20.0, 15.0, 28.0, 6.0, 6.0, 17.0, 17.0]
        }
    ) # note this is with origin in top left
    df_source_locs["y_s"] = box_dims[1] - df_source_locs["y_s"] # translate so that origin in bottom left
    df_source_locs
    return (df_source_locs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## read in sensor responses""")
    return


@app.cell
def _(csv, n_sensors, pd):
    def read_detector_outputs(file_path):
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            rows = []
            header = next(csv_reader)  # Read header separately

            # Create new header: keep first 6 columns and remove the rest
            new_header = header[:6]

            for row in csv_reader:
                # Take first 5 columns as is
                new_row = row[:6]
                # Convert columns 6 through 1029 into a vector and store in 6th position
                vector = [float(x) for x in row[5:1029]]
                new_row[5] = vector
                rows.append(new_row)

            df = pd.DataFrame(rows, columns=new_header)

            # keep only the last eight rows because these are the latest measurements,
            #    corresponding to response to the source
            start = len(df) - n_sensors
            df = df[start:]

            return df
    return (read_detector_outputs,)


@app.cell
def _():
    folder_path = "forward model data/"
    return (folder_path,)


@app.cell
def _():
    n_expts = 100
    return (n_expts,)


@app.cell
def _(folder_path, n_expts, n_sensors, os, read_detector_outputs):
    def read_all_data(n_expts, exp_to_filename):
        dataframes = {}

        for exp in range(n_expts):
            filename = exp_to_filename(exp)
            file_path = os.path.join(folder_path, filename)
            print(f"\nReading {filename}...")
            # Use the filename (or file_path) as the key
            dataframes[exp] = read_detector_outputs(file_path)

            # unique sensors
            assert dataframes[exp]["SN"].nunique() == n_sensors
        return dataframes

    detector_outputs = read_all_data(n_expts, lambda exp: f"pos_{exp}.csv")
    return detector_outputs, read_all_data


@app.cell
def _(detector_outputs):
    detector_outputs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## re-work data into an ML-friendly format

    ::lucide:lightbulb:: source locations paired with sensor network response vectors

    first, get list of sensors in the network.
    """
    )
    return


@app.cell
def _(detector_outputs, n_sensors):
    sensors = detector_outputs[0]["SN"].unique()
    sensor_to_nice_int = dict(zip(sensors, range(len(sensors))))
    assert len(sensors) == n_sensors
    sensors
    return sensor_to_nice_int, sensors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""next, from a sensor network response data frame, extract the count rate of a particular sensor.""")
    return


@app.function
def grab_sensor_response(detector_output, sensor):
    return float(detector_output.loc[detector_output["SN"] == sensor, "ICR"].item())


@app.cell
def _(detector_outputs, np, sensors):
    # e.g. reading of third sensor from second experiment
    assert np.isclose(grab_sensor_response(detector_outputs[1], sensors[2]), 14.307) # looked up
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🎯 goal:

    * each row is an experiment where we place the source at a location and observe the sensor network response
    * the row lists the source location paired with the sensor response vector.
    """
    )
    return


@app.cell
def _(detector_outputs, df_source_locs, pd, sensors):
    def make_data_nice(detector_outputs, df_source_locs):
        data = pd.DataFrame(columns=sensors) # nice data for ML
        # sensor network data
        for exp in range(len(detector_outputs)):
            new_row = {sensor: grab_sensor_response(detector_outputs[exp], sensor) for sensor in sensors}
            data.loc[len(data)] = new_row
        # join source locs
        data = pd.concat([df_source_locs, data], axis=1)
        return data

    data = make_data_nice(detector_outputs, df_source_locs)
    data
    return data, make_data_nice


@app.cell
def _(data, np, sensors):
    assert np.isclose(data.loc[1, sensors[2]], 14.307) # see above
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ::lucide:sailboat:: visually explore the data

    first, where are the source locations over all experiments?
    """
    )
    return


@app.cell
def _(
    box_dims,
    data,
    plt,
    sensor_to_loc,
    sensor_to_nice_int,
    sensors,
    setup_environment,
    thing_to_color,
):
    def viz_source_locs(data, ids=None, title=""):
        if ids is None:
            ids = range(data.shape[0])

        setup_environment(box_dims)

        # sensors
        plt.scatter(
            # x locs of sensors
            [sensor_to_loc[sensor][0] for sensor in sensors], 
            # y locs of sensors
            [sensor_to_loc[sensor][1] for sensor in sensors],
            c="white",
            s=65,
            edgecolor="black",
            marker="s",
            label="sensor"
        )

        # sensor names
        for sensor in sensors:
            plt.annotate(
                f"{sensor_to_nice_int[sensor]}",
                (sensor_to_loc[sensor][0], sensor_to_loc[sensor][1]),
                xytext=(5, 5),
                textcoords="offset points", 
                ha='left',
                va='bottom'
            )

        # source locations
        plt.scatter(
            data.loc[ids, "x_s"], data.loc[ids, "y_s"], 
            clip_on=False, color=thing_to_color["true source loc"], s=65, marker="o", 
            label="radiation\nsource"
        )

        plt.title(title)

        # legend w unique entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 0.8), loc='upper left', borderaxespad=0)

        plt.show()

    viz_source_locs(data)
    return (viz_source_locs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""second, visualize the sensor readout and source location for a single experiment.""")
    return


@app.cell
def _(draw_obstacles, plt):
    def setup_environment(box_dims):
        fig, ax = plt.subplots()

        draw_obstacles(ax)

        ax.set_aspect('equal', 'box')

        plt.xlabel("x [in]")
        plt.ylabel("y [in]")

        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])

        return fig, ax
    return (setup_environment,)


@app.cell
def _(box_dims, patches, thing_to_color):
    def draw_obstacles(ax):
        # circles (all lead)
        r = 3.125 / 2
        x_centers = [
            [7, box_dims[1] - 17], 
            [10, box_dims[1] - 17], 
            [13, box_dims[1] - 17]
        ]
        for x_center in x_centers:
            circle = patches.Circle(
                x_center, r, 
                color=thing_to_color["lead"], alpha=0.5
            )
            ax.add_patch(circle)

        # rectangles (different materials)
        x_bs = [
            # lead
            [0, box_dims[1]-10.5], 
            [0, box_dims[1]-18.5], 
            [0, box_dims[1]-28], 
            [2, box_dims[1]-10.5], 
            [2, box_dims[1]-18.5], 
            [32, box_dims[1]-33], 
            [34, box_dims[1]-27],
            # cardboard
            [20, box_dims[1]-25],
            # MDF
            [6, box_dims[1]-11], 
            [28, box_dims[1]-7], 
            [28, box_dims[1]-8],
            [38, box_dims[1]-7],
            # pine
            [0, box_dims[1]-20]
        ]
        # bottom left
        ws = [
            # lead
            2, 2, 2, 4, 4, 2, 8, 
            # cardboard
            20,
            # MDF
            7, 
            1, 
            14, 
            1,
            # pine
            10
        ]
        hs = [
            # lead
            8, 8, 8, 8, 8, 8, 2, 
            # cardboard
            0.5,
            # MDF
            1, 
            7, 
            1, 
            7,
            # pine
            1.5
        ]

        mat = ["lead" for i in range(7)] + [
            "cardboard"] + ["MDF" for i in range(4)] + ["pine"]
        for r in range(len(x_bs)):
            rectangle = patches.Rectangle(
                x_bs[r], ws[r], hs[r], 
                color=thing_to_color[mat[r]], alpha=0.5, 
                label=mat[r]
            )
            ax.add_patch(rectangle)
    return (draw_obstacles,)


@app.cell
def _(
    box_dims,
    data,
    plt,
    sensor_colormap,
    sensor_to_loc,
    sensor_to_nice_int,
    sensors,
    setup_environment,
    thing_to_color,
):
    def viz_sensor_readout(data, exp, max_response=75.0, label_sensors=False):
        setup_environment(box_dims)

        # plot sensors; color acc to response
        plt.scatter(
            # x locs of sensors
            [sensor_to_loc[sensor][0] for sensor in sensors], 
            # y locs of sensors
            [sensor_to_loc[sensor][1] for sensor in sensors],
            c=[data.loc[exp, sensor] for sensor in sensors],
            s=65,
            edgecolor="black",
            vmin=0,
            marker="s",
            vmax=max_response,
            cmap=sensor_colormap,
            label="sensor"
        )

        if label_sensors:
            for sensor in sensors:
                plt.annotate(
                    f"{sensor_to_nice_int[sensor]}",
                    (sensor_to_loc[sensor][0], sensor_to_loc[sensor][1]),
                    xytext=(5, 5),
                    textcoords="offset points", 
                    ha='left',
                    va='bottom'
                )

        plt.colorbar(label="count rate [CPS]", extend="max")

        # plot source location
        plt.scatter(
            data.loc[exp, "x_s"], data.loc[exp, "y_s"], marker="o", 
            s=65, color=thing_to_color["true source loc"], label="source\nlocation", clip_on=False
        )

        plt.title(f"experiment {exp}")

        # legend w unique entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.25, 0.8), loc='upper left', borderaxespad=0)

        plt.show()

    viz_sensor_readout(data, 11, label_sensors=True)
    return (viz_sensor_readout,)


@app.cell
def _(data, viz_source_locs):
    viz_source_locs(data, ids=[1, 40, 41, 57, 63], title="false negatives")
    return


@app.cell
def _(data, viz_source_locs):
    viz_source_locs(data, ids=[], title="the environment and sensor network")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""the distribution of the detector responses over all experiments and correlations between them. use a log scale.""")
    return


@app.cell
def _(data, plt, sensor_to_nice_int, sensors, sns, theme_colors):
    sns.swarmplot(data[sensors].rename(columns=sensor_to_nice_int), size=2, color=theme_colors[3])
    plt.axhline(13.2, linestyle="--", color="gray", zorder=0) # background
    plt.yscale("log")
    plt.xlabel("sensor")
    plt.ylabel("response [CPS]")
    plt.title("Sensor Response Distributions")
    return


@app.cell
def _(data, np, plt, sensor_to_nice_int, sensors, sns):
    _bins = np.geomspace(data[sensors].min().min(), data[sensors].max().max(), 30)
    _g = sns.pairplot(data[sensors].rename(columns=sensor_to_nice_int), corner=True, diag_kws={'bins': _bins})
    for _ax in _g.axes.flat:
        if not _ax == None:
            _ax.set_xscale('log')
            _ax.set_yscale('log')
    plt.show()
    return


@app.cell
def _(data, np, plt, sensor_to_nice_int, sensors, sns):
    def correlation_matrix(data):
        # Compute the correlation matrix
        corr = data.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.color_palette("vlag", as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, vmin=-0.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}
        )
        plt.title("Detector Correlation Matrix")
        plt.show()

    correlation_matrix(data[sensors].rename(columns=sensor_to_nice_int))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ☢️ source location predictor

    ML task: predict source location from sensor network response


    **input**: 8D sensor network response

    **output**: 2D source location

    ## leave-one-out cross validation
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        flowchart LR
            B[tree ensemble]
            direction LR
            a1[ ] -.-|"sensor network response vector"| B
            B -.-|"source location"| a2[ ]
            style a1 fill:none, stroke:none
            style a2 fill:none, stroke:none
        """
    )
    return


@app.cell
def _(np):
    def calculate_errors(data):
        data["error_x"] = np.abs(data["x_s"] - data["x_s_pred"])
        data["error_y"] = np.abs(data["y_s"] - data["y_s_pred"])
        data["error"] = np.sqrt(
            (data["x_s"] - data["x_s_pred"]) ** 2 + (data["y_s"] - data["y_s_pred"]) ** 2
        )
    return (calculate_errors,)


@app.cell
def _(
    ExtraTreesRegressor,
    LeaveOneOut,
    box_dims,
    calculate_errors,
    np,
    sensors,
):
    # a multi-output tree ensemble model. maps 8D vectors to 2D vectors.
    #  maps sensor network readout to source location
    def do_loo_cv(
        data, n_estimators=250, verbose=True, very_verbose=False, delta_modeling=False, uq=True
    ):
        target_prepend = "delta_" if delta_modeling else "" # handle delta modeling

        data_loo = data.copy()
        # store predicted source locations in data frame.
        #  ok bc each data point is test point ONCE.
        data_loo[target_prepend + "x_s_pred"] = np.zeros((len(data)))
        data_loo[target_prepend + "y_s_pred"] = np.zeros((len(data)))
        if uq:
            data_loo[target_prepend + "ensemble pred source locs"] = [np.zeros(n_estimators) for _ in range(len(data_loo))]

        loo = LeaveOneOut()
        for i, (_train_index, _test_index) in enumerate(loo.split(data_loo)):
            # account for non 0, ..., n_row indexing (NaN's dropped for delta learning)
            train_index = data_loo.index[_train_index]
            test_index  = data_loo.index[_test_index]

            assert test_index.size == 1
            if verbose:
                print("fold :", i, " / ", data.shape[0])

                if very_verbose:
                    print("\ttest expt: ", test_index)
                    print("\ttrain expt: ", train_index)
                    print("\t\ttraining the tree ensemble.")

            # build X_train, y_train
            sensor_network_readout = data_loo.loc[train_index, sensors]
            source_locs = data_loo.loc[train_index, [target_prepend + "x_s", target_prepend + "y_s"]]

            # train tree ensemble on training data
            tree_ensemble = ExtraTreesRegressor(n_estimators=n_estimators)
            tree_ensemble.fit(sensor_network_readout, source_locs)

            # test tree ensemble on test data
            # first, build X_test, y_test
            if very_verbose:
                print("\t\ttesting the tree ensemble.")
            sensor_network_readout_test = data_loo.loc[test_index, sensors]
            source_locs_test_pred = tree_ensemble.predict(sensor_network_readout_test)[0]

            # store prediction of source location on test network readout.
            data_loo.loc[test_index, target_prepend + "x_s_pred"] = source_locs_test_pred[0]
            data_loo.loc[test_index, target_prepend + "y_s_pred"] = source_locs_test_pred[1]

            # also store predictions by each tree for UQ
            for tree in tree_ensemble.estimators_: # back to suppress warning
                tree.feature_names_in_ = tree_ensemble.feature_names_in_

            if uq:
                data_loo.loc[test_index, target_prepend + "ensemble pred source locs"] = [
                    np.array(
                        [tree.predict(sensor_network_readout_test)[0] for tree in tree_ensemble.estimators_]
                    )
                ]

        if delta_modeling:
            for ii, xy in enumerate(["x", "y"]):
                # the prediction is really LS pred plus the delta
                data_loo[xy + "_s_pred"] = data_loo[f"{xy}_s_LS_pred"] + data_loo[f"delta_{xy}_s_pred"]
                # avoid going outside the box
                data_loo[xy + "_s_pred"] = data_loo[xy + "_s_pred"].apply(lambda x: max(0, x))
                data_loo[xy + "_s_pred"] = data_loo[xy + "_s_pred"].apply(lambda x: min(box_dims[ii], x))

        # DONE! compute and store error = distance from true to predicted source
        calculate_errors(data_loo)

        return data_loo
    return (do_loo_cv,)


@app.cell
def _(mo):
    don_run_loo_cv = mo.ui.checkbox(label="don't run LOO CV?", value=False)
    don_run_loo_cv
    return (don_run_loo_cv,)


@app.cell
def _(data, do_loo_cv, don_run_loo_cv):
    if not don_run_loo_cv.value:
        data_loo = do_loo_cv(data)
    data_loo
    return (data_loo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""analyze error := norm of true source location vector minus predicted source location vector.""")
    return


@app.cell
def _(box_dims, data_loo, np):
    mean_error = data_loo["error"].mean()
    print("mean error: ", mean_error, "in.")
    print("\tfrom area perspective: ", np.pi * mean_error ** 2 / (box_dims[0] * box_dims[1]) * 100, "% of box")
    return


@app.cell
def _(data_loo, plt):
    plt.figure()
    plt.hist(data_loo["error"])
    plt.xlabel("error [in]")
    plt.ylabel("# experiments")
    plt.title("LOO-CV error")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""parity plot over cross-validation procedure.""")
    return


@app.cell
def _(box_dims, data_loo, plt):
    def xy_parity_plot(data):
        f, (ax1, ax2) = plt.subplots(1, 2)
        plt.subplots_adjust(wspace=0.3)
        for ax in [ax1, ax2]:
            ax.set_aspect('equal', 'box')

        ax1.set_xlim(0, box_dims[0])
        ax1.set_ylim(0, box_dims[0])
        ax2.set_xlim(0, box_dims[1])
        ax2.set_ylim(0, box_dims[1])

        ax1.set_xticks([0, 10, 20, 30, 40])
        ax1.set_yticks([0, 10, 20, 30, 40])

        ax2.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax2.set_yticks([0, 5, 10, 15, 20, 25, 30])

        ax1.set_xlabel("x$_s$ [in]")
        ax2.set_xlabel("y$_s$ [in]")
        ax1.set_ylabel("predicted x$_s$ [in]")
        ax2.set_ylabel("predicted y$_s$ [in]")

        ax1.plot([0, box_dims[0]], [0, box_dims[0]], color="black", linestyle="--")
        ax2.plot([0, box_dims[1]], [0, box_dims[1]], color="black", linestyle="--")

        ax1.scatter(data["x_s"], data["x_s_pred"], clip_on=False)
        ax2.scatter(data["y_s"], data["y_s_pred"], clip_on=False)

        x_err = data["error_x"].mean()
        y_err = data["error_y"].mean()
        ax1.legend(title=f"error = {x_err:.2f} in")
        ax2.legend(title=f"error = {y_err:.2f} in")

        plt.show()

    xy_parity_plot(data_loo)
    return (xy_parity_plot,)


@app.cell
def _(
    box_dims,
    data_loo,
    draw_obstacles,
    plt,
    sensor_to_loc,
    sensors,
    setup_environment,
    thing_to_color,
):
    def explain_errors(data_loo):
        fig, ax = setup_environment(box_dims)
        draw_obstacles(ax)

        # plot sensors
        plt.scatter(
            [sensor_to_loc[sensor][0] for sensor in sensors],
            [sensor_to_loc[sensor][1] for sensor in sensors],
            color=thing_to_color["sensor"], s=50, edgecolor="black", marker="s", label="sensor",
            clip_on=False
        )

        # source locs. color by error.
        plt.scatter(
            data_loo["x_s"], data_loo["y_s"],
            c=data_loo["error"], vmin=0, vmax=data_loo["error"].max(),
            marker="o", 
            s=65,
            clip_on=False, edgecolors="black", label="true\nsource location",
            cmap="Reds"
        )
        plt.colorbar(label="error [in]")

        # show predicted locs
        plt.scatter(
            data_loo["x_s_pred"], data_loo["y_s_pred"], s=65, marker="+",
            clip_on=False, color=thing_to_color["pred source loc"],
            label="predicted\nsource location"
        )
        for i in range(len(data_loo)):
            plt.plot(
                [data_loo.loc[i, "x_s"], data_loo.loc[i, "x_s_pred"]],
                [data_loo.loc[i, "y_s"], data_loo.loc[i, "y_s_pred"]],
                color="gray", linestyle="--", alpha=0.3
            )

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[-3:], labels[-3:], bbox_to_anchor=(1.3, 0.5), loc='upper left', borderaxespad=0)

        plt.show()

    explain_errors(data_loo)
    return


@app.cell
def _(Ellipse, np, transforms):
    # from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    def draw_confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    return (draw_confidence_ellipse,)


@app.cell
def _(
    box_dims,
    data_loo,
    draw_confidence_ellipse,
    draw_obstacles,
    np,
    plt,
    sensor_colormap,
    sensor_to_loc,
    sensors,
    setup_environment,
    theme_colors,
    thing_to_color,
):
    def viz_prediction(data_loo, exp, n_samples=25, incl_ellipse=True):
        max_response = 75.0 

        fig, ax = setup_environment(box_dims)
        draw_obstacles(ax)

        # source locs. color by error.
        plt.scatter(
            data_loo.loc[exp, "x_s"], data_loo.loc[exp, "y_s"],
            clip_on=False, edgecolors="black", color=theme_colors[4],
            s=65, marker="o",
            label="true\nsource\nlocation"
        )

        # plot sensors
        plt.scatter(
            [sensor_to_loc[sensor][0] for sensor in sensors],
            [sensor_to_loc[sensor][1] for sensor in sensors],
            s=50, edgecolor="black", marker="s",
            clip_on=False,
            c=[data_loo.loc[exp, sensor] for sensor in sensors],
            vmin=0,
            vmax=max_response,
            label="sensor",
            cmap=sensor_colormap
        )

        plt.colorbar(label="count rate [CPS]", extend="max")

        # viz uncertainty quantification by looking at whole ensemble.
        xs_preds = np.array([xs[0] for xs in data_loo.loc[exp, "ensemble pred source locs"]])
        ys_preds = np.array([xs[1] for xs in data_loo.loc[exp, "ensemble pred source locs"]])
        if n_samples > 0:
            # plot samples of predicted responses
            ids_trees = np.random.choice(np.arange(len(xs_preds)), n_samples)
            plt.scatter(
                xs_preds[ids_trees],
                ys_preds[ids_trees],
                marker="+", color=thing_to_color["pred source loc"], 
                s=65,
                label="predicted\nsource\nlocation\nsample", clip_on=False
            )

        if incl_ellipse:
            draw_confidence_ellipse(xs_preds, ys_preds, ax, facecolor=thing_to_color["pred source loc"], alpha=0.2)
            draw_confidence_ellipse(xs_preds, ys_preds, ax, n_std=2.0, facecolor=thing_to_color["pred source loc"], alpha=0.2)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[-3:], labels[-3:], bbox_to_anchor=(1.3, 0.5), loc='upper left', borderaxespad=0)

        plt.show()

    _exp = 12
    viz_prediction(data_loo, _exp, n_samples=25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### sensor importance""")
    return


@app.cell
def _(ExtraTreesRegressor, sensors):
    def train_tree_ensemble(data, n_estimators=100):
        sensor_network_readout = data.loc[:, sensors] # X_train
        source_locs = data.loc[:, ["x_s", "y_s"]]     # y_train
        tree_ensemble = ExtraTreesRegressor(n_estimators=n_estimators)
        tree_ensemble.fit(sensor_network_readout, source_locs)
        return tree_ensemble
    return (train_tree_ensemble,)


@app.cell
def _(data, n_sensors, np, plt, train_tree_ensemble):
    # train on ALL data
    tree_ensemble = train_tree_ensemble(data)

    plt.figure()
    plt.xlabel("sensor")
    plt.ylabel("feature importance")
    plt.xticks(np.arange(n_sensors))
    plt.bar(np.arange(n_sensors), tree_ensemble.feature_importances_)
    return (tree_ensemble,)


@app.cell
def _(
    box_dims,
    data,
    draw_obstacles,
    n_sensors,
    plt,
    sensor_to_loc,
    sensors,
    setup_environment,
    tree_ensemble,
):
    def viz_sensor_importance(data, tree_ensemble):
        fig, ax = setup_environment(box_dims)
        draw_obstacles(ax)

        # plot sensors
        plt.scatter(
            [sensor_to_loc[sensor][0] for sensor in sensors],
            [sensor_to_loc[sensor][1] for sensor in sensors],
            s=50, edgecolor="black", marker="s",
            clip_on=False,
            c=[tree_ensemble.feature_importances_[sensor] for sensor in range(n_sensors)],
            vmin=0,
            vmax=tree_ensemble.feature_importances_.max(),
            label="sensor",
            cmap="plasma"
        )

        plt.colorbar(label="sensor importance")

        # plt.title("sensor importance")
        plt.show()

    viz_sensor_importance(data, tree_ensemble)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## learning curve""")
    return


@app.cell(hide_code=True)
def _(mo):
    run_learning_curve = mo.ui.checkbox(label="compute the learning curve?")
    run_learning_curve
    return (run_learning_curve,)


@app.cell
def _(data, do_loo_cv, np, pd, run_learning_curve):
    if run_learning_curve.value:
        n_repeats = 3

        nb_datas = [5 * i for i in range(1, 21)]
        assert nb_datas[-1] <= len(data)

        loo_errors_mu  = []
        loo_errors_std = []
        for nb_data in nb_datas:
            print(f"running with {nb_data} data points.")
            loo_errors = []
            for r in range(n_repeats):
                data_loo_lc = do_loo_cv(data.sample(nb_data).reset_index(), verbose=False)
                loo_errors.append(data_loo_lc["error"].mean())
            loo_errors_mu.append(np.mean(loo_errors))
            loo_errors_std.append(np.std(loo_errors))

    if run_learning_curve.value:
        learning_curve = pd.DataFrame(
            {"# data": nb_datas, "loo error [in]": loo_errors_mu, "loo error std [in]": loo_errors_std}
        )
    return (learning_curve,)


@app.cell
def _(learning_curve, plt, run_learning_curve):
    if run_learning_curve.value:
        plt.figure()
        plt.xlabel("# data")
        plt.ylabel("LOO error [in]")
        plt.errorbar(
            learning_curve["# data"], learning_curve["loo error [in]"], 
            yerr=learning_curve["loo error std [in]"], marker="s"
        )
        plt.title("learning curve")
        plt.ylim(ymin=0)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ☢️ source presence classifier

    distinguish source presence from background.

    ##  background data
    """
    )
    return


@app.cell
def _(csv, pd):
    def read_background_data(filename):
        bgk_df = {}
        print(f"\nReading {filename}...")
        # Use the filename (or file_path) as the key
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            rows = []
            header = next(csv_reader)  # Read header separately

            # Create new header: keep first 6 columns and remove the rest
            new_header = header[:6]

            for row in csv_reader:
                if row[1] != '':
                    # Take first 5 columns as is
                    new_row = row[:6]
                    # Convert columns 6 through 1029 into a vector and store in 6th position
                    vector = [float(x) for x in row[5:1029]]
                    new_row[5] = vector
                    rows.append(new_row)

        bkg_df = pd.DataFrame(rows, columns=new_header)
        return bkg_df

    raw_background_data = read_background_data("Background_for_localization_model.csv")
    raw_background_data
    return (raw_background_data,)


@app.cell
def _(pd, raw_background_data):
    def reshape_bkg_data(data):
        # Convert ICR to numeric first, then group and apply list
        reshaped_dict = data.groupby('SN')['ICR'].apply(lambda x: pd.to_numeric(x, errors='coerce').tolist()).to_dict()

        reshaped_df = pd.DataFrame.from_dict(reshaped_dict, orient='index').T
        return reshaped_df

    data_bkg = reshape_bkg_data(raw_background_data)
    data_bkg
    return (data_bkg,)


@app.cell
def _(data_bkg):
    # per sensor background stats
    bkg_avg = data_bkg.mean()
    bkg_std = data_bkg.std()
    bkg_var = data_bkg.var()
    bkg_avg
    return (bkg_avg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""classification. concatentate the background with non-background.""")
    return


@app.cell
def _(data, data_bkg, np, pd):
    data_c = pd.concat([data, data_bkg], ignore_index=True)
    data_c["background"] = data_c["x_s"].map(lambda x_s : "background" if np.isnan(x_s) else "non-background")
    data_c
    return (data_c,)


@app.cell
def _(ExtraTreesClassifier, LeaveOneOut, np, sensors):
    # a multi-output tree ensemble model. maps 8D vectors to 2D vectors.
    #  maps sensor network readout to source location
    def do_loo_cv_classification(data_c, n_estimators=100, verbose=True, very_verbose=False):
        data_loo = data_c.copy()
        # store prediction of whether or not a source in the data frame.
        #  ok bc each data point is test point ONCE.
        data_loo["pred_safe"] = np.zeros((len(data_c)))

        loo = LeaveOneOut()
        for i, (_train_index, _test_index) in enumerate(loo.split(data_loo)):
            # account for index missing after remove NaN
            train_index = data_loo.index[_train_index]
            test_index  = data_loo.index[_test_index]

            assert test_index.size == 1
            if verbose:
                print("fold :", i, " / ", data_loo.shape[0])

                if very_verbose:
                    print("\ttest expt: ", test_index)
                    print("\ttrain expt: ", train_index)
                    print("\t\ttraining the tree ensemble.")

            # build X_train, y_train
            sensor_network_readout = data_loo.loc[train_index, sensors]
            safety = data_loo.loc[train_index, "background"]

            # train tree ensemble on training data
            tree_ensemble = ExtraTreesClassifier(n_estimators=n_estimators)
            tree_ensemble.fit(sensor_network_readout, safety)

            # test tree ensemble on test data
            # first, build X_test, y_test
            if very_verbose:
                print("\t\ttesting the tree ensemble.")
            sensor_network_readout_test = data_loo.loc[test_index, sensors]
            safety_pred = tree_ensemble.predict(sensor_network_readout_test)[0]

            # store prediction on test network readout.
            data_loo.loc[test_index, "pred_safe"] = safety_pred

        data_loo["agreement"] = data_loo["background"] == data_loo["pred_safe"]

        return data_loo
    return (do_loo_cv_classification,)


@app.cell
def _(data_c, do_loo_cv_classification):
    data_loo_c = do_loo_cv_classification(data_c)
    data_loo_c
    return (data_loo_c,)


@app.cell
def _(data, data_loo_c, viz_source_locs):
    incorrect_expts = data_loo_c.index[
        data_loo_c["background"] != data_loo_c["pred_safe"]
    ]

    viz_source_locs(data, ids=incorrect_expts, title="incorrect experiments")
    return


@app.cell
def _(ConfusionMatrixDisplay, data_loo_c, plt):
    cm = ConfusionMatrixDisplay.from_predictions(
        data_loo_c["background"].values, data_loo_c["pred_safe"].values,
        text_kw={'fontsize': 20}
    )
    cm.ax_.tick_params(axis='x', labelsize=16)
    cm.ax_.tick_params(axis='y', labelsize=16)
    cm.ax_.set_xlabel('predicted label', fontsize=20)
    cm.ax_.set_ylabel('true label', fontsize=20)
    cm.im_.colorbar.set_label('# instances', fontsize=18)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""viz the ones we got wrong""")
    return


@app.cell
def _(data_loo_c, np):
    ids_wrong = np.where(~ data_loo_c["agreement"])[0]
    ids_wrong
    return (ids_wrong,)


@app.cell
def _(ids_wrong, mo):
    disagreement_selector = mo.ui.slider(steps=ids_wrong)
    disagreement_selector
    return (disagreement_selector,)


@app.cell
def _(data_loo_c, disagreement_selector, viz_sensor_readout):
    viz_sensor_readout(data_loo_c, disagreement_selector.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ☢️ traditional least squares estimation of the source location

    first, calibrate a curve that gives sensor response to the source as a function of distance from the source.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## calibration curve

    output of sensor as a function of distance from source.
    """
    )
    return


@app.cell
def _(np, pd):
    output_distance_data = pd.DataFrame(
        {
            "detector output": np.array(
                [366.832, 325.406, 236.634, 157.014, 110.397, 103.229, 
                 50.095, 54.271, 38.758, 42.934, 34.583, 33.39, 35.775, 22.655, 
                 16.692, 22.059, 23.848, 23.848, 25.637, 15.5, 13.115
                ]
            ),
            "distance [in]": np.array(
                np.array(
                    [
                        26, 34, 60, 85, 136, 161, 186, 237, 262, 288, 313,
                        338, 364, 389, 415, 440, 466, 491, 517, 542, 568 # mm
                    ]
                ) / 10 / 2.54
            )
        }
    )
    output_distance_data
    return (output_distance_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### fit detector response function to data""")
    return


@app.cell
def _(np):
    def response_fun(d, S, e_ip, a, b):
        #a = 14
        #b = 28
        alpha = a / (2 * d)
        beta = b / (2 * d)
        omega = 4 * np.arctan(alpha * beta / np.sqrt(1 + alpha**2 + beta**2)) 

        return (S * 0.3 * omega) / (4 * np.pi) + 12
    return (response_fun,)


@app.cell
def _(np, optimize, output_distance_data, response_fun):
    fit_results = optimize.curve_fit(
        response_fun, output_distance_data["distance [in]"], output_distance_data["detector output"], 
        bounds=(0, np.inf), p0=[15.984e+6, 0.3, 14, 28]
    )
    opt_params = fit_results[0]
    return (opt_params,)


@app.cell
def _(np, opt_params, output_distance_data, plt, response_fun):
    plt.figure()
    plt.xlabel("distance [in]")
    plt.ylabel("detector output")
    plt.scatter(
        output_distance_data["distance [in]"],
        output_distance_data["detector output"], 
        label="data"
    )
    ds = np.linspace(0.1, 25.0)
    plt.ylim(ymax=output_distance_data["detector output"].max() * 1.02)
    plt.plot(
        ds, [response_fun(d, *opt_params) for d in ds], label="model fit"
    )
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### retreive distance from detector output

    second, given a sensor's reading, predict the distance of the source from it. this puts a circle around the sensor.
    """
    )
    return


@app.cell
def _(box_dims, np, response_fun, root_scalar):
    def find_distance_to_detector(detetector_output, opt_params):
        # max distance is corner to corner
        max_d = np.linalg.norm(box_dims)
        if response_fun(max_d, *opt_params) > detetector_output:
            return max_d    

        root_finding_res = root_scalar(
            lambda d: response_fun(d, *opt_params) - detetector_output, 
            bracket=[0.1, max_d], 
            method='brentq'
        )
        assert root_finding_res.converged
        return root_finding_res.root
    return (find_distance_to_detector,)


@app.cell
def _(find_distance_to_detector, opt_params):
    find_distance_to_detector(100.0, opt_params) # eyball: should be around 5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## least squares estimator for location of source

    third, given the response of the sensor network, select the subset of sensors that come into play with a detectable response.
    """
    )
    return


@app.cell
def _(bkg_avg, np):
    # Calculated the significant threshold for each detector based on the measured background
    Nds = 2.326*np.sqrt(bkg_avg) + bkg_avg # Currie eqn
    Nds
    return (Nds,)


@app.cell
def _(Nds):
    def select_responsive_sensors(data, exp, sensors):
        return [sensor for sensor in sensors if data.loc[exp, sensor] > Nds[sensor]]
    return (select_responsive_sensors,)


@app.cell
def _(data, select_responsive_sensors, sensors):
    select_responsive_sensors(data, 0, sensors)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""some of the experiments led to no detectable response for ANY sensor. these are false negatives.""")
    return


@app.cell
def _(data, select_responsive_sensors, sensors):
    ids_false_neg_trad = [exp for exp in range(data.shape[0]) if len(select_responsive_sensors(data, exp, sensors)) == 0]
    ids_false_neg_trad # exclude these for error analysis
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""when a SINGLE sensor in the network elicits a warning, we consider that a positive. but we need three for triangulation. so we select the top three sensors in terms of response from baseline.""")
    return


@app.cell
def _(Nds):
    def select_top_three_sensors(data, exp, sensors):
        output_minus_Nds = data.loc[0, sensors].copy()
        for sensor in sensors:
            output_minus_Nds[sensor] -= Nds[sensor]
        return list(output_minus_Nds.sort_values(ascending=False).index[0:3])
    return (select_top_three_sensors,)


@app.cell
def _(data, select_top_three_sensors, sensors):
    select_top_three_sensors(data, 20, sensors)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""finally, the traditional localization method that finds source location most consistent with the "measured" distance. here measured means we look at the response and infer the distance from the calibration curve. now we search for source location that is most consistent with those distances we measure.""")
    return


@app.cell
def _(
    box_dims,
    data,
    differential_evolution,
    find_distance_to_detector,
    np,
    opt_params,
    select_responsive_sensors,
    select_top_three_sensors,
    sensor_to_loc,
    sensors,
):
    def trad_localize(data, exp, sensors, opt_params, verbose=False):
        if verbose:
            print(f"expt {exp}")

        # what sensors are in the game?
        game_sensors = select_responsive_sensors(data, exp, sensors)

        if len(game_sensors) == 0:
            print(f"FALSE NEGATIVE: expt {exp} has no detectors with significant response.")
            return np.array([np.nan, np.nan])

        if len(game_sensors) < 3: # we need three...
            # just select three highest above Nds
            print(f"WARNING: expt {exp} has only {len(game_sensors)} detectors with significant response. selecting top 3.")
            game_sensors = select_top_three_sensors(data, exp, sensors)

        if verbose:
            print("sensor data in the game:", data.loc[exp, game_sensors])

        # predicted distance from these sensors
        measured_distances = [
            find_distance_to_detector(data.loc[exp, sensor], opt_params) for sensor in game_sensors
        ]

        game_sensors_xy = [
            np.array(sensor_to_loc[sensor]) for sensor in game_sensors
        ]

        # find location most consistent with this
        def objective(x):
            # residual := ~ distance if source were at x - measured distance
            return sum(
                # residual between sensor-source-loc distance and what we measure it to be
                (np.linalg.norm(x - game_sensors_xy[i]) - measured_distances[i]) ** 2 # residual
                for i in range(len(game_sensors))
            )

        opt_res = differential_evolution(
            objective,
            bounds=[(0.0, box_dims[0]), (0.0, box_dims[1])],
            popsize=250
        ) # better exploration to ensure we are not trapped in a local minimum

        assert opt_res.success

        xy_pred = opt_res.x

        assert xy_pred[0] >= 0 and xy_pred[0] <= box_dims[0]
        assert xy_pred[1] >= 0 and xy_pred[1] <= box_dims[1] 

        return xy_pred

    trad_localize(data, 3, sensors, opt_params)
    return (trad_localize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""🥞 let's do it!""")
    return


@app.cell
def _(calculate_errors, data, opt_params, sensors, trad_localize):
    # do traditional localization
    xy_trad_preds = [trad_localize(data, exp, sensors, opt_params) for exp in range(data.shape[0])]

    # put results into data frame
    data_trad = data.copy()
    data_trad["x_s_pred"] = [xy_trad_preds[exp][0] for exp in range(data.shape[0])]
    data_trad["y_s_pred"] = [xy_trad_preds[exp][1] for exp in range(data.shape[0])]

    # calculate errors
    calculate_errors(data_trad)
    data_trad
    return (data_trad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ☢️ a sub-baseline: simple triangulation

    weighted average of significant sensor locations.
    """
    )
    return


@app.cell
def _(np, select_top_three_sensors, sensor_to_loc):
    def dumb_triangulation(data, exp, sensors, opt_params):
        game_sensors = select_top_three_sensors(data, exp, sensors)
        wts = data.loc[exp, game_sensors] / data.loc[exp, game_sensors].sum()
        return np.sum([np.array(sensor_to_loc[sensor]) * wts[sensor] for sensor in game_sensors], axis=0)
    return (dumb_triangulation,)


@app.cell
def _(data, dumb_triangulation, opt_params, sensors):
    dumb_triangulation(data, 0, sensors, opt_params)
    return


@app.cell
def _(calculate_errors, data, dumb_triangulation, opt_params, sensors):
    xy_triangulation_preds = [dumb_triangulation(data, exp, sensors, opt_params) for exp in range(data.shape[0])]

    data_triangulation = data.copy()
    data_triangulation["x_s_pred"] = [xy_triangulation_preds[exp][0] for exp in range(data.shape[0])]
    data_triangulation["y_s_pred"] = [xy_triangulation_preds[exp][1] for exp in range(data.shape[0])]

    calculate_errors(data_triangulation)
    data_triangulation
    return (data_triangulation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Vizualizes the source locations where no solution was able to be found b/c no significant sensors""")
    return


@app.cell
def _(data, data_trad, np, viz_source_locs):
    _ids_sources_to_viz = data_trad.index[np.isnan(data_trad["y_s_pred"])]
    viz_source_locs(data, ids=_ids_sources_to_viz)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""...when the errors are huge.""")
    return


@app.cell
def _(data, data_trad, viz_source_locs):
    _ids_sources_to_viz = data_trad.index[data_trad["error"] > 20.0]
    viz_source_locs(data, ids=_ids_sources_to_viz)
    _ids_sources_to_viz
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## compare errors among methods""")
    return


@app.cell
def _(data, ids_wrong):
    expts_to_compare = [exp for exp in range(data.shape[0]) if not exp in ids_wrong]
    return (expts_to_compare,)


@app.cell
def _(data_loo, data_trad, data_triangulation, expts_to_compare, pd):
    all_errors = pd.merge(
        pd.merge(
            data_loo["error"].rename("tree\nensemble"), 
            data_trad["error"].rename("trad\nlocalization"), 
            left_index=True, right_index=True
        ),
        data_triangulation["error"].rename("naive\ntriangulation"),
        left_index=True, right_index=True
    )
    all_errors = all_errors.loc[expts_to_compare, :]
    all_errors = all_errors.melt(var_name='method', value_name='error')
    all_errors
    return (all_errors,)


@app.cell
def _(all_errors, plt, pt, sns):
    # see https://github.com/pog87/PtitPrince/blob/master/tutorial_python/raincloud_tutorial_python.ipynb
    _f, _ax = plt.subplots(figsize=(7, 5))
    pt.RainCloud(
        x="method", data=all_errors.rename(columns={"error": "absolute error [in]"}), 
        y="absolute error [in]", hue="method", bw=0.25, ax=_ax, orient="h", palette=sns.color_palette("Set2")[:3]#, cut=2
    )
    _ax.set_xlim(xmin=0.0)
    for i, method in enumerate(all_errors["method"].unique()):
        mean_err = all_errors[all_errors["method"] == method]["error"].mean()
        _ax.text(
            x=_ax.get_xlim()[1],  # right edge
            y=i-0.4,           # slightly above each cloud
            s=f"mean: {mean_err:.3f} in",
            ha="right", va="bottom",
            fontsize=9, color="black"
        )
    plt.savefig("error_rainclouds.pdf", format="pdf")
    plt.show()
    return (i,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 🥔 delta learning.""")
    return


@app.cell
def _(data, data_trad, n_expts):
    data_delta = data.copy()

    # load with deltas
    for exp in range(n_expts):
        for coord in ["x", "y"]:
            # x_{pred, ML} = x_{pred, LS} + delta.
            #   hope: x_{pred, ML} = x_true. so delta := x_true - x_{pred, LS}
            data_delta.loc[exp, "delta_" + coord + "_s"] = data.loc[exp,  coord + "_s"] - data_trad.loc[exp, coord + "_s_pred"]

            # store trad pred too
            data_delta.loc[exp, f"{coord}_s_LS_pred"] = data_trad.loc[exp, coord + "_s_pred"]

    # remove the NaNs which are false negatives
    print(f"dropping {data_delta.isna().any(axis=1).sum()} rows with NaN corresponding with false negatives.")
    data_delta = data_delta.dropna()

    data_delta
    return (data_delta,)


@app.cell
def _(data_delta, do_loo_cv, don_run_loo_cv):
    if not don_run_loo_cv.value:
        data_delta_loo = do_loo_cv(data_delta, delta_modeling=True, very_verbose=False, uq=False)
    data_delta_loo
    return (data_delta_loo,)


@app.cell
def _(data_delta_loo, xy_parity_plot):
    xy_parity_plot(data_delta_loo)
    return


@app.cell
def _(data_delta_loo, np, plt):
    def deltas_parity_plot(data, scale=None):
        if not scale:
            scale = np.max(
                [
                    data["delta_x_s_pred"].abs().max(), data["delta_y_s_pred"].abs().max(),
                    data["delta_x_s"].abs().max(), data["delta_y_s"].abs().max()
                ]
            )

        f, (ax1, ax2) = plt.subplots(1, 2)
        plt.subplots_adjust(wspace=0.3)
        for ax in [ax1, ax2]:
            ax.set_aspect('equal', 'box')

        ax1.set_xlim(-scale, scale)
        ax1.set_ylim(-scale, scale)
        ax2.set_xlim(-scale, scale)
        ax2.set_ylim(-scale, scale)

        ax1.plot([-scale, scale], [-scale, scale], color="black", linestyle="--")
        ax2.plot([-scale, scale], [-scale, scale], color="black", linestyle="--")

        ax1.set_xlabel("$\delta x_s$ [in]")
        ax2.set_xlabel("$\delta y_s$ [in]")
        ax1.set_ylabel("predicted $\delta x_s$ [in]")
        ax2.set_ylabel("predicted $\delta y_s$ [in]")

        ax1.scatter(data["delta_x_s"], data["delta_x_s_pred"], clip_on=False, s=5)
        ax2.scatter(data["delta_y_s"], data["delta_y_s_pred"], clip_on=False, s=5)

        plt.show()

    deltas_parity_plot(data_delta_loo, scale=20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 🚓 tracking""")
    return


@app.cell
def _():
    n_tracks = 22
    return (n_tracks,)


@app.cell
def _(n_tracks, read_all_data):
    tracking_outputs = read_all_data(n_tracks, lambda exp: f"tracking/20260303_track_{exp+1}.csv")
    return (tracking_outputs,)


@app.cell
def _(box_dims, pd):
    tracking_locs = pd.DataFrame(
        {
            'x_s': [40, 35, 30, 25, 22, 21, 19, 17, 16, 11, 6, 4, 4, 5, 7, 12, 10, 10, 15, 20, 25, 30],
            'y_s': [14, 14, 14, 11, 6, 1, 6, 11, 16, 17, 17, 13, 8, 13, 16, 17, 22, 27, 28, 29, 27, 27]
        }
    ) # note this is with origin in top left
    tracking_locs["y_s"] = box_dims[1] - tracking_locs["y_s"] # translate so that origin in bottom left
    tracking_locs
    return (tracking_locs,)


@app.cell
def _(make_data_nice, tracking_locs, tracking_outputs):
    tracking_data = make_data_nice(tracking_outputs, tracking_locs)
    tracking_data
    return (tracking_data,)


@app.cell
def _(tracking_data, viz_source_locs):
    viz_source_locs(tracking_data)
    return


@app.cell
def _(i, sensors, tracking_data):
    tracking_data.loc[i, sensors]
    return


@app.cell
def _(tracking_data):
    tracking_data
    return


@app.cell
def _(sensors, tracking_data, tree_ensemble):
    def predict_track(tracking_data, tree_ensemble):
        pred_source_locs = tree_ensemble.predict(tracking_data.loc[:, sensors])
        for i in range(len(tracking_data)):
            tracking_data.loc[i, "x_s_pred"] = pred_source_locs[i][0]
            tracking_data.loc[i, "y_s_pred"] = pred_source_locs[i][1]
        return tracking_data
    
    tracking_data_loo = predict_track(tracking_data, tree_ensemble)
    return


@app.cell
def _(
    box_dims,
    draw_obstacles,
    plt,
    sensor_colormap,
    sensor_to_loc,
    sensors,
    setup_environment,
    theme_colors,
    thing_to_color,
):
    def viz_track(data_loo):
        max_response = 75.0 

        fig, ax = setup_environment(box_dims)
        draw_obstacles(ax)

        for exp in range(len(data_loo)):
            # source locs. color by error.
            plt.scatter(
                data_loo.loc[exp, "x_s"], data_loo.loc[exp, "y_s"],
                clip_on=False, edgecolors="black", color=theme_colors[4],
                s=65, marker="o",
                label="true\nsource\nlocation"
            )

        # plot sensors
        plt.scatter(
            [sensor_to_loc[sensor][0] for sensor in sensors],
            [sensor_to_loc[sensor][1] for sensor in sensors],
            s=50, edgecolor="black", marker="s",
            clip_on=False,
            #c=[data_loo.loc[exp, sensor] for sensor in sensors],
            vmin=0,
            vmax=max_response,
            label="sensor",
            cmap=sensor_colormap
        )

        plt.colorbar(label="count rate [CPS]", extend="max")

        plt.scatter(
            data_loo["x_s_pred"], data_loo["y_s_pred"], s=65, marker="+",
            clip_on=False, color=thing_to_color["pred source loc"],
            label="predicted\nsource location"
        )
        """
        for i in range(len(data_loo)):
            plt.plot(
                [data_loo.loc[i, "x_s"], data_loo.loc[i, "x_s_pred"]],
                [data_loo.loc[i, "y_s"], data_loo.loc[i, "y_s_pred"]],
                color="gray", linestyle="--", alpha=0.3
            )
        """
        for i in range(0, len(data_loo)-1):
            plt.plot(
                [data_loo.loc[i, "x_s"], data_loo.loc[i+1, "x_s"]],
                [data_loo.loc[i, "y_s"], data_loo.loc[i+1, "y_s"]],
                color="grey", linestyle="-", alpha=0.3#, label="True path"
            )

        for i in range(0, len(data_loo)-1):
            plt.plot(
                [data_loo.loc[i, "x_s_pred"], data_loo.loc[i+1, "x_s_pred"]],
                [data_loo.loc[i, "y_s_pred"], data_loo.loc[i+1, "y_s_pred"]],
                color="red", linestyle="-", alpha=0.3#, label="Predicted  path"
            )

        # viz uncertainty quantification by looking at whole ensemble.
        #xs_preds = np.array([xs[0] for xs in data_loo.loc[exp, "ensemble pred source locs"]])
        #ys_preds = np.array([xs[1] for xs in data_loo.loc[exp, "ensemble pred source locs"]])
        """
        if n_samples > 0:
            # plot samples of predicted responses
            ids_trees = np.random.choice(np.arange(len(xs_preds)), n_samples)
            plt.scatter(
                xs_preds[ids_trees],
                ys_preds[ids_trees],
                marker="+", color=thing_to_color["pred source loc"], 
                s=65,
                label="predicted\nsource\nlocation\nsample", clip_on=False
            )
        """

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[-3:], labels[-3:], bbox_to_anchor=(1.3, 0.5), loc='upper left', borderaxespad=0)

        plt.show()
    return (viz_track,)


@app.cell
def _(tracking_data, viz_track):
    viz_track(tracking_data)
    return


if __name__ == "__main__":
    app.run()
