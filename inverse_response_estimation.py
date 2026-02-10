import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import csv
    import os
    import math
    import pandas as pd
    import marimo as mo
    import localization as lx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    import matplotlib.font_manager as fm
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib as mpl
    import seaborn as sns
    from scipy.optimize import least_squares
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
        IsolationForest,
        LeaveOneOut,
        csv,
        least_squares,
        math,
        mo,
        np,
        os,
        patches,
        pd,
        plt,
        sns,
        train_test_split,
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
        'pine': theme_colors[1]
    }
    sensor_colormap = "Purples"

    # theme_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # sns.color_palette(theme_colors)
    sns.color_palette("Set2")
    return sensor_colormap, theme_colors, thing_to_color


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::icon-park:data:: read in sensor network response data

    * each data set is the sensor network response to a source in a particular location.
    * `SN` gives sensor network ID
    * `ICR` gives count rate
    * the list of source locations are below.

    ## locations of detectors in the environment
    """)
    return


@app.cell
def _(box_dims, np):
    # sensor locs. origin is top left here. (x, y) positions
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

    for (sensor, loc) in sensor_to_loc.items():
        # convert to numpy array
        sensor_to_loc[sensor] = np.array(loc)
        # convert to origin in bottom left
        sensor_to_loc[sensor][1] = box_dims[1] - sensor_to_loc[sensor][1]
    return (sensor_to_loc,)


@app.cell
def _():
    box_dims = (42.0, 33.0) # box size, in
    return (box_dims,)


@app.cell
def _(sensor_to_loc):
    n_sensors = 8
    assert len(sensor_to_loc) == n_sensors
    return (n_sensors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## location of radioactive source placed in the environment

    the first 25 rows come from Latin Hypercube sampling.

    the next 25 rows are manually-selected on the corners/boundary.

    the remaining 25 are from another round of Latin Hypercube sampling.
    """)
    return


@app.cell
def _(box_dims, pd):
    df_source_locs = pd.DataFrame(
        {
            'x_s': [28.0, 38.0, 4.0, 2.0, 10.0, 41.0,32.0, 5.0, 40.0, 24.0, 16.0, 7.0, 18.0, 35.0, 33.0, 30.0, 23.0, 26.0, 19.0, 12.0, 0.0, 9.0, 21.0, 37.0, 14.0, 2, 5,  7,  7, 10, 10, 12, 16, 40, 41, 31, 31, 34, 41, 41, 41, 36, 28, 17, 12,  0,  7,  7,  7, 10, 13.0, 38.0, 41.0, 19.0,  7.0, 17.0, 39.0, 36.0,  9.0, 25.0, 32.0, 16.0,  6.0, 42.0,  0.0,  2.0, 26.0, 20.0, 28.0, 30.0,  3.0, 29.0, 14.0, 23.0, 22.0],
            'y_s': [14.0,30.0,28.0,21.0,16.0,10.0,4.0, 6.0, 18.0,32.0,19.0,22.0,26.0,12.0,8.0, 6.0, 11.0,29.0,15.0,3.0, 0.0, 1.0, 23.0,23.0,32.0,0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 4.0, 0.0, 4.0, 4.0, 0.0, 0.0, 15.0,20.0,28.0,32.0,32.0,32.0,32.0,32.0,32.0,29.0,25.0,25.0, 20.0, 10.0, 24.0, 11.0, 15.0, 30.0, 14.0, 32.0, 0.0 , 32.0, 9.0 , 13.0, 6.0 , 27.0, 29.0, 23.0, 10.0, 2.0 , 18.0, 27.0, 16.0, 1.0 , 17.0, 19.0, 8.0 ]
        }
    ) # note this is with origin in top left
    df_source_locs["y_s"] = box_dims[1] - df_source_locs["y_s"] # accounts for coordinate system
    df_source_locs
    return (df_source_locs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## read in detector outputs
    """)
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
    n_expts = 75
    return (n_expts,)


@app.cell
def _(folder_path, n_expts, n_sensors, os, read_detector_outputs):
    def read_all_data(n_expts):
        dataframes = {}

        for exp in range(n_expts):
            filename = f"pos_{exp}.csv"
            file_path = os.path.join(folder_path, filename)
            print(f"\nReading {filename}...")
            # Use the filename (or file_path) as the key
            dataframes[exp] = read_detector_outputs(file_path)

            # unique sensors
            assert dataframes[exp]["SN"].nunique() == n_sensors
        return dataframes

    dataframes = read_all_data(n_expts)
    return (dataframes,)


@app.cell
def _(dataframes):
    dataframes[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## re-work data into an ML-friendly format

    ::lucide:lightbulb:: source locations paired with sensor network response vectors

    first, get list of sensors in the network
    """)
    return


@app.cell
def _(dataframes):
    sensors = dataframes[0]["SN"].unique()
    sensors
    return (sensors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    next, from a sensor network response data frame, extract the count rate of a particular sensor.
    """)
    return


@app.function
def grab_sensor_response(df, sensor):
    return float(df.loc[df["SN"] == sensor, "ICR"].item())


@app.cell
def _(dataframes, sensors):
    grab_sensor_response(dataframes[0], sensors[2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    🎯 goal:

    * each row is an experiment where we place the source at a location and observe the sensor network response
    * the row lists the source location paired with the sensor response vector.
    """)
    return


@app.cell
def _(dataframes, df_source_locs, n_expts, pd, sensors):
    def make_data_nice(dataframes, df_source_locs):
        data = pd.DataFrame(columns=sensors) # nice data for ML
        # sensor network data
        for exp in range(n_expts):
            new_row = {sensor : grab_sensor_response(dataframes[exp], sensor) for sensor in sensors}
            data.loc[len(data)] = new_row
        # join source locs
        data = pd.concat([df_source_locs, data], axis=1)
        return data

    data = make_data_nice(dataframes, df_source_locs)
    data
    return (data,)


@app.cell
def _(dataframes, sensors):
    assert grab_sensor_response(dataframes[0], sensors[2]) == 31.6
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ::lucide:sailboat:: visually explore the data

    first, where are the source locations over all experiments?
    """)
    return


@app.cell
def _(box_dims, data, plt, thing_to_color):
    def viz_source_locs(data):
        plt.figure()
        plt.scatter(data["x_s"], data["y_s"], clip_on=False, color=thing_to_color["true source loc"], s=65, marker="o")
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])
        plt.title("source locations")
        plt.show()

    viz_source_locs(data)
    return (viz_source_locs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    second, visualize the sensor readout and source location for a single experiment.
    """)
    return


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
    draw_obstacles,
    plt,
    sensor_colormap,
    sensor_to_loc,
    sensors,
    thing_to_color,
):
    def viz_sensor_readout(data, exp):
        max_response = 75.0 # data.loc[exp, sensors].max()

        plt.figure()

        draw_obstacles(plt.gca())

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

        plt.colorbar(label="count rate [CPS]", extend="max")

        # plot source location
        plt.scatter(
            data.loc[exp, "x_s"], data.loc[exp, "y_s"], marker="o", 
            s=65, color=thing_to_color["true source loc"], label="source location", clip_on=False
        )

        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])
        plt.title(f"experiment {exp}")

        # unique legend
        # Get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create a dictionary to store unique labels and their corresponding handles
        by_label = dict(zip(labels, handles))

        # Create the legend using only the unique entries
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.3, 0.5), loc='upper left', borderaxespad=0)

        plt.show()

    viz_sensor_readout(data, 10)
    return (viz_sensor_readout,)


@app.cell
def _(
    box_dims,
    data,
    draw_obstacles,
    plt,
    sensor_to_loc,
    sensors,
    thing_to_color,
):
    def viz_source_locs_with_data(data):
        FNs = [1, 40, 41, 57, 63]
        plt.figure()
        draw_obstacles(plt.gca())
        plt.scatter(
            # x locs of sensors
            [sensor_to_loc[sensor][0] for sensor in sensors], 
            # y locs of sensors
            [sensor_to_loc[sensor][1] for sensor in sensors],
            s=65,
            edgecolor="black",
            marker="s",
            label="sensor"
        )
        for i in FNs:
            plt.scatter(
                data.loc[i, "x_s"], data.loc[i, "y_s"], marker="o", 
                s=65, color=thing_to_color["true source loc"], label="source location", clip_on=False
            )
        #plt.scatter(data["x_s"], data["y_s"], clip_on=False, color=thing_to_color["true source loc"], s=65, marker="o")
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])
        plt.title("False Negative Locations")
        plt.show()
    viz_source_locs_with_data(data)
    return


@app.cell
def _(box_dims, data, draw_obstacles, plt, sensor_to_loc, sensors):
    def viz_env(data):

        plt.figure()
        draw_obstacles(plt.gca())
        plt.scatter(
            # x locs of sensors
            [sensor_to_loc[sensor][0] for sensor in sensors], 
            # y locs of sensors
            [sensor_to_loc[sensor][1] for sensor in sensors],
            s=65,
            edgecolor="black",
            marker="s",
            label="sensor"
        )

        #plt.scatter(data["x_s"], data["y_s"], clip_on=False, color=thing_to_color["true source loc"], s=65, marker="o")
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])
        plt.title("Test Environment")
        plt.show()
    viz_env(data)
    return


@app.cell
def _():
    return


@app.cell
def _(data):
    print(data.loc[[1, 40, 41, 57, 63],'x_s'], data.loc[[1, 40, 41, 57, 63],'y_s'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    the distribution of the detector responses over all experiments and correlations between them. use a log scale.
    """)
    return


@app.cell
def _(data, plt, sensors, sns, theme_colors):
    sns.swarmplot(data[sensors], size=2, color=theme_colors[3])
    plt.axhline(13.2, linestyle="--", color="gray", zorder=0) # background
    plt.yscale("log")
    plt.xlabel("sensor")
    plt.ylabel("response [CPS]")
    plt.title("Sensor Response Distributions")
    return


@app.cell
def _(data, plt, sensors, sns):
    _g = sns.pairplot(data[sensors], corner=True)
    for _ax in _g.axes.flat:
        if not _ax == None:
            _ax.set_xscale('log')
            _ax.set_yscale('log')
    plt.show()
    return


@app.cell
def _(data, np, plt, sns):
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
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title("Detector Correlation Matrix")
        plt.show()
    correlation_matrix(data.iloc[:,3::])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ☢️ source location predictor

    ML task: predict source location from sensor network response


    **input**: 8D sensor network response

    **output**: 2D source location

    ## leave-one-out cross validation
    """)
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
def _(ExtraTreesRegressor, LeaveOneOut, np, sensors):
    # a multi-output tree ensemble model. maps 8D vectors to 2D vectors.
    #  maps sensor network readout to source location
    def do_loo_cv(data, n_estimators=100, verbose=True, very_verbose=False):
        data_loo = data.copy()
        # store predicted source locations in data frame.
        #  ok bc each data point is test point ONCE.
        data_loo["x_s_pred"] = np.zeros((len(data)))
        data_loo["y_s_pred"] = np.zeros((len(data)))
        data_loo["ensemble pred source locs"] = [np.zeros(n_estimators) for _ in range(len(data_loo))]

        loo = LeaveOneOut()
        for i, (train_index, test_index) in enumerate(loo.split(data_loo)):
            assert test_index.size == 1
            if verbose:
                print("fold :", i, " / ", data.shape[0])

                if very_verbose:
                    print("\ttest expt: ", test_index)
                    print("\ttrain expt: ", train_index)
                    print("\t\ttraining the tree ensemble.")

            # build X_train, y_train
            sensor_network_readout = data_loo.loc[train_index, sensors]
            source_locs = data_loo.loc[train_index, ["x_s", "y_s"]]

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
            data_loo.loc[test_index, "x_s_pred"] = source_locs_test_pred[0]
            data_loo.loc[test_index, "y_s_pred"] = source_locs_test_pred[1]

            # also store predictions by each tree for UQ
            for tree in tree_ensemble.estimators_: # back to suppress warning
                tree.feature_names_in_ = tree_ensemble.feature_names_in_

            data_loo.loc[test_index, "ensemble pred source locs"] = [
                np.array(
                    [tree.predict(sensor_network_readout_test)[0] for tree in tree_ensemble.estimators_]
                )
            ]

        # DONE! compute and store error = distance from true to predicted source
        data_loo["error_x"] = np.abs(data_loo["x_s"] - data_loo["x_s_pred"])
        data_loo["error_y"] = np.abs(data_loo["y_s"] - data_loo["y_s_pred"])
        data_loo["error"] = np.sqrt(
            (data_loo["x_s"] - data_loo["x_s_pred"]) ** 2 + (data_loo["y_s"] - data_loo["y_s_pred"]) ** 2
        )

        return data_loo
    return (do_loo_cv,)


@app.cell
def _(mo):
    run_loo_cv = mo.ui.checkbox(label="run LOO CV?")
    run_loo_cv
    return (run_loo_cv,)


@app.cell
def _(data, do_loo_cv, run_loo_cv):
    if run_loo_cv.value:
        data_loo = do_loo_cv(data)
    data_loo
    return (data_loo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    analyze error := norm of true source location vector minus predicted source location vector.
    """)
    return


@app.cell
def _(data_loo):
    mean_error = data_loo["error"].mean()
    return (mean_error,)


@app.cell
def _(mean_error):
    print("mean error: ", mean_error, "in.")
    return


@app.cell
def _(box_dims, mean_error):
    print("from area perspective: ", mean_error ** 2 / (box_dims[0] * box_dims[1]) * 100, "% error")
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
    mo.md(r"""
    parity plot over cross-validation procedure.
    """)
    return


@app.cell
def _(box_dims, data_loo, plt):
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

    ax1.scatter(data_loo["x_s"], data_loo["x_s_pred"], clip_on=False)
    ax2.scatter(data_loo["y_s"], data_loo["y_s_pred"], clip_on=False)

    x_err = data_loo["error_x"].mean()
    y_err = data_loo["error_y"].mean()
    ax1.legend(title=f"error = {x_err:.2f} in")
    ax2.legend(title=f"error = {y_err:.2f} in")

    plt.show()
    return


@app.cell
def _(
    box_dims,
    data_loo,
    draw_obstacles,
    plt,
    sensor_to_loc,
    sensors,
    thing_to_color,
):
    def explain_errors(data_loo):
        fig, ax = plt.subplots()

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


        # plot sensors
        plt.scatter(
            [sensor_to_loc[sensor][0] for sensor in sensors],
            [sensor_to_loc[sensor][1] for sensor in sensors],
            color=thing_to_color["sensor"], s=50, edgecolor="black", marker="s", label="sensor",
            clip_on=False
        )

        # plot obstacles TODO

        plt.gca().set_aspect('equal', 'box')

        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])

        plt.legend(bbox_to_anchor=(1.3, 0.5), loc='upper left', borderaxespad=0)
        draw_obstacles(plt.gca())
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
    np,
    plt,
    sensor_colormap,
    sensor_to_loc,
    sensors,
    theme_colors,
    thing_to_color,
):
    def viz_prediction(data_loo, exp, n_samples=25, incl_ellipse=True):
        max_response = 75.0 

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
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


        # plot obstacles TODO

        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])

        plt.legend(bbox_to_anchor=(1.3, 0.5), loc='upper left', borderaxespad=0)

        plt.show()

    _exp = 6
    viz_prediction(data_loo, _exp, n_samples=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### sensor importance
    """)
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
def _(data, n_sensors, np, plt, sensors, train_tree_ensemble):
    tree_ensemble = train_tree_ensemble(data)

    plt.figure()
    plt.xlabel("sensor")
    plt.ylabel("feature importance")
    plt.xticks(np.arange(n_sensors), sensors)
    plt.bar(np.arange(n_sensors), tree_ensemble.feature_importances_)
    return (tree_ensemble,)


@app.cell
def _(tree_ensemble):
    tree_ensemble.feature_importances_
    return


@app.cell
def _(
    box_dims,
    data,
    draw_obstacles,
    plt,
    sensor_colormap,
    sensor_to_loc,
    sensors,
    tree_ensemble,
):
    def viz_importance(data):

        plt.figure()
        draw_obstacles(plt.gca())

        # plot sensors


        plt.scatter(
            [sensor_to_loc[sensor][0] for sensor in sensors],
            [sensor_to_loc[sensor][1] for sensor in sensors],
            s=50, edgecolor="black", marker="s",
            clip_on=False,
            c=[tree_ensemble.feature_importances_[sensor] for sensor in range(0,len(sensors))],
            vmin=0,
            vmax=0.24,
            label="sensor",
            cmap=sensor_colormap
        )

        #plt.scatter(data["x_s"], data["y_s"], clip_on=False, color=thing_to_color["true source loc"], s=65, marker="o")
        plt.colorbar(label="Importance", extend="max")
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])
        plt.title("Sensor Impotance")
        plt.show()

    viz_importance(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## learning curve
    """)
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

        nb_datas = [5 * i for i in range(1, 16)]
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
    mo.md(r"""
    # ☢️ source presence classifier

    distinguish source presence from background.

    ##  background data
    """)
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
    def find_avg_bkgs(data_bkg):
        avg_bkgs = []
        for det in data_bkg:
            avg_bkgs.append(sum(data_bkg[det][:])/len(data_bkg[det][:]))
        return avg_bkgs
    avg_bkgs = find_avg_bkgs(data_bkg)
    avg_bkgs
    return (avg_bkgs,)


@app.cell
def _(avg_bkgs, data_bkg):
    print(data_bkg['16513']-avg_bkgs[1])
    return


@app.cell
def _(avg_bkgs, data_bkg):
    bkg_var = []
    for _det in data_bkg:
        bkg_var.append(sum((data_bkg[_det]-avg_bkgs[0])**2)/(len(data_bkg[_det])-1))
        print(f"Detector: {_det} variance is {sum((data_bkg[_det]-avg_bkgs[0])**2)/len(data_bkg[_det])}")
    return (bkg_var,)


@app.cell
def _(bkg_var):
    bkg_var
    return


@app.cell
def _(bkg_var, np):
    bkg_std = np.sqrt(bkg_var)
    bkg_std
    return


app._unparsable_cell(
    r"""
    Looks at the statistical anomaly detection
    """,
    name="_"
)


@app.cell
def _(avg_bkgs, data_bkg, np):
    _c = 0
    _d = 0
    _a = 0 
    _t = 0
    fp_rates = []
    for det in data_bkg:
        print(f"Average Bkg count rate for detector: {det} is {avg_bkgs[_c]}")
        print(f"Anomalous count rate threshold, nd, is {(4.65*np.sqrt(60*avg_bkgs[_c])+2.71)/60+avg_bkgs[_c]}")
        for i in range(len(data_bkg[det])):
            _t += 1
            if data_bkg[det][i]>((4.65*np.sqrt(60*avg_bkgs[_c])+2.71)/60+avg_bkgs[_c]):
                _a+=1
                _d+=1
                print(f"{det}_{i}:{data_bkg[det][i]}")
        print(f"Number of false positives for detector: {det} is {_a}")
        fp_rates.append(_a/len(data_bkg[det]))
        _a = 0
        _c += 1
    print(f"Total false positive percent: {(_d/_t)*100}%")
    return (fp_rates,)


@app.cell
def _(fp_rates):
    fp_rates
    return


@app.cell
def _(fp_rates, plt):
    plt.bar(range(len(fp_rates)), fp_rates)
    return


@app.cell
def _(bkg_var, fp_rates, plt):
    plt.bar(range(len(fp_rates)), bkg_var)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    classification. concatentate the background with non-background.
    """)
    return


@app.cell
def _(data, data_bkg, np, pd):
    data_c = pd.concat([data, data_bkg], ignore_index=True)
    data_c["background"] = data_c["x_s"].map(lambda x_s : "background" if np.isnan(x_s) else "non-background")
    data_c
    return (data_c,)


@app.cell
def _(ExtraTreesClassifier, LeaveOneOut, data, np, sensors):
    # a multi-output tree ensemble model. maps 8D vectors to 2D vectors.
    #  maps sensor network readout to source location
    def do_loo_cv_classification(data_c, n_estimators=100, verbose=True, very_verbose=False):
        data_loo = data_c.copy()
        # store prediction of whether or not a source in the data frame.
        #  ok bc each data point is test point ONCE.
        data_loo["pred_safe"] = np.zeros((len(data_c)))

        loo = LeaveOneOut()
        for i, (train_index, test_index) in enumerate(loo.split(data_loo)):
            assert test_index.size == 1
            if verbose:
                print("fold :", i, " / ", data.shape[0])

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
def _(data_loo_c):
    def _():
        for i in range(1,len(data_loo_c)):
            if data_loo_c["background"][i] != data_loo_c['pred_safe'][i]:
                print(data_loo_c["pred_safe"][i])
                print(i)


    _()
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


@app.cell
def _():
    5/75
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    viz the ones we got wrong
    """)
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
    mo.md(r"""
    # ☢️ detection of sensor tampering

    suppose we think a source is present, and we want to predict its location.
    now, we use anomaly detector to make sure a sensor is not being tampered with.

    ## read in sensor tampering data.
    """)
    return


@app.cell
def _(folder_path, n_sensors, os, read_detector_outputs):
    def read_tampering_data():
        n_expts = 10
        dets = [12, 13, 19] # detectors tampered with
        tamp_data = {
            "12" : {},
            "13" : {},
            "19" : {}
        }
        for det in dets:
            for exp in range(1, n_expts+1):
                filename = f"det{det}_tamp{exp}.csv"
                file_path = os.path.join(folder_path, filename)
                print(f"\nReading {filename}...")
                # Use the filename (or file_path) as the key
                tamp_data[f"{det}"][exp] = read_detector_outputs(file_path)

            # unique sensors
            assert tamp_data[f"{det}"][exp]["SN"].nunique() == n_sensors
        return tamp_data

    raw_tampering_data = read_tampering_data()
    raw_tampering_data
    return (raw_tampering_data,)


@app.cell
def _(raw_tampering_data):
    raw_tampering_data
    return


@app.cell
def _(pd):
    tampered_source_locs = pd.DataFrame(
        {
            'tampered_sensor' : ['16512', '16512', '16512', '16512', '16512', '16512', '16512', '16512', '16512', 
                                 '16512', '16513', '16513', '16513', '16513', '16513', '16513', '16513', '16513', 
                                 '16513', '16513', '16519', '16519', '16519', '16519', '16519', '16519', '16519', 
                                 '16519', '16519', '16519'
            ],
            'x_s' : [39.0, 37.0, 35.0, 33.0, 31.0, 40.0, 38.0, 36.0, 34.0, 32.0, 18.0, 20.0, 21.0, 22.0, 
                    21.0, 23.0, 24.0, 24.0, 23.0, 18.0, 7.0, 7.0, 7.0, 9.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0
            ],
            'y_s' : [18.0, 18.0, 18.0, 18.0, 18.0, 15.0, 15.0, 15.0, 15.0, 15.0, 17.0, 15.0, 12.0, 10.0, 
                    7.0, 5.0, 8.0, 13.0, 16.0, 10.0, 29.0, 26.0, 24.0, 22.0, 25.0, 28.0, 31.0, 29.0, 26.0, 23.0
            ]
        }
    )
    return (tampered_source_locs,)


@app.cell
def _(pd, raw_tampering_data, sensors, tampered_source_locs):
    def format_tampering_data(raw_tampering_data):
        data = pd.DataFrame(columns=sensors)
        for tamp in raw_tampering_data:
            for exp in range(1,len(raw_tampering_data[tamp])+1):
                df = raw_tampering_data[tamp][exp]
                new_row = {sensor : grab_sensor_response(df,sensor) for sensor in sensors}
                data.loc[len(data)] = new_row
        data = pd.concat([tampered_source_locs, data], axis=1)
        return data

    data_tamp = format_tampering_data(raw_tampering_data)
    data_tamp
    return (data_tamp,)


@app.cell
def _(data_tamp, viz_source_locs):
    viz_source_locs(data_tamp)
    return


@app.cell
def _(data, viz_source_locs):
    viz_source_locs(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## anomaly detection
    train isolation forest on normal data. get anomaly scores for normal and tampering data.
    """)
    return


@app.cell
def _(IsolationForest, pd, plt, sensors, sns, train_test_split):
    def train_test_anomaly_detection(data, data_tamp):
        # split normal data into train and test
        data_train, data_test = train_test_split(data, test_size=0.5, shuffle=True)

        iso_f = IsolationForest()
        iso_f.fit(data_train[sensors])

        ascore_normal = iso_f.decision_function(data_test[sensors])
        a_normal = iso_f.predict(data_test[sensors])
        ascore_tamp   = iso_f.decision_function(data_tamp[sensors])
        a_tamp = iso_f.predict(data_tamp[sensors])

        adata = pd.concat(
            [
                pd.DataFrame({"anomaly score": ascore_normal, 'sensor tampering': 'no', 'anomaly': a_normal}),
                pd.DataFrame({"anomaly score": ascore_tamp,   'sensor tampering': 'yes', 'anomaly': a_tamp})
            ]
        )

        sns.kdeplot(data=adata, x="anomaly score", hue="sensor tampering", common_norm=False)
        # plt.axvline(iso_f.offset_, color="gray", linestyle="--", zorder=1)
        plt.show()
        return adata
    return (train_test_anomaly_detection,)


@app.cell
def _(data, data_tamp, train_test_anomaly_detection):
    anomaly_data = train_test_anomaly_detection(data, data_tamp)
    return (anomaly_data,)


@app.cell
def _(anomaly_data):
    anomaly_data["anomaly"].value_counts()
    return


@app.cell
def _(anomaly_data):
    anomaly_data["anomaly"]
    return


@app.cell
def _(ConfusionMatrixDisplay, data_loo_c):
    ConfusionMatrixDisplay.from_predictions(
        data_loo_c["background"].values, data_loo_c["pred_safe"].values,
        text_kw={'fontsize': 22}
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Baseline Localization
    """)
    return


@app.cell
def _(np):
    m = np.array([366.832,325.406,236.634, 157.014,110.397, 103.229, 120.558, 50.095, 54.271, 38.758, 42.934, 34.583, 33.39, 35.775, 22.655, 16.692, 22.059, 23.848, 23.848, 25.637, 15.5, 13.115])
    return


@app.cell
def _(np):
    d = np.array([26,34,60,85,136,161,186,212,237,262,288,313,338,364,389,415,440,466,491,517,542,568])
    return


@app.cell
def _(mo):
    mo.md(r"""
    $d=\frac{\sqrt{-4a\tan ^2\left(\frac{\pi N}{S}\right)-4b\tan ^2\left(\frac{\pi N}{S}\right)+\sqrt{\left(4a\tan ^2\left(\frac{\pi N}{S}\right)+4b\tan ^2\left(\frac{\pi N}{S}\right)\right)^2+64a^2b^2\tan ^2\left(\frac{\pi N}{S}\right)}}}{4\sqrt{2}\tan \left(\frac{\pi N}{S}\right)},\:$

    $d=-\frac{\sqrt{-4a\tan ^2\left(\frac{\pi N}{S}\right)-4b\tan ^2\left(\frac{\pi N}{S}\right)+\sqrt{\left(4a\tan ^2\left(\frac{\pi N}{S}\right)+4b\tan ^2\left(\frac{\pi N}{S}\right)\right)^2+64a^2b^2\tan ^2\left(\frac{\pi N}{S}\right)}}}{4\sqrt{2}\tan \left(\frac{\pi N}{S}\right)},\:$

    $d=\frac{\sqrt{-4a\tan ^2\left(\frac{\pi N}{S}\right)-4b\tan ^2\left(\frac{\pi N}{S}\right)-\sqrt{\left(4a\tan ^2\left(\frac{\pi N}{S}\right)+4b\tan ^2\left(\frac{\pi N}{S}\right)\right)^2+64a^2b^2\tan ^2\left(\frac{\pi N}{S}\right)}}}{4\sqrt{2}\tan \left(\frac{\pi N}{S}\right)},\:$

    $d=-\frac{\sqrt{-4a\tan ^2\left(\frac{\pi N}{S}\right)-4b\tan ^2\left(\frac{\pi N}{S}\right)-\sqrt{\left(4a\tan ^2\left(\frac{\pi N}{S}\right)+4b\tan ^2\left(\frac{\pi N}{S}\right)\right)^2+64a^2b^2\tan ^2\left(\frac{\pi N}{S}\right)}}}{4\sqrt{2}\tan \left(\frac{\pi N}{S}\right)}$
    """)
    return


@app.cell
def _(np, sensor_locations):
    def compute_omega(x_s, y_s):
        """
        Compute solid angle factor Ω_n for all detectors.
    
        Returns:
        --------
        omega : array, shape (N,)
            Solid angle factors for each detector
        """
        detectors = np.array(sensor_locations)
        a=1.4/2.54
        b=2.8/2.54
        # Distance from source to each detector
        dx = x_s - detectors[:, 0]
        dy = y_s - detectors[:, 1]
        d = np.sqrt(dx**2 + dy**2)
    
        # Avoid division by zero
        d = np.maximum(d, 1e-10)
    
        # Compute alpha and beta
        alpha = a / (2 * d)
        beta = b / (2 * d)
    
        # Compute omega
        numerator = alpha * beta
        denominator = np.sqrt(1 + alpha**2 + beta**2)
    
        # Ensure argument is in valid range for arcsin
        arg = numerator / denominator
        arg = np.clip(arg, -1, 1)
    
        omega = 4 * np.arcsin(arg)
    
        return omega
    return (compute_omega,)


@app.cell
def _(compute_omega, least_squares, np):
    def localize_source(detector_positions, counts):
        """
        Localize a radiation source using detector array measurements.
    
        Model: N_n = (S/ε_ip) * (Ω_n/4π)
        where Ω_n = 4*arcsin(α_n*β_n / sqrt(1 + α_n² + β_n²))
        with α_n = a/(2*d_n), β_n = b/(2*d_n)
        and d_n = sqrt((x_s - x_n)² + (y_s - y_n)²)
    
        Parameters:
        -----------
        detector_positions : array-like, shape (N, 2)
            Positions of detectors [[x_1, y_1], [x_2, y_2], ...]
        counts : array-like, shape (N,)
            Measured counts at each detector [N_1, N_2, ..., N_N]
        a : float
            Detector geometry constant
        b : float
            Detector geometry constant
        initial_guess : array-like, shape (2,) or (3,), optional
            Initial guess [x_s, y_s] or [x_s, y_s, k]
            If None, uses weighted centroid of detectors
    
        Returns:
        --------
        result : dict
            Dictionary containing:
            - 'x_s': estimated x position
            - 'y_s': estimated y position
            - 'k': estimated scaling factor S/ε_ip
            - 'success': whether optimization succeeded
        """
        detectors = np.array(detector_positions)
        counts = np.array(counts)
        a=1.4/2.54
        b=2.8/2.54
        def residuals(params):
            """Compute residuals: r_n = N_n - k*R_n(x_s, y_s)"""
            x_s, y_s, k = params
        
            # Compute omega for the source position
            omega = compute_omega(x_s, y_s)
        
            # Compute response R = 4π/Ω
            R = omega / (4 * np.pi)
        
            return counts - k * R
    
        # Initial guess using weighted centroid
    
        weights = counts / np.sum(counts)
        x_init = np.sum(weights * detectors[:, 0])
        y_init = np.sum(weights * detectors[:, 1])
        k_init = np.max(counts)  # Rough estimate
        initial_guess = [x_init, y_init, k_init]

    
        # Solve least squares problem
        result = least_squares(residuals, x0=initial_guess, method='lm')
    
        return {
            'x_s': result.x[0],
            'y_s': result.x[1],
            'k': result.x[2],
            'success': result.success
        }

    return (localize_source,)


@app.cell
def _(data, np, sensors):
    measurements = np.array(data.loc[:,sensors])
    return (measurements,)


@app.cell
def _(np, sensor_to_loc):
    sensor_locations = np.array(list(sensor_to_loc.values()))
    return (sensor_locations,)


@app.cell
def _(localize_source, measurements, sensor_locations):
    results = [localize_source(sensor_locations, measurements[i]) for i in range(len(measurements))]
    return (results,)


@app.cell
def _(df_source_locs, results):
    bl_y_pred = [results[i]['y_s'] for i in range(len(results))]
    bl_x_pred = [results[i]['y_s'] for i in range(len(results))]
    bl_y_true = [df_source_locs['y_s'][i] for i in range(len(results))]
    bl_x_true = [df_source_locs['x_s'][i] for i in range(len(results))]
    return bl_x_pred, bl_x_true, bl_y_pred, bl_y_true


@app.cell
def _(df_source_locs, np, results):
    bl_x_errors = [np.abs(results[i]['x_s'] - df_source_locs['x_s'][i]) for i in range(len(results))]
    return (bl_x_errors,)


@app.cell
def _(df_source_locs, np, results):
    bl_y_errors = [np.abs(results[i]['y_s'] - df_source_locs['y_s'][i]) for i in range(len(results))]
    return (bl_y_errors,)


@app.cell
def _(bl_x_errors, bl_y_errors, math):
    bl_dist_error = [math.sqrt(pow(bl_x_errors[i],2) + pow(bl_y_errors[i],2)) for i in range(len(bl_x_errors))]
    return (bl_dist_error,)


@app.cell
def _(bl_dist_error, plt):
    plt.figure()
    plt.hist(bl_dist_error)
    plt.xlabel("error [in]")
    plt.ylabel("# experiments")
    plt.title("LOO-CV error")
    plt.show()
    return


@app.cell
def _(bl_x_errors, plt):
    plt.figure()
    plt.hist(bl_x_errors)
    plt.xlabel("error [in]")
    plt.ylabel("# experiments")
    plt.title("LOO-CV error")
    plt.show()
    return


@app.cell
def _(bl_y_errors, plt):
    plt.figure()
    plt.hist([error for error in bl_y_errors if error < 50])
    plt.xlabel("error [in]")
    plt.ylabel("# experiments")
    plt.title("LOO-CV error")
    plt.show()
    return


@app.cell
def _(
    bl_x_errors,
    bl_x_pred,
    bl_x_true,
    bl_y_errors,
    bl_y_pred,
    bl_y_true,
    box_dims,
    np,
    plt,
):
    def _():
        f, (ax1, ax2) = plt.subplots(1, 2)
        #plt.subplots_adjust(wspace=0.3)
        #for ax in [ax1, ax2]:
        #    ax.set_aspect('equal', 'box')

        ax1.set_xlim(0, 32)
        ax1.set_ylim(0, 80)
        ax2.set_xlim(0, 32)
        ax2.set_ylim(0, 80)

        ax1.set_xlabel("x$_s$ [in]")
        ax2.set_xlabel("y$_s$ [in]")
        ax1.set_ylabel("predicted x$_s$ [in]")
        ax2.set_ylabel("predicted y$_s$ [in]")

        ax1.plot([0, box_dims[0]], [0, box_dims[0]], color="black", linestyle="--")
        ax2.plot([0, box_dims[1]], [0, box_dims[1]], color="black", linestyle="--")

        ax1.scatter(bl_x_true, bl_x_pred, clip_on=False)
        ax2.scatter(bl_y_true, bl_y_pred, clip_on=False)

        bl_x_err = np.mean(bl_x_errors)
        bl_y_err = np.mean(bl_y_errors)

        return plt.show()


    _()
    return


@app.cell
def _(compute_omega, df_source_locs, np):
    def make_unshielded_responses(x_true, y_true, k_true = 807e+3):

        omega = compute_omega(x_true, y_true)
        R = omega/(4 * np.pi)
        ideal_counts = k_true*R
        measured_counts = np.random.poisson(ideal_counts)
        return measured_counts
    print(make_unshielded_responses(df_source_locs['x_s'][0], df_source_locs['y_s'][0]))
    return (make_unshielded_responses,)


@app.cell
def _(
    df_source_locs,
    localize_source,
    make_unshielded_responses,
    sensor_locations,
):
    unshielded_results = [localize_source(sensor_locations, make_unshielded_responses(df_source_locs['x_s'][i], df_source_locs['y_s'][i])) for i in range(len(df_source_locs))]
    return (unshielded_results,)


@app.cell
def _(df_source_locs, results):
    US_bl_y_pred = [results[i]['y_s'] for i in range(len(results))]
    US_bl_x_pred = [results[i]['y_s'] for i in range(len(results))]
    US_bl_y_true = [df_source_locs['y_s'][i] for i in range(len(results))]
    US_bl_x_true = [df_source_locs['x_s'][i] for i in range(len(results))]
    return US_bl_x_pred, US_bl_x_true, US_bl_y_pred, US_bl_y_true


@app.cell
def _(df_source_locs, math, np, unshielded_results):
    US_bl_x_errors = [np.abs(unshielded_results[i]['x_s'] - df_source_locs['x_s'][i]) for i in range(len(unshielded_results))]
    US_bl_y_errors = [np.abs(unshielded_results[i]['y_s'] - df_source_locs['y_s'][i]) for i in range(len(unshielded_results))]
    US_bl_dist_error = [math.sqrt(pow(US_bl_x_errors[i],2) + pow(US_bl_y_errors[i],2)) for i in range(len(US_bl_x_errors))]
    return US_bl_dist_error, US_bl_x_errors, US_bl_y_errors


@app.cell
def _(US_bl_dist_error, plt):
    plt.figure()
    plt.hist(US_bl_dist_error)
    plt.xlabel("error [in]")
    plt.ylabel("# experiments")
    plt.title("LOO-CV error")
    plt.show()
    return


@app.cell
def _(
    US_bl_x_errors,
    US_bl_x_pred,
    US_bl_x_true,
    US_bl_y_errors,
    US_bl_y_pred,
    US_bl_y_true,
    box_dims,
    np,
    plt,
):
    def _():
        f, (ax1, ax2) = plt.subplots(1, 2)
        #plt.subplots_adjust(wspace=0.3)
        #for ax in [ax1, ax2]:
        #    ax.set_aspect('equal', 'box')

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

        ax1.scatter(US_bl_x_true, US_bl_x_pred, clip_on=False)
        ax2.scatter(US_bl_y_true, US_bl_y_pred, clip_on=False)

        bl_x_err = np.mean(US_bl_x_errors)
        bl_y_err = np.mean(US_bl_y_errors)
        ax1.legend(title=f"error = {bl_x_err:.2f} in")
        ax2.legend(title=f"error = {bl_y_err:.2f} in")
        return plt.show()


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
