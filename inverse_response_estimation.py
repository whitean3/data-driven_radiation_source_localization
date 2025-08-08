import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import csv
    import os
    import pandas as pd
    import marimo as mo

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    import matplotlib.font_manager as fm
    import matplotlib as mpl
    import seaborn as sns
    plt.style.use('rose-pine-dawn.mplstyle')
    fm.fontManager.addfont("/Users/cokes/Library/Fonts/SourceCodePro-Regular.ttf")
    plt.rcParams["font.family"] = "Source Code Pro"

    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.model_selection import LeaveOneOut
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    return (
        Ellipse,
        ExtraTreesRegressor,
        LeaveOneOut,
        csv,
        mo,
        np,
        os,
        pd,
        plt,
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
        'sensor': theme_colors[2]
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
    mo.md(
        r"""
    ## location of radioactive source placed in the environment

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
            'x_s': [28.0, 38.0, 4.0, 2.0, 10.0, 41.0,32.0, 5.0, 40.0, 24.0, 16.0, 7.0, 18.0, 35.0, 33.0, 30.0, 23.0, 26.0, 19.0, 12.0, 0.0, 9.0, 21.0, 37.0, 14.0, 2, 5,  7,  7, 10, 10, 12, 16, 40, 41, 31, 31, 34, 41, 41, 41, 36, 28, 17, 12,  0,  7,  7,  7, 10, 13.0, 38.0, 41.0, 19.0,  7.0, 17.0, 39.0, 36.0,  9.0, 25.0, 32.0, 16.0,  6.0, 42.0,  0.0,  2.0, 26.0, 20.0, 28.0, 30.0,  3.0, 29.0, 14.0, 23.0, 22.0],
            'y_s': [14.0, 30.0, 28.0, 21.0, 16.0, 10.0, 4.0, 6.0, 18.0, 32.0, 19.0, 22.0, 26.0, 12.0,  8.0, 6.0, 11.0,29.0, 15.0, 3.0, 0.0, 1.0, 23.0, 23.0, 32.0,  0, 0,  0,  4,  0,  4,  4,  4,  0,  4,  4,  0,  0, 15, 20, 28, 32, 32, 32, 32, 32, 32, 29, 25, 25, 20.0, 10.0, 24.0, 11.0, 15.0, 30.0, 14.0, 32.0, 0.0 , 32.0, 9.0 , 13.0, 6.0 , 27.0, 29.0, 23.0, 10.0, 2.0 , 18.0, 27.0, 16.0, 1.0 , 17.0, 19.0, 8.0]
        }
    ) # note this is with origin in top left
    df_source_locs["y_s"] = box_dims[1] - df_source_locs["y_s"] # accounts for coordinate system
    df_source_locs
    return (df_source_locs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## read in detector outputs""")
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
    mo.md(
        r"""
    ## re-work data into an ML-friendly format

    ::lucide:lightbulb:: source locations paired with sensor network response vectors

    first, get list of sensors in the network
    """
    )
    return


@app.cell
def _(dataframes):
    sensors = dataframes[0]["SN"].unique()
    sensors
    return (sensors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""next, from a sensor network response data frame, extract the count rate of a particular sensor.""")
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
    mo.md(
        r"""
    🎯 goal: 

    * each row is an experiment where we place the source at a location and observe the sensor network response
    * the row lists the source location paired with the sensor response vector.
    """
    )
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
    return data, make_data_nice


@app.cell
def _(dataframes, sensors):
    assert grab_sensor_response(dataframes[0], sensors[2]) == 31.6
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""second, visualize the sensor readout and source location for a single experiment.""")
    return


@app.cell
def _(
    box_dims,
    data,
    plt,
    sensor_colormap,
    sensor_to_loc,
    sensors,
    thing_to_color,
):
    def viz_sensor_readout(data, exp):
        max_response = 75.0 # data.loc[exp, sensors].max()

        plt.figure()

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
            s=65, color=thing_to_color["true source loc"], label="source location"
        )

        # TODO draw obstacles

        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])
        plt.title(f"experiment {exp}")
        plt.legend()
        plt.show()

    viz_sensor_readout(data, 19)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""the distribution of the detector responses over all experiments and correlations between them. use a log scale.""")
    return


@app.cell
def _(data, plt, sensors, sns):
    sns.swarmplot(data[sensors], size=2)
    plt.axhline(13.2, linestyle="--", color="gray") # background
    plt.yscale("log")
    plt.xlabel("sensor")
    plt.ylabel("response [CPS]")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ::icon-park:new-computer:: ML

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
def _(run_loo_cv):
    #run_loo_cv = mo.ui.checkbox(label="run LOO CV?")
    run_loo_cv
    return


@app.cell
def _(data, do_loo_cv, run_loo_cv):
    if run_loo_cv.value:
        data_loo = do_loo_cv(data)
    data_loo
    return (data_loo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""analyze error := norm of true source location vector minus predicted source location vector.""")
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
    mo.md(r"""parity plot over cross-validation procedure.""")
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
def _(box_dims, data_loo, plt, sensor_to_loc, sensors, thing_to_color):
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
def _(data, n_sensors, np, plt, sensors, train_tree_ensemble):
    _tree_ensemble = train_tree_ensemble(data)

    plt.figure()
    plt.xlabel("sensor")
    plt.ylabel("feature importance")
    plt.xticks(np.arange(n_sensors), sensors)
    plt.bar(np.arange(n_sensors), _tree_ensemble.feature_importances_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## learning curve""")
    return


@app.cell
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

    learning_curve = pd.DataFrame(
        {"# data": nb_datas, "loo error [in]": loo_errors_mu, "loo error std [in]": loo_errors_std}
    )
    return (learning_curve,)


@app.cell
def _(learning_curve, plt):
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
    mo.md(r"""## live demo-ing""")
    return


@app.cell
def _(data, train_tree_ensemble):
    tree_ensemble = train_tree_ensemble(data)
    return (tree_ensemble,)


@app.cell
def _(dataframes):
    demo_sensors = dataframes[0]["SN"].unique()
    demo_sensors
    return (demo_sensors,)


@app.cell
def _(demo):
    demo_response = demo["ICR"]
    demo_response
    return (demo_response,)


@app.cell
def _(dataframes, demo_response, demo_sensors, pd):
    def prep_demo_data(demo_sensors, demo_response):
        demo_input = pd.DataFrame(columns=demo_sensors) 
        new_row = {sensor : grab_sensor_response(dataframes[1], sensor) for sensor in demo_sensors}
        demo_input.loc[1] = new_row
        return demo_input


    demo_input = prep_demo_data(demo_sensors, demo_response)
    demo_input
    return (demo_input,)


@app.cell
def _():
    return


@app.cell
def _():
    #demo_pred = rf.predict(demo_input)
    #demo_pred
    return


@app.cell
def _(demo_pred):
    33.0-demo_pred[0][1]
    return


@app.cell
def _(demo_input, tree_ensemble):
    demo_input.iloc[0, :] = [
        29.812,
    23.847,
    51.884,
    20.271,
    32.2,
    31.005,
    19.077,
    78.744,


    ]
    demo_pred = tree_ensemble.predict(demo_input)
    demo_pred
    return (demo_pred,)


@app.cell
def _():
    29.812
    23.847
    51.884
    20.271
    32.2
    31.005
    19.077
    78.744

    return


@app.cell
def _(mo):
    mo.md(r"""# background data""")
    return


@app.cell
def _(mo):
    mo.md(r"""# sensor tampering data""")
    return


@app.cell
def _():
    tampered_dets_SN = [16512,16513,16519]
    return


@app.cell
def _(folder_path, n_sensors, os, read_detector_outputs):
    def read_tampering_data(n_expts, dets):
        tamp_data = {
            "12" : {},
            "13" : {},
            "19" : {}
        }
        for det in dets:
            for exp in range(1,n_expts+1):

                filename = f"det{det}_tamp{exp}.csv"
                file_path = os.path.join(folder_path, filename)
                print(f"\nReading {filename}...")
                # Use the filename (or file_path) as the key
                tamp_data[f"{det}"][exp] = read_detector_outputs(file_path)

            # unique sensors
            assert tamp_data[f"{det}"][exp]["SN"].nunique() == n_sensors
        return tamp_data

    tampering_data = read_tampering_data(10, [12,13,19])
    tampering_data
    return (tampering_data,)


@app.cell
def _(make_data_nice, tampering_data):
    make_data_nice(tampering_data)
    return


@app.cell
def _(tampering_data):
    tampering_data
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
