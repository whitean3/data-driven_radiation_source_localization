import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import csv
    import os
    import pandas as pd
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    return csv, mo, np, os, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ::icon-park:data:: data read in sensor network response data

    * each data set is the sensor network response to a source in a particular location.
    * `SN` gives sensor network ID
    * `ICR` gives count rate
    * the list of source locations are below.
    """
    )
    return


@app.cell
def _(box_dims, np):
    # sensor locs. orgigin is top left here.
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
def _():
    n_sensors = 8
    return (n_sensors,)


@app.cell
def _(box_dims, pd):
    df_source_locs = pd.DataFrame(
        {
            'x_s': [28.0, 38.0, 4.0, 2.0, 10.0, 41.0,32.0, 5.0, 40.0, 24.0, 16.0, 7.0, 18.0, 35.0, 33.0, 30.0, 23.0, 26.0, 19.0, 12.0, 0.0, 9.0, 21.0, 37.0, 14.0],
            'y_s': [14.0, 30.0, 28.0, 21.0, 16.0, 10.0, 4.0, 6.0, 18.0, 32.0, 19.0, 22.0, 26.0, 12.0,  8.0, 6.0, 11.0,29.0, 15.0, 3.0, 0.0, 1.0, 23.0, 23.0, 32.0]
        }
    ) # note this is with origin in top left
    df_source_locs["y_s"] = box_dims[1] - df_source_locs["y_s"]
    df_source_locs
    return (df_source_locs,)


@app.cell
def _(csv, n_sensors, pd):
    def read_csv(file_path):
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
    return (read_csv,)


@app.cell
def _():
    folder_path = "forward model data/"
    return (folder_path,)


@app.cell
def _():
    n_expts = 25
    return (n_expts,)


@app.cell
def _(folder_path, n_expts, n_sensors, os, read_csv):
    def read_all_data(n_expts):
        dataframes = {}

        for exp in range(n_expts):
            filename = f"pos_{exp}.csv"
            file_path = os.path.join(folder_path, filename)
            print(f"\nReading {filename}...")
            # Use the filename (or file_path) as the key
            dataframes[exp] = read_csv(file_path)

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
    mo.md(r"""## assemble into an easier format for ML.""")
    return


@app.cell
def _(dataframes):
    # list of sensors in the network
    sensors = dataframes[0]["SN"].unique()
    sensors
    return (sensors,)


@app.function
# from a sensor network response data frame, extract the count rate of a particular sensor.
def grab_sensor_response(df, sensor):
    return float(df.loc[df["SN"] == sensor, "ICR"].item())


@app.cell
def _(dataframes, sensors):
    grab_sensor_response(dataframes[0], sensors[2])
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


@app.cell
def _(mo):
    mo.md(r"""## viz""")
    return


@app.cell
def _(box_dims, data, plt):
    def viz_source_locs(data):
        plt.figure()
        plt.scatter(data["x_s"], data["y_s"])
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])
        plt.show()

    viz_source_locs(data)
    return


@app.cell
def _():
    return


@app.cell
def _(box_dims, data, plt, sensor_to_loc, sensors):
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
            vmax=max_response
        )

        plt.colorbar(label="count rate [CPS]", extend="max")
    
        # plot source location
        plt.scatter(data.loc[exp, "x_s"], data.loc[exp, "y_s"], marker="+", s=65, color="red", label="source location")

        # TODO draw obstacles
    
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("x [in]")
        plt.ylabel("y [in]")
        plt.xlim(0, box_dims[0])
        plt.ylim(0, box_dims[1])
        plt.show()
    
    viz_sensor_readout(data, 4)
    return


@app.cell
def _(data, sensors, sns):
    sns.histplot(data[sensors])
    return


@app.cell
def _(data, plt, sensors, sns):
    g = sns.pairplot(data[sensors])
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
