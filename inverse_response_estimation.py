import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import numpy as np
    import os
    import pandas as pd
    return csv, os, pd


@app.cell
def _(csv, pd):
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
            return df
    return (read_csv,)


@app.cell
def _():
    folder_path = "./inverse_response_estimation/forward model data/"
    return (folder_path,)


@app.cell
def _():
    dataframes = {} 
    return (dataframes,)


@app.cell
def _(dataframes, folder_path, os, read_csv):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f"\nReading {filename}:")
            # Use the filename (or file_path) as the key
            dataframes[filename] = read_csv(file_path)
    return


@app.cell
def _(dataframes):
    dataframes['pos_0.csv']
    return


@app.cell
def _(dataframes):
    for df in dataframes:
        start = len(dataframes[df])-8
        dataframes[df] = dataframes[df][start:]
    return


@app.cell
def _():
    x_s = [28.0, 38.0, 4.0, 2.0, 10.0, 41.0,32.0, 5.0, 40.0, 24.0, 16.0, 7.0, 18.0, 35.0, 33.0, 30.0, 23.0, 26.0, 19.0, 12.0, 0.0, 9.0, 21.0, 37.0, 14.0]
    y_s = [14.0, 30.0, 28.0, 21.0, 16.0, 10.0, 4.0, 6.0, 18.0, 32.0, 19.0, 22.0, 26.0, 12.0,  8.0, 6.0, 11.0,29.0, 15.0, 3.0, 0.0, 1.0, 23.0, 23.0, 32.0]
    return


@app.cell
def _(dataframes):
    def filter_by_SN(SN):
        values = []
        for df in dataframes:
            # Filter rows where filter_column equals filter_value
            filtered_df = df[df["SN"] == SN]
            # Add the values from target_column to our list
            values.extend(filtered_df["ICR"].tolist())
        return values
    return


if __name__ == "__main__":
    app.run()
