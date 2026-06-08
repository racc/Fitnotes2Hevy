import pandas as pd
import numpy as np
import json
from config import (
    MAP_PATH, FIT_NOTES_PATH, STRONG_PATH, RESULT_PATH,
    DEFAULT_TRAINING_TIME, DEFAULT_WORKOUT_NAME,
    DEFAULT_DURATION, DEFAULT_REST_TIME
)

MAX_ROWS = 1999

STRONG_COLUMNS = [
    "Date",
    "Workout Name",
    "Exercise Name",
    "Set Order",
    "Weight",
    "Weight Unit",
    "Reps",
    "RPE",
    "Distance",
    "Distance Unit",
    "Seconds",
    "Notes",
    "Workout Notes",
    "Workout Duration",
]

if __name__ == "__main__":

    # Load mapping FitNotes → Strong exercise names
    with open(MAP_PATH, 'r', encoding='utf-8') as f:
        map_fn2strong = json.load(f)

    # Load FitNotes CSV
    df_fn = pd.read_csv(FIT_NOTES_PATH)

    # Sort chronologically
    df = df_fn.sort_values("Date").copy()

    # Add workout metadata
    df["Date"] = df["Date"] + " " + DEFAULT_TRAINING_TIME
    df["Workout Name"] = DEFAULT_WORKOUT_NAME
    df["Workout Notes"] = np.nan
    df["Workout Duration"] = DEFAULT_DURATION

    # Map exercise names
    df["Exercise Name"] = df["Exercise"].map(map_fn2strong).fillna("Other")

    # Base set fields
    df["Weight"] = df["Weight"]
    df["Weight Unit"] = "kg"
    df["Reps"] = df["Reps"].astype(float)
    df["Distance"] = np.nan
    df["Distance Unit"] = np.nan
    df["Seconds"] = np.nan
    df["Notes"] = ""
    df["RPE"] = np.nan

    # Compute Set Order for training rows
    df["Set Order"] = (
        df.groupby(["Date", "Exercise Name"])
        .cumcount() + 1
    ).astype("Int64")

    # -------------------------------
    # REMOVE REST ROWS
    # -------------------------------
    # Rest rows are the ones created artificially earlier:
    #   - Exercise Name = "Other"
    #   - Seconds = DEFAULT_REST_TIME
    #   - Weight/Reps are NaN
    # We remove ONLY those.
    df = df[~(
        (df["Exercise Name"] == "Other") &
        (df["Seconds"] == DEFAULT_REST_TIME)
    )]

    # Ensure correct column order
    df_final = df[STRONG_COLUMNS].reset_index(drop=True)
    df_final = df_final[df_final["Exercise Name"] != "Other"]
    
    # Split into multiple files
    num_chunks = int(np.ceil(len(df_final) / MAX_ROWS))

    for i in range(num_chunks):
        start = i * MAX_ROWS
        end = start + MAX_ROWS
        chunk = df_final.iloc[start:end]

        out_path = RESULT_PATH.replace(".csv", f"_{i+1}.csv")
        chunk.to_csv(out_path, index=False, sep=";")

    print(f"Wrote {num_chunks} files, each ≤ {MAX_ROWS} rows.")
