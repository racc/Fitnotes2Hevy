import pandas as pd
import numpy as np
import json
from config import MAP_PATH, FIT_NOTES_PATH, STRONG_PATH, RESULT_PATH, DEFAULT_TRAINING_TIME, DEFAULT_WORKOUT_NAME, DEFAULT_DURATION, DEFAULT_REST_TIME 
MAX_ROWS = 1999

if __name__ == "__main__":

    with open(MAP_PATH, 'r', encoding='utf-8') as f:
        map_fn2strong = json.load(f)

    df_fn = pd.read_csv(FIT_NOTES_PATH)
    # df_strong = pd.read_csv(STRONG_PATH, sep = ";")

    df_fn_mod = df_fn.copy()
    df_fn_mod = df_fn_mod.sort_values("Date")
    df_fn_mod["Workout #"] = df_fn_mod["Date"].rank(method="dense").astype(int)

    df_fn_mod['Date'] = df_fn_mod['Date'] + ' ' + DEFAULT_TRAINING_TIME

    df_fn_mod['Workout Name'] = DEFAULT_WORKOUT_NAME

    df_fn_mod['Duration (sec)'] = DEFAULT_DURATION

    df_fn_mod["Exercise Name"] = df_fn_mod["Exercise"].map(map_fn2strong)
    df_fn_mod["Exercise Name"] = df_fn_mod["Exercise Name"].fillna('Other')

    df_fn_mod['Set Order'] = df_fn_mod.groupby(['Date', 'Exercise Name']).cumcount() + 1

    df_fn_mod['Weight (kg)'] = df_fn_mod['Weight']

    df_fn_mod['Reps'] = df_fn_mod['Reps'].astype(float)

    df_fn_mod['RPE'] = np.nan

    df_fn_mod['Distance (meters)'] = np.nan

    df_fn_mod['Seconds'] = np.nan

    df_fn_mod['Notes'] = ""

    df_fn_mod['Workout Notes'] = np.nan

    df_fn_mod_rest = df_fn_mod.copy()
    df_fn_mod['training'] = True
    df_fn_mod_rest['training'] = False
    df_fn_mod = pd.concat([df_fn_mod, df_fn_mod_rest])
    df_fn_mod["index"] = df_fn_mod.index
    df_fn_mod = df_fn_mod.sort_values(["index", "training"], ascending=[True, False])
    df_fn_mod.loc[~df_fn_mod.training, ["Set Order", "Weight (kg)", "Reps", "Seconds", "Notes"]] = [
        np.nan, np.nan, np.nan, DEFAULT_REST_TIME, np.nan
    ]

    df_final = df_fn_mod.reset_index(drop=True)

    num_chunks = int(np.ceil(len(df_final) / MAX_ROWS))

    for i in range(num_chunks):
        start = i * MAX_ROWS
        end = start + MAX_ROWS
        chunk = df_final.iloc[start:end]

        out_path = RESULT_PATH.replace(".csv", f"_{i+1}.csv")
        chunk.to_csv(out_path, index=False)

    print(f"Wrote {num_chunks} files, each ≤ {MAX_ROWS} rows.")