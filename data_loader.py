import pandas as pd

def load_data(filepath):

    df = pd.read_csv(filepath, sep="\t")

    # bersihkan nama kolom
    df.columns = [c.replace("<","").replace(">","").capitalize() for c in df.columns]

    # buat datetime
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

    df.set_index("datetime", inplace=True)

    numeric_cols = ["Open","High","Low","Close"]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    print("Data loaded:", len(df))

    return df