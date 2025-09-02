import pandas as pd

def read_temperatures(txt_path: str) -> pd.DataFrame:
    """Read temperatures: col0=id, col1..M=temperatures by time step."""
    import pandas as pd, numpy as np
    try:
        df = pd.read_csv(txt_path, sep=None, engine="python", header=None, comment="#")
    except Exception:
        df = pd.read_csv(txt_path, sep=r'\s+', header=None, comment="#")
    if not np.issubdtype(df.iloc[0, 0].dtype, np.number):
        try:
            df = pd.read_csv(txt_path, sep=None, engine="python", header=0, comment="#")
        except Exception:
            df = pd.read_csv(txt_path, sep=r'\s+', header=0, comment="#")
    df.columns = ["id"] + [f"t{k}" for k in range(df.shape[1]-1)]
    df["id"] = df["id"].astype(int)
    return df

def read_positions(csv_path: str, has_header: bool = False) -> pd.DataFrame:
    """Read positions: col0=id, col1..3=x,y,z. Returns ['id','x','y','z']."""
    header = 0 if has_header else None
    pos = pd.read_csv(csv_path, sep=None, engine="python", header=header, comment="#")
    pos = pos.iloc[:, :4].copy()
    pos.columns = ["id", "x", "y", "z"]
    pos["id"] = pos["id"].astype(int)
    return pos

# --------------- Load & merge ---------------
def load_df(txt_path: str, csv_path: str, has_header: bool = False) -> pd.DataFrame:
    """Load wide-format df with columns [x,y,z,t0,t1,...]; index can be 'id'."""
    pos_df  = read_positions(csv_path, has_header=has_header).set_index("id")
    temp_df = read_temperatures(txt_path).set_index("id")
    
    df = pos_df.join(temp_df, how="inner")
    if df.empty:
        raise ValueError("No matching ids between positions and temperatures.")
    if "id" in df.columns:
        df = df.set_index("id")
    return df
