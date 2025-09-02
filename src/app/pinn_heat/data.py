import numpy as np
import torch
import pandas as pd

def extract_time_columns(df: pd.DataFrame, dt: float, N_use: int):
    """Pick first N_use time columns (t0, t1, ...) and build time values array."""
    tcols = [c for c in df.columns if c.lower().startswith("t")]
    def suf(c): 
        import re
        d = re.findall(r"\d+", c)
        return int(d[0]) if d else 10**9
    tcols = sorted(tcols, key=suf)[:N_use]
    tvals = np.array([i*dt for i in range(len(tcols))], dtype=np.float32)
    return tcols, tvals

def build_data_from_df(df: pd.DataFrame, N_nodes: int, N_use_times: int, dt: float, device):
    """Build supervised tensors (X_data=[x,z,t], T_data=[T]) from a wide df."""
    tcols, tvals = extract_time_columns(df, dt, N_use_times)
    ids = np.random.choice(df.index.values, size=min(N_nodes, len(df)), replace=False)
    sub = df.loc[ids, ['x','z'] + tcols].copy()

    xz = sub[['x','z']].to_numpy(dtype=np.float32)
    Nt = len(tcols)
    xz_rep = np.repeat(xz, Nt, axis=0)
    t_tile = np.tile(tvals.reshape(-1,1), (xz.shape[0],1)).reshape(-1,1)

    T_vals = sub[tcols].to_numpy(dtype=np.float32).reshape(-1,1)

    X_data = np.concatenate([xz_rep, t_tile], axis=1).astype(np.float32)
    return (torch.tensor(X_data, dtype=torch.float32, device=device),
            torch.tensor(T_vals, dtype=torch.float32, device=device))

def sample_interior_collocation(Nf, t_max, x_left, x_right, z_bottom, z_top, device):
    """Uniform random interior collocation points (x,z,t)."""
    x = x_left + np.random.rand(Nf,1)*(x_right-x_left)
    z = z_bottom + np.random.rand(Nf,1)*(z_top-z_bottom)
    t = np.random.rand(Nf,1)*t_max
    Xf = np.concatenate([x,z,t], axis=1).astype(np.float32)
    return torch.tensor(Xf, dtype=torch.float32, device=device)

def sample_bc_collocation(Nb, t_max, x_left, x_right, z_bottom, z_top, device):
    """Boundary collocation points dict, shapes [Nb,3] with keys:
       'z0_flux', 'zL_conv', 'x0_conv'(left), 'xL_conv'(right)."""
    t = np.random.rand(Nb,1)*t_max

    x_z0 = x_left + np.random.rand(Nb,1)*(x_right-x_left)
    z_z0 = np.full((Nb,1), z_bottom, np.float32)
    X_z0 = np.concatenate([x_z0, z_z0, t], axis=1).astype(np.float32)

    x_zL = x_left + np.random.rand(Nb,1)*(x_right-x_left)
    z_zL = np.full((Nb,1), z_top, np.float32)
    X_zL = np.concatenate([x_zL, z_zL, t], axis=1).astype(np.float32)

    x_xL = np.full((Nb,1), x_left, np.float32)
    z_xL = z_bottom + np.random.rand(Nb,1)*(z_top-z_bottom)
    X_xL = np.concatenate([x_xL, z_xL, t], axis=1).astype(np.float32)

    x_xR = np.full((Nb,1), x_right, np.float32)
    z_xR = z_bottom + np.random.rand(Nb,1)*(z_top-z_bottom)
    X_xR = np.concatenate([x_xR, z_xR, t], axis=1).astype(np.float32)

    td = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return {"z0_flux": td(X_z0), "zL_conv": td(X_zL),
            "x0_conv": td(X_xL), "xL_conv": td(X_xR)}

def sample_ic_collocation(Nic, x_left, x_right, z_bottom, z_top, device):
    """Collocation points at t=0 for IC."""
    x = x_left + np.random.rand(Nic,1)*(x_right-x_left)
    z = z_bottom + np.random.rand(Nic,1)*(z_top-z_bottom)
    t = np.zeros((Nic,1), dtype=np.float32)
    Xic = np.concatenate([x,z,t], axis=1).astype(np.float32)
    return torch.tensor(Xic, dtype=torch.float32, device=device)
