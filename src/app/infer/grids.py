from __future__ import annotations
import numpy as np
import torch

def predict_on_grid(model, x_left: float, x_right: float, z_bottom: float, z_top: float,
                    t_values: list[float], nx: int = 101, nz: int = 101,
                    device: torch.device | None = None):
    """Evaluate T(x,z,t) on a regular (x,z) grid for selected times.

    Args:
        model: Trained PINNHeat model (accepts physical X=[x,z,t]).
        x_left, x_right, z_bottom, z_top: Domain bounds (meters).
        t_values: List of times (seconds).
        nx, nz: Grid size along x and z.
        device: Torch device.

    Returns:
        (T_maps, xs, zs) where:
          - T_maps: dict time -> ndarray [nz, nx] (Kelvin)
          - xs: 1D array of x coordinates (m)
          - zs: 1D array of z coordinates (m)
    """
    device = device or next(model.parameters()).device
    xs = np.linspace(x_left, x_right, nx, dtype=np.float32)
    zs = np.linspace(z_bottom, z_top, nz, dtype=np.float32)
    Xg, Zg = np.meshgrid(xs, zs)

    T_maps = {}
    model.eval()
    with torch.no_grad():
        for tt in t_values:
            tgrid = np.full_like(Xg, tt, dtype=np.float32)
            X_in = np.stack([Xg.ravel(), Zg.ravel(), tgrid.ravel()], axis=1)
            X_in = torch.tensor(X_in, dtype=torch.float32, device=device)
            u = model(X_in)                          # state
            T = model.nondim.state_to_T(u).reshape(nz, nx).cpu().numpy()
            T_maps[float(tt)] = T
    return T_maps, xs, zs
