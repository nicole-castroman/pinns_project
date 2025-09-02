import os
import json
import yaml
import torch
import numpy as np
from .factory import build_model
from .etl.samplers import prepare_tensors
from .infer.grids import predict_on_grid
from .viz.plots import plot_temperature_maps

def run_inference(config_path: str, checkpoint_path: str, out_dir: str = "models/infer",
                  nx: int = 81, nz: int = 81, times: list[float] | None = None,
                  save_plots: bool = True):
    """Load a trained model and produce T(x,z,t) maps on a regular grid.

    Args:
        config_path: YAML config used to build the model (variant/domain/physics).
        checkpoint_path: Path to a saved model .pt (state_dict).
        out_dir: Output folder to store npz and figures.
        nx, nz: Grid resolution along x and z.
        times: Optional list of times (seconds). If None, use [0, mid, t_max].
        save_plots: Whether to save PNG heatmaps.

    Returns:
        dict mapping time -> ndarray [nz, nx] of temperatures in Kelvin.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model, (phys, dom, timec, trainc), device = build_model(cfg)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Default sample times
    if times is None:
        times = [timec.t_min, 0.5*(timec.t_min + timec.t_max), timec.t_max]

    # Predict maps (Kelvin)
    T_maps, xs, zs = predict_on_grid(
        model=model,
        x_left=dom.x_left, x_right=dom.x_right,
        z_bottom=dom.z_bottom, z_top=dom.z_top,
        t_values=times, nx=nx, nz=nz, device=device
    )

    # Save npz
    npz_path = os.path.join(out_dir, "predictions.npz")
    np.savez_compressed(npz_path, xs=xs, zs=zs, **{f"t_{t:.6f}": T_maps[float(t)] for t in times})

    # Save simple plots
    if save_plots:
        fig_path = os.path.join(out_dir, "maps.png")
        plot_temperature_maps(T_maps, xs, zs, path=fig_path, title="PINN T(x,z,t) predictions")

    # Dump metadata
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({"config": config_path, "checkpoint": checkpoint_path, "times": times,
                   "nx": nx, "nz": nz, "out": npz_path}, f, indent=2)

    return T_maps
