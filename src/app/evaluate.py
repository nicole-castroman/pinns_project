import os
import json
import yaml
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .factory import build_model
from .etl.loaders import load_df
from .pinn_heat.data import build_data_from_df  # reuse core builder

def evaluate_model(config_path: str, checkpoint_path: str, csv_override: str | None = None,
                   N_nodes_eval: int | None = None, out_dir: str = "models/eval"):
    """Evaluate a trained model on random (x,z,t) data points from df.

    Args:
        config_path: YAML used to build the model.
        checkpoint_path: Path to model state_dict (.pt).
        csv_override: Optional path to a CSV for evaluation (else use config's).
        N_nodes_eval: Optional number of spatial nodes for evaluation sampling.
        out_dir: Folder to save metrics and samples.

    Returns:
        dict with MAE, RMSE and counts.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Build model skeleton
    model, (phys, dom, timec, trainc), device = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load df
    if csv_override:
        # Si se proporciona csv_override, asumir que es el archivo de temperaturas
        # y buscar el archivo de mesh en la misma carpeta
        import os
        temp_path = csv_override
        mesh_path = os.path.join(os.path.dirname(csv_override), "mesh.csv")
        has_header = cfg["data"].get("has_header", False)
    else:
        temp_path = cfg["data"]["temperature_path"]
        mesh_path = cfg["data"]["mesh_path"]
        has_header = cfg["data"].get("has_header", False)
    
    df = load_df(temp_path, mesh_path, has_header=has_header)

    # Build evaluation data (keep same N_time_use, dt; change N_nodes if requested)
    N_nodes = N_nodes_eval or cfg["data"]["N_data_nodes"]
    X_eval, T_eval = build_data_from_df(df, N_nodes, timec.N_time_use, timec.dt, device=device)

    # Predict at evaluation points
    with torch.no_grad():
        u_pred = model(X_eval)                   # state (T or θ)
        T_pred = model.nondim.state_to_T(u_pred) # always Kelvin

    y_true = T_eval.cpu().numpy().ravel()
    y_pred = T_pred.cpu().numpy().ravel()

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {"mae": mae, "rmse": rmse, "N": len(y_true)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Optional small sample dump
    k = min(5000, len(y_true))
    idx = np.random.choice(len(y_true), size=k, replace=False)
    np.savez_compressed(os.path.join(out_dir, "sample_preds.npz"),
                        X_eval=X_eval.cpu().numpy()[idx],
                        T_true=y_true[idx], T_pred=y_pred[idx])

    print(f"[EVAL] MAE={mae:.4e} | RMSE={rmse:.4e} | N={len(y_true)}")
    return metrics
