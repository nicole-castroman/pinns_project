import os, json, yaml, torch
from .etl.loaders import load_df
from .etl.samplers import prepare_tensors
from .factory import build_model
from .train.loops import train_adam, train_lbfgs

def run_training(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model, (phys, dom, timec, trainc), device = build_model(cfg)

    temp_path = cfg["data"]["temperature_path"]
    mesh_path = cfg["data"]["mesh_path"]
    has_header = cfg["data"].get("has_header", False)
    df = load_df(temp_path, mesh_path, has_header=has_header)
    X_data, T_data, X_f, BCs, X_ic = prepare_tensors(
        df, cfg_data=cfg["data"], cfg_time=cfg["time"], cfg_domain=cfg["domain"], device=device
    )

    # ADAM
    train_adam(model, X_data, T_data, X_f, BCs, X_ic,
               n_iter=trainc.n_iter_adam, lr=trainc.lr_adam)

    # LBFGS
    train_lbfgs(model, X_data, T_data, X_f, BCs, X_ic,
                n_iter=trainc.n_iter_lbfgs, lr=trainc.lr_lbfgs)

    # guardar artefactos
    out_dir = cfg.get("artifacts", {}).get("out_dir", "models/run")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(model.history, f)

    print(f"[OK] Saved to {out_dir}")
