from ..pinn_heat.data import (build_data_from_df, sample_interior_collocation,
                            sample_bc_collocation, sample_ic_collocation)

def prepare_tensors(df, cfg_data, cfg_time, cfg_domain, device):
    X_data, T_data = build_data_from_df(
        df, cfg_data["N_data_nodes"], cfg_time["N_time_use"], cfg_time["dt"], device
    )
    X_f  = sample_interior_collocation(
        cfg_data["N_f_interior"], cfg_time["t_max"],
        cfg_domain["x_left"], cfg_domain["x_right"],
        cfg_domain["z_bottom"], cfg_domain["z_top"], device
    )
    BCs  = sample_bc_collocation(
        cfg_data["N_b_per_face"], cfg_time["t_max"],
        cfg_domain["x_left"], cfg_domain["x_right"],
        cfg_domain["z_bottom"], cfg_domain["z_top"], device
    )
    X_ic = sample_ic_collocation(
        cfg_data["N_ic"], cfg_domain["x_left"], cfg_domain["x_right"],
        cfg_domain["z_bottom"], cfg_domain["z_top"], device
    )
    return X_data, T_data, X_f, BCs, X_ic
