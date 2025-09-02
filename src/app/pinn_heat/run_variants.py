import torch
from .configs import PhysicsConfig, DomainConfig, TimeConfig, TrainingConfig, LossConfig
from .data import (build_data_from_df, sample_interior_collocation,
                            sample_bc_collocation, sample_ic_collocation)
from .normalizers import IdentityNormalizer, AffineNormalizer
from .nondim import IdentityNondim, HeatNondim
from .nets import DNN
from .losses import StaticLossWeights, AdaptiveUncertaintyWeights
from .models import PINNHeat
from .train import train_adam, train_lbfgs

def build_and_run(df, variant: int = 1, device=None):
    """
    Build and train one of 4 variants:
      1: no normalization, no nondimensionalization
      2: input normalization only
      3: normalization + nondimensionalization
      4: normalization + nondimensionalization + adaptive weights
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phys = PhysicsConfig()
    dom  = DomainConfig()
    timec = TimeConfig()
    trainc = TrainingConfig()
    lossc = LossConfig(adaptive=(variant==4))

    alpha = phys.k/(phys.rho*phys.Cp)

    # --- data tensors (all in PHYSICAL units) ---
    X_data, T_data = build_data_from_df(df, trainc.N_data_nodes, timec.N_time_use, timec.dt, device)
    X_f  = sample_interior_collocation(trainc.N_f_interior, timec.t_max, dom.x_left, dom.x_right, dom.z_bottom, dom.z_top, device)
    BCs  = sample_bc_collocation(trainc.N_b_per_face, timec.t_max, dom.x_left, dom.x_right, dom.z_bottom, dom.z_top, device)
    X_ic = sample_ic_collocation(trainc.N_ic, dom.x_left, dom.x_right, dom.z_bottom, dom.z_top, device)

    # --- normalizer & nondim ---
    if variant in (1,):
        normalizer = IdentityNormalizer()
        nondim = IdentityNondim(alpha, phys.k, phys.h, phys.T_inf, phys.Ti)
    elif variant in (2,):
        normalizer = AffineNormalizer.from_bounds(dom.x_left, dom.x_right, dom.z_bottom, dom.z_top, timec.t_min, timec.t_max, device)
        nondim = IdentityNondim(alpha, phys.k, phys.h, phys.T_inf, phys.Ti)
    else:
        # choose a reference length and temperature scale for nondim
        L_ref = max(dom.Lx, dom.Lz)
        dT_ref = max(abs(phys.qpp*dom.Lz/phys.k), 1.0)  # robust default
        nondim = HeatNondim(alpha, phys.k, phys.h, phys.T_inf, phys.Ti, L_ref, dT_ref)
        if variant in (3,4):
            # normalize inputs AFTER nondim mapping (best with tanh)
            # bounds in X* are approximately:
            #   x* in [x_left/L_ref, x_right/L_ref], z* in [0, Lz/L_ref], τ in [0, t_max/t_ref]
            import numpy as np
            t_ref = (L_ref**2)/alpha
            normalizer = AffineNormalizer.from_bounds(
                x_min=dom.x_left/L_ref, x_max=dom.x_right/L_ref,
                z_min=dom.z_bottom/L_ref, z_max=dom.z_top/L_ref,
                t_min=timec.t_min/t_ref, t_max=timec.t_max/t_ref,
                device=device
            )

    # --- loss aggregator ---
    if lossc.adaptive:
        loss_agg = AdaptiveUncertaintyWeights()
    else:
        loss_agg = StaticLossWeights(lossc.w_data, lossc.w_pde, lossc.w_bc, lossc.w_ic)

    # --- network ---
    layers = [3] + [trainc.nneurons]*trainc.nlayers_hidden + [1]
    net = DNN(layers, activation=trainc.act).to(device)

    # --- model ---
    model = PINNHeat(net=net, normalizer=normalizer, nondim=nondim,
                     loss_aggregator=loss_agg, qpp=phys.qpp, device=device).to(device)

    # --- train ---
    train_adam(model, X_data, T_data, X_f, BCs, X_ic,
               n_iter=trainc.n_iter_adam, lr=trainc.lr_adam)
    train_lbfgs(model, X_data, T_data, X_f, BCs, X_ic,
                n_iter=trainc.n_iter_lbfgs, lr=trainc.lr_lbfgs)

    return model

# Example usage:
# model1 = build_and_run(df, variant=1)  # base
# model2 = build_and_run(df, variant=2)  # normalized inputs
# model3 = build_and_run(df, variant=3)  # normalized + nondim
# model4 = build_and_run(df, variant=4)  # normalized + nondim + adaptive weights
