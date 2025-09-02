import torch
from .pinn_heat.configs import PhysicsConfig, DomainConfig, TimeConfig, TrainingConfig, LossConfig
from .pinn_heat.normalizers import IdentityNormalizer, AffineNormalizer
from .pinn_heat.nondim import IdentityNondim, HeatNondim
from .pinn_heat.losses import StaticLossWeights, AdaptiveUncertaintyWeights
from .pinn_heat.nets import DNN
from .pinn_heat.models import PINNHeat

def build_model(cfg: dict, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- configs ---
    phys = PhysicsConfig(**cfg["physics"])
    dom  = DomainConfig(**cfg["domain"])
    timec = TimeConfig(**{**cfg["time"], "N_time_use": cfg["data"]["N_time_use"], "dt": cfg["data"]["dt"]})
    trainc = TrainingConfig(**cfg["train"])
    lossc = LossConfig(**cfg["loss"])

    alpha = phys.k/(phys.rho*phys.Cp)
    variant = cfg.get("variant", "base").lower()

    # --- normalizer & nondim ---
    if variant == "base":
        normalizer = IdentityNormalizer()
        nondim = IdentityNondim(alpha, phys.k, phys.h, phys.T_inf, phys.Ti)
    elif variant == "norm_in":
        normalizer = AffineNormalizer.from_bounds(dom.x_left, dom.x_right, dom.z_bottom, dom.z_top, timec.t_min, timec.t_max, device)
        nondim = IdentityNondim(alpha, phys.k, phys.h, phys.T_inf, phys.Ti)
    else:
        # nondimensional
        nd = cfg.get("nondim", {})
        L_ref = nd.get("L_ref", max(dom.Lx, dom.Lz))
        dT_ref = nd.get("dT_ref", max(abs(phys.qpp*dom.Lz/phys.k), 1.0))
        nondim = HeatNondim(alpha, phys.k, phys.h, phys.T_inf, phys.Ti, L_ref, dT_ref)
        # normalizer works on star-space (x*,z*,τ)
        t_ref = (L_ref**2)/alpha
        normalizer = AffineNormalizer.from_bounds(dom.x_left/L_ref, dom.x_right/L_ref, dom.z_bottom/L_ref, dom.z_top/L_ref, timec.t_min/t_ref, timec.t_max/t_ref, device)

    # --- loss aggregator ---
    if variant == "full" or lossc.adaptive:
        loss_agg = AdaptiveUncertaintyWeights()
    else:
        loss_agg = StaticLossWeights(lossc.w_data, lossc.w_pde, lossc.w_bc, lossc.w_ic)

    # --- net ---
    layers = [3] + [trainc.nneurons]*trainc.nlayers_hidden + [1]
    net = DNN(layers, activation=trainc.act).to(device)

    model = PINNHeat(net=net, normalizer=normalizer, nondim=nondim, loss_aggregator=loss_agg, qpp=phys.qpp, device=device).to(device)
    return model, (phys, dom, timec, trainc), device
