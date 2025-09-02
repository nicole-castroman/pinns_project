from dataclasses import dataclass

@dataclass
class PhysicsConfig:
    k: float = 50.0
    rho: float = 1200.0
    Cp: float = 100.0
    h: float = 750.0
    T_inf: float = 300.0
    qpp: float = 15000.0
    Ti: float = 300.0

@dataclass
class DomainConfig:
    Lx: float = 0.1
    Lz: float = 0.1
    x_left: float = -0.05
    x_right: float = +0.05
    z_bottom: float = 0.0
    z_top: float = 0.1

@dataclass
class TimeConfig:
    t_min: float = 0.0
    t_max: float = 60.0
    dt: float = 0.1
    N_time_use: int = 25

@dataclass
class TrainingConfig:
    nneurons: int = 64
    nlayers_hidden: int = 8
    act: str = "tanh"
    n_iter_adam: int = 20000
    lr_adam: float = 5e-4
    n_iter_lbfgs: int = 5000
    lr_lbfgs: float = 1e-2
    N_data_nodes: int = 500
    N_f_interior: int = 10000
    N_b_per_face: int = 2000
    N_ic: int = 4000

@dataclass
class LossConfig:
    w_data: float = 1.0
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_ic: float = 1.0
    adaptive: bool = False  # True -> uncertainty weighting
