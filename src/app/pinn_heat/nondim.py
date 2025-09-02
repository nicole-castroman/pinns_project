import torch
import torch.nn as nn

class IdentityNondim(nn.Module):
    """No nondimensionalization: state u == T [K]; PDE in physical variables."""
    def __init__(self, alpha: float, k: float, h: float, T_inf: float, Ti: float):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.k     = torch.tensor(k,     dtype=torch.float32)
        self.h     = torch.tensor(h,     dtype=torch.float32)
        self.T_inf = torch.tensor(T_inf, dtype=torch.float32)
        self.Ti    = torch.tensor(Ti,    dtype=torch.float32)

    # --- mappings ---
    def encode_inputs(self, X_phys):  # here: pass-through
        return X_phys
    def state_to_T(self, u):  # u is T
        return u
    def ic_target(self):
        return self.Ti

    # --- residuals (expect physical derivs of u=T) ---
    def pde_residual_from_phys_derivs(self, u_t, u_xx, u_zz):
        return u_t - self.alpha*(u_xx + u_zz)

    def bc_residual_z_bottom(self, u, u_z):
        # z=0: T_z + q''/k = 0  (qpp must be provided externally)
        raise NotImplementedError("Provide qpp at call site; see models.PINNHeat")

    def bc_residual_z_top(self, u, u_z):
        # z=Lz: T_z + (h/k)*(T - T_inf) = 0
        return u_z + (self.h/self.k)*(u - self.T_inf)

    def bc_residual_x_left(self, u, u_x):
        # x=x_left: T_x - (h/k)*(T - T_inf) = 0
        return u_x - (self.h/self.k)*(u - self.T_inf)

    def bc_residual_x_right(self, u, u_x):
        # x=x_right: T_x + (h/k)*(T - T_inf) = 0
        return u_x + (self.h/self.k)*(u - self.T_inf)


class HeatNondim(nn.Module):
    """Nondimensionalization for transient heat conduction.

    Uses:
      x* = x / L_ref,  z* = z / L_ref,  τ = t / t_ref,   t_ref = L_ref^2 / α
      θ = (T - T_inf)/ΔT_ref
    PDE (dimensionless):
      θ_τ - (θ_xx + θ_zz) = 0
    BCs:
      z*=0:      θ_z* + β_q = 0,   β_q = q'' L_ref / (k ΔT_ref)
      z*=z_top*: θ_z* + Bi θ = 0,  Bi = h L_ref / k
      x*=x_L*:   θ_x* - Bi θ = 0
      x*=x_R*:   θ_x* + Bi θ = 0
    IC:
      θ(x*,z*,0) = (Ti - T_inf)/ΔT_ref
    """
    def __init__(self, alpha, k, h, T_inf, Ti, L_ref, dT_ref):
        super().__init__()
        self.alpha  = torch.tensor(alpha, dtype=torch.float32)
        self.k      = torch.tensor(k,     dtype=torch.float32)
        self.h      = torch.tensor(h,     dtype=torch.float32)
        self.T_inf  = torch.tensor(T_inf, dtype=torch.float32)
        self.Ti     = torch.tensor(Ti,    dtype=torch.float32)
        self.Lref   = torch.tensor(L_ref, dtype=torch.float32)
        self.dTref  = torch.tensor(dT_ref, dtype=torch.float32)

        # dimensionless groups
        self.tref = (self.Lref**2)/self.alpha       # s
        self.Bi   = (self.h*self.Lref)/self.k       # -
        # β_q depends on q'' -> completed at call with qpp

    # --- mappings ---
    def encode_inputs(self, X_phys):
        """Map physical X=[x,z,t] to dimensionless X*=[x*,z*,τ]."""
        Xs = X_phys.clone()
        Xs[:,0:1] = Xs[:,0:1]/self.Lref
        Xs[:,1:2] = Xs[:,1:2]/self.Lref
        Xs[:,2:3] = Xs[:,2:3]/self.tref
        return Xs

    def state_to_T(self, u):  # u is θ
        return self.T_inf + self.dTref*u

    def ic_target(self):
        return (self.Ti - self.T_inf)/self.dTref

    # --- helpers for scaling derivatives to dimensionless form ---
    # u_x* = Lref * u_x_phys,  u_xx* = Lref^2 * u_xx_phys,  u_τ = tref * u_t_phys
    def pde_residual_from_phys_derivs(self, u_t, u_xx, u_zz):
        return self.tref*u_t - self.Lref**2*(u_xx + u_zz)

    def bc_residual_z_bottom(self, u, u_z, qpp):
        beta_q = (qpp*self.Lref)/(self.k*self.dTref)
        return self.Lref*u_z + beta_q

    def bc_residual_z_top(self, u, u_z):
        return self.Lref*u_z + self.Bi*u

    def bc_residual_x_left(self, u, u_x):
        return self.Lref*u_x - self.Bi*u

    def bc_residual_x_right(self, u, u_x):
        return self.Lref*u_x + self.Bi*u
