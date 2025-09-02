import torch
import torch.nn as nn

class PINNHeat(nn.Module):
    """Composable PINN for transient heat in (x,z,t).

    Pipeline:
      - Receives physical inputs X_phys=[x,z,t] (SI).
      - (Optional) Nondim: X* = nondim.encode_inputs(X_phys)
      - Normalize: X_in = normalizer.encode(X* or X_phys)
      - Net predicts state u:
          * IdentityNondim: u = T [K]
          * HeatNondim    : u = θ [-]
      - Data loss compares T_pred = nondim.state_to_T(u) vs T_data [K]
      - PDE/BC/IC residuals computed via nondim using physical derivatives of u.
    """
    def __init__(self, net, normalizer, nondim, loss_aggregator, qpp: float, device):
        super().__init__()
        self.net = net
        self.normalizer = normalizer
        self.nondim = nondim
        self.loss_agg = loss_aggregator
        self.qpp = torch.tensor(qpp, dtype=torch.float32, device=device)
        self.mse = nn.MSELoss(reduction='mean')
        self.device = device
        self.iter = 0
        self.history = {"Total": [], "Data": [], "PDE": [], "BC": [], "IC": []}

    def forward(self, X_phys):
        X_star = self.nondim.encode_inputs(X_phys)          # may be identity/dimensionless
        X_in   = self.normalizer.encode(X_star)             # may be identity/affine
        u      = self.net(X_in)                             # state
        return u

    # --- autograd helpers ---
    @staticmethod
    def gradients(y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   retain_graph=True, create_graph=True)[0]

    def _first_second_derivs(self, u, X_phys):
        du = self.gradients(u, X_phys)  # [ux, uz, ut]
        ux, uz, ut = du[:,0:1], du[:,1:2], du[:,2:3]
        uxx = self.gradients(ux, X_phys)[:,0:1]
        uzz = self.gradients(uz, X_phys)[:,1:2]
        return ut, ux, uz, uxx, uzz

    # --- losses ---
    def loss_data(self, X_data, T_data):
        u = self.forward(X_data)
        T_pred = self.nondim.state_to_T(u)
        return self.mse(T_pred, T_data)

    def loss_ic(self, X_ic):
        u = self.forward(X_ic)
        u_i = torch.ones_like(u)*self.nondim.ic_target()
        return self.mse(u, u_i)

    def loss_bc(self, BCs):
        loss = 0.0
        # z=0 flux
        X = BCs["z0_flux"].clone().requires_grad_(True)
        u = self.forward(X)
        _, _, uz, _, _ = self._first_second_derivs(u, X)
        # IdentityNondim needs qpp injected at call
        if hasattr(self.nondim, "bc_residual_z_bottom") and self.nondim.__class__.__name__ == "HeatNondim":
            res = self.nondim.bc_residual_z_bottom(u, uz, self.qpp)
        else:
            # IdentityNondim: residual = T_z + q''/k
            # k is inside nondim; re-use its k
            k = self.nondim.k if hasattr(self.nondim,"k") else torch.tensor(1.0, dtype=torch.float32, device=self.device)
            res = uz + (self.qpp/k)
        loss = loss + self.mse(res, torch.zeros_like(res))

        # z=Lz convection
        X = BCs["zL_conv"].clone().requires_grad_(True)
        u = self.forward(X)
        _, _, uz, _, _ = self._first_second_derivs(u, X)
        res = self.nondim.bc_residual_z_top(u, uz)
        loss = loss + self.mse(res, torch.zeros_like(res))

        # x=x_left convection
        X = BCs["x0_conv"].clone().requires_grad_(True)
        u = self.forward(X)
        _, ux, _, _, _ = self._first_second_derivs(u, X)
        res = self.nondim.bc_residual_x_left(u, ux)
        loss = loss + self.mse(res, torch.zeros_like(res))

        # x=x_right convection
        X = BCs["xL_conv"].clone().requires_grad_(True)
        u = self.forward(X)
        _, ux, _, _, _ = self._first_second_derivs(u, X)
        res = self.nondim.bc_residual_x_right(u, ux)
        loss = loss + self.mse(res, torch.zeros_like(res))

        return loss

    def loss_pde(self, X_f):
        X = X_f.clone().requires_grad_(True)
        u = self.forward(X)
        ut, _, _, uxx, uzz = self._first_second_derivs(u, X)
        res = self.nondim.pde_residual_from_phys_derivs(ut, uxx, uzz)
        return self.mse(res, torch.zeros_like(res))

    def total_loss(self, X_data, T_data, X_f, BCs, X_ic, save=True):
        Ld  = self.loss_data(X_data, T_data)
        Lp  = self.loss_pde(X_f)
        Lbc = self.loss_bc(BCs)
        Lic = self.loss_ic(X_ic)
        total = self.loss_agg(Ld, Lp, Lbc, Lic)
        if save:
            self.history["Data"].append(Ld.item())
            self.history["PDE"].append(Lp.item())
            self.history["BC"].append(Lbc.item())
            self.history["IC"].append(Lic.item())
            self.history["Total"].append(total.item())
        return total
