import torch
import torch.nn as nn

class StaticLossWeights(nn.Module):
    """Static linear combination of loss terms."""
    def __init__(self, w_data=1.0, w_pde=1.0, w_bc=1.0, w_ic=1.0):
        super().__init__()
        self.wd = float(w_data); self.wp = float(w_pde)
        self.wb = float(w_bc);   self.wi = float(w_ic)
    def forward(self, Ld, Lpde, Lbc, Lic):
        return self.wd*Ld + self.wp*Lpde + self.wb*Lbc + self.wi*Lic

class AdaptiveUncertaintyWeights(nn.Module):
    """Uncertainty weighting (Kendall & Gal, 2018) for multi-task losses.

    total = Σ [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ], with s_i = log σ_i^2 (trainable)
    """
    def __init__(self, init_logvars=(0.0,0.0,0.0,0.0)):
        super().__init__()
        self.s_data = nn.Parameter(torch.tensor(init_logvars[0], dtype=torch.float32))
        self.s_pde  = nn.Parameter(torch.tensor(init_logvars[1], dtype=torch.float32))
        self.s_bc   = nn.Parameter(torch.tensor(init_logvars[2], dtype=torch.float32))
        self.s_ic   = nn.Parameter(torch.tensor(init_logvars[3], dtype=torch.float32))

    def forward(self, Ld, Lpde, Lbc, Lic):
        def term(L, s): return 0.5*torch.exp(-s)*L + 0.5*s
        return term(Ld, self.s_data) + term(Lpde, self.s_pde) + term(Lbc, self.s_bc) + term(Lic, self.s_ic)
