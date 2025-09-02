import torch
import torch.nn as nn

class IdentityNormalizer(nn.Module):
    """No-op normalizer for inputs X=[x,z,t]."""
    def __init__(self):
        super().__init__()
    def encode(self, X): return X
    def decode(self, X): return X

class AffineNormalizer(nn.Module):
    """Feature-wise affine normalizer to map each input to [-1,1]."""
    def __init__(self, centers: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.register_buffer("centers", centers)  # [1,3]
        self.register_buffer("scales",  scales)   # [1,3]
    @classmethod
    def from_bounds(cls, x_min, x_max, z_min, z_max, t_min, t_max, device):
        cx, sx = 0.5*(x_max+x_min), 0.5*max(1e-12, x_max-x_min)
        cz, sz = 0.5*(z_max+z_min), 0.5*max(1e-12, z_max-z_min)
        ct, st = 0.5*(t_max+t_min), 0.5*max(1e-12, t_max-t_min)
        centers = torch.tensor([[cx,cz,ct]], dtype=torch.float32, device=device)
        scales  = torch.tensor([[sx,sz,st]], dtype=torch.float32, device=device)
        return cls(centers, scales)
    def encode(self, X): return (X - self.centers) / self.scales
    def decode(self, Xn): return Xn * self.scales + self.centers
