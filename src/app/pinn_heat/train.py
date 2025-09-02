import time
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt

def train_adam(model, X_data, T_data, X_f, BCs, X_ic, n_iter=20000, lr=5e-4, betas=(0.9,0.999), show_every=200):
    """Train with ADAM."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, amsgrad=True)
    t0 = time.time()
    for epoch in range(1, n_iter+1):
        opt.zero_grad()
        loss = model.total_loss(X_data, T_data, X_f, BCs, X_ic, save=True)
        loss.backward()
        opt.step()
        if epoch % show_every == 0:
            clear_output(wait=True)
            print(f"[ADAM] epoch={epoch:6d} | total={loss.item():.4e} "
                  f"| D={model.history['Data'][-1]:.3e} P={model.history['PDE'][-1]:.3e} "
                  f"BC={model.history['BC'][-1]:.3e} IC={model.history['IC'][-1]:.3e}")
            # quick preview
            plt.figure(figsize=(8,3))
            for k in ["Total","Data","PDE","BC","IC"]:
                plt.semilogy(model.history[k], label=k)
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    print(f"[ADAM] done in {time.time()-t0:.1f}s")

def train_lbfgs(model, X_data, T_data, X_f, BCs, X_ic, n_iter=5000, lr=1e-2, show_every=20):
    """Finalize with LBFGS."""
    opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=n_iter,
                            tolerance_grad=1e-8, tolerance_change=1e-12,
                            history_size=100, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = model.total_loss(X_data, T_data, X_f, BCs, X_ic, save=True)
        loss.backward()
        if model.iter % show_every == 0:
            clear_output(wait=True)
            print(f"[LBFGS] iter={model.iter:6d} | total={loss.item():.4e}")
        model.iter += 1
        return loss
    opt.step(closure)
