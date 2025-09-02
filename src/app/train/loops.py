import time, torch
from IPython.display import clear_output

def train_adam(model, X_data, T_data, X_f, BCs, X_ic, n_iter=20000, lr=5e-4, show_every=200):
    opt = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    t0 = time.time()
    for epoch in range(1, n_iter+1):
        opt.zero_grad()
        loss = model.total_loss(X_data, T_data, X_f, BCs, X_ic, save=True)
        loss.backward()
        opt.step()
        if epoch % show_every == 0:
            clear_output(wait=True)
            print(f"[ADAM] epoch={epoch:6d} total={loss.item():.4e}")
    print(f"[ADAM] done in {time.time()-t0:.1f}s")

def train_lbfgs(model, X_data, T_data, X_f, BCs, X_ic, n_iter=5000, lr=1e-2, show_every=20):
    opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=n_iter,
                            tolerance_grad=1e-8, tolerance_change=1e-12,
                            history_size=100, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = model.total_loss(X_data, T_data, X_f, BCs, X_ic, save=True)
        loss.backward()
        return loss
    opt.step(closure)
