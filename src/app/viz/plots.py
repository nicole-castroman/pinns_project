from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_history(history: dict, path: str | None = None, title: str = "Training Loss"):
    """Plot loss history curves (Total/Data/PDE/BC/IC) on a semilog scale.

    Args:
        history: dict with lists under keys 'Total','Data','PDE','BC','IC'.
        path: Optional file path to save figure.
        title: Plot title.
    """
    plt.figure(figsize=(9,4))
    for key in ["Total","Data","PDE","BC","IC"]:
        if key in history and len(history[key])>0:
            plt.semilogy(history[key], label=key)
    plt.grid(True); plt.legend(); plt.title(title)
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=160)
        plt.close()
    else:
        plt.show()

def plot_temperature_maps(T_maps: dict[float, np.ndarray], xs: np.ndarray, zs: np.ndarray,
                          path: str | None = None, title: str = "Temperature fields [K]"):
    """Plot a row of heatmaps for several times.

    Args:
        T_maps: Dict time-> 2D array [nz,nx] (Kelvin).
        xs, zs: 1D coordinate arrays (meters).
        path: Optional path to save PNG.
        title: Figure title.
    """
    times = list(T_maps.keys())
    n = len(times)
    fig, axs = plt.subplots(1, n, figsize=(5*n, 4), constrained_layout=True)
    axs = np.atleast_1d(axs)

    Xmin, Xmax = xs.min(), xs.max()
    Zmin, Zmax = zs.min(), zs.max()

    for ax, tt in zip(axs, times):
        im = ax.imshow(T_maps[tt], origin="lower", extent=[Xmin, Xmax, Zmin, Zmax],
                       aspect="auto")
        ax.set_title(f"t = {tt:.2f} s")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("T [K]")

    fig.suptitle(title)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
    else:
        plt.show()

def scatter_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray,
                         path: str | None = None, title: str = "True vs Predicted Temperature"):
    """Scatter plot of T_true vs T_pred with y=x reference.

    Args:
        y_true: 1D array of ground-truth temperatures [K].
        y_pred: 1D array of predictions [K].
        path: Optional file path to save figure.
        title: Plot title.
    """
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    lo, hi = np.min([y_true, y_pred]), np.max([y_true, y_pred])
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("True T [K]"); plt.ylabel("Pred T [K]")
    plt.title(title); plt.grid(True); plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=160)
        plt.close()
    else:
        plt.show()
