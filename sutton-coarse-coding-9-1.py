"""
Coarse Coding - Example 9.1 from Sutton & Barto
================================================
Reproduces Figure 9.4: effect of feature width on learning a 1D square-wave
function using linear gradient-descent function approximation.

Usage:
    python coarse_coding.py

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Target function ──────────────────────────────────────────────────────────

def target_fn(x):
    """Square-wave: 1 inside two bumps, 0 elsewhere."""
    return ((x > 0.2) & (x < 0.4)) | ((x > 0.6) & (x < 0.8))


# ── Feature construction ─────────────────────────────────────────────────────

def make_features(width, n_features=50, x_min=0.0, x_max=1.0):
    """
    Returns an array of feature centers evenly spaced over [x_min, x_max].
    Each feature is an interval of given width centred on its center.
    """
    centers = np.linspace(x_min, x_max, n_features)
    return centers  # width is stored separately


def phi(x, centers, width):
    """
    Binary feature vector for scalar x.
    phi_i(x) = 1 if |x - center_i| <= width/2, else 0.
    """
    return (np.abs(x - centers) <= width / 2).astype(float)


# ── Linear function approximation ────────────────────────────────────────────

def predict(x, w, centers, width):
    """v̂(x, w) = w^T φ(x)"""
    return np.dot(w, phi(x, centers, width))


def train(centers, width, n_examples, alpha_base=0.2, seed=42):
    """
    Online gradient-descent update (eq. 9.3):
        w ← w + α [V_t - v̂(S_t, w)] φ(S_t)

    Step size: α = alpha_base / m  where m = number of active features.
    Training points drawn uniformly at random.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros(len(centers))

    for _ in range(n_examples):
        x = rng.uniform(0, 1)
        v_target = float(target_fn(x))
        p = phi(x, centers, width)
        m = p.sum()
        if m == 0:
            continue
        alpha = alpha_base / m
        v_hat = np.dot(w, p)
        error = v_target - v_hat
        # Gradient-descent update: gradient of linear approx = φ(x)
        w += alpha * error * p

    return w


def rmse(w, centers, width, n_eval=500):
    """Root mean squared error over a uniform grid."""
    xs = np.linspace(0, 1, n_eval)
    preds = np.array([predict(x, w, centers, width) for x in xs])
    targets = target_fn(xs).astype(float)
    return np.sqrt(np.mean((targets - preds) ** 2))


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_figure(n_examples_list, width_configs, n_features=50, seed=42):
    """
    Reproduces Figure 9.4.
    Rows = different numbers of training examples.
    Columns = narrow / medium / broad features.
    """
    xs = np.linspace(0, 1, 400)
    target = target_fn(xs).astype(float)

    n_rows = len(n_examples_list)
    n_cols = len(width_configs)

    fig = plt.figure(figsize=(10, 2.2 * n_rows + 1.2))
    fig.suptitle(
        "Coarse Coding — Effect of Feature Width (Sutton & Barto Fig. 9.4)",
        fontsize=13, y=1.01
    )

    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.55, wspace=0.3)

    for col_idx, (label, width) in enumerate(width_configs.items()):
        centers = make_features(width, n_features)

        for row_idx, n_ex in enumerate(n_examples_list):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Train
            w = train(centers, width, n_ex, seed=seed)
            approx = np.clip(
                np.array([predict(x, w, centers, width) for x in xs]), 0, 1
            )
            err = rmse(w, centers, width)

            # Draw feature intervals as faint shaded bands
            for c in centers:
                ax.axvspan(
                    max(0, c - width / 2), min(1, c + width / 2),
                    alpha=0.04, color='steelblue'
                )

            ax.plot(xs, target, color='#888', lw=1.5,
                    linestyle='--', label='target')
            ax.plot(xs, approx, color='#3266ad', lw=2,
                    label=f'approx (RMSE={err:.3f})')

            ax.set_ylim(-0.15, 1.25)
            ax.set_xlim(0, 1)
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=8)

            # Row label (left column only)
            if col_idx == 0:
                ax.set_ylabel(f'n={n_ex}', fontsize=9)

            # Column header (top row only)
            if row_idx == 0:
                ax.set_title(f'{label}\n(width={width})', fontsize=10)

            ax.text(0.98, 1.12, f'RMSE={err:.3f}',
                    transform=ax.transAxes, fontsize=7,
                    ha='right', color='#3266ad')

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color='#888', lw=1.5, linestyle='--', label='target'),
        plt.Line2D([0], [0], color='#3266ad', lw=2, label='approximation'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.savefig('coarse_coding_figure.png', dpi=150, bbox_inches='tight')
    print("Saved: coarse_coding_figure.png")
    plt.show()


# ── Interactive exploration ───────────────────────────────────────────────────

def interactive_demo():
    """
    Single-panel demo: adjust width and n_examples interactively via sliders.
    Run this in a Jupyter notebook or with matplotlib's TkAgg / Qt backend.
    """
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots(figsize=(9, 4))
    plt.subplots_adjust(bottom=0.28)

    xs = np.linspace(0, 1, 400)
    target = target_fn(xs).astype(float)
    n_features = 50
    init_width = 0.12
    init_n = 40

    centers = make_features(init_width, n_features)
    w = train(centers, init_width, init_n)
    approx = np.clip([predict(x, w, centers, init_width) for x in xs], 0, 1)

    line_target, = ax.plot(xs, target, '--', color='#888', lw=1.5, label='target')
    line_approx, = ax.plot(xs, approx, color='#3266ad', lw=2, label='approximation')
    err_text = ax.text(0.01, 1.1, '', transform=ax.transAxes,
                       fontsize=9, color='#3266ad')

    ax.set_ylim(-0.15, 1.3)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title("Coarse Coding — Linear Function Approximation", fontsize=11)

    ax_width = plt.axes([0.15, 0.14, 0.7, 0.03])
    ax_n     = plt.axes([0.15, 0.07, 0.7, 0.03])

    s_width = Slider(ax_width, 'feature width', 0.02, 0.4,
                     valinit=init_width, valstep=0.01)
    s_n     = Slider(ax_n,     'n examples',    1,    500,
                     valinit=init_n,    valstep=1)

    def update(_):
        w_ = np.zeros(n_features)
        width_ = s_width.val
        n_     = int(s_n.val)
        centers_ = make_features(width_, n_features)
        w_ = train(centers_, width_, n_)
        approx_ = np.clip([predict(x, w_, centers_, width_) for x in xs], 0, 1)
        line_approx.set_ydata(approx_)
        err = rmse(w_, centers_, width_)
        err_text.set_text(f'RMSE = {err:.4f}')
        fig.canvas.draw_idle()

    s_width.on_changed(update)
    s_n.on_changed(update)
    update(None)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # --- Figure 9.4 reproduction ---
    width_configs = {
        'narrow':  0.05,
        'medium':  0.12,
        'broad':   0.25,
    }

    n_examples_list = [10, 40, 160, 640, 2560, 10240]

    plot_figure(n_examples_list, width_configs)

    # --- Uncomment for the interactive slider demo ---
    # interactive_demo()