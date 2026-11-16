"""Plot helpers shared by the notebooks."""
import matplotlib.pyplot as plt
import numpy as np

BASE_COLOR = "#3b6ea8"
REF_COLOR = "#c0504d"


def rate_panel(tables: dict, base_rate: float, path=None, ncols=3):
    """Grid of default-rate-by-bucket bar charts with 95% Wilson CI whiskers."""
    n = len(tables)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (title, t) in zip(axes.ravel(), tables.items()):
        x = np.arange(len(t))
        err = np.vstack([t["default_rate"] - t["ci_low"], t["ci_high"] - t["default_rate"]])
        ax.bar(x, t["default_rate"], color=BASE_COLOR, yerr=err, capsize=3)
        ax.axhline(base_rate, color=REF_COLOR, ls="--", lw=1, label=f"base {base_rate:.1%}")
        ax.set_xticks(x, t.index, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("default rate")
        for xi, (r, nn) in enumerate(zip(t["default_rate"], t["n"])):
            ax.text(xi, r + 0.01, f"n={nn}", ha="center", fontsize=6.5, color="#555")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig
