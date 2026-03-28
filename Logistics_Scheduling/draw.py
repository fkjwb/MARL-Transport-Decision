"""Plot curves exported from TensorBoard CSV files."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def tb_smooth(values, smooth=0.6):
    """TensorBoard-style exponential smoothing."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values

    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = smooth * smoothed[i - 1] + (1 - smooth) * values[i]
    return smoothed


def setup_plot_style():
    """Set a clean default matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _validate_limit(limit, name):
    if limit is None:
        return None
    if len(limit) != 2:
        raise ValueError(f"{name} must be a (min, max) tuple.")

    lower, upper = float(limit[0]), float(limit[1])
    if lower >= upper:
        raise ValueError(f"{name} must satisfy min < max.")
    return lower, upper


def _format_tick_value(value, decimals=None):
    if decimals is None:
        if np.isclose(value, round(value)):
            return str(int(round(value)))
        return f"{value:g}"
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _make_tick_formatter(scale=1.0, suffix="", decimals=None):
    if np.isclose(scale, 0.0):
        raise ValueError("Tick scale must not be 0.")

    def _formatter(value, _):
        scaled_value = value / scale
        return f"{_format_tick_value(scaled_value, decimals=decimals)}{suffix}"

    return FuncFormatter(_formatter)


def _build_ticks(values, limit=None, tick_step=None):
    if tick_step is None:
        return None
    if tick_step <= 0:
        raise ValueError("Tick step must be positive.")

    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None

    if limit is None:
        start = float(finite_values.min())
        end = float(finite_values.max())
    else:
        start, end = limit

    return np.arange(start, end + tick_step * 0.5, tick_step, dtype=float)


def _configure_axis(
    ax,
    values,
    axis="x",
    limit=None,
    tick_step=None,
    tick_scale=1.0,
    tick_suffix="",
    tick_decimals=None,
):
    axis_name = axis.lower()
    if axis_name not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'.")

    set_limit = ax.set_xlim if axis_name == "x" else ax.set_ylim
    set_ticks = ax.set_xticks if axis_name == "x" else ax.set_yticks
    axis_obj = ax.xaxis if axis_name == "x" else ax.yaxis

    if limit is not None:
        set_limit(limit)

    ticks = _build_ticks(values=values, limit=limit, tick_step=tick_step)
    if ticks is not None:
        set_ticks(ticks)

    axis_obj.set_major_formatter(
        _make_tick_formatter(
            scale=tick_scale,
            suffix=tick_suffix,
            decimals=tick_decimals,
        )
    )


def plot_tb_csv(
    csv_path,
    x_col="Step",
    y_col="Value",
    smooth=0.6,
    label="Smoothed",
    raw_label="Raw",
    title="Average Episode Length",
    xlabel="Timesteps",
    ylabel="Average Episode Length (steps)",
    color="#1f77b4",
    save_dir="./figures",
    save_name="ep_len_mean",
    show_raw=True,
    raw_alpha=0.25,
    raw_linewidth=1.0,
    show_grid=True,
    grid_linestyle="--",
    show_legend=True,
    xlim=None,
    ylim=None,
    xtick_step=None,
    ytick_step=None,
    x_tick_scale=1.0,
    y_tick_scale=1.0,
    x_tick_suffix="",
    y_tick_suffix="",
    x_tick_decimals=None,
    y_tick_decimals=None,
):
    """Plot a TensorBoard-exported CSV curve with configurable axes."""
    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    xlim = _validate_limit(xlim, "xlim")
    ylim = _validate_limit(ylim, "ylim")

    df = pd.read_csv(csv_path)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Missing required columns. Available columns: {list(df.columns)}")

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    y_smooth = tb_smooth(y, smooth=smooth)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    if show_raw:
        ax.plot(
            x,
            y,
            color=color,
            alpha=raw_alpha,
            linewidth=raw_linewidth,
            label=raw_label,
        )

    ax.plot(x, y_smooth, color=color, linewidth=2.2, label=label)

    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    _configure_axis(
        ax,
        values=x,
        axis="x",
        limit=xlim,
        tick_step=xtick_step,
        tick_scale=x_tick_scale,
        tick_suffix=x_tick_suffix,
        tick_decimals=x_tick_decimals,
    )
    _configure_axis(
        ax,
        values=y,
        axis="y",
        limit=ylim,
        tick_step=ytick_step,
        tick_scale=y_tick_scale,
        tick_suffix=y_tick_suffix,
        tick_decimals=y_tick_decimals,
    )

    if show_grid:
        ax.grid(True, linestyle=grid_linestyle)
    else:
        ax.grid(False)

    if show_legend:
        ax.legend()

    plt.tight_layout()

    pdf_path = save_dir / f"{save_name}.pdf"
    png_path = save_dir / f"{save_name}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()

    plt.close(fig)
    print(f"Saved to:\n  {pdf_path}\n  {png_path}")


if __name__ == "__main__":
    setup_plot_style()

    plot_tb_csv(
        csv_path=Path("runs") / "20260327-093517" / "run-.-tag-train_episodes_return_mean.csv",
        smooth=0.6,
        label="PPO",
        raw_label="PPO Raw",
        title="Average Episode Length",
        xlabel="Timesteps",
        ylabel="Average Episode Length (steps)",
        color="#3399FF",
        save_dir="./figures",
        save_name="ep_len_mean",
        show_raw=True,
        show_grid=False,
        x_tick_scale=1000,
        x_tick_suffix="k",
        x_tick_decimals=1,
    )
