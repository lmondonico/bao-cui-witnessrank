"""Plot the main paper figures from data/; --from-results first rebuilds the Figure 2 data from the audits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from experiments.paths import RESULTS


FIGURES = Path(__file__).resolve().parent
DATA = FIGURES / "data"
BUDGETS = np.array([8, 16, 32, 64, 128, 256])

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
GREEN = "#2ca02c"
BLACK = "#252525"
TEXT_GREY = "#4d4d4d"

LANDSCAPES = ["tf_bind_8", "trpb", "gb1", "av_gfp"]
AUDIT_ORDER = ["av_gfp", "gb1", "tf_bind_8", "trpb"]
LANDSCAPE_LABELS = {"tf_bind_8": "TF-Bind-8", "trpb": "TrpB", "gb1": "GB1", "av_gfp": "avGFP"}
LANDSCAPE_COLORS = {"tf_bind_8": BLUE, "trpb": ORANGE, "gb1": GREEN, "av_gfp": PURPLE}
MARKERS = {"av_gfp": "o", "gb1": "s", "tf_bind_8": "^", "trpb": "D"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "lines.linewidth": 1.35,
            "lines.markersize": 5.0,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def light_grid(axis: plt.Axes, direction: str = "both") -> None:
    axis.grid(
        axis=direction,
        which="major",
        color="#dedede",
        lw=.4,
        linestyle=(0, (2, 2)),
        alpha=.68,
    )


def panel_label(
    axis: plt.Axes,
    letter: str,
    x: float = -0.18,
    y: float = 1.04,
    fontsize: float = 13,
) -> None:
    axis.text(
        x,
        y,
        f"({letter})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        clip_on=False,
    )


def save(figure: plt.Figure, filename: str) -> None:
    figure.savefig(FIGURES / filename, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def bottom_legend(figure: plt.Figure, handles, labels, columns: int) -> None:
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=columns,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.45,
        handlelength=1.5,
    )


def set_budget_axis(axis: plt.Axes) -> None:
    budgets = [8, 16, 32, 64, 128, 256]
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, [str(value) for value in budgets])
    axis.set_xlabel("trusted budget $B$")


def _primary_margin(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame.delta_family.eq("sd") & np.isclose(frame.delta_multiplier, .05) & frame.delta_applicable
    ].copy()


def figure_2_data() -> dict[str, pd.DataFrame]:
    margin = _primary_margin(pd.read_csv(RESULTS / "margin_nomination/margin_metrics.csv"))
    random = _primary_margin(pd.read_csv(RESULTS / "random_order_null/metrics.csv"))
    finite = margin[margin.R_delta.gt(0)]
    ranks = finite[["landscape", "R_delta"]].sort_values(["landscape", "R_delta"], kind="stable")
    recall = pd.DataFrame([
        {
            "landscape": landscape,
            "budget": int(budget),
            "proxy_recall": margin.loc[margin.landscape.eq(landscape), f"recall_{budget}"].mean(),
            "random_recall": random.loc[random.landscape.eq(landscape), f"recall_random_{budget}"].mean(),
        }
        for landscape in AUDIT_ORDER
        for budget in BUDGETS
    ])
    return {"fig2_audit_ranks": ranks, "fig2_audit_recall": recall}


def plot_figure_2(frames: dict[str, pd.DataFrame]) -> None:
    ranks, recall = frames["fig2_audit_ranks"], frames["fig2_audit_recall"]
    figure, (axis_a, axis_b) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.8),
        gridspec_kw={"width_ratios": [1.02, 1.10], "wspace": .30},
    )
    plot_order = AUDIT_ORDER[::-1]
    for row, landscape in enumerate(plot_order):
        values = np.sort(ranks.loc[ranks.landscape.eq(landscape), "R_delta"].to_numpy(float))
        for rank in np.unique(values):
            count = int((values == rank).sum())
            offsets = np.linspace(-.19, .19, count) if count > 1 else np.array([0.0])
            axis_a.scatter(
                np.full(count, rank),
                row + offsets,
                s=27,
                color=LANDSCAPE_COLORS[landscape],
                alpha=.62,
                edgecolors="white",
                linewidths=.3,
                zorder=2,
            )
        axis_a.scatter(
            float(np.median(values)),
            row,
            s=68,
            marker="D",
            facecolor="white",
            edgecolor=BLACK,
            linewidth=1.15,
            zorder=4,
        )
    axis_a.set_xscale("log")
    axis_a.set_xlim(.8, 650)
    axis_a.set_ylim(-.52, 3.52)
    axis_a.set_xticks([1, 10, 100], ["1", "10", "100"])
    axis_a.set_yticks(range(4), [LANDSCAPE_LABELS[name] for name in plot_order])
    axis_a.set_xlabel(r"first $\Delta$-witness rank $R_\Delta$", fontsize=16)
    axis_a.tick_params(axis="both", labelsize=14)
    light_grid(axis_a, "x")

    for landscape in AUDIT_ORDER:
        part = recall[recall.landscape.eq(landscape)]
        axis_b.plot(
            BUDGETS,
            part.proxy_recall.to_numpy(),
            color=LANDSCAPE_COLORS[landscape],
            marker=MARKERS[landscape],
            markeredgecolor="white",
            markeredgewidth=.55,
            markersize=7,
            linewidth=2.1,
            zorder=3,
        )
        axis_b.plot(
            BUDGETS,
            part.random_recall.to_numpy(),
            color="#8a8a8a",
            marker=MARKERS[landscape],
            markerfacecolor="white",
            markeredgecolor="#777777",
            markeredgewidth=.65,
            markersize=6.5,
            linestyle=(0, (4, 2)),
            linewidth=1.55,
            zorder=2,
        )
    axis_b.axvline(16, color=TEXT_GREY, linestyle=(0, (2, 2)), linewidth=.9, zorder=1)
    axis_b.text(
        16,
        1.025,
        r"$B=16$ trusted evaluations",
        transform=axis_b.get_xaxis_transform(),
        color=BLACK,
        fontsize=14.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        clip_on=False,
    )
    set_budget_axis(axis_b)
    axis_b.set_xlim(7, 290)
    axis_b.set_ylim(0, 1.035)
    axis_b.set_ylabel(r"certification recall $\Pr(R_\Delta\leq B)$", fontsize=16)
    axis_b.xaxis.label.set_size(16)
    axis_b.tick_params(axis="both", labelsize=14)
    light_grid(axis_b, "y")
    handles = [
        Line2D([0], [0], color=LANDSCAPE_COLORS[name], marker=MARKERS[name], label=LANDSCAPE_LABELS[name])
        for name in AUDIT_ORDER
    ]
    handles.append(Line2D([0], [0], color="#8a8a8a", linestyle=(0, (4, 2)), label="exact random null"))
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(.5, .015),
        fontsize=13,
        ncol=5,
        handlelength=2.1,
        columnspacing=1.15,
        handletextpad=.5,
        frameon=False,
    )

    panel_label(axis_a, "a", x=-.25, y=1.11, fontsize=22)
    panel_label(axis_b, "b", x=-.20, y=1.11, fontsize=22)
    figure.subplots_adjust(left=.085, right=.99, top=.84, bottom=.24)
    save(figure, "fig2_audit.png")


def plot_figure_3(frames: dict[str, pd.DataFrame]) -> None:
    removal = frames["fig3_regret_removal"]
    gain = frames["fig3_gain_over_random"].set_index("landscape")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.18, 1], "wspace": .38})
    for landscape in LANDSCAPES:
        part = removal[removal.landscape == landscape]
        axes[0].fill_between(
            BUDGETS, part["low"].tolist(), part["high"].tolist(),
            color=LANDSCAPE_COLORS[landscape], alpha=.10, linewidth=0, zorder=1,
        )
        axes[0].plot(
            BUDGETS,
            part["mean"].tolist(),
            color=LANDSCAPE_COLORS[landscape],
            marker="o",
            markeredgecolor="white",
            markeredgewidth=.5,
            zorder=3,
        )
    axes[0].axhline(0, color=TEXT_GREY, ls="--", lw=1.1)
    axes[0].text(9, .035, "non-degradation guarantee", color=TEXT_GREY, ha="left", va="bottom", fontsize=10)
    set_budget_axis(axes[0])
    axes[0].set_ylabel(r"mean $L_B/G_{\rm verifier}$")
    axes[0].set_ylim(-.03, 1.0)
    light_grid(axes[0], "y")

    for row, landscape in enumerate(LANDSCAPES[::-1]):
        values = gain.loc[landscape]
        axes[1].errorbar(
            values.mean_difference,
            row,
            xerr=[[values.mean_difference - values.ci95_low], [values.ci95_high - values.mean_difference]],
            fmt="o",
            color=LANDSCAPE_COLORS[landscape],
            capsize=3,
            lw=1.2,
        )
    for row in range(4):
        if row % 2 == 0:
            axes[1].axhspan(row - .45, row + .45, color="#f5f5f5", zorder=0)
    axes[1].axvline(0, color=TEXT_GREY, ls="--", lw=1.1)
    axes[1].set_yticks(range(4), [LANDSCAPE_LABELS[name] for name in LANDSCAPES[::-1]])
    axes[1].set_xlabel("gain in selection-regret removal\n(Top-$B$ over random, $B=256$)")
    axes[1].set_xlim(-.08, .71)
    axes[1].set_ylim(-.55, 3.55)
    axes[1].annotate(
        "no difference", xy=(0, 3.35), xytext=(5, 0), textcoords="offset points",
        color=TEXT_GREY, ha="left", va="center", fontsize=10, fontweight="bold"
    )
    light_grid(axes[1], "x")

    panel_label(axes[0], "a", x=-.20)
    panel_label(axes[1], "b", x=-.34)
    handles = [
        Line2D([0], [0], color=LANDSCAPE_COLORS[name], marker="o", label=LANDSCAPE_LABELS[name])
        for name in LANDSCAPES
    ]
    bottom_legend(figure, handles, [handle.get_label() for handle in handles], 4)
    figure.subplots_adjust(left=.11, right=.985, top=.92, bottom=.32)
    save(figure, "fig3_regret_removal.png")


def read_data(names: Iterable[str]) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(DATA / f"{name}.csv", float_precision="round_trip") for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-results", action="store_true", help="rebuild the Figure 2 data from the audits")
    arguments = parser.parse_args()
    if arguments.from_results:
        for name, frame in figure_2_data().items():
            frame.to_csv(DATA / f"{name}.csv", index=False)
    configure_style()
    plot_figure_2(read_data(["fig2_audit_ranks", "fig2_audit_recall"]))
    plot_figure_3(read_data(["fig3_regret_removal", "fig3_gain_over_random"]))


if __name__ == "__main__":
    main()
