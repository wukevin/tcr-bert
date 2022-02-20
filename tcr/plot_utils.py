from typing import *
import collections
import logging

import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics as metrics
from scipy import stats
from adjustText import adjust_text

import featurization as ft
import custom_metrics
import utils

SAVEFIG_DPI = 300


def plot_sequence_diversity(
    sequences: Sequence[str],
    title: str = "Sequence diversity",
    xlabel: str = "Position in sequence",
    fname: str = "",
):
    """
    Plot sequence diversity
    """
    fixed_len = set([len(s) for s in sequences])
    assert len(fixed_len) == 1
    fixed_len = fixed_len.pop()
    per_pos_counts = custom_metrics.per_position_aa_count(sequences)

    fig, ax = plt.subplots(dpi=SAVEFIG_DPI)
    bottom = np.zeros(fixed_len)
    for i in range(len(ft.AMINO_ACIDS)):
        ax.bar(
            np.arange(fixed_len),
            per_pos_counts.values[:, i],
            bottom=bottom,
            label=ft.AMINO_ACIDS[i],
        )
        bottom += per_pos_counts.values[:, i]
    ax.set_xticks(np.arange(fixed_len))
    ax.set_xticklabels(np.arange(fixed_len) + 1)
    ax.set(title=title, xlabel=xlabel)
    if fname:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_auroc(
    truth,
    preds,
    title_prefix: str = "Receiver operating characteristic",
    ax=None,
    fname: str = "",
    label_prefix: str = "",
    **kwargs,
):
    """
    Plot AUROC after flattening inputs
    """
    truth = utils.ensure_arr(truth).flatten()
    preds = utils.ensure_arr(preds).flatten()
    fpr, tpr, _thresholds = metrics.roc_curve(truth, preds)
    auc = metrics.auc(fpr, tpr)
    logging.info(f"AUROC of {auc:.4f}")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(dpi=SAVEFIG_DPI, figsize=(7, 5))
    if label_prefix:
        ax.plot(fpr, tpr, label=label_prefix + f" (AUROC={auc:.3f})", **kwargs)
    else:
        ax.plot(fpr, tpr, **kwargs)
    ax.set(
        xlim=(-0.01, 1.0),
        ylim=(0, 1.03),
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    if fig is not None:
        ax.set(title=f"{title_prefix} (n={len(truth)}, AUROC={auc:.3f})")
    if fname:
        assert fig is not None
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_auprc(
    truth,
    preds,
    title_prefix: str = "Precision recall curve",
    ax=None,
    fname: str = "",
    label_prefix: str = "",
    **kwargs,
):
    """Plot AUPRC"""
    truth = utils.ensure_arr(truth).flatten()
    preds = utils.ensure_arr(preds).flatten()

    precision, recall, _thresholds = metrics.precision_recall_curve(truth, preds)
    average_precision = metrics.average_precision_score(truth, preds)
    logging.info(f"AUPRC of {average_precision:.4f}")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(dpi=SAVEFIG_DPI, figsize=(7, 5))
    if label_prefix:
        ax.plot(
            recall,
            precision,
            label=label_prefix + f" (AUPRC={average_precision:.3f})",
            **kwargs,
        )
    else:
        ax.plot(recall, precision, **kwargs)
    ax.set(
        xlabel="Recall", ylabel="Precision",
    )
    if fig is not None:
        ax.set(title=f"{title_prefix} (n={len(truth)}, AUPRC={average_precision:.3f})")
    if fname:
        assert fig is not None
        fig.savefig(fname, bbox_inches="tight")

    return fig


def plot_residue_atoms(
    atoms: str,
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    z_coords: Sequence[float],
    ax,
    prev_terminal: Optional[Tuple[float, float, float]] = None,
    alpha: float = 0.9,
    label: Optional[str] = None,
    annot_atoms: bool = False,
    scatter_atoms_size: Optional[Sequence[float]] = None,
    **kwargs,
) -> Tuple[float, float, float]:
    """
    Plot each atom in the residue
    Automatically extracts the "backbone" from the sidechain
    Returns the terminal "O" coords to interact with the next aa
    """
    assert atoms.startswith("N-C-C-O")  # Backbone
    if prev_terminal is not None:  # Connec to previous if previous is given
        ax.plot3D(
            [prev_terminal[0], x_coords[0]],
            [prev_terminal[1], y_coords[0]],
            [prev_terminal[2], z_coords[0]],
            alpha=0.5 * alpha,
            **kwargs,
        )
    # Plot backbone
    ax.plot3D(
        x_coords[:4],
        y_coords[:4],
        z_coords[:4],
        alpha=0.6 * alpha,
        label=label,  # Only label here to avoid duplicated labels
        **kwargs,
    )
    # Plot sidechain
    ax.plot3D(
        [x_coords[1]] + list(x_coords[4:]),
        [y_coords[1]] + list(y_coords[4:]),
        [z_coords[1]] + list(z_coords[4:]),
        alpha=alpha,
        lw=2,
        **kwargs,
    )
    if scatter_atoms_size:
        ax.scatter3D(
            [x_coords[1]] + list(x_coords[4:]),
            [y_coords[1]] + list(y_coords[4:]),
            [z_coords[1]] + list(z_coords[4:]),
            alpha=alpha,
            s=scatter_atoms_size,
            **kwargs,
        )
    # Label each atom
    if annot_atoms:
        for atom, x, y, z in zip(atoms.split("-"), x_coords, y_coords, z_coords):
            ax.text(x, y, z, atom)
    return x_coords[3], y_coords[3], z_coords[3]


def plot_chain_atoms(
    chain: pd.DataFrame, color, ax, s: Optional[Sequence[float]] = None, **kwargs
):
    """
    Takes as input, the output dataframe from get_chain_to_coords
    """
    prev_last_coord = None
    for i, (_i, row) in enumerate(chain.iterrows()):
        curr_size = s[i] if s is not None else None
        prev_last_coord = plot_residue_atoms(
            row["atoms"],
            row["x_coord"],
            row["y_coord"],
            row["z_coord"],
            ax,
            annot_atoms=False,
            prev_terminal=prev_last_coord,
            label=row["residue"],
            color=color,
            scatter_atoms_size=curr_size,
            **kwargs,
        )


def plot_anndata_rep(
    a: AnnData,
    color: str,
    representation: str = "umap",
    representation_axes_label: str = "",
    swap_axes: bool = False,
    cmap: Callable = plt.get_cmap("tab20"),
    direct_label: bool = True,
    adjust: bool = False,
    ax_tick: bool = False,
    legend_size: Optional[int] = None,
    figsize: Tuple[float, float] = (6.4, 4.8),
    fname: str = "",
    **kwargs,
):
    """
    Plot the given adata's representation, directly labelling instead of using
    a legend
    """
    rep_key = "X_" + representation
    assert (
        rep_key in a.obsm
    ), f"Representation {representation} not fount in keys {a.obsm.keys()}"

    coords = a.obsm[rep_key]
    if swap_axes:
        coords = coords[:, ::-1]  # Reverse the columns
    assert isinstance(coords, np.ndarray) and len(coords.shape) == 2
    assert coords.shape[0] == a.n_obs
    assert color in a.obs
    color_vals = a.obs[color]
    unique_val = np.unique(color_vals.values)
    color_idx = [sorted(list(unique_val)).index(i) for i in color_vals]
    # Vector of colors for each point
    color_vec = [cmap(i) for i in color_idx]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.scatter(
        coords[:, 0], coords[:, 1], s=12000 / coords.shape[0], c=color_vec, alpha=0.9
    )

    if direct_label:
        # Label each cluster
        texts = []
        for v in unique_val:
            v_idx = np.where(color_vals.values == v)
            # Median behaves better with outliers than mean
            v_coords = np.median(coords[v_idx], axis=0)
            t = ax.text(
                *v_coords,
                v,
                horizontalalignment="center",
                verticalalignment="center",
                size=legend_size,
            )
            texts.append(t)
        if adjust:
            adjust_text(
                texts,
                only_move={"texts": "y"},
                force_text=0.01,
                autoalign="y",
                avoid_points=False,
            )
    else:
        patches = []
        for i, val in enumerate(unique_val):
            p = mpatches.Patch(color=cmap(i), label=val)
            patches.append(p)
        ax.legend(handles=patches, prop={"size": legend_size})

    rep_str = representation_axes_label if representation_axes_label else representation
    if not swap_axes:
        ax.set(
            xlabel=f"{rep_str.upper()}1", ylabel=f"{rep_str.upper()}2",
        )
    else:
        ax.set(
            xlabel=f"{rep_str.upper()}2", ylabel=f"{rep_str.upper()}1",
        )
    ax.set(**kwargs)
    if not ax_tick:
        ax.set(xticks=[], yticks=[])

    if fname:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_perf_over_params(
    bot_dfs: Dict[str, pd.DataFrame],
    top_dfs: Iterable[pd.DataFrame],
    metric: str,
    bot_label="GLIPH global covergence distance cutoff",
    top_label="TCR-BERT Leiden resolution",
    bot_reverse: bool = False,
    top_reverse: bool = True,
    fname: Optional[str] = None,
    rand_line: Optional[float] = None,
    **kwargs,
):
    """
    Dual-axis (on x) line plots comparing performance (yaxis) across several
    parameter configs (shown on x). Originally written for comparing TCR-BERT
    and GLIPH performance on clustering.
    """

    def is_logarithmic(vals) -> bool:
        """Returns true of the vals are probably logarithmic"""
        lin_r = stats.linregress(np.arange(len(vals)), vals).rvalue
        log_r = stats.linregress(np.arange(len(vals)), np.log(vals)).rvalue
        return log_r > lin_r

    # Create a mapping from plotted metrics to readable label
    label_readable = {
        "perc_clustered": "Percent clustered",
        "perc_correct": "Percent correctly clustered",
    }
    markers = ["*", "<", "H"]

    fig, ax1 = plt.subplots(dpi=300)
    # Plot the bottom axis
    for i, (k, df) in enumerate(bot_dfs.items()):
        ax1.plot(
            df.index,
            df[metric],
            marker=markers[i],
            color="tab:orange",
            label=k,
            alpha=0.6,
        )
        if is_logarithmic(df.index):
            ax1.set_xscale("log", base=2)
    if len(bot_dfs) > 1:
        ax1.legend()

    ax1.set_xlabel(bot_label, color="tab:orange")
    ax1.tick_params(axis="x", colors="tab:orange")
    if bot_reverse:
        ax1.invert_xaxis()
    ax1.set(**kwargs, ylabel=label_readable[metric])  # Set axes properties

    ax2 = ax1.twiny()
    for i, (k, df) in enumerate(top_dfs.items()):
        ax2.plot(
            df.index,
            df[metric],
            label=k,
            color="tab:blue",
            marker=markers[i],
            alpha=0.6,
        )
        if is_logarithmic(df.index):
            ax2.set_xscale("log", base=2)
    if len(top_dfs) > 1:
        ax2.legend()

    ax2.set_xlabel(top_label, color="tab:blue")
    ax2.tick_params(axis="x", colors="tab:blue")
    if top_reverse:
        ax2.invert_xaxis()

    # Plot a random line
    if rand_line:
        ax2.axhline(
            rand_line, label="Random", color="tab:grey", alpha=0.5, linestyle="--"
        )

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig

