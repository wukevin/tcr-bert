"""
Custom metrics
"""
import os
import functools
import json
import subprocess, shlex
import logging
import collections
import itertools
import tempfile
from typing import *

import numpy as np
import pandas as pd
from sklearn import metrics, mixture
from scipy import stats, spatial
import anndata as ad
from Bio import PDB
import logomaker

import data_loader as dl
import featurization as ft
import muscle
import utils


def get_chain_bfactors(fname: str, chain_key: str = "H") -> pd.DataFrame:
    """
    Read in pdb file, getting the bfactors for the given chain
    """
    assert fname.endswith(".pdb")
    parser = PDB.PDBParser()
    structure = parser.get_structure("foo", fname)
    model = structure[0]
    logging.info(f"{fname} contains chains {list(model.get_chains())}")
    chain = model[chain_key]  # P, H, G, A, B for 5m00

    residues_df = pd.DataFrame(
        0, index=[], columns=["residue", "bfactor_mean", "bfactor_sd"]
    )
    for i, residue in enumerate(chain.get_residues()):
        res = residue.resname
        if res not in ft.AA_TRIPLET_TO_SINGLE:
            logging.warning(f"Skipping unrecognized AA triplet: {res}")
            continue
        bfactors = np.array([atom.get_bfactor() for atom in residue.get_atoms()])
        residues_df.loc[i] = (
            ft.AA_TRIPLET_TO_SINGLE[res],
            np.mean(bfactors),
            np.std(bfactors),
        )
    return residues_df


def min_matrix_dist(x: np.ndarray, y: np.ndarray) -> float:
    """Return the minimum distance between any two points in input point matrices"""
    assert len(x.shape) == len(y.shape) == 2
    assert x.shape[-1] == y.shape[-1]
    dists = np.min(metrics.pairwise_distances_argmin_min(x, y)[1])
    return np.min(dists)


def distance_to_antigen(
    fname: str, antigen_chain: str = "P"
) -> Dict[str, pd.DataFrame]:
    """Calculate the distance to antigen"""
    assert fname.endswith(".pdb")
    parser = PDB.PDBParser()
    structure = parser.get_structure("foo", fname)
    # From TCRpMHCmodels, expect that the antigen is "P"
    # Other chains include A (TRA), B (TRB), M (MHC)

    model = structure[0]
    antigen = model[antigen_chain]

    # Get the antigen points
    antigen_points, antigen_residues = [], []
    for residue in antigen.get_residues():
        res = residue.resname
        if res not in ft.AA_TRIPLET_TO_SINGLE:
            logging.warning(
                f"Skipping unrecognized AA triplet in antigen chain {antigen_chain}: {res}"
            )
            continue
        antigen_residues.append(ft.AA_TRIPLET_TO_SINGLE[res])
        antigen_points.extend([atom.get_coord() for atom in residue.get_atoms()])
    logging.info(f"Loaded antigen {''.join(antigen_residues)}")
    antigen_points = np.stack(antigen_points)

    # Walk through the chains and calculate minimum distance to each
    retval = {}
    for chain in model.get_chains():
        if chain.id == antigen_chain:
            continue
        # Collect points in this chain
        chain_dists = pd.DataFrame(0, index=[], columns=["residue", "min_dist"])
        for i, residue in enumerate(chain.get_residues()):
            res = residue.resname
            if res not in ft.AA_TRIPLET_TO_SINGLE:
                logging.warning(f"Skipping unrecognized AA triplet: {res}")
                continue
            residue_points = np.stack(
                [atom.get_coord() for atom in residue.get_atoms()]
            )
            # Distance b/w each residue and antigen
            d = min_matrix_dist(residue_points, antigen_points)
            chain_dists.loc[i] = ft.AA_TRIPLET_TO_SINGLE[res], d
        retval[chain.id] = chain_dists
    return retval


def get_chain_to_coords(fname: str, average: bool = True) -> Dict[str, pd.DataFrame]:
    """
    For each chain in the pdb fname, return the coords of the chain's amino acids
    Returns a dataframe mapping chain name to a dataframe of residue and coord columns
    https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    """
    assert fname.endswith(".pdb")
    parser = PDB.PDBParser()
    structure = parser.get_structure("foo", fname)

    model = structure[0]
    retval = {}
    for chain in model.get_chains():
        chain_coords = pd.DataFrame(
            0,
            index=[0],
            columns=["residue", "atoms", "x_coord", "y_coord", "z_coord"],
            dtype=float if average else object,
        )
        for i, residue in enumerate(chain.get_residues()):
            res = residue.resname
            if res not in ft.AA_TRIPLET_TO_SINGLE:
                logging.warning(f"Skipping unrecognized AA triplet: {res}")
                continue
            residue_points = np.array(
                [atom.get_coord() for atom in residue.get_atoms()]
            )
            residue_atoms = "-".join([atom.element for atom in residue.get_atoms()])
            if average:
                residue_coord = np.mean(residue_points, axis=0).tolist()
                chain_coords.loc[i] = (
                    ft.AA_TRIPLET_TO_SINGLE[res],
                    residue_atoms,
                    *residue_coord,
                )
            else:
                # tolist gets each row, and we rearrange to be columns
                x_coords, y_coords, z_coords = list(zip(*residue_points.tolist()))
                chain_coords.loc[i] = (
                    ft.AA_TRIPLET_TO_SINGLE[res],
                    residue_atoms,
                    x_coords,
                    y_coords,
                    z_coords,
                )
        retval[chain.id] = chain_coords
    return retval


def per_position_aa_count(
    sequences: Sequence[str], *, normalize: bool = False, psuedocount: int = 0,
) -> pd.DataFrame:
    """
    Return a count matrix of (seq_len, n_amino_acids) reflecting
    counts of each amino acid at each base
    """
    seq_mat = np.stack([np.array(list(x)) for x in sequences])
    num_seq, fixed_len = seq_mat.shape
    assert num_seq == len(sequences)
    per_pos_counts = []
    for j in range(fixed_len):
        col = seq_mat[:, j]
        assert len(col) == num_seq
        counter = collections.Counter(col)
        count_vec = np.array([counter[aa] for aa in ft.AMINO_ACIDS])
        count_vec += psuedocount
        per_pos_counts.append(count_vec)
    per_pos_counts = np.stack(per_pos_counts)
    assert per_pos_counts.shape == (seq_mat.shape[1], len(ft.AMINO_ACIDS))
    # assert np.allclose(np.sum(per_pos_counts, 1), np.sum(per_pos_counts[0]))
    if normalize:
        row_sums = set(per_pos_counts.sum(axis=1).tolist())
        assert len(row_sums) == 1
        per_pos_counts = per_pos_counts.astype(np.float64)
        per_pos_counts /= row_sums.pop()
        assert np.all(np.isclose(per_pos_counts.sum(axis=1), 1.0))
    retval = pd.DataFrame(per_pos_counts, columns=list(ft.AMINO_ACIDS))
    return retval


def positional_entropy(sequences: Sequence[str]) -> np.ndarray:
    """
    Return the per-position entropy
    """
    per_pos_counts = per_position_aa_count(sequences)
    retval = stats.entropy(per_pos_counts, base=2, axis=1)
    return retval


def motif_from_sequences(
    sequences: Iterable[str], normalize_pwm: bool = True, dedup: bool = False, **kwargs
) -> Tuple[pd.DataFrame, logomaker.Logo]:
    """
    Computes a motif for the sequences by running sequences through
    MUSCLE and visualizing using logomaker
    kwargs are forwarded to logomaker

    Returns both the per-position counts and the logo
    """
    if dedup:
        sequences = utils.dedup(sequences)
    msa_seqs = muscle.run_muscle(sequences)
    msa_pwm = per_position_aa_count(msa_seqs, normalize=normalize_pwm)
    logo = logomaker.Logo(msa_pwm, **kwargs)
    return msa_pwm, logo


def pairwise_dist_by_group(
    adata: ad.AnnData,
    groupby: str,
    reduction: Callable = np.mean,
    excl_self_comparison: bool = True,
) -> pd.DataFrame:
    """
    Pairwise distance between elements in groupby
    """
    levels = utils.dedup(adata.obs[groupby])
    retval = pd.DataFrame(index=levels, columns=levels)
    for (i, j) in itertools.product(levels, levels):
        i_idx = np.where(adata.obs[groupby] == i)[0]
        j_idx = np.where(adata.obs[groupby] == j)[0]
        dists = metrics.pairwise_distances(adata.X[i_idx], adata.X[j_idx], n_jobs=-1)
        if i == j and excl_self_comparison:
            dists = dists[~np.eye(dists.shape[0], dtype=bool)].flatten()
        retval.loc[i, j] = reduction(dists.flatten())
    return retval


def find_centroids(data, n: int) -> List[int]:
    """
    Find n centroids in the data via group labels
    Do this by running a GMM, finding the centers,
    and returning the indices of the points closest to those centers.
    """
    gmm = mixture.GaussianMixture(n_components=n)
    gmm.fit(data)
    means = gmm.means_
    centroid_indices = []
    for mean in means:
        distances = metrics.pairwise.euclidean_distances(
            mean.reshape(1, -1), data
        ).squeeze()
        min_idx = np.argmin(distances)
        centroid_indices.append(min_idx)
    return centroid_indices


def percent_and_correct_clustered(
    truth_mapping: Dict[str, Any],
    clusters: Iterable[Collection[str]],
    min_cluster_size: int = 3,
) -> Tuple[float, float]:
    """
    Returns percent clustered, and percent correctly clustered
    Assumes that truth mapping contains all sequences in question
    and thus will be used as "denominator" for percentage clustered

    Takes in
    - a mapping from antigen -> label
    - cluster assignments in the form [(x, y, z), (a, b), (c)]
      where each group inside list is a cluster

    >>> percent_and_correct_clustered(
    ...     {
    ...         "x": "foo",
    ...         "y": "foo",
    ...         "z": "foo",
    ...     },
    ...     [("x", "y", "z")],
    ... )
    (1.0, 1.0)

    >>> percent_and_correct_clustered(
    ...     {
    ...         "x": "foo",
    ...         "y": "foo",
    ...         "z": "foo",
    ...         "a": "foo",
    ...     },
    ...     [("x", "y", "z")],
    ... )
    (0.75, 1.0)

    >>> percent_and_correct_clustered(
    ...     {
    ...         "x": "foo",
    ...         "y": "foo",
    ...         "z": "foo",
    ...         "a": "foo",
    ...     },
    ...     [("x", "y")],  # Cluster too small
    ... )
    (0.0, nan)

    >>> percent_and_correct_clustered(
    ...     {
    ...         "w": "foo",
    ...         "x": "foo",
    ...         "y": "foo",
    ...         "z": "foo",
    ...         "a": "bar",
    ...         "b": "bar",
    ...         "c": "bar",
    ...         "d": "bar",
    ...     },
    ...     [('x', 'y', 'a', 'w'), ('b', 'c', 'z', 'd')],
    ... )
    (1.0, 0.75)

    >>> percent_and_correct_clustered(  # Test for no dominant label
    ...     {
    ...         "w": "foo",
    ...         "x": "bar",
    ...         "y": "baz",
    ...         "z": "fu",
    ...         "a": "b4r",
    ...         "b": "b4z",
    ...         "c": "f00",
    ...         "d": "f11",
    ...     },
    ...     [('x', 'y', 'a', 'w'), ('b', 'c', 'z', 'd')],
    ... )
    (1.0, 0.0)
    """
    correct_clustered, total_clustered = 0, 0
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        cluster_truths = [truth_mapping[i] for i in cluster]
        cnt = collections.Counter(cluster_truths)
        _most_common, most_common_count = cnt.most_common(1)[0]
        assert most_common_count <= len(cluster)
        if most_common_count / len(cluster) >= 0.5:
            correct_clustered += most_common_count
        total_clustered += len(cluster)
    if total_clustered > 0:
        perc_correct = correct_clustered / total_clustered
    else:  # Avoid zero division error
        perc_correct = np.nan
    perc_cluster = total_clustered / len(truth_mapping)
    return perc_cluster, perc_correct


def tukey_outlier_cutoffs(
    x: np.ndarray, k: int = 3, direction: str = "higher"
) -> Tuple[float, float]:
    """
    Uses tukey method to return the outliers in x
    https://en.wikipedia.org/wiki/Outlier
    Given quarties Q1, Q2 (median), and Q3, outlier cutoffs are
    [Q1 - k(Q3-Q1), Q3 + k(Q3-Q1)]
    Values of k are typically 1.5 for "outlier" and 3 for "far out"

    >>> tukey_outlier_cutoffs(np.array(list(range(10)) + [1000]))
    (-12.5, 22.5)
    """
    if direction not in ("higher", "lower", "both"):
        raise ValueError(f"Unrecognized direction: {direction}")
    q1, q3 = np.percentile(x, [25, 75])  # Q1 and Q3
    iqr = stats.iqr(x)
    assert np.isclose(q3 - q1, iqr)
    bottom_cutoff = q1 - k * iqr
    top_cutoff = q3 + k * iqr
    # idx = np.logical_or(x < bottom_cutoff, x > top_cutoff)
    # return x[idx]
    return bottom_cutoff, top_cutoff


@functools.cache
def load_blosum(
    fname: str = os.path.join(dl.LOCAL_DATA_DIR, "blosum62.json")
) -> pd.DataFrame:
    """Return the blosum matrix as a dataframe"""
    with open(fname) as source:
        d = json.load(source)
        retval = pd.DataFrame(d)
    retval = pd.DataFrame(0, index=list(ft.AMINO_ACIDS), columns=list(ft.AMINO_ACIDS))
    for x, y in itertools.product(retval.index, retval.columns):
        if x == "U" or y == "U":
            continue
        retval.loc[x, y] = d[x][y]
    retval.drop(index="U", inplace=True)
    retval.drop(columns="U", inplace=True)
    return retval


def auc_score_dual_vectors(y_true, y_false, curve: str = "auroc", **kwargs) -> float:
    """
    Compute the ROC AUC score but instead of having a vector of y_true and y_pred
    this takes as input two vectors of preds corrresponding to true and false
    """
    labels = np.array([1] * len(y_true) + [0] * len(y_false))
    preds = np.concatenate([y_true, y_false])
    if curve == "auroc":
        return metrics.roc_auc_score(labels, preds, **kwargs)
    elif curve == "auprc":
        return metrics.average_precision_score(labels, preds, **kwargs)
    else:
        raise ValueError(f"Unrecognized curve: {curve}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    # aa_counts, logo = motif_from_sequences(["CASSA", "CASSRR"])
    # print(logo)
    # demo_file = "/home/wukevin/projects/tcr/tcr/data/stcrdab/raw/5m00.pdb"
    # print(get_chain_to_coords(demo_file, average=False)["B"])
