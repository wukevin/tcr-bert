"""
Code to perform MCMC sampling of sequences
"""
import os
import json
import logging
from joblib import load
from typing import *

import numpy as np
import sklearn

from torch.utils.data import Dataset
import skorch

import custom_metrics
import featurization as ft
import model_utils

MCMC_RNG = np.random.default_rng(seed=12947)


def sample_sequence(
    n: int,
    p: Optional[np.ndarray] = None,
    *,
    blacklist_aa: Optional[Collection[str]] = None,
) -> str:
    """
    Return a amino acid chain of length n, sampled according to per-position
    probability vector p
    """
    vocab = list(ft.AMINO_ACIDS)
    if p is None:
        p = np.array([np.ones(len(vocab)) / len(vocab) for _ in range(n)])
    assert p.shape == (n, len(vocab))
    # Remove blacklisted amino acids if given
    if blacklist_aa:
        for aa in blacklist_aa:
            assert aa in vocab
            aa_idx = vocab.index(aa)
            p[:, aa_idx] = 0
        p = (p.T / p.sum(axis=1)).T  # Renormalize
    # Generate the sequence
    retval = "".join([MCMC_RNG.choice(vocab, p=p_vec) for p_vec in p])
    assert len(retval) == n
    return retval


def sample_sequence_mlm(
    seed_seqs: Sequence[str],
    n_seqs: int,
    n_mutations: int = 2,
    model: str = "wukevin/tcr-bert-mlm-only",
    *,
    blacklist_aa: Optional[Collection[str]] = None,
    device: int = 0,
) -> List[str]:
    """
    Returns a list of n_seqs amino acids that are each mutations of a sequeunce
    present in seed_seqs. Each of these amino acid chains undergoes n_mutations cycles of mutation
    using the MLM model given to sample from the landscape
    of naturally occurring TCRs
    """
    mlm_pipeline = model_utils.load_fill_mask_pipeline(model, device=device)
    # Choose an amino acid to modify and generate modifications
    retval = []
    for _i in range(n_seqs):
        # Create a copy of the starting sequence that we then modify
        this_seq = MCMC_RNG.choice(seed_seqs)
        for _j in range(n_mutations):  # Apply series of mutations
            # Choose an index mutate
            # TODO consider using a better approach here than random?
            # TODO consider not having every new sequence being mutated? or mutated so deeply
            # Consider a sequence that is really good that id discovered in an earlier iteration
            # It will be mutated into something possibly worse if every iteration everytning mutates
            idx = MCMC_RNG.integers(len(this_seq))
            # Build input of sequence and a mask
            masked_input = this_seq[:idx] + ft.MASK + this_seq[idx + 1 :]
            # returns the top 5 predictions and corresponding probs
            mlm_preds_dict = mlm_pipeline(ft.insert_whitespace(masked_input))
            mlm_residues = [d["token_str"] for d in mlm_preds_dict]
            mlm_probs = np.array([d["score"] for d in mlm_preds_dict])
            # If there are blacklisted amino acids, remove them
            if blacklist_aa:
                whitelist_idx = np.array(
                    [i for i, aa in enumerate(mlm_residues) if aa not in blacklist_aa]
                )
                mlm_residues = [mlm_residues[i] for i in whitelist_idx]
                mlm_probs = mlm_probs[whitelist_idx]
            # Normalize probabilities and sample
            mlm_probs_norm = mlm_probs / np.sum(mlm_probs)  # Normalize
            res = MCMC_RNG.choice(mlm_residues, p=mlm_probs_norm)
            # Updates the current sequence with the mutated res
            this_seq = this_seq[:idx] + res + this_seq[idx + 1 :]
        retval.append(this_seq)
    assert len(retval) == n_seqs
    return retval


def generate_random_sequences(
    seq_pairs: Collection[Tuple[str, str]],
    *,
    method: str = "pwm",
    n: int = 256,
    **kwargs,
) -> List[Tuple[str, str]]:
    """
    Given some sequence pairs (TRA, TRB), generate n random sequence pairs
    based on these sequences
    """
    if method == "pwm":
        tra_sequences, trb_sequences = zip(*seq_pairs)
        tra_per_pos_freq = custom_metrics.per_position_aa_count(
            tra_sequences, psuedocount=0, normalize=True
        )
        tras_sampled = [
            sample_sequence(len(tra_per_pos_freq), tra_per_pos_freq.values, **kwargs)
            for _ in range(n)
        ]
        trb_per_pos_freq = custom_metrics.per_position_aa_count(
            trb_sequences, psuedocount=0, normalize=True
        )
        trbs_sampled = [
            sample_sequence(len(trb_per_pos_freq), trb_per_pos_freq.values, **kwargs)
            for _ in range(n)
        ]
        return list(zip(tras_sampled, trbs_sampled))

    elif method == "mlm":
        # Use masked language modelling to generate the new sequences
        # using the seed sequences as a sstarting point
        tra_sequences, trb_sequences = zip(*seq_pairs)
        tras_sampled = sample_sequence_mlm(
            tra_sequences, n_seqs=n, n_mutations=2, **kwargs
        )
        trbs_sampled = sample_sequence_mlm(
            trb_sequences, n_seqs=n, n_mutations=2, **kwargs
        )
        return list(zip(tras_sampled, trbs_sampled))

    else:
        raise ValueError(f"Unrecognized method: {method}")


def generate_random_sequences_single(
    seqs: Collection[str], *, method="pwm", n: int = 256, **kwargs,
) -> List[str]:
    """
    Given some sequences (e.g. TRBs), generate n random sequence pairs based on these
    This is the single sequence analog of the above function to generate for pairs
    """
    if method == "pwm":
        per_pos_freq = custom_metrics.per_position_aa_count(
            seqs, psuedocount=0, normalize=True
        )
        seqs_sampled = [
            sample_sequence(len(per_pos_freq), per_pos_freq.values, **kwargs)
            for _ in range(n)
        ]
        return seqs_sampled
    else:
        raise ValueError(f"Unrecognized method: {method}")


def generate_binding_sequences_using_nsp(
    transformer: str,
    seed_seqs: List[Tuple[str, str]],
    n_iter: int = 10,
    sample_ratio: float = 0.5,
    min_prob: Optional[float] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Run n_iter to generate sequences that maximize binding
    if min_prob is given, will stop early after the worst of the top
    sequences has at least min_prob predicted binding probability
    If this never occurs, run all n_iter iterations
    **kwargs are passed on to generate_random_sequences

    sample_ratio determines the proportion of top binders used
    as seeds for the next round of generation

    Return the predicted top p-values over iterations, as well as
    the corresponding sequences
    """
    if seed is not None:
        global MCMC_RNG
        MCMC_RNG = np.random.default_rng(seed=seed)
    sequences = seed_seqs.copy()
    sample_n = int(round(len(seed_seqs) * sample_ratio))

    per_iteration_pvalues = []
    per_iteration_best_seqs = []
    for _ in range(n_iter + 1):  # +1 bc the first round is really just seeding
        # Predict the binding for each
        _labels, preds = model_utils.get_transformer_nsp_preds(
            transformer, sequences, as_probs=True, device=3,
        )
        largest_idx = np.argsort(preds[:, 1])[-sample_n:]
        per_iteration_pvalues.append(preds[largest_idx, 1])
        good_seqs = [sequences[i] for i in largest_idx]
        # The first entry corresponds to top 50 of starting
        per_iteration_best_seqs.append(good_seqs)
        # If min_prob is given, check if we should break early
        if min_prob is not None and np.all(per_iteration_pvalues[-1] >= min_prob):
            logging.info(
                f"Stopping generation, min predicted prob {np.min(per_iteration_pvalues[-1])} exceeds floor of {min_prob}"
            )
            break
        # The last iteration has this "discarded" i.e. not used
        sequences = generate_random_sequences(good_seqs, n=100, **kwargs)
    return np.array(per_iteration_pvalues), per_iteration_best_seqs


def generate_binding_sequences_skorch(
    net: skorch.NeuralNet,
    seed_seqs: List[Tuple[str, str]],
    dset_obj: Dataset,
    dset_kwargs: Dict[str, Any],
    n_iter: int = 10,
    sample_ratio=0.5,
    min_prob: Optional[float] = None,
    seed: Optional[int] = None,
    neg_baseline: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, List[List[Tuple[str, str]]]]:
    """
    Run n_iterate to generate sequences that maximize binding
    if min_prob is given, stop early after the lowest of the top
    sequences has at least the given binding

    Compared to other functions, this attempts to be more general
    by accepting a skorch neuralnet rather than a dir to load from

    If neg_baseline is True, then optimize for NONbinding instead of binding

    **kwargs passed to generate_random_sequences

    Returns:
    - matrix of (num_iters, num_seqs) shape with predicted probability of
      sequences at each iteration
    - List of length num_iters, each entry is a list of TRA/TRB pairs
    """
    if seed is not None:
        global MCMC_RNG
        MCMC_RNG = np.random.default_rng(seed)
    sequences = seed_seqs.copy()
    sample_n = int(round(len(seed_seqs) * sample_ratio))

    per_iteration_pvals, per_iteration_best_seqs = [], []
    for _ in range(n_iter + 1):
        # Create a dataset for the current sequences
        # This is somewhat inelegant but since we don't do this for many iterations
        # it should be fine.
        tra, trb = zip(*sequences)
        this_iter_dset = dset_obj(tra, trb, np.zeros(len(sequences)), **dset_kwargs)
        # generate predictions and rank the sequences
        preds = net.predict_proba(this_iter_dset)[:, 0 if neg_baseline else 1]
        largest_idx = np.argsort(preds)[-sample_n:]
        per_iteration_pvals.append(preds[largest_idx])
        good_seqs = [sequences[i] for i in largest_idx]
        per_iteration_best_seqs.append(good_seqs)
        if min_prob is not None and np.all(per_iteration_pvals[-1] >= min_prob):
            logging.info(
                f"Stopping generation, min predicted prob exceeds floor of {min_prob}"
            )
            break

        sequences = generate_random_sequences(good_seqs, n=len(seed_seqs), **kwargs)
    return np.array(per_iteration_pvals), per_iteration_best_seqs


def generate_binding_sequences_using_pcasvm(
    pca_svm_dir: str,
    seed_seqs: List[str],
    n_iter: int = 10,
    sample_ratio=0.5,
    seed: Optional[int] = None,
    **kwargs,
):
    """
    Run n_iter to generate sequences that maximize binding
    Instead of using NSP to jointly generate TRA/TRB, we use the PCA-SVM
    to generate just TRBs
    **kwargs are passed to generate_random_sequences
    """
    if seed is not None:
        global MCMC_RNG
        MCMC_RNG = np.random.default_rng(seed=seed)
    sequences = seed_seqs.copy()
    sample_n = int(round(len(seed_seqs) * sample_ratio))

    # Additional logic to sanity check and load in the PCA-SVM model
    with open(os.path.join(pca_svm_dir, "metadata.json")) as source:
        metadict = json.load(source)
    assert metadict["sklearn"] == sklearn.__version__
    assert os.path.isdir(metadict["transformer"])
    cls = load(os.path.join(pca_svm_dir, "pcasvm.sklearn"))

    per_iteration_pvalues = []
    per_iteration_best_seqs = []
    for _ in range(n_iter + 1):  # +1 bc the first round is really just seeding
        # Predict the binding for each
        embeds = model_utils.get_transformer_embeddings(
            model_dir=metadict["transformer"],
            seqs=sequences,
            layers=[-7],
            method="mean",
            device=3,
        )
        preds = cls.predict_proba(embeds)
        largest_idx = np.argsort(preds[:, 1])[-sample_n:]
        per_iteration_pvalues.append(preds[largest_idx, 1])
        good_seqs = [sequences[i] for i in largest_idx]
        # The first entry corresponds to top 50 of starting
        per_iteration_best_seqs.append(good_seqs)
        # The last iteration has this "discarded" i.e. not used
        sequences = generate_random_sequences_single(
            good_seqs, n=len(sequences), **kwargs
        )
    return np.array(per_iteration_pvalues), per_iteration_best_seqs


if __name__ == "__main__":
    print(sample_sequence_mlm("CASSRKDESYF", 5))
