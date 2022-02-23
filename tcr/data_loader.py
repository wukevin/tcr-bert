"""
Code for loading data
"""

import os, sys
import shutil
import argparse
import functools
import multiprocessing
import gzip
import inspect
import glob
import json
import itertools
import collections
import logging
from typing import *

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import Levenshtein

import featurization as ft
import utils

LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
assert os.path.isdir(LOCAL_DATA_DIR)

EXTERNAL_EVAL_DIR = os.path.join(os.path.dirname(LOCAL_DATA_DIR), "external_eval")
assert os.path.join(EXTERNAL_EVAL_DIR)

# Names of datasets
DATASET_NAMES = {"LCMV", "VDJdb", "PIRD", "TCRdb"}


logging.basicConfig(level=logging.INFO)


class TcrABSupervisedIdxDataset(Dataset):
    """Dataset that returns TcrAB and label"""

    def __init__(
        self,
        tcr_table: pd.DataFrame,
        label_col: str = "tetramer",
        pos_labels: Collection[str] = ["TetMid", "TetPos"],
        idx_encode: bool = False,
        max_a_len: Optional[int] = None,
        max_b_len: Optional[int] = None,
        disambiguate_labels: bool = True,
    ):
        self.table = tcr_table
        self.label_col = label_col

        if disambiguate_labels:
            logging.info("Deduping and removing examples with conflicting labels")
            lcmv_dedup_ab, self.labels = dedup_lcmv_table(tcr_table)
            self.tras, self.trbs = zip(*lcmv_dedup_ab)
        else:
            raise NotImplementedError(
                "Running withough disambiguating labels causes duplicated and conflicting labels! This was the prior behavior, but is now deprecated"
            )

        tcr_a_lengths = [len(self.get_ith_tcr_a(i)) for i in range(len(self))]
        tcr_b_lengths = [len(self.get_ith_tcr_b(i)) for i in range(len(self))]
        self.max_a_len = max(tcr_a_lengths) if max_a_len is None else max_a_len
        self.max_b_len = max(tcr_b_lengths) if max_b_len is None else max_b_len
        self.idx_encode = idx_encode
        logging.info(f"Maximum TCR A/B lengths: {self.max_a_len} {self.max_b_len}")

        self.pos_labels = pos_labels
        logging.info(f"Positive {label_col} labels: {pos_labels}")

    def __len__(self) -> int:
        return len(self.labels)

    def get_ith_tcr_a(self, idx: int, pad: bool = False) -> str:
        """Gets the ith TRA sequence"""
        seq = self.tras[idx]
        if pad:
            seq = ft.pad_or_trunc_sequence(seq, self.max_a_len, right_align=False)
        return seq

    def get_ith_tcr_b(self, idx: int, pad: bool = False) -> str:
        """Gets the ith TRB sequence"""
        seq = self.trbs[idx]
        if pad:
            seq = ft.pad_or_trunc_sequence(seq, self.max_b_len, right_align=False)
        return seq

    def get_ith_sequence(self, idx: int) -> Tuple[str, str]:
        """Get the ith TRA/TRB pair"""
        return self.tras[idx], self.trbs[idx]

    def get_ith_label(self, idx: int, idx_encode: Optional[bool] = None) -> np.ndarray:
        """Get the ith label"""
        label = self.labels[idx]
        retval = float(np.any([l in label for l in self.pos_labels]))
        retval = np.array([1.0 - retval, retval], dtype=np.float32)
        idx_encode = self.idx_encode if idx_encode is None else idx_encode
        if idx_encode:
            retval = np.where(retval)[0]
        return retval

    def __getitem__(self, idx: int):
        tcr_a_idx = ft.idx_encode(self.get_ith_tcr_a(idx, pad=True))
        tcr_b_idx = ft.idx_encode(self.get_ith_tcr_b(idx, pad=True))

        label = self.get_ith_label(idx)
        return (
            {
                "tcr_a": torch.from_numpy(tcr_a_idx),
                "tcr_b": torch.from_numpy(tcr_b_idx),
            },
            torch.from_numpy(label).type(torch.long).squeeze(),
        )


class TcrABSupervisedOneHotDataset(TcrABSupervisedIdxDataset):
    """Dataset that encodes tcrAB as one hot encoded vectors"""

    def __getitem__(self, idx: int):
        tcr_a_idx = ft.one_hot(self.get_ith_tcr_a(idx, pad=True))
        tcr_b_idx = ft.one_hot(self.get_ith_tcr_b(idx, pad=True))

        label = self.get_ith_label(idx)
        return (
            {
                "tcr_a": torch.from_numpy(tcr_a_idx),
                "tcr_b": torch.from_numpy(tcr_b_idx),
            },
            torch.from_numpy(label).type(torch.long).squeeze(),
        )


class TCRSupervisedIdxDataset(Dataset):
    """Dataset meant for either TRA or TRB supervised learning"""

    def __init__(
        self,
        tcrs: Sequence[str],
        labels: Sequence[bool],
        idx_encode_labels: bool = True,
        max_len: Optional[int] = None,
    ):
        self.tcrs = tcrs
        self.labels = labels
        assert len(self.tcrs) == len(self.labels)

        self.max_len = max_len  # Defaults to None
        determined_max_len = max([len(t) for t in tcrs])
        if self.max_len is not None:
            # If a max_len is explicitly given, check that it is greater than the actual max len
            assert isinstance(self.max_len, int)
            assert determined_max_len <= self.max_len
            logging.info(
                f"Given max_len of {self.max_len} exceeds (as expected) empirical max_len of {determined_max_len}"
            )
        else:
            # If max_len is not given, directly set the max_len
            logging.info(
                f"Max len not set, using empirical max len of {determined_max_len}"
            )
            self.max_len = determined_max_len

        logging.info(f"Using maximum length of {self.max_len}")
        self.idx_encode_labels = idx_encode_labels

    def all_labels(self) -> Sequence[bool]:
        """Return all labels"""
        return self.labels

    def __len__(self) -> int:
        return len(self.tcrs)

    def get_ith_tcr(self, idx: int, pad: bool = True) -> str:
        """Returns the ith tcr sequence, padded with null residues"""
        retval = self.tcrs[idx]
        if pad:
            retval = ft.pad_or_trunc_sequence(retval, self.max_len, right_align=False)
        return retval

    def get_ith_sequence(self, idx: int) -> str:
        return self.tcrs[idx]

    def get_ith_label(self, idx: int) -> np.ndarray:
        retval = float(self.labels[idx])
        if not self.idx_encode_labels:
            retval = np.array([1.0 - retval, retval], dtype=np.float32)
        return np.atleast_1d(retval)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        tcr_idx = ft.idx_encode(self.get_ith_tcr(idx, pad=True))
        label = self.get_ith_label(idx)
        return (
            {"seq": torch.from_numpy(tcr_idx)},
            torch.from_numpy(label).type(torch.long).squeeze(),
        )


class TcrSelfSupervisedDataset(TcrABSupervisedIdxDataset):
    """
    Mostly for compatibility with transformers library
    LineByLineTextDataset returns a dict of "input_ids" -> input_ids
    """

    # Reference: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/data/datasets/language_modeling.py
    def __init__(self, tcr_seqs: Iterable[str], tokenizer, round_len: bool = True):
        self.tcr_seqs = utils.dedup(tcr_seqs)
        logging.info(
            f"Creating self supervised dataset with {len(self.tcr_seqs)} sequences"
        )
        self.max_len = max([len(s) for s in self.tcr_seqs])
        logging.info(f"Maximum sequence length: {self.max_len}")
        if round_len:
            self.max_len = int(utils.min_power_greater_than(self.max_len, 2))
            logging.info(f"Rounded maximum length to {self.max_len}")
        self.tokenizer = tokenizer
        self._has_logged_example = False

    def __len__(self) -> int:
        return len(self.tcr_seqs)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        tcr = self.tcr_seqs[i]
        retval = self.tokenizer.encode(ft.insert_whitespace(tcr))
        if not self._has_logged_example:
            logging.info(f"Example of tokenized input: {tcr} -> {retval}")
            self._has_logged_example = True
        return {"input_ids": torch.tensor(retval, dtype=torch.long)}

    def merge(self, other):
        """Merge this dataset with the other dataset"""
        all_tcrs = utils.dedup(self.tcr_seqs + other.tcr_seqs)
        logging.info(
            f"Merged two self-supervised datasets of sizes {len(self)} {len(other)} for dataset of {len(all_tcrs)}"
        )
        return TcrSelfSupervisedDataset(all_tcrs)


class TcrNextSentenceDataset(Dataset):
    """
    Dataset for next sentence prediction. Input is two lists of pairwise
    corresponding TRA TRB sequences
    Note that the labelling scheme here is (False, True)
    This DIFFERS from the convention used in the transformers library for NSP

    Note that TRA/TRB naming convention is somewhat of a minomoer - in reality, these are
    just first/second pairs.

    This also supports generating NEGATIVE examples dynamically. This is automatically
    enabled when this is wrapped in a DatasetSplit object with training split. This
    may yield improved sampling of the negative manifold and yield a more general model
    """

    def __init__(
        self,
        tra_seqs: List[str],
        trb_seqs: List[str],
        neg_ratio: float = 1.0,
        labels: Optional[Iterable[bool]] = None,
        tra_blacklist: Optional[Iterable[str]] = None,
        mlm: float = 0.0,
        max_len: int = 64,
        seed: int = 4242,
        remove_null: bool = True,
        shuffle: bool = True,
    ):
        assert len(tra_seqs) == len(trb_seqs)
        # Remove cases of nan
        logging.info(f"Build NSP dataset with {len(tra_seqs)} pairs")
        if remove_null:
            bad_idx_a = [
                i
                for i, seq in enumerate(tra_seqs)
                if seq is None or pd.isnull(seq) or seq == ""
            ]
            bad_idx_b = [
                i
                for i, seq in enumerate(trb_seqs)
                if seq is None or pd.isnull(seq) or seq == ""
            ]
            bad_idx = set(bad_idx_a).union(bad_idx_b)
            logging.info(
                f"Removing {len(bad_idx)} bad pairs: {len(bad_idx_a)} union {len(bad_idx_b)}"
            )
            tra_seqs = [a for i, a in enumerate(tra_seqs) if i not in bad_idx]
            trb_seqs = [a for i, a in enumerate(trb_seqs) if i not in bad_idx]
        if tra_blacklist is not None:
            bad_idx = [i for i, seq in enumerate(tra_seqs) if seq in set(tra_blacklist)]
            logging.info(f"Removing {len(bad_idx)} blacklisted items")
            tra_seqs = [a for i, a in enumerate(tra_seqs) if i not in bad_idx]
            trb_seqs = [a for i, a in enumerate(trb_seqs) if i not in bad_idx]
        logging.info(f"Building NSP datset with {len(tra_seqs)} pairs after filtering")
        # Insert whitespace as we store the sequences
        # Whitespace separated inputs is expected by tokenizer
        # These are never shuffled, regardless of the shuffle param
        self.tra = [ft.insert_whitespace(aa) for aa in tra_seqs]
        self.trb = [ft.insert_whitespace(aa) for aa in trb_seqs]
        assert 0.0 <= mlm <= 1.0
        self.mlm = mlm

        self.neg_ratio = neg_ratio
        self.rng = np.random.default_rng(seed=seed)
        if self.neg_ratio > 0:
            assert labels is None, "Cannot sample negatives if labels are given"
            pos_pairs = list(zip(self.tra, self.trb))
            num_negs = int(round(len(pos_pairs) * neg_ratio))
            logging.info(f"Sampling {num_negs} negatives")
            neg_pairs = [self.__sample_negative() for _i in range(num_negs)]
            logging.info(f"Positive pairs: {len(pos_pairs)}")
            logging.info(f"Sampled negative pairs: {len(neg_pairs)}")
            # WARNING in tokenizers convention, output is (True, False)
            # This means that a correct pair is a "0" and a wrong pair is a "1"
            # we DO NOT adhere to this convention, rather using a conventional labelling
            self.labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
            self.all_pairs = pos_pairs + neg_pairs
        elif labels is not None:
            logging.info(f"Taking given labels with {np.mean(labels)} positive rate")
            self.labels = labels
            self.all_pairs = list(zip(self.tra, self.trb))
        else:
            # raise RuntimeError("Must provide either neg_ratio or labels argument")
            logging.warn(
                "No labels or negative ratio provided, defaulting to all negative labels"
            )
            self.all_pairs = list(zip(self.tra, self.trb))
            self.labels = np.array([0.0] * len(self.all_pairs))
        assert len(self.labels) == len(self.all_pairs)

        self.max_len = max_len
        max_len_actual = max(
            max([len(aa.split()) for aa in self.tra]),
            max([len(aa.split()) for aa in self.trb]),
        )
        logging.info(f"Maximum length of NSP single sequence: {max_len_actual}")
        self.tok = ft.get_aa_bert_tokenizer(max_len=max_len_actual)
        # Shuffle the examples
        if shuffle:
            logging.info("Shuffling NSP dataset")
            shuf_idx = np.arange(len(self.labels))
            self.rng.shuffle(shuf_idx)
            self.labels = self.labels[shuf_idx]  # Contains whether this is a valid pair
            self.all_pairs = [self.all_pairs[i] for i in shuf_idx]

        logging.info(
            f"NSP dataset of {len(self.all_pairs)} pairs, {np.sum(self.labels)} positive examples"
        )
        logging.info(f"Example training example")
        for k, v in self[0].items():
            logging.info(f"{k}: {v}")

    def __sample_negative(self) -> Tuple[str, str]:
        """
        Generate a negative example
        """
        if self.neg_ratio <= 0.0:
            raise RuntimeError("Cannot sample negatives for labelled dataset")
        i, j = self.rng.integers(len(self.tra), size=2)
        while self.tra[i] == self.tra[j]:  # Is not a valid pair
            j = self.rng.integers(len(self.tra))
        return self.tra[i], self.trb[j]

    def __len__(self) -> int:
        assert len(self.labels) == len(self.all_pairs)
        return len(self.labels)

    def get_ith_label(self, idx):
        return self.labels[idx]

    def get_ith_sequence(self, idx) -> Tuple[str, str]:
        return self.all_pairs[idx]

    def __getitem__(self, idx: int, dynamic: bool = False) -> Dict[str, torch.Tensor]:
        """
        dynamic is a general flag for generating examples dynamically
        """
        label = self.labels[idx]
        label_tensor = torch.LongTensor(np.atleast_1d(label))
        if dynamic and label == 0:
            # Dynamically generate a negative example
            pair = self.__sample_negative()
        else:  # Positive example OR not dynamic
            pair = self.all_pairs[idx]
        if self.mlm > 0.0:
            # Mask out each sequence BEFORE we pad/concatenate them
            # This ensures that the mask is always an amino acid
            mlm_targets, pair = zip(*[ft.mask_for_training(a) for a in pair])
            t = np.atleast_1d(-100).astype(np.int64)
            # CLS seq1 SEP seq2 SEP
            mlm_targets_combined = np.concatenate(
                [t, mlm_targets[0], t, mlm_targets[1], t]
            )
            mlm_targets_padded = torch.LongTensor(
                np.pad(
                    mlm_targets_combined,
                    (0, self.max_len - len(mlm_targets_combined)),
                    mode="constant",
                    constant_values=-100,
                )
            )

        enc = self.tok(
            text=pair[0],
            text_pair=pair[1],
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        # Default tokenization has (batch, ...) as first dim
        # Since __getitem__ only gets a single example, remove this
        enc = {k: v.squeeze() for k, v in enc.items()}
        if self.mlm > 0.0:  # NSP + MLM
            assert (
                mlm_targets_padded.size() == enc["input_ids"].size()
            ), f"Mismatched sizes {mlm_targets_padded.size()} {enc['input_ids'].size()}"
            enc["next_sentence_label"] = label_tensor
            enc["labels"] = torch.LongTensor(mlm_targets_padded)
        else:  # NSP only
            enc["labels"] = label_tensor
        return enc

    def get_all_items(self) -> Dict[str, torch.Tensor]:
        """
        Get all the data instead of individual entries
        """
        collector = collections.defaultdict(list)
        for i in range(len(self)):
            x = self[i]
            for k, v in x.items():
                collector[k].append(v.reshape(1, -1))
        retval = {k: torch.cat(v, dim=0) for k, v in collector.items()}
        return retval


class TcrFineTuneSingleDataset(TcrSelfSupervisedDataset):
    """Dataset for fine tuning from only TRA or TRB sequences"""

    def __init__(
        self,
        aa: Sequence[str],
        labels: MutableSequence[float],
        label_continuous: bool = False,
        label_labels: Optional[Sequence[str]] = None,
        drop_rare_labels: bool = True,
    ):
        assert len(aa) == len(
            labels
        ), f"Got differing lengths for aa and labels: {len(aa)}, {len(labels)}"
        self.aa = [ft.insert_whitespace(item) for item in aa]
        self.tokenizer = ft.get_aa_bert_tokenizer(64)

        self.continuous = label_continuous
        label_dtype = np.float32 if self.continuous else np.int64
        self.labels = np.array(labels, dtype=label_dtype).squeeze()
        assert len(self.labels) == len(self.aa)
        self.label_labels = label_labels
        if self.continuous:
            assert self.label_labels is None

        if drop_rare_labels and not self.continuous and not self.is_multilabel:
            # Get the mean positive rate for each label
            labels_expanded = np.zeros((len(labels), np.max(labels) + 1))
            labels_expanded[np.arange(len(labels)), self.labels] = 1
            per_label_prop = np.mean(labels_expanded, axis=0)
            # Find the labels with high enough positive rate
            good_idx = np.where(per_label_prop >= 1e-3)[0]
            if len(good_idx) < labels_expanded.shape[1]:
                logging.info(
                    f"Retaining {len(good_idx)}/{labels_expanded.shape[1]} labels with sufficient examples"
                )
                # Reconstruct labels based only on retained good_idx
                # nonzero returns indices of element that are nonzero
                self.labels = np.array(
                    [
                        np.nonzero(good_idx == label)[0][0]
                        if label in good_idx
                        else len(good_idx)  # "other" labels
                        for label in self.labels
                    ],
                    dtype=label_dtype,
                )
                assert np.max(self.labels) == len(good_idx)
                # Subset label labels
                self.label_labels = [self.label_labels[i] for i in good_idx] + ["other"]
                assert len(self.label_labels) == len(good_idx) + 1

    @property
    def is_multilabel(self) -> bool:
        """Return True if labels represent multilabel classification"""
        return len(self.labels.shape) > 1

    def get_ith_sequence(self, idx: int) -> str:
        """Get the ith sequence"""
        return self.aa[idx]

    def get_ith_label(self, idx: int) -> np.ndarray:
        """Gets the ith label"""
        return np.atleast_1d(self.labels[idx])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        label = torch.tensor(self.get_ith_label(idx))
        if self.is_multilabel:
            # Multilabel -> BCEWithLogitsLoss which wants float target
            label = label.float()
        # We already inserted whitespaces in init
        enc = self.tokenizer(
            self.aa[idx], padding="max_length", max_length=64, return_tensors="pt"
        )
        enc = {k: v.squeeze() for k, v in enc.items()}
        enc["labels"] = label
        return enc


class TcrFineTuneDataset(TcrSelfSupervisedDataset):
    """Can supply tokenizer to work with ESM"""

    def __init__(
        self,
        tcr_a_seqs: Sequence[str],
        tcr_b_seqs: Sequence[str],
        labels: Optional[np.ndarray] = None,
        label_continuous: bool = False,
        tokenizer: Optional[Callable] = None,
        skorch_mode: bool = True,
        idx_encode: bool = False,
    ):
        assert len(tcr_a_seqs) == len(tcr_b_seqs)
        self.tcr_a = list(tcr_a_seqs)
        self.tcr_b = list(tcr_b_seqs)
        self.max_len = max([len(s) for s in self.tcr_a + self.tcr_b]) + 2

        if tokenizer is None:
            tokenizer = ft.get_aa_bert_tokenizer(self.max_len)
            self.tcr_a_tokenized = [
                tokenizer.encode(
                    ft.insert_whitespace(aa),
                    padding="max_length",
                    max_length=self.max_len,
                )
                for aa in self.tcr_a
            ]
            self.tcr_b_tokenized = [
                tokenizer.encode(
                    ft.insert_whitespace(aa),
                    padding="max_length",
                    max_length=self.max_len,
                )
                for aa in self.tcr_b
            ]
        else:
            logging.info(f"Using pre-supplied tokenizer: {tokenizer}")
            _label, _seq, self.tcr_a_tokenized = tokenizer(list(enumerate(self.tcr_a)))
            _label, _seq, self.tcr_b_tokenized = tokenizer(list(enumerate(self.tcr_b)))

        if labels is not None:
            assert len(labels) == len(tcr_a_seqs)
            self.labels = np.atleast_1d(labels.squeeze())
        else:
            logging.warning(
                "Labels not given, defaulting to False labels (DO NOT USE FOR TRAINING)"
            )
            self.labels = None
        self.continuous = label_continuous
        self.skorch_mode = skorch_mode
        self.idx_encode = idx_encode

    def get_ith_sequence(self, idx: int) -> Tuple[str, str]:
        """Get the ith TRA/TRB pair"""
        return self.tcr_a[idx], self.tcr_b[idx]

    def get_ith_label(self, idx: int, idx_encode: Optional[bool] = None) -> np.ndarray:
        """Get the ith label"""
        if self.labels is None:
            return np.array([0])  # Dummy value
        if not self.continuous:
            label = self.labels[idx]
            if not isinstance(label, np.ndarray):
                label = np.atleast_1d(label)
            if self.skorch_mode and len(label) == 1:
                label = np.array([1.0 - label, label]).squeeze()
            # Take given value if supplied, else default to self.idx_encode
            idx_encode = self.idx_encode if idx_encode is None else idx_encode
            if idx_encode:
                label = np.where(label)[0]
            return label
        else:
            # For the continuous case we simply return the ith value(s)
            return self.labels[idx]

    def __len__(self) -> int:
        return len(self.tcr_a)

    def __getitem__(
        self, idx: int
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        label_dtype = torch.float if self.continuous else torch.long
        tcr_a = self.tcr_a_tokenized[idx]
        tcr_b = self.tcr_b_tokenized[idx]
        label = self.get_ith_label(idx)
        if not self.skorch_mode:
            retval = {
                "tcr_a": utils.ensure_tensor(tcr_a, dtype=torch.long),
                "tcr_b": utils.ensure_tensor(tcr_b, dtype=torch.long),
                "labels": utils.ensure_tensor(label, dtype=label_dtype),
            }
        else:
            model_inputs = {
                "tcr_a": utils.ensure_tensor(tcr_a, dtype=torch.long),
                "tcr_b": utils.ensure_tensor(tcr_b, dtype=torch.long),
            }
            retval = (model_inputs, torch.tensor(label, dtype=label_dtype).squeeze())
        return retval


class DatasetSplit(Dataset):
    """
    Dataset split. Thin wrapper on top a dataset to provide data split functionality.
    Can also enable dynamic example generation for train fold if supported by
    the wrapped dataset (NOT for valid/test folds) via dynamic_training flag

    kwargs are forwarded to shuffle_indices_train_valid_test
    """

    def __init__(
        self,
        full_dataset: Dataset,
        split: str,
        dynamic_training: bool = False,
        **kwargs,
    ):
        self.dset = full_dataset
        split_to_idx = {"train": 0, "valid": 1, "test": 2}
        assert split in split_to_idx
        self.split = split
        self.dynamic = dynamic_training
        if self.split != "train":
            assert not self.dynamic, "Cannot have dynamic examples for valid/test"
        self.idx = shuffle_indices_train_valid_test(
            np.arange(len(self.dset)), **kwargs
        )[split_to_idx[self.split]]
        logging.info(f"Split {self.split} with {len(self)} examples")

    def all_labels(self, **kwargs) -> np.ndarray:
        """Get all labels"""
        if not hasattr(self.dset, "get_ith_label"):
            raise NotImplementedError("Wrapped dataset must implement get_ith_label")
        labels = [
            self.dset.get_ith_label(self.idx[i], **kwargs) for i in range(len(self))
        ]
        return np.stack(labels)

    def all_sequences(self, **kwargs) -> Union[List[str], List[Tuple[str, str]]]:
        """Get all sequences"""
        if not hasattr(self.dset, "get_ith_sequence"):
            raise NotImplementedError(
                f"Wrapped dataset {type(self.dset)} must implement get_ith_sequence"
            )
        # get_ith_sequence could return a str or a tuple of two str (TRA/TRB)
        sequences = [
            self.dset.get_ith_sequence(self.idx[i], **kwargs) for i in range(len(self))
        ]
        return sequences

    def to_file(self, fname: str, compress: bool = True) -> str:
        """
        Write to the given file
        """
        if not (
            hasattr(self.dset, "get_ith_label")
            and hasattr(self.dset, "get_ith_sequence")
        ):
            raise NotImplementedError(
                "Wrapped dataset must implement both get_ith_label & get_ith_sequence"
            )
        assert fname.endswith(".json")
        all_examples = []
        for idx in range(len(self)):
            seq = self.dset.get_ith_sequence(self.idx[idx])
            label_list = self.dset.get_ith_label(self.idx[idx]).tolist()
            all_examples.append((seq, label_list))

        with open(fname, "w") as sink:
            json.dump(all_examples, sink, indent=4)

        if compress:
            with open(fname, "rb") as source:
                with gzip.open(fname + ".gz", "wb") as sink:
                    shutil.copyfileobj(source, sink)
            os.remove(fname)
            fname += ".gz"
        assert os.path.isfile(fname)
        return os.path.abspath(fname)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, idx: int):
        if (
            self.dynamic
            and self.split == "train"
            and "dynamic" in inspect.getfullargspec(self.dset.__getitem__).args
        ):
            return self.dset.__getitem__(self.idx[idx], dynamic=True)
        return self.dset.__getitem__(self.idx[idx])


class DatasetSplitByAttribute(DatasetSplit):
    """
    Dataset split. Thin wrapper on top of a datset to provide data split functionality.
    Unlike the above, which is a purely random split, this splits by a given attribute.
    attr_getter function should take the dataset and return a list of attrs to split by
    """

    def __init__(
        self,
        full_datset: Dataset,
        attr_getter: Callable,
        split: str,
        dynamic_training: bool = False,
        valid: float = 0.15,
        test: float = 0.15,
        seed: int = 1234,
    ):
        self.dset = full_datset
        self.dynamic = dynamic_training
        self.split = split
        self.split_attr = attr_getter(self.dset)
        assert len(self.split_attr) == len(self.dset)

        # Get the unique attrs and count occurrences of each
        split_attr_counts = collections.Counter(self.split_attr)
        assert (
            len(split_attr_counts) >= 2
        ), f"Must have at least two classes of attribute to split, but got {len(split_attr_counts)}"
        # Sort the attrs by most counts to least
        _, self.train_attrs = zip(
            *sorted(
                [(count, attr) for attr, count in split_attr_counts.items()],
                reverse=True,
            )
        )
        self.train_attrs = list(self.train_attrs)

        # Build valid, then test sets, by greedily taking the largest groups
        # until we have at least the required number of examples
        valid_n, test_n = len(self.dset) * valid, len(self.dset) * test
        self.valid_attrs, self.test_attrs = [], []
        while sum([split_attr_counts[a] for a in self.valid_attrs]) < valid_n:
            # Take the biggest item in the list
            self.valid_attrs.append(self.train_attrs.pop(0))
        while sum([split_attr_counts[a] for a in self.test_attrs]) < test_n:
            # Take the biggest item in the list
            self.test_attrs.append(self.train_attrs.pop(0))

        train_idx = np.array(
            [
                i
                for i, attr in enumerate(self.split_attr)
                if attr in set(self.train_attrs)
            ]
        )
        valid_idx = np.array(
            [
                i
                for i, attr in enumerate(self.split_attr)
                if attr in set(self.valid_attrs)
            ]
        )
        test_idx = np.array(
            [
                i
                for i, attr in enumerate(self.split_attr)
                if attr in set(self.test_attrs)
            ]
        )
        assert len(train_idx) + len(valid_idx) + len(test_idx) == len(self.dset)
        logging.info(
            f"Train split with {len(train_idx)} examples across {len(self.train_attrs)} attrs"
        )
        logging.info(
            f"Valid split with {len(valid_idx)} examples across {len(self.valid_attrs)} attrs"
        )
        logging.info(
            f"Test split with {len(test_idx)} examples across {len(self.test_attrs)} attrs"
        )
        rng = np.random.default_rng(seed)
        rng.shuffle(train_idx)
        rng.shuffle(valid_idx)
        rng.shuffle(test_idx)

        self.idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}[split]


class DownsampledDataset(Dataset):
    """
    Downsampled and shuffled dataset. Useful for evaluating impact of having less data.
    Downsampling is done to a *fixed* subset of the original dataset
    """

    def __init__(self, dset: Dataset, downsample: float = 0.1, seed: int = 3939):
        assert 0.0 < downsample < 1.0
        self.dset = dset
        self.downsample = downsample
        self.idx = np.arange(len(self.dset))
        np.random.seed(seed)
        np.random.shuffle(self.idx)
        self.idx = self.idx[: int(np.round(downsample * len(self.dset)))]
        logging.info(f"Downsampled from {len(self.dset)} -> {len(self)} samples")

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, idx: int):
        return self.dset[self.idx[idx]]


def shuffle_indices_train_valid_test(
    idx: np.ndarray, valid: float = 0.15, test: float = 0.15, seed: int = 1234
) -> Tuple[np.ndarray]:
    """
    Given an array of indices, return indices partitioned into train, valid, and test indices
    The following tests ensure that ordering is consistent across different calls
    >>> np.all(shuffle_indices_train_valid_test(np.arange(100))[0] == shuffle_indices_train_valid_test(np.arange(100))[0])
    True
    >>> np.all(shuffle_indices_train_valid_test(np.arange(10000))[1] == shuffle_indices_train_valid_test(np.arange(10000))[1])
    True
    >>> np.all(shuffle_indices_train_valid_test(np.arange(20000))[2] == shuffle_indices_train_valid_test(np.arange(20000))[2])
    True
    >>> np.all(shuffle_indices_train_valid_test(np.arange(1000), 0.1, 0.1)[1] == shuffle_indices_train_valid_test(np.arange(1000), 0.1, 0.1)[1])
    True
    """
    np.random.seed(seed)  # For reproducible subsampling
    indices = np.copy(idx)  # Make a copy because shuffling occurs in place
    np.random.shuffle(indices)  # Shuffles inplace
    num_valid = int(round(len(indices) * valid)) if valid > 0 else 0
    num_test = int(round(len(indices) * test)) if test > 0 else 0
    num_train = len(indices) - num_valid - num_test
    assert num_train > 0 and num_valid >= 0 and num_test >= 0
    assert num_train + num_valid + num_test == len(
        indices
    ), f"Got mismatched counts: {num_train} + {num_valid} + {num_test} != {len(indices)}"

    indices_train = indices[:num_train]
    indices_valid = indices[num_train : num_train + num_valid]
    indices_test = indices[-num_test:]
    assert indices_train.size + indices_valid.size + indices_test.size == len(idx)

    return indices_train, indices_valid, indices_test


def split_arr(
    arr: Union[np.ndarray, pd.DataFrame, list, tuple],
    split: Literal["train", "valid", "test"],
    **kwargs,
) -> Union[np.ndarray, pd.DataFrame, list]:
    """
    Randomly split the array into the given split
    kwargs are fed to shuffle_indices_train_valid_test
    """
    split_to_idx = {"train": 0, "valid": 1, "test": 2}
    assert split in split_to_idx, f"Unrecognized split: {split}"

    n = len(arr) if isinstance(arr, (list, tuple)) else arr.shape[0]
    indices = np.arange(n)
    keep_idx = shuffle_indices_train_valid_test(indices, **kwargs)[split_to_idx[split]]
    if isinstance(arr, pd.DataFrame):
        return arr.iloc[keep_idx]
    if isinstance(arr, (list, tuple)):
        return [arr[i] for i in keep_idx]
    return arr[keep_idx]


def sample_unlabelled_tcrdb_trb(
    n: int, blacklist: Optional[Collection[str]] = None, seed: int = 6464
) -> List[str]:
    """
    Convenience function to sample the given number of TRBs from TCRdb
    Blacklist can be given to exclude certain sequences from sampling

    The following tests ensure reproducibility
    >>> all([a == b for a, b in zip(sample_unlabelled_tcrdb_trb(10), sample_unlabelled_tcrdb_trb(10))])
    True
    """
    tcrdb = load_tcrdb()
    rng = np.random.default_rng(seed=seed)
    if blacklist is None:
        # Maintain separate code paths for back compatibility
        idx = rng.choice(np.arange(len(tcrdb)), size=n, replace=False)
    else:
        # Removing whitespace has no effect if there was no whitespace to start
        blacklist_set = set([ft.remove_whitespace(b) for b in blacklist])
        # Oversample in case of overlap with blacklist
        idx = rng.choice(np.arange(len(tcrdb)), size=n + len(blacklist), replace=False)
        # Filter out blacklist overlaps and take the first n
        idx = [i for i in idx if tcrdb.iloc[i]["AASeq"] not in blacklist_set][:n]

    assert len(idx) == n
    retval = [tcrdb.iloc[i]["AASeq"] for i in idx]
    if blacklist:  # No overlap
        assert not set(retval).intersection(blacklist)
    return retval


def load_lcmv_vdj(
    vdj_fname: str = os.path.join(LOCAL_DATA_DIR, "lcmv_tcr_vdj_unsplit.txt.gz")
) -> Dict[str, Dict[str, Dict[str, Tuple[str, str, str]]]]:
    """
    Load the vdj table and return it in a 3-level dictionary
    identifier -> TRA/TRB -> cdr3 sequence -> (v, d, j)
    {
        tcr_cdr3s_aa_identifier : {
            "TRA": {
                cdr3_sequence: (v, d, j),
                ...
            }
            "TRB": ...
        }
    }
    v d or j may be None if not provided
    """
    check_none: Callable[
        [str], Union[str, None]
    ] = lambda x: None if x.lower() == "none" or not x else x
    df = pd.read_csv(vdj_fname, delimiter="\t", low_memory=False)
    retval = collections.defaultdict(lambda: {"TRA": dict(), "TRB": dict()})
    for i, row in df.iterrows():
        k = row["tcr_cdr3s_aa"]
        retval[k][row["chain"]][row["cdr3"]] = (
            check_none(row["v_gene"]),
            check_none(row["d_gene"]),
            check_none(row["j_gene"]),
        )
    return retval

@functools.lru_cache(maxsize=16)
def load_lcmv_table(
    fname: str = os.path.join(LOCAL_DATA_DIR, "lcmv_tetramer_tcr.txt"),
    metadata_fname: str = os.path.join(LOCAL_DATA_DIR, "lcmv_all_metadata.txt.gz"),
    vdj_fname: str = os.path.join(LOCAL_DATA_DIR, "lcmv_tcr_vdj_unsplit.txt.gz"),
    drop_na: bool = True,
    drop_unsorted: bool = True,
) -> pd.DataFrame:
    """Load the LCMV data table"""
    table = pd.read_csv(fname, delimiter="\t")
    logging.info(f"Loaded in table of {len(table)} entries")
    if drop_na:
        table.dropna(axis=0, how="any", subset=["tetramer", "TRB", "TRA"], inplace=True)
        logging.info(f"{len(table)} entries remain after dropping na")
    if drop_unsorted:
        drop_idx = table.index[table["tetramer"] == "Unsorted"]
        table.drop(index=drop_idx, inplace=True)
        logging.info(f"{len(table)} entries remain after dropping unsorted")

    # Take entires with multiple TRA or TRB sequences and split them, carrying over
    # all of the other metadata to each row
    dedup_rows = []
    for _i, row in table.iterrows():
        # For this row, determine nucleotide sequences
        tcr_nt = collect_tra_trb(row["tcr_cdr3s_nt"])
        # The nucleotide and the protein sequences should match up correctly
        tcr_aa_combos = list(
            itertools.product(row["TRA"].split(";"), row["TRB"].split(";"))
        )
        tcr_nt_combos = list(itertools.product(tcr_nt["TRA"], tcr_nt["TRB"]))
        assert len(tcr_aa_combos) == len(tcr_nt_combos)

        for ((tra_aa, trb_aa), (tra_nt, trb_nt)) in zip(tcr_aa_combos, tcr_nt_combos):
            new_row = row.copy(deep=True)
            # Check that nucleotide and protein sequences match up
            assert utils.nt2aa(tra_nt) == tra_aa
            assert utils.nt2aa(trb_nt) == trb_aa
            new_row["TRA"] = tra_aa
            new_row["TRB"] = trb_aa
            new_row["TRA_nt"] = tra_nt
            new_row["TRB_nt"] = trb_nt
            dedup_rows.append(new_row)

    dedup_table = pd.DataFrame(dedup_rows)
    logging.info(f"{len(dedup_table)} entries after expanding multiple entries")
    gp33_antigen = utils.read_newline_file(
        os.path.join(os.path.dirname(fname), "lcmv_antigen.txt")
    ).pop()
    dedup_table["antigen.sequence"] = gp33_antigen  # gp33 tetramer

    # Load metadata and match it up with prior table
    metadata_df = pd.read_csv(metadata_fname, delimiter="\t", low_memory=False)
    if drop_na:
        metadata_df.dropna(axis=0, how="any", subset=["TRA", "TRB"], inplace=True)
    table_ab_pairs = list(dedup_table["tcr_cdr3s_aa"])
    metadata_ab_pairs = list(metadata_df["tcr_cdr3s_aa"])
    idx_map = np.array([metadata_ab_pairs.index(p) for p in table_ab_pairs])
    metadata_df_reorder = metadata_df.iloc[idx_map]
    assert (
        all(
            [
                i == j
                for i, j in zip(
                    metadata_df_reorder["tcr_cdr3s_aa"], dedup_table["tcr_cdr3s_aa"]
                )
            ]
        )
        and metadata_df_reorder.shape[0] == dedup_table.shape[0]
    )
    metadata_df_reorder = metadata_df_reorder.drop(
        columns=[
            col for col in metadata_df_reorder.columns if col in dedup_table.columns
        ]
    )
    metadata_df_reorder.index = dedup_table.index

    # Load in VDJ annotations and match it up with prior table
    vdj_mapping = load_lcmv_vdj(vdj_fname)  # 3 layer dict
    vdj_df_reorder_rows = []
    for i, row in dedup_table.iterrows():
        b_vdj = vdj_mapping[row["tcr_cdr3s_aa"]]["TRB"][row["TRB"]]
        a_vdj = vdj_mapping[row["tcr_cdr3s_aa"]]["TRA"][row["TRA"]]
        s = pd.Series(
            [row["tcr_cdr3s_aa"], *a_vdj, *b_vdj],
            index=[
                "tcr_cdr3s_aa",
                "v_a_gene",
                "d_a_gene",
                "j_a_gene",
                "v_b_gene",
                "d_b_gene",
                "j_b_gene",
            ],
        )
        vdj_df_reorder_rows.append(s)
    vdj_df_reorder = pd.DataFrame(vdj_df_reorder_rows)
    assert all(
        [
            i == j
            for i, j in zip(vdj_df_reorder["tcr_cdr3s_aa"], dedup_table["tcr_cdr3s_aa"])
        ]
    )
    vdj_df_drop_cols = [
        col
        for col in vdj_df_reorder.columns
        if col in dedup_table.columns or col in metadata_df_reorder.columns
    ]
    logging.debug(f"Dropping cols from VDJ info: {vdj_df_drop_cols}")
    vdj_df_reorder = vdj_df_reorder.drop(columns=vdj_df_drop_cols)
    vdj_df_reorder.index = dedup_table.index
    retval = pd.concat([dedup_table, metadata_df_reorder, vdj_df_reorder], axis=1)

    # Check that the TRA/TRB are the same as the "dedup_table" object that we were previously returning
    assert all([i == j for i, j in zip(retval["TRA"], dedup_table["TRA"])])
    assert all([i == j for i, j in zip(retval["TRB"], dedup_table["TRB"])])

    # Report basic metadata
    cnt = collections.Counter(dedup_table["tetramer"])
    for k, v in cnt.items():
        logging.info(f"Class {k}: {v}")

    return retval


def dedup_lcmv_table(
    lcmv_tab: pd.DataFrame,
    blacklist_label_combos: Sequence[str] = (
        "TetMid,TetNeg",
        "TetNeg,TetPos",
        "TetMid,TetNeg,TetPos",
    ),
    return_mode: Literal["nt", "aa", "full"] = "aa",
) -> Tuple[Union[List[Tuple[str, str]], pd.DataFrame], List[str]]:
    """
    Return TRA and TRB pairs that are deduped according to their AA sequence and removes
    pairs with ambiguous labels

    This was implemented to centrally solve the issue where the LCMV table had duplicate rows and
    a few cases of ambiguous labels

    Returns two items of equal length:
    - List of (TRA, TRB) pairs either in AA form or NT form, or a subset of the full dataframe
    - List of corresponding labels (may be merged)
    """
    lcmv_ab = ["|".join(p) for p in zip(lcmv_tab["TRA"], lcmv_tab["TRB"])]
    # Create a mapping from amino acid to NT sequence
    lcmv_ab_to_nt = {
        n: "|".join(p)
        for n, p in zip(lcmv_ab, zip(lcmv_tab["TRA_nt"], lcmv_tab["TRB_nt"]))
    }
    lcmv_ab_to_full = {
        "|".join([row["TRA"], row["TRB"]]): row for i, row in lcmv_tab.iterrows()
    }
    lcmv_ab_dedup, lcmv_labels_dedup = dedup_and_merge_labels(
        lcmv_ab, list(lcmv_tab["tetramer"])
    )
    all_label_counter = collections.Counter(lcmv_labels_dedup)
    logging.info(f"Combined labels {all_label_counter.most_common()}")
    logging.info(f"Filtering out labels {blacklist_label_combos}")
    good_label_idx = [
        i for i, l in enumerate(lcmv_labels_dedup) if l not in blacklist_label_combos
    ]
    logging.info(f"Retaining {len(good_label_idx)} pairs with unambiguous labels")
    lcmv_ab_good = [lcmv_ab_dedup[i] for i in good_label_idx]
    lcmv_labels_good = [lcmv_labels_dedup[i] for i in good_label_idx]
    assert len(lcmv_ab_good) == len(lcmv_labels_good) == len(good_label_idx)
    label_counter = collections.Counter(lcmv_labels_good)
    logging.info(f"LCMV deduped labels: {label_counter.most_common()}")

    # Resplit into pairs
    if return_mode == "nt":
        lcmv_ab_good_split = [tuple(lcmv_ab_to_nt[p].split("|")) for p in lcmv_ab_good]
    elif return_mode == "aa":
        lcmv_ab_good_split = [tuple(p.split("|")) for p in lcmv_ab_good]
    elif return_mode == "full":
        lcmv_ab_good_split = pd.DataFrame([lcmv_ab_to_full[p] for p in lcmv_ab_good])
    else:
        raise ValueError(f"Unrecognized return mode: {return_mode}")
    return lcmv_ab_good_split, lcmv_labels_good


def dedup_lcmv_table_trb_only(
    lcmv_tab: pd.DataFrame,
    blacklist_label_combos: Sequence[str] = (
        "TetMid,TetNeg",
        "TetNeg,TetPos",
        "TetMid,TetNeg,TetPos",
    ),
) -> Tuple[List[str], List[str]]:
    """
    Return a list of unique TRBs and corresponding set of labels
    """
    trb_dedup, labels_dedup = dedup_and_merge_labels(
        list(lcmv_tab["TRB"]), list(lcmv_tab["tetramer"])
    )
    assert utils.is_all_unique(trb_dedup)
    all_label_counter = collections.Counter(labels_dedup)
    logging.info(f"Combined labels {all_label_counter.most_common()}")
    logging.info(f"Filtering out labels {blacklist_label_combos}")
    good_label_idx = [
        i for i, l in enumerate(labels_dedup) if l not in blacklist_label_combos
    ]
    logging.info(f"Retaining {len(good_label_idx)} sequences with unambiguous labels")

    trb_good = [trb_dedup[i] for i in good_label_idx]
    labels_good = [labels_dedup[i] for i in good_label_idx]
    assert len(trb_good) == len(labels_good) == len(good_label_idx)
    label_counter = collections.Counter(labels_good)
    logging.info(f"LCMV TRB only deduped labels: {label_counter.most_common()}")

    return trb_good, labels_good


def load_vdjdb(
    fname: str = os.path.join(LOCAL_DATA_DIR, "vdjdb-2021-02-02", "vdjdb.slim.txt"),
    species_filter: Optional[Iterable[str]] = ["MusMusculus", "HomoSapiens"],
    tra_trb_filter: Optional[Iterable[str]] = ["TRA", "TRB"],
    addtl_filters: Optional[Dict[str, Iterable[str]]] = None,
    drop_null: bool = True,
    vocab_check: bool = True,
) -> pd.DataFrame:
    """
    Load VDJdb as a dataframe. 'cdr3' column is the column containing sequences
    ~62k examples, spanning 352 distinct antigens

    Additional filters can be provided in the format
    {column_name: ['acceptableValue1', 'acceptableValue2', ...]}
    """
    df = pd.read_csv(fname, sep="\t")
    if species_filter is not None:
        logging.info(f"Filtering VDJdb species to: {species_filter}")
        keep_idx = [i for i in df.index if df.loc[i, "species"] in species_filter]
        df = df.loc[keep_idx]
    logging.info(f"Species distribution: {collections.Counter(df['species'])}")
    if drop_null:
        keep_idx = [~pd.isnull(aa) for aa in df["cdr3"]]
        logging.info(
            f"VDJdb: dropping {np.sum(keep_idx==False)} entries for null cdr3 sequence"
        )
        df = df.iloc[np.where(keep_idx)]
    if vocab_check:
        pass_idx = np.array([ft.adheres_to_vocab(aa) for aa in df["cdr3"]])
        logging.info(
            f"VDJdb: dropping {np.sum(pass_idx==False)} entries for unrecognized AAs"
        )
        df = df.iloc[np.where(pass_idx)]
    nonnull_antigens_df = df.loc[~pd.isnull(df["antigen.epitope"])]
    logging.info(
        f"Entries with antigen sequence: {nonnull_antigens_df.shape[0]}/{df.shape[0]}"
    )
    logging.info(
        f"Unique antigen sequences: {len(set(nonnull_antigens_df['antigen.epitope']))}"
    )
    if tra_trb_filter is not None:
        logging.info(f"Filtering TRA/TRB to: {tra_trb_filter}")
        keep_idx = [i for i in df.index if df.loc[i, "gene"] in tra_trb_filter]
        df = df.loc[keep_idx]

    # For each of the additional fitlers
    if addtl_filters is not None:
        for colname, keep_vals in addtl_filters.items():
            logging.info(f"Filtering {colname} to {keep_vals}")
            keep_idx = [i for i in df.index if df.loc[i, colname] in keep_vals]
            df = df.loc[keep_idx]

    ab_counter = collections.Counter(df["gene"])
    logging.info(f"TRA: {ab_counter['TRA']} | TRB: {ab_counter['TRB']}")
    return df


def load_pird(
    fname: str = os.path.join(LOCAL_DATA_DIR, "pird", "pird_tcr_ab.csv"),
    tra_trb_only: bool = True,
    vocab_check: bool = True,
    addtl_filters: Optional[Dict[str, Iterable[str]]] = None,
    with_antigen_only: bool = False,
) -> pd.DataFrame:
    """
    Load PIRD (pan immune repertoire database) TCR A/B data
    https://db.cngb.org/pird/tbadb/
    For TRA we want the column CDR3.alpha.aa
    For TRB we want the column CDR3.beta.aa
    The PIRD dataset also has ~8k examples with antigens (73 unique)
    """
    if not tra_trb_only:
        raise NotImplementedError
    df = pd.read_csv(fname, na_values="-", low_memory=False)
    # df_orig = pd.read_csv(fname, na_values="-", low_memory=False)
    # df = df_orig.dropna(axis=0, how="all", subset=["CDR3.alpha.aa", "CDR3.beta.aa"])
    # logging.info(
    #     f"Dropped {len(df_orig) - len(df)} entires with null sequence in both TRA/TRB"
    # )
    antigen_null_rate = np.sum(pd.isnull(df["Antigen.sequence"])) / df.shape[0]
    logging.info(
        f"PIRD data {1.0 - antigen_null_rate:.4f} data labelled with antigen sequence"
    )
    # Filter out entries that have weird characters in their aa sequences
    if vocab_check:
        tra_pass = [
            pd.isnull(aa) or ft.adheres_to_vocab(aa) for aa in df["CDR3.alpha.aa"]
        ]
        trb_pass = [
            pd.isnull(aa) or ft.adheres_to_vocab(aa) for aa in df["CDR3.beta.aa"]
        ]
        both_pass = np.logical_and(tra_pass, trb_pass)
        logging.info(
            f"PIRD: Removing {np.sum(both_pass == False)} entires with non amino acid residues"
        )
        df = df.iloc[np.where(both_pass)]
    # Collect instances where we have antigen information
    nonnull_antigens_df = df.loc[~pd.isnull(df["Antigen.sequence"])]
    nonnull_antigens = nonnull_antigens_df["Antigen.sequence"]
    logging.info(
        f"Entries with antigen sequence: {len(nonnull_antigens)}/{df.shape[0]}"
    )
    logging.info(f"Unique antigen sequences: {len(set(nonnull_antigens))}")
    logging.info(f"PIRD data TRA/TRB instances: {collections.Counter(df['Locus'])}")
    retval = nonnull_antigens_df if with_antigen_only else df

    # Perform additional filtering
    if addtl_filters is not None:
        for colname, keep_vals in addtl_filters.items():
            logging.info(f"Filtering {colname} to {keep_vals}")
            keep_idx = [i for i in retval.index if retval.loc[i, colname] in keep_vals]
            retval = retval.loc[keep_idx]

    # Report metrics
    # print(df.loc[:, ["CDR3.alpha.aa", "CDR3.beta.aa"]])
    has_tra = ~pd.isnull(df["CDR3.alpha.aa"])
    has_trb = ~pd.isnull(df["CDR3.beta.aa"])
    has_both = np.logical_and(has_tra, has_trb)
    logging.info(f"PIRD entries with TRB sequence: {np.sum(has_tra)}")
    logging.info(f"PIRD entries with TRB sequence: {np.sum(has_trb)}")
    logging.info(f"PIRD entries with TRA and TRB:  {np.sum(has_both)}")
    # print(retval.iloc[np.where(has_both)[0]].loc[:, ["CDR3.alpha.aa", "CDR3.beta.aa"]])

    return retval


def _tcrdb_df_to_entries(fname: str) -> List[tuple]:
    """Helper function for processing TCRdb tables"""

    def tra_trb_from_str(s: str) -> str:
        if s.startswith("TRA"):
            return "TRA"
        elif s.startswith("TRB"):
            return "TRB"
        return "UNK"

    def infer_row_tra_trb(row) -> str:
        """Takes in a row from itertuples and return inferred TRA/TRB"""
        infers = []
        if "Vregion" in row._fields:
            infers.append(tra_trb_from_str(row.Vregion))
        if "Dregion" in row._fields:
            infers.append(tra_trb_from_str(row.Dregion))
        if "Jregion" in row._fields:
            infers.append(tra_trb_from_str(row.Jregion))
        if len(infers) == 0:
            return "UNK"
        # Use majority voting
        cnt = collections.Counter(infers)
        consensus, consensus_prop = cnt.most_common(1).pop()
        if consensus_prop / len(infers) > 0.5:
            return consensus
        return "UNK"  # No majority

    acc = os.path.basename(fname).split(".")[0]
    df = pd.read_csv(fname, delimiter="\t")
    entries = [
        (acc, row.RunId, row.AASeq, row.cloneFraction, infer_row_tra_trb(row))
        for row in df.itertuples(index=False)
    ]
    return entries


@functools.lru_cache()
def load_tcrdb(
    dirname: str = os.path.join(LOCAL_DATA_DIR, "tcrdb"),
    drop_unk: bool = True,
    vocab_check: bool = True,
) -> pd.DataFrame:
    """
    Load TCRdb
    https://academic.oup.com/nar/article/49/D1/D468/5912818
    http://bioinfo.life.hust.edu.cn/TCRdb/#/
    """
    accessions_list_fname = os.path.join(dirname, "tcrdb_accessions_21_03_22.txt")
    with open(accessions_list_fname, "r") as source:
        accessions = [line.strip() for line in source if not line.startswith("#")]
    # Load in each accession
    accession_fnames = [os.path.join(dirname, f"{acc}.tsv.gz") for acc in accessions]
    pool = multiprocessing.Pool(8)
    entries = pool.map(_tcrdb_df_to_entries, accession_fnames)
    pool.close()
    pool.join()
    retval = pd.DataFrame(
        itertools.chain.from_iterable(entries),
        columns=["accession", "RunId", "AASeq", "cloneFraction", "tra_trb"],
    )
    if drop_unk:
        drop_idx = np.where(retval["tra_trb"] == "UNK")[0]
        logging.info(
            f"Dropping {len(drop_idx)} TCRdb entries for unknown TRA TRB status"
        )
        retval.drop(index=drop_idx, inplace=True)
    if vocab_check:
        is_valid_aa = np.array([ft.adheres_to_vocab(aa) for aa in retval["AASeq"]])
        logging.info(
            f"TCRdb: Removing {np.sum(is_valid_aa == False)} entries with non-amino acid residues"
        )
        retval = retval.iloc[np.where(is_valid_aa)]
    return retval


def collect_tra_trb(s: str) -> Dict[str, List[str]]:
    """
    Given semicolon separated TRA/TRB listings in a string, separate them and return a mapping
    If either TRA/TRB is missing, corresponding return will be a list with a single empty string

    >>> collect_tra_trb("TRA:foo;TRA:baz;TRB:bar")
    {'TRA': ['foo', 'baz'], 'TRB': ['bar']}
    >>> collect_tra_trb("TRB:bar")
    {'TRA': [''], 'TRB': ['bar']}
    >>> collect_tra_trb("TRB:bar;TRA:foo")
    {'TRA': ['foo'], 'TRB': ['bar']}
    """
    retval = {"TRA": [], "TRB": []}
    for part in s.split(";"):
        k, v = part.split(":")
        retval[k].append(v)
    # Return empty strings if TRA/TRB are not found
    if not retval["TRA"]:
        retval["TRA"].append("")
    if not retval["TRB"]:
        retval["TRB"].append("")
    return retval


def dedup_and_merge_labels(
    sequences: Sequence[str], labels: Sequence[str], sep: str = ","
) -> Tuple[List[str], List[str]]:
    """
    Remove duplicates in sequences and merge labels accordingly
    sep is the label separator, used to split and rejoin labels
    Return is sorted!

    >>> dedup_and_merge_labels(['a', 'b', 'a'], ['x', 'y', 'y'])
    (['a', 'b'], ['x,y', 'y'])
    >>> dedup_and_merge_labels(['a', 'b', 'a', 'a'], ['x', 'y', 'y,x', 'z'])
    (['a', 'b'], ['x,y,z', 'y'])
    >>> dedup_and_merge_labels(['a', 'b', 'd', 'c'], ['x', 'z', 'y', 'n'])
    (['a', 'b', 'c', 'd'], ['x', 'z', 'n', 'y'])
    """
    # unique returns the *sorted* unique elements of an array
    uniq_sequences, inverse_idx, uniq_seq_counts = np.unique(
        sequences, return_inverse=True, return_counts=True
    )
    uniq_labels, agg_count = [], 0
    # Walk through all unique sequences and fetch/merge corresponding labels
    for i, (seq, c) in enumerate(zip(uniq_sequences, uniq_seq_counts)):
        orig_idx = np.where(inverse_idx == i)[0]
        match_labels = utils.dedup([labels[i] for i in orig_idx])
        if len(match_labels) == 1:
            uniq_labels.append(match_labels.pop())
        else:  # Aggregate labels
            aggregated_labels = utils.dedup(
                list(
                    itertools.chain.from_iterable([l.split(sep) for l in match_labels])
                )
            )
            logging.debug(f"Merging {match_labels} -> {sep.join(aggregated_labels)}")
            agg_count += 1
            uniq_labels.append(sep.join(sorted(aggregated_labels)))
    assert len(uniq_sequences) == len(uniq_labels)
    logging.info(
        f"Deduped from {len(sequences)} -> {len(uniq_sequences)} merging {agg_count} labels"
    )
    return list(uniq_sequences), uniq_labels


def load_clonotypes_csv_general(fname: str, single_return: bool = True) -> pd.DataFrame:
    """
    Load clonotypes.csv file. This file is expected to be a comma-delimited table with columns
    "clonotype_id" and "cdr3s_aa".

    Returned data frame is the df contained in fname with added columns TRA_aa and TRB_aa
    containing amino acid sequences for TRA/TRB, respectively.

    single_return = True is default/legacy behavior, where in the event that multiple TRA/TRB
    sequences are listed, we take the last listed one from each. Setting this to false returns
    a ;-delimited series of TCRs when multiple values are encountered.
    """
    # Read file
    df = pd.read_csv(fname, index_col=0)
    # Expand out the TRA/TRBs
    tra_seqs, trb_seqs = [], []
    for i, row in df.iterrows():
        tra_trb_mapping = collect_tra_trb(row["cdr3s_aa"])
        if single_return:
            tra_seqs.append(tra_trb_mapping["TRA"][-1])
            trb_seqs.append(tra_trb_mapping["TRB"][-1])
        else:
            tra_seqs.append(";".join(tra_trb_mapping["TRA"]))
            trb_seqs.append(";".join(tra_trb_mapping["TRB"]))
    df["TRA_aa"] = tra_seqs
    df["TRB_aa"] = trb_seqs
    return df


def load_10x(
    celltype: str = "CD8_healthy", exclude_singles: bool = True
) -> pd.DataFrame:
    """
    Load 10x data. Columns of interest are TRA_aa and TRB_aa
    """

    def split_to_tra_trb(s: Iterable[str]):
        """Split into two lists of TRA and TRB"""
        # TODO this does NOT correctly handle cases where there are say
        # multiple TRA sequences in a single row
        tra_seqs, trb_seqs = [], []
        for entry in s:
            sdict = dict([part.split(":") for part in entry.split(";")])
            tra = sdict["TRA"] if "TRA" in sdict else ""
            trb = sdict["TRB"] if "TRB" in sdict else ""
            tra_seqs.append(tra)
            trb_seqs.append(trb)
        return tra_seqs, trb_seqs

    dirname = os.path.join(LOCAL_DATA_DIR, "10x", celltype)
    assert os.path.isdir(dirname), f"Unrecognized celltype: {celltype}"
    if celltype == "CD8_healthy":
        fnames = glob.glob(
            os.path.join(dirname, "vdj_v1_hs_aggregated_donor*_clonotypes.csv")
        )
    else:
        fnames = glob.glob(os.path.join(dirname, "*_t_clonotypes.csv"))
    assert fnames
    fnames = sorted(fnames)
    dfs = []
    for fname in fnames:
        df = pd.read_csv(fname)
        tra_seqs, trb_seqs = split_to_tra_trb(df["cdr3s_aa"])
        df["TRA_aa"] = tra_seqs
        df["TRB_aa"] = trb_seqs
        tra_nt, trb_nt = split_to_tra_trb(df["cdr3s_nt"])
        df["TRA_nt"] = tra_nt
        df["TRB_nt"] = trb_nt

        if exclude_singles:
            is_single_idx = np.where(
                np.logical_or(df["TRA_aa"] == "", df["TRB_aa"] == "")
            )
            logging.info(
                f"Dropping {len(is_single_idx[0])} entries for unmatched TRA/TRB"
            )
            df.drop(index=is_single_idx[0], inplace=True)

        dfs.append(df)
    retval = pd.concat(dfs, axis=0)
    return retval


def load_glanville() -> pd.DataFrame:
    """Load in the Glanville GLIPH dataset"""
    fname = os.path.join(LOCAL_DATA_DIR, "glanville", "glanville_curated.csv")
    df = pd.read_csv(fname, low_memory=False, header="infer")
    df["CDR3b_spaced"] = [ft.insert_whitespace(aa) for aa in df["CDR3b"]]
    return df


def load_bcc(
    dirname: str = os.path.join(LOCAL_DATA_DIR, "GSE123813_bcc"),
    require_tra: bool = False,
    require_trb: bool = False,
) -> pd.DataFrame:
    """
    Load the BCC TCR data
    Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123813
    """
    # Load in the tables
    tcr_table = pd.read_csv(
        os.path.join(dirname, "GSE123813_bcc_tcr.txt"), sep="\t", index_col=0,
    )
    tcr_table.index.name = "cell.id"
    metadata_table = pd.read_csv(
        os.path.join(dirname, "GSE123813_bcc_tcell_metadata.txt"),
        sep="\t",
        index_col=0,
    )
    # Intersect and concatenate
    overlapped_idx = [i for i in metadata_table.index if i in tcr_table.index]
    table = metadata_table.loc[overlapped_idx]
    tcr_table = tcr_table.loc[overlapped_idx]
    for col in tcr_table.columns:
        table[col] = tcr_table[col]

    # Create dedicated columns for TRA and TRB
    tra_trb_pairs = []
    for item in table["cdr3s_aa"]:
        d = {"TRA": "", "TRB": ""}  # Null sequences for both
        d.update(dict([i.split(":") for i in item.split(";")]))
        tra_trb_pairs.append((d["TRA"], d["TRB"]))
    tra, trb = zip(*tra_trb_pairs)

    table["TRA_aa"] = tra
    table["TRB_aa"] = trb

    if require_tra:
        keep_idx = np.array([i for i, aa in enumerate(table["TRA_aa"]) if aa])
        logging.info(
            f"BCC: Retaining {len(keep_idx)}/{table.shape[0]} entries with TRA sequence"
        )
        table = table.iloc[keep_idx]
    if require_trb:
        keep_idx = np.array([i for i, aa in enumerate(table["TRB_aa"]) if aa])
        logging.info(
            f"BCC: retaining {len(keep_idx)}/{table.shape[0]} entries with TRB sequence"
        )
        table = table.iloc[keep_idx]

    return table


def load_immuneaccess_mira_covid(
    dirname: str = os.path.join(
        LOCAL_DATA_DIR, "immuneaccess/ImmuneCODE-MIRA-Release002.1"
    ),
    basename: str = "peptide-detail-ci.csv",
) -> pd.DataFrame:
    """
    Load the immuneaccess data
    https://clients.adaptivebiotech.com/pub/covid-2020
    https://www.researchsquare.com/article/rs-51964/v1

    Dataset includes three panels:
    minigene_set1, minigene_set2 target large protein sequences
    C19_cI targets individual peptides or small groups of peptides

    subject-metadata.csv - metadata
    orfs.csv - genomic location of MIRA targets as per GenBank11
    minigene-hits.csv - counts of number of unique TCRs bound to targets in minigene panels
    minigene-detail.csv - describes identity of TCRs bound per target for both minigene panels
    peptide-hits.csv - counts of number of unique TCRs bound to targets within C19_cI panel
    peptide-detail.csv - describes identity of TCRs bound per target for C19_cI MIRA panel

    Formerly used minigene-detail.csv, now use peptide-detail-ci
    """
    fname = os.path.join(dirname, basename)
    df = pd.read_csv(fname, delimiter=",", low_memory=False)
    tcr_seqs = [i.split("+")[0] for i in df["TCR BioIdentity"]]
    good_idx = []
    for i, seq in enumerate(tcr_seqs):
        if ft.adheres_to_vocab(seq):
            good_idx.append(i)
        else:
            logging.debug(f"immuneaccess MIRA: Got anomalous sequence: {seq}")
    logging.info(f"Retaining {len(good_idx)}/{len(df)} fully specified TCR sequences")
    df = df.iloc[np.array(good_idx)]
    df["TCR_aa"] = [i.split("+")[0] for i in df["TCR BioIdentity"]]

    # Load in metadata and attach it
    metadata_fname = os.path.join(dirname, "subject-metadata.csv")
    metadata_df = pd.read_csv(metadata_fname, index_col=0)
    df["cohort"] = [metadata_df.loc[e, "Cohort"] for e in df["Experiment"]]
    df["celltype"] = [metadata_df.loc[e, "Cell Type"] for e in df["Experiment"]]
    df["patient"] = [metadata_df.loc[e, "Subject"] for e in df["Experiment"]]
    df["target"] = [metadata_df.loc[e, "Target Type"] for e in df["Experiment"]]

    return df


def load_longitudinal_covid_trbs(
    dirname: str = os.path.join(LOCAL_DATA_DIR, "covid_longitudinal/beta"),
    check_vocab: bool = True,
) -> pd.DataFrame:
    """
    Load longitudinal covid data

    References:
    https://zenodo.org/record/4065547
    https://elifesciences.org/articles/63502
    """
    filenames = glob.glob(os.path.join(dirname, "*/*_beta.txt.gz"))
    dataframes = []
    for fname in filenames:
        # Parse out metadata from the filename
        bname = os.path.basename(fname)
        tokens = bname.split(".")[0].split("_")
        if len(tokens) != 4:
            logging.warn(f"Could not parse metadata from {bname}, skipping")
            continue
        patient, timepoint, celltype, tcr_segment = tokens
        assert tcr_segment == "beta"
        if not utils.is_numeric_scalar(timepoint):
            timepoint, celltype = celltype, timepoint
        timepoint = int(timepoint)
        if celltype in ("F1", "F2"):
            celltype = "PBMC_rep1" if celltype == "F1" else "PBMC_rep2"
        # Read in the TCRs
        df = pd.read_csv(fname, sep="\t", index_col=0)
        if check_vocab:
            passes_vocab = [ft.adheres_to_vocab(aa) for aa in df["aaSeqCDR3"]]
            keep_idx = np.where(passes_vocab)
            df = df.iloc[keep_idx]
        df["celltype"] = celltype
        df["timepoint"] = timepoint
        df["patient"] = patient
        dataframes.append(df)
    # https://stackoverflow.com/questions/41181779/merging-2-dataframes-vertically
    retval = pd.concat(dataframes, ignore_index=True)
    return retval


def _load_reuben_df_with_label_col(
    fname: str, label_col: str, drop_na: bool = True, drop_illegal: bool = True
) -> pd.DataFrame:
    """Helper function for loading in parallel"""
    assert os.path.isfile(fname)
    df = pd.read_csv(fname, sep="\t", low_memory=False)
    df["label"] = label_col
    if drop_na:
        df.drop(index=df.index[pd.isna(df["aminoAcid"])], inplace=True)
    if drop_illegal:
        illegal_idx = [
            df.index[i]
            for i, aa in enumerate(df["aminoAcid"])
            if not ft.adheres_to_vocab(aa)
        ]
        df.drop(index=illegal_idx, inplace=True)
    return df


def load_reuben_nsclc(
    dirname: str = os.path.join(LOCAL_DATA_DIR, "reuben_nsclc")
) -> pd.DataFrame:
    """
    Load the TRB sequences for NSCLC tumor and normal
    """
    assert os.path.isdir(dirname)
    metadata_df = pd.read_csv(
        os.path.join(dirname, "SampleOverview_06-29-2021_9-05-38_PM.tsv"),
        sep="\t",
        index_col=0,
    )
    metadata_tokens = {
        i: utils.dedup([tok.strip() for tok in row["sample_tags"].split(",")])
        for i, row in metadata_df.iterrows()
    }

    # Find the names the correspond to normal and tumor and their overlap
    norm_fnames = [
        k for k, v in metadata_tokens.items() if "Normal adjacent tissue" in v
    ]
    tumor_fnames = [k for k, v in metadata_tokens.items() if "Tumor" in v]
    assert not set(norm_fnames).intersection(tumor_fnames)
    strip_tail_suffix = lambda x: "-".join(x.split("-")[:-1])
    norm_prefixes = {strip_tail_suffix(x): x for x in norm_fnames}
    assert len(norm_prefixes) == len(norm_fnames)
    tumor_prefixes = {strip_tail_suffix(x): x for x in tumor_fnames}
    assert len(tumor_prefixes) == len(tumor_fnames)
    # Find overlap
    shared_prefixes = sorted(
        list(set(norm_prefixes.keys()).intersection(tumor_prefixes.keys()))
    )
    norm_fnames = [
        os.path.join(dirname, norm_prefixes[p] + ".tsv.gz") for p in shared_prefixes
    ]
    tumor_fnames = [
        os.path.join(dirname, tumor_prefixes[p] + ".tsv.gz") for p in shared_prefixes
    ]
    assert len(norm_fnames) == len(tumor_fnames)

    # Load in the data
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pfunc_norm = functools.partial(
        _load_reuben_df_with_label_col, label_col="normal_adj_tissue"
    )
    pfunc_tumor = functools.partial(_load_reuben_df_with_label_col, label_col="tumor")
    normal_dfs = pool.map(pfunc_norm, norm_fnames)
    tumor_dfs = pool.map(pfunc_tumor, tumor_fnames)
    pool.close()
    pool.join()
    combo_df = pd.concat(normal_dfs + tumor_dfs)
    return combo_df


def load_chiou_nsclc_yeast_screen(
    dirname: str = os.path.join(LOCAL_DATA_DIR, "chiou_nsclc_yeast_screen")
) -> pd.DataFrame:
    """
    Paper performs:
    1) Take patient TCR data, find specificity groups (returned here)
    2) Take specificity groups, focus on motif S%DGMNTE
    3) Use a yeast screen to identify antigens binding to group
    4) Validate that antigen and its cross-reactivity patterns

    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7960510/#mmc1
    Paper focuses on motif: S%DGMNTE
    """
    # TODO investigate what our model predicts as similar to these sequences
    # TODO potentially investigate whether our model's embedding can capture similar groups
    assert os.path.isdir(dirname)
    spec_groups_df = pd.read_csv(
        os.path.join(dirname, "nsclc_share_specificity_groups.txt"), sep="\t"
    )
    return spec_groups_df


def load_aa_properties(
    fname: str = os.path.join(LOCAL_DATA_DIR, "aa_properties.csv")
) -> pd.DataFrame:
    """Load aa properties"""
    df = pd.read_csv(fname, index_col=0, header=0)
    assert all([x == y for x, y in zip(df.index, ft.AMINO_ACIDS)])
    return df


def chunkify(x: Sequence[Any], chunk_size: int = 128):
    """
    Split list into chunks of given size
    >>> chunkify([1, 2, 3, 4, 5, 6, 7], 3)
    [[1, 2, 3], [4, 5, 6], [7]]
    >>> chunkify([(1, 10), (2, 20), (3, 30), (4, 40)], 2)
    [[(1, 10), (2, 20)], [(3, 30), (4, 40)]]
    """
    retval = [x[i : i + chunk_size] for i in range(0, len(x), chunk_size)]
    return retval


def chunkify_dict(
    x: Dict[str, Sequence[Any]], chunk_size: int = 128
) -> List[Dict[str, Sequence[Any]]]:
    """
    Apply chunkify to each item in the dictionary
    """
    # Dict of chunkified lists
    chunkified = {k: chunkify(v, chunk_size) for k, v in x.items()}
    # List of chunkfiied dicts
    retval = []
    for i in range(len(chunkified[list(chunkified.keys())[0]])):
        chunk = {k: chunkified[k][i] for k in x.keys()}
        assert len(set([len(v) for v in chunk.values()])) == 1
        retval.append(chunk)
    return retval


def min_dist_train_test_pairs(
    train_pairs: List[Tuple[str, str]], test_pairs: List[Tuple[str, str]]
) -> np.ndarray:
    """
    For each training pair, find the minimum edit distance to any test pair
    summed across the two elements in the pair
    """
    retval = []
    test_x, test_y = zip(*test_pairs)
    for x, y in train_pairs:
        x_dists = np.array([Levenshtein.distance(x, item) for item in test_x])
        y_dists = np.array([Levenshtein.distance(y, item) for item in test_y])
        d = np.min(x_dists + y_dists)
        retval.append(d)
    return np.array(retval)


def min_dist_train_test_seqs(
    train_seqs: Sequence[str], test_seqs: Collection[str]
) -> np.ndarray:
    """
    For each training sequence, finding the minimum edit distance
    to any test sequence.
    """
    retval = []
    for seq in train_seqs:
        # Calculate the edit distance to the most similar test sequence
        d = min([Levenshtein.distance(seq, test_seq) for test_seq in test_seqs])
        retval.append(d)
    return np.array(retval)


def sanitize_train_sequences(
    train_seqs: Sequence[str],
    train_labels: Sequence[str],
    test_seqs: Collection[str],
    min_edit_dist: int = 2,
) -> Tuple[List[str], List[str]]:
    """
    Return the training seqs/labels that are at least a given edit distance from
    any test sequence
    """
    assert len(train_seqs) == len(train_labels)
    train_dist = min_dist_train_test_seqs(train_seqs, test_seqs)

    passing_idx = np.where(train_dist >= min_edit_dist)[0]
    logging.info(
        f"Passing >= {min_edit_dist} edit dist cutoff: {len(passing_idx)}/{len(train_seqs)}"
    )
    return [train_seqs[i] for i in passing_idx], [train_labels[i] for i in passing_idx]


def write_lcmv_subsampled_benchmark_data():
    """Write the LCMV subsampled data for benchmarking runtime"""
    tab = load_lcmv_table()
    # Write out the table at varying sizes
    tab_condensed = tab["TRB"]
    for s in [500, 1000, 1500, 2000, 2500, 5000, 10000]:
        # Write GLIPH inputs
        t = tab_condensed.iloc[:s]
        t.to_csv(
            os.path.join(
                LOCAL_DATA_DIR, "lcmv_runtime_benchmark_files", f"lcmv_sub_{s}.tsv"
            ),
            sep="\t",
            index=False,
        )

        # Write TCRDist3 inputs
        write_lcmv_tcrdist3_input(
            fname=os.path.join(
                LOCAL_DATA_DIR,
                "lcmv_runtime_benchmark_files",
                f"tcrdist3_beta_lcmv_sub_{s}.tsv",
            ),
            dual_chain=False,
            subset=s,
        )

    return


def write_lcmv_tcrdist_input(fname: str = "temp.tsv"):
    """
    Write the LCMV data in format for TCRDist, which expects a tsv file with the columns:
    id  epitope subject a_nucseq    b_nucseq    a_quals b_quals
    Output is written to fname
    """
    lcmv = load_lcmv_table()
    seqs, labels = dedup_lcmv_table(lcmv, return_mode="nt")
    tra, trb = zip(*seqs)
    df = pd.DataFrame(
        {
            "id": np.arange(len(seqs)),
            "epitope": ["foo"] * len(seqs),
            "subject": ["bar"] * len(seqs),
            "a_nucseq": tra,
            "b_nucseq": trb,
        }
    )
    df.to_csv(fname, sep="\t", index=False)


def write_lcmv_tcrdist3_input(
    fname: str = os.path.join(EXTERNAL_EVAL_DIR, "lcmv_test_tcrdist3.tsv"),
    dual_chain: bool = True,
    subset: Union[str, int] = "test",
) -> pd.DataFrame:
    """
    Write the LCMV data in a format for TCRDist, which expects 3 columns per chain
    cdr3_b_aa   v_b_gene    j_b_gene    # example for b chain
    v/j genes expect the *01 suffix

    Important processing:
    if a v/j gene has a "+" indicating its two or more genes, we take the last one
    """

    def sanitize_vj(s: str) -> str:
        if "+" in s:
            s = s.split("+")[-1]
        if not s.endswith("*01"):
            s += "*01"
        return s

    lcmv = load_lcmv_table()
    df, labels = dedup_lcmv_table(lcmv, return_mode="full")
    if isinstance(subset, str):
        labels_sub = split_arr(labels, "test")
        df_sub = split_arr(df, "test")  # We only evaluate test set clustering
    else:
        labels_sub = labels[:subset]
        df_sub = df.iloc[:subset]
    if dual_chain:
        retval = df_sub.loc[
            :, ["TRA", "v_a_gene", "j_a_gene", "TRB", "v_b_gene", "j_b_gene"]
        ]
        retval.columns = [
            "cdr3_a_aa",
            "v_a_gene",
            "j_a_gene",
            "cdr3_b_aa",
            "v_b_gene",
            "j_b_gene",
        ]
    else:
        # Single chain
        retval = df_sub.loc[:, ["TRB", "v_b_gene", "j_b_gene"]]
        retval.columns = ["cdr3_b_aa", "v_b_gene", "j_b_gene"]
    for colname in ["v_a_gene", "j_a_gene", "v_b_gene", "j_b_gene"]:
        if colname in retval.columns:
            retval[colname] = [sanitize_vj(s) for s in retval[colname]]
    # Attach the "truth" column
    retval["gp33_binding"] = ["TetPos" in l or "TetMid" in l for l in labels_sub]
    retval.to_csv(fname, sep="\t", index=False)
    return retval


def on_the_fly():
    """On the fly testing"""
    # table = load_longitudinal_covid_trbs()
    # print(table)
    # print(collections.Counter(table["patient"]).most_common())
    # print(collections.Counter(table["celltype"]).most_common())
    # df = load_clonotypes_csv_general(sys.argv[1])
    # print(df)
    # print(write_lcmv_tcrdist3_input())
    write_lcmv_subsampled_benchmark_data()


if __name__ == "__main__":
    import doctest

    # doctest.testmod()
    # main()
    on_the_fly()
