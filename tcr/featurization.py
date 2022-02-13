"""
Featurization code
"""

import os, sys
import logging
import tempfile
from functools import cache, lru_cache
import itertools
import collections
from typing import *
from functools import cached_property
from math import floor

import numpy as np
import pandas as pd

from transformers import BertTokenizer

import muscle
import utils

#
AA_TRIPLET_TO_SINGLE = {
    "ARG": "R",
    "HIS": "H",
    "LYS": "K",
    "ASP": "D",
    "GLU": "E",
    "SER": "S",
    "THR": "T",
    "ASN": "N",
    "GLN": "Q",
    "CYS": "C",
    "SEC": "U",
    "GLY": "G",
    "PRO": "P",
    "ALA": "A",
    "VAL": "V",
    "ILE": "I",
    "LEU": "L",
    "MET": "M",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
}
AA_SINGLE_TO_TRIPLET = {v: k for k, v in AA_TRIPLET_TO_SINGLE.items()}

# 21 amino acids
AMINO_ACIDS = "RHKDESTNQCUGPAVILMFYW"
assert len(AMINO_ACIDS) == 21
assert all([x == y for x, y in zip(AMINO_ACIDS, AA_TRIPLET_TO_SINGLE.values())])
AMINO_ACIDS_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Pad with $ character
PAD = "$"
MASK = "."
UNK = "?"
SEP = "|"
CLS = "*"
AMINO_ACIDS_WITH_ALL_ADDITIONAL = AMINO_ACIDS + PAD + MASK + UNK + SEP + CLS
AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX = {
    aa: i for i, aa in enumerate(AMINO_ACIDS_WITH_ALL_ADDITIONAL)
}


class SequenceMasker:
    """Mask one position in each sequence for evaluation (NOT FOR TRAINING)"""

    def __init__(self, seq: Union[str, List[str]], seed: int = 4581):
        self._seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.unmasked = [seq] if isinstance(seq, str) else seq
        self._masked_indices = []
        self.unmasked_msa = muscle.run_muscle(self.unmasked)

    @cached_property
    def masked(self) -> List[str]:
        retval = []
        for unmasked in self.unmasked:
            aa = list(unmasked)
            mask_idx = self.rng.integers(0, len(aa))
            assert 0 <= mask_idx < len(aa)
            self._masked_indices.append(mask_idx)
            aa[mask_idx] = MASK
            retval.append(" ".join(aa))  # Space is necessary for tokenizer
        assert len(self._masked_indices) == len(self)
        return retval

    @cached_property
    def masked_truth(self) -> List[str]:
        """Return the masked amino acids"""
        _ = self.masked  # Ensure that this has been generated
        return [
            self.unmasked[i][mask_idx]
            for i, mask_idx in enumerate(self._masked_indices)
        ]

    def __len__(self) -> int:
        return len(self.unmasked)

    def get_naive_predictions(
        self,
        k: int,
        method: Literal[
            "most_common", "random", "most_common_positional"
        ] = "most_common",
    ) -> List[List[str]]:
        """
        Return naive predictions for each of the masked sequences
        Each entry in the list is a list of the top k predictions
        """
        if method == "most_common":
            cnt = collections.Counter()
            for seq in self.unmasked:
                cnt.update(seq)
            top_k = [k for k, v in cnt.most_common(k)]
            return [top_k] * len(self)
        elif method == "most_common_positional":
            # Create a matrix where each row corresponds to a position
            max_len = len(self.unmasked_msa[0])
            seqs_matrix = np.stack([np.array(list(s)) for s in self.unmasked_msa]).T
            assert seqs_matrix.shape == (max_len, len(self))

            # Per-position predictions
            per_pos_most_common = []
            for i in range(max_len):
                # Excludes padding bases
                cnt = collections.Counter(
                    [aa for aa in seqs_matrix[i] if aa in AMINO_ACIDS]
                )
                per_pos_most_common.append([aa for aa, _n, in cnt.most_common(k)])
            #
            retval = [per_pos_most_common[i] for i in self._masked_indices]
            return retval
        elif method == "random":
            baseline_naive_rng = np.random.default_rng(seed=self._seed)
            retval = []
            for _i in range(len(self)):
                idx = [
                    baseline_naive_rng.integers(0, len(AMINO_ACIDS)) for _j in range(k)
                ]
                retval.append([AMINO_ACIDS[i] for i in idx])
            return retval
        else:
            raise ValueError(f"Unrecognized method: {method}")


def adheres_to_vocab(s: str, vocab: str = AMINO_ACIDS) -> bool:
    """
    Returns whether a given string contains only characters from vocab
    >>> adheres_to_vocab("RKDES")
    True
    >>> adheres_to_vocab(AMINO_ACIDS + AMINO_ACIDS)
    True
    """
    return set(s).issubset(set(vocab))


def write_vocab(vocab: Iterable[str], fname: str) -> str:
    """
    Write the vocabulary to the fname, one entry per line
    Mostly for compatibility with transformer BertTokenizer
    """
    with open(fname, "w") as sink:
        for v in vocab:
            sink.write(v + "\n")
    return fname


def get_aa_bert_tokenizer(
    max_len: int = 64, d=AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX
) -> BertTokenizer:
    """
    Tokenizer for amino acid sequences. Not *exactly* the same as BertTokenizer
    but mimics its behavior, encoding start with CLS and ending with SEP

    >>> get_aa_bert_tokenizer(10).encode(insert_whitespace("RKDES"))
    [25, 0, 2, 3, 4, 5, 24]
    """
    with tempfile.TemporaryDirectory() as tempdir:
        vocab_fname = write_vocab(d, os.path.join(tempdir, "vocab.txt"))
        tok = BertTokenizer(
            vocab_fname,
            do_lower_case=False,
            do_basic_tokenize=True,
            tokenize_chinese_chars=False,
            pad_token=PAD,
            mask_token=MASK,
            unk_token=UNK,
            sep_token=SEP,
            cls_token=CLS,
            model_max_len=max_len,
            padding_side="right",
        )
    return tok


def get_pretrained_bert_tokenizer(path: str) -> BertTokenizer:
    """Get the pretrained BERT tokenizer from given path"""
    tok = BertTokenizer.from_pretrained(
        path,
        do_basic_tokenize=False,
        do_lower_case=False,
        tokenize_chinese_chars=False,
        unk_token=UNK,
        sep_token=SEP,
        pad_token=PAD,
        cls_token=CLS,
        mask_token=MASK,
        padding_side="right",
    )
    return tok


def mask_for_training(seq: str, prob: float = 0.15):
    """
    Manually mask for training. Expects 'spaced' input
    """
    aa = np.array(seq.split())
    assert len(aa) > 1
    target = np.zeros_like(aa, dtype=np.int64) - 100
    mask = np.random.random(len(aa)) <= prob
    masked_aa = aa[mask]
    target[mask] = [AMINO_ACIDS_TO_IDX[a] for a in masked_aa]
    aa[mask] = MASK
    assert target.shape == aa.shape
    return target, " ".join(aa)


def one_hot(seq: str, alphabet: Optional[str] = AMINO_ACIDS) -> np.ndarray:
    """
    One-hot encode the input string. Since pytorch convolutions expect
    input of (batch, channel, length), we return shape (channel, length)
    When one hot encoding, we ignore the pad characters, encoding them as
    a vector of 0's instead
    """
    if not seq:
        assert alphabet
        return np.zeros((len(alphabet), 1), dtype=np.float32)
    if not alphabet:
        alphabet = utils.dedup(seq)
        logging.info(f"No alphabet given, assuming alphabet of: {alphabet}")
    seq_arr = np.array(list(seq))
    # This implementation naturally ignores the pad character if not provided
    # in the alphabet
    retval = np.stack([seq_arr == char for char in alphabet]).astype(float).T
    assert len(retval) == len(seq), f"Mismatched lengths: {len(seq)} {retval.shape}"
    return retval.astype(np.float32).T


def idx_encode(
    seq: str, alphabet_idx: Dict[str, int] = AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX
) -> np.ndarray:
    """
    Encode the sequence as the indices in the alphabet
    >>> idx_encode("CAFEVVGQLTF")
    array([ 9, 13, 18,  4, 14, 14, 11,  8, 16,  6, 18], dtype=int32)
    """
    retval = np.array([alphabet_idx[aa] for aa in seq], dtype=np.int32)
    return retval


def pad_or_trunc_sequence(seq: str, l: int, right_align: bool = False, pad=PAD) -> str:
    """
    Pad the given sequence to the given length
    >>> pad_or_trunc_sequence("RKDES", 8, right_align=False)
    'RKDES$$$'
    >>> pad_or_trunc_sequence("RKDES", 8, right_align=True)
    '$$$RKDES'
    >>> pad_or_trunc_sequence("RKDESRKRKR", 3, right_align=False)
    'RKD'
    >>> pad_or_trunc_sequence("RKDESRRK", 3, right_align=True)
    'RRK'
    """
    delta = len(seq) - l
    if len(seq) > l:
        if right_align:
            retval = seq[delta:]
        else:
            retval = seq[:-delta]
    elif len(seq) < l:
        insert = pad * np.abs(delta)
        if right_align:
            retval = insert + seq
        else:
            retval = seq + insert
    else:
        retval = seq
    assert len(retval) == l, f"Got mismatched lengths: {len(retval)} {l}"
    return retval


def is_whitespaced(seq: str) -> bool:
    """
    Return whether the sequence has whitespace inserted
    >>> is_whitespaced("R K D E S")
    True
    >>> is_whitespaced("RKDES")
    False
    >>> is_whitespaced("R K D ES")
    False
    >>> is_whitespaced("R")
    True
    >>> is_whitespaced("RK")
    False
    >>> is_whitespaced("R K")
    True
    """
    tok = list(seq)
    spaces = [t for t in tok if t.isspace()]
    if len(spaces) == floor(len(seq) / 2):
        return True
    return False


def insert_whitespace(seq: str) -> str:
    """
    Return the sequence of characters with whitespace after each char
    >>> insert_whitespace("RKDES")
    'R K D E S'
    """
    return " ".join(list(seq))


def remove_whitespace(seq: str) -> str:
    """
    Remove whitespace from the given sequence
    >>> remove_whitespace("R K D E S")
    'RKDES'
    >>> remove_whitespace("R K D RR K")
    'RKDRRK'
    >>> remove_whitespace("RKIL")
    'RKIL'
    """
    return "".join(seq.split())


@cache
def all_possible_kmers(alphabet: Iterable[str] = AMINO_ACIDS, k: int = 3) -> List[str]:
    """
    Return all possible kmers
    """
    return ["".join(k) for k in itertools.product(*[alphabet for _ in range(k)])]


@lru_cache(maxsize=128)
def kmer_ft(
    seq: str, k: int = 3, size_norm: bool = False, alphabet: Iterable[str] = AMINO_ACIDS
) -> np.ndarray:
    """
    Kmer featurization to sequence
    """
    kmers = [seq[i : i + k] for i in range(0, len(seq) - k + 1)]
    kmers_to_idx = {
        k: i for i, k in enumerate(all_possible_kmers(alphabet=alphabet, k=k))
    }
    kmers = [k for k in kmers if k in kmers_to_idx]
    idx = np.array([kmers_to_idx[k] for k in kmers])
    retval = np.zeros(len(kmers_to_idx))
    np.add.at(retval, idx, 1)
    assert np.sum(retval) == len(kmers)
    if size_norm:
        retval /= len(kmers)
    return retval


def main():
    """On the fly testing"""
    np.random.seed(123456)
    print(mask_for_training(insert_whitespace("RKDES")))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
