"""
Misc. utility functions
"""
import os
import logging
import json
import random
import math
from typing import *

import torch

import numpy as np
import pandas as pd
import scipy
import Bio.Data.CodonTable as CodonTable


def min_power_greater_than(
    value: SupportsFloat, base: SupportsFloat = 2
) -> SupportsFloat:
    """
    Return the lowest power of the base that exceeds the given value
    >>> min_power_greater_than(3, 4)
    4.0
    >>> min_power_greater_than(48, 2)
    64.0
    """
    p = math.ceil(math.log(value, base))
    return math.pow(base, p)


def load_json_params(param_fname: str, **kwargs) -> Dict[str, Union[int, float, str]]:
    """Load in the param_fname, overriding with given kwargs"""
    with open(param_fname) as source:
        params = json.load(source)
    for k, v in kwargs.items():
        if k not in params:
            logging.warning(f"Key {k} not in original parameters")
        params[k] = v
    return params


def ensure_arr(x: Any) -> np.ndarray:
    """Return x as a np.array"""
    if isinstance(x, np.matrix):
        return np.squeeze(np.asarray(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, scipy.sparse.spmatrix):
        return x.toarray()
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    elif np.isscalar(x):
        return np.array([x])  # at least 1 dimensional
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")


def ensure_tensor(x: Any, **kwargs) -> torch.Tensor:
    """Return x as a torch tensor, kwargs are passed through"""
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, (np.ndarray, list, tuple)):
        return torch.tensor(x, **kwargs)
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")


def is_numeric_scalar(x: Any) -> bool:
    """
    Returns if the given item is numeric
    >>> is_numeric_scalar("hello")
    False
    >>> is_numeric_scalar("234")
    True
    >>> is_numeric_scalar("1e-5")
    True
    >>> is_numeric_scalar(2.5)
    True
    """
    if isinstance(x, (float, int)):
        return True
    elif isinstance(x, str):
        try:
            _ = float(x)
            return True
        except ValueError:
            return False
    return False


def is_all_unique(x: Collection[Any]) -> bool:
    """
    Return whether the given iterable is all unique
    >>> is_all_unique(['x', 'y'])
    True
    >>> is_all_unique(['x', 'x', 'y'])
    False
    """
    return len(set(x)) == len(x)


def dedup(x: Iterable[Any]) -> List[Any]:
    """
    Dedup the given iterable, preserving order of occurrence
    >>> dedup([1, 2, 0, 1, 3, 2])
    [1, 2, 0, 3]
    >>> dedup(dedup([1, 2, 0, 1, 3, 2]))
    [1, 2, 0, 3]
    """
    # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    # Python 3.7 and above guarantee that dict is insertion ordered
    # sets do NOT do this, so list(set(x)) will lose order information
    return list(dict.fromkeys(x))


def get_device(i: Optional[int] = None) -> torch.device:
    """
    Returns the i-th GPU if GPU is available, else CPU
    A negative value or a float will default to CPU
    """
    if torch.cuda.is_available() and i is not None and isinstance(i, int) and i >= 0:
        devices = list(range(torch.cuda.device_count()))
        device_idx = devices[i]
        torch.cuda.set_device(device_idx)
        d = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(d)
    else:
        logging.warn("Defaulting to CPU")
        d = torch.device("cpu")
    return d


def set_visible_device(devices: List[int] = [0]):
    """
    Set the visible GPU(s) using env variable.
    """
    assert len(devices) > 0, "Cannot set no visible devices"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in devices])


def is_json_file(fname: str) -> bool:
    """
    Return whether the given file is a json
    >>> is_json_file("hello.json")
    True
    >>> is_json_file("/usr/local/lol.json")
    True
    >>> is_json_file("foo.txt")
    False
    """
    if os.path.splitext(os.path.basename(fname))[1] == ".json":
        return True
    return False


def read_newline_file(fname: str, comment_char: str = "#") -> List[str]:
    """
    Read the newline delimited file, ignoring lines starting with
    comment_char
    """
    with open(fname) as source:
        retval = [line.strip() for line in source if line[0] != comment_char]
    return retval


def read_fasta(fname: str) -> Dict[str, str]:
    """Read fa file, returning a dict of seq names to seqs"""
    retval = {}
    curr_key, curr_seq = "", ""
    with open(fname) as source:
        for line in source:
            line = line.strip()
            if line.startswith(">"):
                if curr_key:  # Previous record
                    assert curr_seq
                    retval[curr_key] = curr_seq
                curr_key = line.strip(">").strip()
                curr_seq = ""  # Reset
            else:
                curr_seq += line.strip()
    if curr_key:  # Store trailing record
        assert curr_seq
        retval[curr_key] = curr_seq
    return retval


def nt2aa(seq: str) -> str:
    """
    Translate a nucleotide sequence to an amino acid sequence
    >>> nt2aa("TGT")
    'C'
    >>> nt2aa("TGTGCCAGCAGTTTCAGGGACAGCTCCTATGAACAGTACTTC")
    'CASSFRDSSYEQYF'
    """
    table = CodonTable.standard_dna_table
    seq = seq.upper()
    protein = ""
    assert len(seq) % 3 == 0, f"Sequence of length {len(seq)} not divisible by 3"
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        protein += table.forward_table[codon]
    return protein


def isnotebook() -> bool:
    """
    Returns True if the current execution environment is a jupyter notebook
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def seed_all(seed: int):
    """Seed all RNGs"""
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed + 3)
    torch.manual_seed(seed + 1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
