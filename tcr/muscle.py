"""
Wrapper and utility functions for running MUSCLE
"""
import os
import tempfile
import subprocess
import shlex
import logging
from typing import *

import utils


def run_muscle(sequences: Iterable[str], fast: bool = False) -> List[str]:
    """
    Run MUSCLE on the given input sequences
    > run_muscle(["DEASV", "KKDEASV", "KKVVVSV"])
    ['--DEASV', 'KKDEASV', 'KKVVVSV']
    """
    with tempfile.TemporaryDirectory() as tempdir:
        # Write the MUSCLE input
        logging.debug(f"Running MUSCLE for MSA in {tempdir}")
        muscle_input_fname = os.path.join(tempdir, "msa_input.fa")
        with open(muscle_input_fname, "w") as sink:
            sink.write("\n".join([f"> {i}\n{seq}" for i, seq in enumerate(sequences)]))
            sink.write("\n")
        # Call MUSCLE
        muscle_output_fname = os.path.join(tempdir, "msa_output.fa")
        muscle_cmd = f"muscle -in {muscle_input_fname} -out {muscle_output_fname}"
        if fast:
            muscle_cmd += " -maxiters 2"
        retval = subprocess.call(
            shlex.split(muscle_cmd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert retval == 0, f"Exit code {retval} when running muscle"
        msa_seqs = list(utils.read_fasta(muscle_output_fname).values())
    return msa_seqs


def main():
    """On the fly testing"""
    m = run_muscle(["DEASV", "KKDEASV", "KKVVVSV",])
    print(m)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
