"""
Utility functions for fetching and parsing RefSeq entries
"""
import os
import json
import subprocess
import shlex
import logging
from sys import stderr
import tempfile
import functools
from urllib.error import HTTPError
from typing import *

import pandas as pd

from Bio import Entrez

MOUSE_TRB_BLASTDB = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/mus_musculus_trb_blastdb/mus_musculus_tcr_beta_chains_071721.fasta",
)
assert os.path.isfile(MOUSE_TRB_BLASTDB), f"Cannot find {MOUSE_TRB_BLASTDB}"

def run_blast(
    seqs: Iterable[str], db: str, e_val: float = 0.01, fetch_hit_metadata: bool = True
) -> pd.DataFrame:
    """Wrapper for running blast and return a table of hits"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logging.info(f"Running BLAST with E-value of {e_val} under: {tmpdir}")

        # Write the input fasta file
        fasta_map = {f"query-{i}": seq for i, seq in enumerate(seqs)}
        input_fa = os.path.join(tmpdir, "input_fasta.fa")
        with open(input_fa, "w") as sink:
            for k, v in fasta_map.items():
                sink.write(f">{k}\n")
                sink.write(v + "\n")

        # Run blast
        out_fname = os.path.join(tmpdir, "blast_out.txt")
        blast_cmd = f"blastp -query {input_fa} -db {db} -out {out_fname} -evalue {e_val} -outfmt 7"
        retval = subprocess.call(
            shlex.split(blast_cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        assert retval == 0, "BLAST failed"

        # Read BLAST output
        blastp_hits = pd.read_csv(
            out_fname,
            names=[
                "query_acc",
                "subject_acc",
                "perc_identity",
                "alignment_length",
                "mismatches",
                "gap_opens",
                "query_start",
                "query_end",
                "subject_start",
                "subject_end",
                "evalue",
                "bit_score",
            ],
            sep="\t",
            comment="#",
        )
        blastp_hits["query_seq"] = [fasta_map[acc] for acc in blastp_hits["query_acc"]]

    if fetch_hit_metadata:
        # Fetch metadata like matching sequence and corresponding manuscript
        logging.info(f"Fetching addtl information for {len(blastp_hits)} matches")
        hit_trb_sequences = [
            get_sequence(acc, start, end)
            for acc, start, end in zip(
                blastp_hits["subject_acc"],
                blastp_hits["subject_start"],
                blastp_hits["subject_end"],
            )
        ]
        blastp_hits["subject_seq"] = hit_trb_sequences
        hit_titles = [get_title(acc) for acc in blastp_hits["subject_acc"]]
        blastp_hits["ms_title"] = hit_titles

    return blastp_hits


def setup_entrez_credentials(
    config_fname: str = os.path.join(
        os.path.dirname(__file__), "entrez_credentials.json"
    )
) -> None:
    """
    Read in the given config fname and set the contained credentials

    Fill in the entrez_credentials.json file in the same directory as this file
    with two keys, email and api_key to configure
    """
    if not os.path.isfile(config_fname):
        logging.warning(f"Cannot find entrez credentials at {config_fname} - skipping")
        return
    with open(config_fname) as source:
        entrez_config = json.load(source)
    if "email" not in entrez_config or "api_key" not in entrez_config:
        logging.warning("Malformed entrez config file - skipping")
        return
    logging.debug(f"Configuring Entrez with account {entrez_config['email']}")
    Entrez.email = entrez_config["email"]
    Entrez.api_key = entrez_config["api_key"]


@functools.lru_cache(maxsize=4096)
def get_protein_record(identifier: str) -> Dict[str, Any]:
    """
    Get the protein record

    Since we frequently fetch multiple attributes from the same record,
    we cache the record itself
    """
    try:
        handle = Entrez.efetch(db="protein", id=identifier, rettype="gb", retmode="xml")
    except HTTPError:
        return ""
    record = Entrez.read(handle)[0]
    handle.close()
    return record


def get_title(identifier: str) -> str:
    """
    Return the title
    >>> get_title("AAC27968")
    'Spectratyping of TCR expressed by CTL-infiltrating male antigen (HY)-disparate allografts'
    """
    record = get_protein_record(identifier)
    try:
        return record["GBSeq_references"][0]["GBReference_title"]
    except (TypeError, KeyError) as e:
        return ""


def get_sequence(
    identifier: str, start: Optional[int] = None, end: Optional[int] = None
) -> str:
    """
    Return the sequence for the protein given identifier
    If the identifier is null
    >>> get_sequence("AFR46509", 10, 23)
    'CASSLGTTNTEVFF'
    """
    record = get_protein_record(identifier)
    try:
        seq = record["GBSeq_sequence"]
    except TypeError:
        logging.warning(f"Cannot parse record: {record} for identifier {identifier}")
        return ""  # Null return value
    if start is not None and end is not None:
        seq = seq[start - 1 : end]
    return seq.upper()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print(run_blast(["casslrrfsgntiyf".upper()], db=HUMAN_TRB_BLASTDB))
