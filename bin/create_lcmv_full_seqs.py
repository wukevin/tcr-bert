"""
Code to create LCMV sequences for experimental validation
"""

import os, sys
import json
import argparse
import collections
import logging
from typing import *

import numpy as np
import pandas as pd
from Bio import pairwise2

from Bio.Align import substitution_matrices
import Bio.Data.CodonTable as CodonTable

import Levenshtein

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tcr")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import data_loader as dl
import utils


logging.basicConfig(level=logging.INFO)

STOP_CODONS = ["TAG", "TGA", "TAA"]


def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_json", type=str, help="json of generated sequences",
    )
    parser.add_argument("reference_file", type=str, help="Reference file in csv form")
    parser.add_argument("outdir", type=str, help="Directory to write output files")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["edit", "blosum"],
        default="blosum",
        help="Select best match based on min edit or best alignment",
    )
    return parser


def find_best_ref_match(
    tcrs: Tuple[str, str], ref_df: pd.DataFrame, mode: str = "edit"
) -> Tuple[int, pd.Series]:
    """
    Find the row in the given reference df that best matches the given tcr pair
    This can be done using either minimm "edit" distance or maximum BLOSUM62 score
    """

    def dist_func(tcrs: Tuple[str, str], row: pd.Series) -> int:
        a_edit = Levenshtein.distance(tcrs[0], row["TRA_cdr3"])
        b_edit = Levenshtein.distance(tcrs[1], row["TRB_cdr3"])
        return a_edit + b_edit

    def blosum_func(
        tcrs: Tuple[str, str],
        row: pd.Series,
        gap_open: float = -5.0,
        gap_extend: float = -0.5,
        scoring=substitution_matrices.load("BLOSUM62"),
    ) -> int:
        a_blosum = pairwise2.align.globalds(
            tcrs[0], row["TRA_cdr3"], scoring, gap_open, gap_extend, score_only=True
        )
        b_blosum = pairwise2.align.globalds(
            tcrs[1], row["TRB_cdr3"], scoring, gap_open, gap_extend, score_only=True
        )
        return int(a_blosum + b_blosum)

    if mode == "edit":
        d = [dist_func(tcrs, row) for _, row in ref_df.iterrows()]
        best_idx = np.argmin(d)
        ref_row = ref_df.iloc[best_idx].copy(deep=True)
        # check_row(ref_row)
        return min(d), ref_row
    elif mode == "blosum":
        d = [blosum_func(tcrs, row) for _, row in ref_df.iterrows()]
        best_idx = np.argmax(d)
        ref_row = ref_df.iloc[best_idx].copy(deep=True)
        # check_row(ref_row)
        return max(d), ref_row
    else:
        raise ValueError(f"Unrecognized mode: {mode}")


def find_best_ref_match_partial(trb: str, ref_df: pd.DataFrame):
    """
    Find the row in the given reference df that best matches the given TRB ONLY
    """
    raise NotImplementedError


def mutate_dna_by_protein(prot_new: str, prot_old: str, nt_old: str) -> str:
    """
    Mutates nt_old to encode prot_new instead of prot_old. Returns nt string
    This is necessary because there is redundancy in the codon table and we want to make
    as few nucleotide changes as possible to get to new protein

    >>> mutate_dna_by_protein("CASSD", "CASSF", "TGTGCCAGCAGTTTC")  # Single point
    'TGTGCCAGCAGTGAC'
    >>> mutate_dna_by_protein("CASS", "CASSF", "TGTGCCAGCAGTTTC")  # Deletion at end
    'TGTGCCAGCAGT'
    >>> mutate_dna_by_protein("CSSF", "CASSF", "TGTGCCAGCAGTTTC")  # Deletion in middle
    'TGTAGCAGTTTC'
    >>> mutate_dna_by_protein("CSS", "CASSF", "TGTGCCAGCAGTTTC")  # Deletion in end and middle
    'TGTAGCAGT'
    >>> mutate_dna_by_protein("CASSFC", "CASSF", "TGTGCCAGCAGTTTC")  # Insert at end
    'TGTGCCAGCAGTTTCTGT'
    >>> mutate_dna_by_protein("CASSCF", "CASSF", "TGTGCCAGCAGTTTC")  # Insert in the middle
    'TGTGCCAGCAGTTGTTTC'
    >>> mutate_dna_by_protein("ASSCF", "CASSF", "TGTGCCAGCAGTTTC")  # Insert with deletion prior
    'GCCAGCAGTTGTTTC'
    >>> mutate_dna_by_protein("CASACSF", "CASSF", "TGTGCCAGCAGTTTC")  # Two insertions
    'TGTGCCAGCGCTTGTAGTTTC'
    """
    table = CodonTable.standard_dna_table
    assert utils.nt2aa(nt_old) == prot_old, "Mismatched prot/nt old sequences"
    nt_codons = [nt_old[i : i + 3] for i in range(0, len(nt_old), 3)]
    edits = Levenshtein.editops(prot_old, prot_new)  # source --> destination string
    for edit in edits[::-1]:  # Avoids messing up indexing
        # positions are in protein land, change to nucleotide land
        op, s_pos_aa, d_pos_aa = edit
        if op == "delete":
            nt_codons[s_pos_aa] = ""
        elif op == "insert":
            nt_codons.insert(s_pos_aa, table.back_table[prot_new[d_pos_aa]])
        elif op == "replace":
            src_codon = nt_codons[s_pos_aa]
            assert utils.nt2aa(src_codon) == prot_old[s_pos_aa]
            # Find all codons that match the destination aa
            dst_codons = [
                codon
                for codon, aa in table.forward_table.items()
                if aa == prot_new[d_pos_aa]
            ]
            dst_codon_dists = [Levenshtein.distance(x, src_codon) for x in dst_codons]
            dst_codon_min_dist = min(dst_codon_dists)
            dst_codon_accept = [
                c
                for dist, c in zip(dst_codon_dists, dst_codons)
                if dist == dst_codon_min_dist
            ]
            nt_codons[s_pos_aa] = dst_codon_accept.pop()
        else:
            raise NotImplementedError(f"Cannot yet handle {op}")
    retval = "".join(nt_codons)
    assert utils.nt2aa(retval) == prot_new
    return retval


def splice_in_seq(new_seq: str, old_seq: str, full_seq: str) -> str:
    """
    Replace old_seq with new_seq in full_seq. full_seq is expected to contain old_seq
    >>> splice_in_seq("CASS", "CASR", "ABCDEFGCASRZZZ")
    'ABCDEFGCASSZZZ'
    >>> splice_in_seq("$$", "CASS", "ABCDEFGCASSYLMZ")
    'ABCDEFG$$YLMZ'
    >>> splice_in_seq("CASSRKDES", "CASSRKDDD", "CASSRKDDD")
    'CASSRKDES'
    """
    # Locate the old sequence in the full sequence
    assert old_seq in full_seq, f"Could not find {old_seq} in {full_seq}"
    assert full_seq.count(old_seq) == 1, "Sequence to be replaced is nonunique"
    old_seq_start_idx = full_seq.index(old_seq)
    old_seq_stop_idx = old_seq_start_idx + len(old_seq)
    assert full_seq[old_seq_start_idx:old_seq_stop_idx] == old_seq

    # replace old seq with new seq
    retval = full_seq[:old_seq_start_idx] + new_seq + full_seq[old_seq_stop_idx:]
    return retval


def splice_in_row(
    tra: str, trb: str, ref_row: pd.Series, src_notes: str = ""
) -> pd.Series:
    """
    Splice the TRA and TRB into the reference row given
    """
    retval = ref_row.copy(deep=True)

    # Update basics
    retval["tcr_cdr3s_aa"] = f"TRA:{tra};TRB:{trb}"
    retval["TRA_cdr3"] = tra
    retval["TRB_cdr3"] = trb

    # Splice in the full TRUST4 seqs for TRA/TRB amino acids
    retval["TRA_cdr3_TRUST4"] = splice_in_seq(
        tra, ref_row["TRA_cdr3"], ref_row["TRA_cdr3_TRUST4"]
    )
    retval["TRB_cdr3_TRUST4"] = splice_in_seq(
        trb, ref_row["TRB_cdr3"], ref_row["TRB_cdr3_TRUST4"]
    )

    # Create new NT sequences
    new_tra_nt_seq = mutate_dna_by_protein(
        tra, ref_row["TRA_cdr3"], ref_row["TRA_cdr3_nt"]
    )
    new_trb_nt_seq = mutate_dna_by_protein(
        trb, ref_row["TRB_cdr3"], ref_row["TRB_cdr3_nt"]
    )
    retval["tcr_cdr3s_nt"] = f"TRA:{new_tra_nt_seq};TRB:{new_trb_nt_seq}"
    retval["TRA_cdr3_nt"] = new_tra_nt_seq
    retval["TRB_cdr3_nt"] = new_trb_nt_seq

    # Get the full nucleotide sequences
    new_tra_nt_full = splice_in_seq(
        new_tra_nt_seq, ref_row["TRA_cdr3_nt"], ref_row["TRA_cdr3_nt_TRUST4"]
    )
    retval["TRA_cdr3_nt_TRUST4"] = new_tra_nt_full
    new_trb_nt_full = splice_in_seq(
        new_trb_nt_seq, ref_row["TRB_cdr3_nt"], ref_row["TRB_cdr3_nt_TRUST4"]
    )
    retval["TRB_cdr3_nt_TRUST4"] = new_trb_nt_full

    # Get full consensus read information
    tra_consensus = splice_in_seq(
        new_tra_nt_full,
        ref_row["TRA_cdr3_nt_TRUST4"],
        ref_row["TRA_longest_consensus_read_TRUST4"],
    )
    retval["TRA_longest_consensus_read_TRUST4"] = tra_consensus
    trb_consensus = splice_in_seq(
        new_trb_nt_full,
        ref_row["TRB_cdr3_nt_TRUST4"],
        ref_row["TRB_longest_consensus_read_TRUST4"],
    )
    retval["TRB_longest_consensus_read_TRUST4"] = trb_consensus

    # Checks and logging
    assert np.all(retval.index == ref_row.index)
    check_row(retval)
    a_dist = Levenshtein.distance(
        ref_row["TRA_longest_consensus_read_TRUST4"],
        retval["TRA_longest_consensus_read_TRUST4"],
    )
    b_dist = Levenshtein.distance(
        ref_row["TRB_longest_consensus_read_TRUST4"],
        retval["TRB_longest_consensus_read_TRUST4"],
    )
    logging.info(
        f"Mismatches from full ref TRA to new TRA: {a_dist}/{len(ref_row['TRA_longest_consensus_read_TRUST4'])}"
    )
    logging.info(
        f"Mismatches from full ref TRB to new TRB: {b_dist}/{len(ref_row['TRB_longest_consensus_read_TRUST4'])}"
    )

    # Insert a column from the notes
    retval["notes"] = src_notes
    retval["TRA_consensus_num_nt_muts"] = a_dist
    retval["TRB_consensus_num_nt_muts"] = b_dist

    return retval


def check_row(row: pd.Series) -> None:
    """
    Checks if the given row is valid. All checks done as assert statments
    so will crash if anything fails
    """
    # We should be building only on tetramer positive cases
    assert row["tetramer_class"] == "TetPos"

    # Check that columns match up
    assert row["tcr_cdr3s_aa"].split(";")[0][4:] == row["TRA_cdr3"]
    assert row["tcr_cdr3s_aa"].split(";")[1][4:] == row["TRB_cdr3"]
    assert row["tcr_cdr3s_nt"].split(";")[0][4:] == row["TRA_cdr3_nt"]
    assert row["tcr_cdr3s_nt"].split(";")[1][4:] == row["TRB_cdr3_nt"]

    # Check that columns encoding each other are consistent
    assert utils.nt2aa(row["tcr_cdr3s_nt"].split(";")[0][4:]) == row["TRA_cdr3"]
    assert utils.nt2aa(row["tcr_cdr3s_nt"].split(";")[1][4:]) == row["TRB_cdr3"]

    # Check that seq1uences are subsets of the TRUST4 sequences
    assert row["TRB_cdr3"] in row["TRB_cdr3_TRUST4"]
    assert row["TRA_cdr3"] in row["TRA_cdr3_TRUST4"]
    assert row["TRA_cdr3_nt"] in row["TRA_cdr3_nt_TRUST4"]
    assert row["TRB_cdr3_nt"] in row["TRB_cdr3_nt_TRUST4"]
    assert row["TRA_cdr3_nt_TRUST4"] in row["TRA_longest_consensus_read_TRUST4"]
    assert row["TRB_cdr3_nt_TRUST4"] in row["TRB_longest_consensus_read_TRUST4"]

    # Check no stop codon
    tra_chunks = dl.chunkify(row["TRA_cdr3_nt"], 3)
    assert not set(tra_chunks).intersection(STOP_CODONS), "Found stop codons in TRA"
    trb_chunks = dl.chunkify(row["TRB_cdr3_nt"], 3)
    assert not set(trb_chunks).intersection(STOP_CODONS), "Found stop codons in TRB"

    # Check construction rules
    # assert (
    #     "gagggcagaggaagtctgctaacatgcggtgacgtcgaggagaatcctggccca".upper()
    #     in row["TRB_cdr3_nt_TRUST4"].upper()
    # ), "Cannot find T2A sequence in TRB"
    # assert (
    #     "GCCACC" in row["TRB_longest_consensus_read_TRUST4"]
    # ), "Cannot find Kozak sequence"
    # assert (
    #     "GGCTCCGGA" in row["TRB_longest_consensus_read_TRUST4"]
    # ), "Cannot find GSG linker"


def main():
    """
    Run script
    """
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Read in the reference data
    ref_df = pd.read_csv(args.reference_file)
    ref_df.dropna(axis="columns", how="all", inplace=True)
    ref_df.dropna(axis="rows", how="all", inplace=True)

    # Read in the generated sequences
    with open(args.input_json) as source:
        eng_tcr = json.load(source)

    # For each, find the match
    match_dists, matches = [], []
    for pair in eng_tcr[-1]:
        match_dist, m = find_best_ref_match(pair, ref_df, args.mode)
        match_dists.append(match_dist)
        matches.append(m)
    assert len(match_dists) == len(matches) == len(eng_tcr[-1])
    print(collections.Counter(match_dists))

    uniq_match_dists = sorted(list(set(match_dists)))
    print(uniq_match_dists)

    if args.mode == "edit":
        best_dist, second_best_dist, third_best_dist = uniq_match_dists[:3]
        assert best_dist < second_best_dist
    elif args.mode == "blosum":
        best_dist, second_best_dist, third_best_dist = uniq_match_dists[::-1][:3]
        assert best_dist > second_best_dist
    else:
        raise ValueError

    # The best matches:
    print(f"Best best: {best_dist}")
    best_mod_rows = []
    idx = np.where(np.array(match_dists) == best_dist)[0]
    for i in idx:
        print(f"Engineered: {eng_tcr[-1][i]}")
        print(f"Original: {matches[i]['TRA_cdr3'], matches[i]['TRB_cdr3']}")
        r = splice_in_row(
            *eng_tcr[-1][i], matches[i], src_notes=f"Best {args.mode} = {best_dist}"
        )
        best_mod_rows.append(r)
    pd.DataFrame(best_mod_rows).to_csv(os.path.join(args.outdir, "best_matches.csv"))

    # Second best matches
    print(f"Second best: {second_best_dist}")
    second_best_mod_rows = []
    idx = np.where(np.array(match_dists) == second_best_dist)[0]
    for i in idx:
        print(eng_tcr[-1][i])
        print(matches[i]["TRA_cdr3"], matches[i]["TRB_cdr3"])
        r = splice_in_row(
            *eng_tcr[-1][i],
            matches[i],
            src_notes=f"Second best {args.mode} = {second_best_dist}",
        )
        second_best_mod_rows.append(r)
    pd.DataFrame(second_best_mod_rows).to_csv(
        os.path.join(args.outdir, "second_best_matches.csv")
    )

    # Third best
    print(f"Third best: {third_best_dist}")
    third_best_mod_rows = []
    for i in np.where(np.array(match_dists) == third_best_dist)[0]:
        print(eng_tcr[-1][i])
        print(matches[i]["TRA_cdr3"], matches[i]["TRB_cdr3"])
        r = splice_in_row(
            *eng_tcr[-1][i],
            matches[i],
            src_notes=f"Third best {args.mode} = {third_best_dist}",
        )
        third_best_mod_rows.append(r)
    pd.DataFrame(third_best_mod_rows).to_csv(
        os.path.join(args.outdir, "third_best_matches.csv")
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
