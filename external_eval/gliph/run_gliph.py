"""
Short script for running gliph and evaluating its output

Inspired by: https://github.com/sidhomj/DeepTCR/blob/master/ancillary_analysis/unsupervised/Run_Gliph.py
"""

import os, sys
import multiprocessing
import tempfile
import argparse
import shutil
import shlex
import subprocess
import logging
from typing import *

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tcr"
)
assert os.path.isdir(SRC_DIR), f"Cannot find src dir: {SRC_DIR}"
sys.path.append(SRC_DIR)
import custom_metrics

logging.basicConfig(level=logging.INFO)


def run_gliph(fname: str, cutoff: int = 0) -> Collection[Tuple[str]]:
    """
    Run gliph. Note that we copy the fname into the current directory to make
    working with GLIPH's outputs a little easier

    Returns the path to the convergence groups txt file output
    """
    assert os.path.isfile(fname)
    gliph_binary = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "bin/gliph-group-discovery.pl")
    )
    assert os.path.isfile(gliph_binary)

    with tempfile.TemporaryDirectory() as tmpdir:
        logging.info(f"Running GLIPH in temp dir: {tmpdir}")
        dst = shutil.copy(fname, os.path.join(tmpdir, "cdr3.txt"))
        cmd = f"{gliph_binary} --tcr={dst} --gccutoff={cutoff}"
        logging.info(f"Calling GLIPH with cutoff {cutoff}: {cmd}")
        retval = subprocess.call(
            shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        assert retval == 0
        cluster_fname = dst.replace(".txt", "-convergence-groups.txt")
        assert os.path.isfile(cluster_fname), f"Cannot find {cluster_fname}"
        clusters = collect_gliph_clusters(cluster_fname)
    return clusters


def collect_gliph_clusters(fname: str) -> Collection[Tuple[str]]:
    """Collect clusters defined by GLIPH"""
    df_gliph_conv_groups = pd.read_csv(fname, sep="\t")
    retval = [
        df_gliph_conv_groups.iloc[i, 2].split()
        for i in range(len(df_gliph_conv_groups))
    ]
    return retval


def build_parser():
    """Build CLI parser"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "filename", type=str, help="File containing a TCR column and a label column"
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        nargs="*",
        default=list(range(1, 21)),
        help="GLIPH cutoff",
    )
    parser.add_argument("-o", "--output", type=str, default="gliph_metrics.csv")
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    reference = pd.read_csv(args.filename, sep="\t")
    ref_mapping = {row["CDR3b"]: row["label"] for _i, row in reference.iterrows()}

    arg_list = [(args.filename, c) for c in args.cutoff]
    pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), len(args.cutoff)))
    per_cutoff_clusters = list(pool.starmap(run_gliph, arg_list))

    perf_df = pd.DataFrame(
        0,
        index=sorted(args.cutoff),
        columns=["perc_clustered", "perc_correct", "n_clusters"],
    )
    for cutoff, gliph_clusters in zip(args.cutoff, per_cutoff_clusters):
        logging.info(f"Cutoff: {cutoff}")
        logging.info(f"GLIPH clusters: {len(gliph_clusters)}")

        clustered, correct = custom_metrics.percent_and_correct_clustered(
            ref_mapping, gliph_clusters
        )
        logging.info(f"Percent clustered: {clustered:.4f}")
        logging.info(f"Percent correct:   {correct:.4f}")
        perf_df.loc[cutoff] = (clustered, correct, len(gliph_clusters))

    perf_df.to_csv(args.output)


if __name__ == "__main__":
    main()
