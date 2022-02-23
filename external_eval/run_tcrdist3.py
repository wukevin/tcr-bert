"""
Primarily for runtime benchmarking
"""

import os, sys
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


from tcrdist.repertoire import TCRrep


def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("infile", type=str, help="Input tsv file")
    parser.add_argument("outfile", type=str, help="File json to write output")
    parser.add_argument("-k", "--kclusters", type=int, default=400)
    return parser


def main():
    """Run script"""
    args = build_parser().parse_args()

    df = pd.read_csv(args.infile, sep="\t")

    tr = TCRrep(
        cell_df=df,
        organism="mouse",
        chains=["beta"],
        db_file="alphabeta_gammadelta_db.tsv",
        compute_distances=True,
    )

    ac = AgglomerativeClustering(
        n_clusters=args.kclusters, affinity="precomputed", linkage="average"
    )
    cluster_labels = ac.fit_predict(tr.pw_beta)
    cluster_groups = []
    for l in set(cluster_labels):  # Aggregate into groups of labels
        idx = np.where(cluster_labels == l)[0]
        cluster_groups.append([str(i) for i in idx])
    with open(args.outfile, "w") as sink:
        json.dump(cluster_groups, sink, indent=4)


if __name__ == "__main__":
    main()
