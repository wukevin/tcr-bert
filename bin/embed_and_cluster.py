"""
Script for embedding and clustering input sequences
"""

import os
import sys
import logging
import argparse
import collections

import pandas as pd
import anndata as ad
import scanpy as sc

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tcr")
)
import featurization as ft
import model_utils
import utils

logging.basicConfig(level=logging.INFO)


def build_parser():
    """CLI parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Input file. If column-delimited, assume first column is sequences",
    )
    parser.add_argument("outfile", type=str, help="Output file to write")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["B", "AB"],
        type=str,
        default="B",
        help="Input TRB or TRA/TRB pairs",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="wukevin/tcr-bert",
        help="Path to transformer or huggingface model identifier",
    )
    parser.add_argument(
        "-l", "--layer", type=int, default=-1, help="Transformer layer to use"
    )
    parser.add_argument(
        "-r", "--res", type=float, default=32, help="Leiden clustering resolution"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        required=False,
        default=None,
        type=int,
        help="GPU to run on. If not given or no GPU available, default to CPU",
    )

    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    # Embed the layers
    embeddings = None
    if args.mode == "B":
        trbs = utils.dedup(
            [trb.split("\t")[0] for trb in utils.read_newline_file(args.infile)]
        )
        trbs = [x for x in trbs if ft.adheres_to_vocab(x)]
        logging.info(f"Read in {len(trbs)} unique valid TCRs from {args.infile}")
        obs_df = pd.DataFrame(trbs, columns=["TCR"])
        embeddings = model_utils.get_transformer_embeddings(
            model_dir=args.transformer,
            seqs=trbs,
            layers=[args.layer],
            method="mean",
            device=args.gpu,
        )
    elif args.mode == "AB":
        raise NotImplementedError
    assert embeddings is not None

    # Create an anndata object to perform clsutering
    embed_adata = ad.AnnData(embeddings, obs=obs_df)
    sc.pp.pca(embed_adata, n_comps=50)
    sc.pp.neighbors(embed_adata)
    sc.tl.leiden(embed_adata, resolution=args.res)

    # Establish groups
    # tcr_groups = embed_adata.obs.groupby("leiden")["TCR"].apply(list)
    clusters_map = collections.defaultdict(list)
    for row in embed_adata.obs.itertuples():
        clusters_map[row.leiden].append(row.TCR)
    logging.info(f"Writing {len(clusters_map)} TCR clusters to: {args.outfile}")
    with open(args.outfile, "w") as sink:
        for group in clusters_map.values():
            sink.write(",".join(group) + "\n")


if __name__ == "__main__":
    main()
