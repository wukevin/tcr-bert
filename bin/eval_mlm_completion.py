"""
For now evaluate on LCMV (hard-coded)
"""

import os, sys
import itertools
import collections
import logging
import argparse
from typing import *

import numpy as np
import pandas as pd

from transformers import BertForMaskedLM, BertTokenizer, FillMaskPipeline

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tcr")
sys.path.append(SRC_DIR)
import data_loader as dl
import model_utils
import featurization as ft
import custom_metrics
import utils

logging.basicConfig(level=logging.INFO)


def load_data(
    identifier: Literal["LCMV", "TCRdb", "PIRD"],
    incl_tra: bool = True,
    incl_trb: bool = True,
) -> List[str]:
    """
    Load data
    """
    seqs = []
    if identifier == "LCMV":
        lcmv_table = dl.load_lcmv_table()
        if incl_tra:
            seqs.extend(list(lcmv_table["TRA"]))
        if incl_trb:
            seqs.extend(list(lcmv_table["TRB"]))
    elif identifier == "TCRdb":
        tcrdb_table = dl.load_tcrdb()
        if incl_tra:
            s = tcrdb_table.loc[tcrdb_table["tra_trb"] == "TRA", "AASeq"]
            logging.info(f"Adding {len(s)} TRA sequences from TCRdb")
            seqs.extend(s)
        if incl_trb:
            s = tcrdb_table.loc[tcrdb_table["tra_trb"] == "TRB", "AASeq"]
            logging.info(f"Adding {len(s)} TRB sequences from TCRdb")
            seqs.extend(s)
    elif identifier == "PIRD":
        pird_table = dl.load_pird()
        if incl_tra:
            seqs.extend([aa for aa in pird_table["CDR3.alpha.aa"] if not pd.isnull(aa)])
        if incl_trb:
            seqs.extend([aa for aa in pird_table["CDR3.beta.aa"] if not pd.isnull(aa)])
    else:
        raise ValueError(f"Unrecognized data identifier: {identifier}")
    return seqs


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="LCMV",
        choices=["TCRdb", "LCMV", "PIRD"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="wukevin/tcr-bert-mlm-only",
        help="Model to evaluate",
    )
    parser.add_argument(
        "-t",
        "--tcr",
        type=str,
        choices=["TRA", "TRB"],
        nargs="*",
        required=True,
        help="Which sequence to evaluate",
    )
    parser.add_argument(
        "-k", "--topk", type=int, default=1, help="Evaluate top-k accuracy"
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=0,
        help="Subsample to this number of points. Default of 0 indicates no subsampling",
    )
    parser.add_argument(
        "-b", "--blosum", action="store_true", help="Report blosum scoring",
    )
    parser.add_argument("--device", type=int, default=0, help="Device to run")
    return parser


def main():
    """Run script"""
    args = build_parser().parse_args()

    blosum = custom_metrics.load_blosum()
    # Load data and retain unique sequences
    seqs = sorted(list(set(load_data(args.data, "TRA" in args.tcr, "TRB" in args.tcr))))

    # Subsample if necessary
    if args.num > 0 and args.num < len(seqs):
        logging.info(f"Subsampling to {args.num} sequences")
        idx = np.arange(len(seqs))
        np.random.seed(1234)
        np.random.shuffle(idx)
        seqs = [seqs[i] for i in idx[: args.num]]
        assert len(seqs) == args.num

    logging.info(f"Evaluating {len(seqs)} sequences ({', '.join(args.tcr)})")
    masker = ft.SequenceMasker(seqs)

    # Logits ouptut by this are of shape (batch, seq_len, vocab)
    pipeline = model_utils.load_fill_mask_pipeline(args.model, args.device)
    # tok = pipeline.tokenizer(ft.insert_whitespace("RKDES"), return_tensors="pt")
    # tok = {k: v.to(utils.get_device(args.device)) for k, v in tok.items()}
    # print(pipeline.model(**tok).logits.shape)
    preds = list(
        itertools.chain.from_iterable(
            [pipeline(chunk) for chunk in dl.chunkify(masker.masked, 512)]
        )
    )

    # Metrics
    mistake_tuples = []
    correct, total = 0, 0
    for t, p in zip(masker.masked_truth, preds):
        assert len(t) == 1
        predicted_aa = [p[i]["token_str"] for i in range(args.topk)]
        if t in predicted_aa:
            correct += 1
        total += 1
        top_pred = p[0]["token_str"]
        if top_pred != t:
            mistake_tuples.append((t, top_pred))

    acc = correct / total
    logging.info(f"Top-{args.topk} accuracy: {acc:.4f}")

    # Also report the most common mistakes
    logging.info("Most common mistakes")
    for k, v in collections.Counter(mistake_tuples).most_common(20):
        logging.info(f"{k[0]} -> {k[1]}: {v}\tBLOSUM score: {blosum.loc[k[0], k[1]]}")

    # Baseline
    for baseline_method in ["random", "most_common", "most_common_positional"]:
        baseline_correct = sum(
            [
                t in p
                for t, p in zip(
                    masker.masked_truth,
                    masker.get_naive_predictions(args.topk, baseline_method),
                )
            ]
        )
        baseline_acc = baseline_correct / total
        logging.info(
            f"Top-{args.topk} {baseline_method} baseline accuracy: {baseline_acc:.4f}"
        )

    # Calculate blosum scores based on top prediction
    if args.blosum:
        blosum_scores = []
        for t, p in zip(masker.masked_truth, preds):
            assert pipeline.tokenizer.decode(p[0]["token"]) == p[0]["token_str"]
            predicted_aa = p[0]["token_str"]
            try:
                blosum_scores.append(blosum.loc[t, predicted_aa].item())
            except KeyError:
                logging.warning(
                    f"Error querying pair {(t, predicted_aa)} in BLOSUM - skipping"
                )
        logging.info(f"Average blosum score: {np.mean(blosum_scores):.3f}")

        for baseline_method in ["random", "most_common", "most_common_positional"]:
            baseline_blosum = [
                # t in p
                blosum.loc[t, p[0]]
                for t, p in zip(
                    masker.masked_truth,
                    masker.get_naive_predictions(args.topk, baseline_method),
                )
                if t in blosum.index and p[0] in blosum.index
            ]
            logging.info(
                f"Average blosum score for {baseline_method} baseline: {np.mean(baseline_blosum):.3f}"
            )


if __name__ == "__main__":
    main()
