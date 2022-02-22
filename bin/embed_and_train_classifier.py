"""
Script for embedding and classifying input sequences
"""

import os
import sys
import logging
import argparse
import json
from joblib import dump
from typing import *

import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.base import BaseEstimator
from sklearn import metrics

import git

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tcr")
)
import featurization as ft
import data_loader as dl
import model_utils
import canonical_models as models

logging.basicConfig(level=logging.INFO)


def get_model(keyword: str, n_components: int) -> BaseEstimator:
    """
    Return a sklearn type model given a keyword
    """
    if keyword == "pcasvm":
        cls = models.ModelOnPCA(
            SVC, n_components=n_components, probability=True, kernel="rbf"
        )
    elif keyword == "svm":
        cls = SVC(probability=True, kernel="rbf", random_state=6489)
    elif keyword == "lr":
        cls = LogisticRegression(penalty="l2", solver="liblinear")
    elif keyword == "gpc":
        cls = GaussianProcessClassifier()
    else:
        raise ValueError(f"Unrecognized classifier: {keyword}")
    logging.info(f"Classifier {cls}")
    return cls


def read_input_files(
    fname: str, has_header: bool = False
) -> Tuple[List[str], List[str]]:
    """Read the tab-delimited input"""
    df = pd.read_csv(fname, delimiter="\t", comment="#")
    seq = list(df.iloc[:, 0])
    label = list(df.iloc[:, 1])
    return seq, label


def build_parser() -> argparse.ArgumentParser:
    """CLI parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Input tab-separated file. First column of TRB sequences, second column of labels. Comments may be prefixed by #",
    )
    parser.add_argument("outdir", type=str, help="Output directory to write results to")

    parser.add_argument(
        "-t",
        "--test",
        type=str,
        required=False,
        help="Test file (formatted as input file). If not given, infile will be randomly split to create a test set (70/30)",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="wukevin/tcr-bert",
        help="Transformer to use, path or huggingface model hub identifier",
    )
    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        choices=["svm", "pcasvm", "lr", "gpc"],
        default="svm",
        help="Classifier to train",
    )
    parser.add_argument(
        "-l", "--layer", type=int, default=-1, help="Transformer layer to use"
    )
    parser.add_argument(
        "-n",
        "--numpcs",
        type=int,
        default=50,
        help="Number of PCs to use uf using pcasvm",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        required=False,
        default=None,
        type=int,
        help="GPU to use for generating embeddings. If not given or no GPU available, defaults to CPU",
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    # Log git version
    repo = git.Repo(
        path=os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True
    )
    logging.info(f"Git commit: {repo.head.object.hexsha}")

    # Load the input sequences
    seqs, labels = read_input_files(args.infile)
    if args.test:
        logging.info("Reading in test set examples")
        test_seqs, test_labels = read_input_files(args.test)
        train_seqs, train_labels = seqs, labels
    else:
        logging.info("Randomly creating test set")
        # Randomly create split
        indices = np.arange(len(seqs))
        train_idx, _, test_idx = dl.shuffle_indices_train_valid_test(
            indices, valid=0, test=0.3
        )
        train_seqs = [seqs[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_seqs = [seqs[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
    logging.info(f"Training: {len(train_seqs)}")
    logging.info(f"Testing:  {len(test_seqs)}")

    # Embed the sequences
    train_embed = model_utils.get_transformer_embeddings(
        model_dir=args.transformer,
        seqs=train_seqs,
        layers=[args.layer],
        method="mean",
        device=args.gpu,
    )
    test_embed = model_utils.get_transformer_embeddings(
        model_dir=args.transformer,
        seqs=test_seqs,
        layers=[args.layer],
        method="mean",
        device=args.gpu,
    )

    cls = get_model(args.classifier, args.numpcs)
    cls.fit(train_embed, train_labels)

    # Report test set performance
    test_preds = cls.predict_proba(test_embed)[:, 1]
    auprc = metrics.average_precision_score(test_labels, test_preds)
    auroc = metrics.roc_auc_score(test_labels, test_preds)
    logging.info(f"Test AUROC: {auroc:.4f}")
    logging.info(f"Test AUPRC: {auprc:.4f}")

    # Save the model
    if not os.path.isdir(args.outdir):
        logging.info(f"Creating output directory: {args.outdir}")
        os.makedirs(args.outdir)
    # Write metadata like sklearn version and transformer used
    metadata_dict = {
        "sklearn": sklearn.__version__,
        "transformer": os.path.abspath(args.transformer)
        if os.path.isdir(args.transformer)
        else args.transformer,
        "commit": repo.head.object.hexsha,
    }
    with open(os.path.join(args.outdir, "metadata.json"), mode="w") as sink:
        json.dump(metadata_dict, sink, indent=4)
    cls_fname = os.path.join(args.outdir, f"{args.classifier}.sklearn")
    logging.info(f"Writing {args.classifier} model to {cls_fname}")
    dump(cls, cls_fname)


if __name__ == "__main__":
    main()
