"""
Finetune transformer that accepts a SINGLE input (either TRA or TRB)
as opposed to two inputs (i.e. TRA and TRB) (see finetune_transformer.py)
"""

### Useful links:
### https://huggingface.co/transformers/custom_datasets.html

from enum import unique
import os, sys
import logging
import json
import itertools
import argparse
from typing import *

import tqdm

import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.special import softmax

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import skorch
import skorch.helper
import neptune

from transformers import (
    BertForSequenceClassification,
    BertConfig,
    BertTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback,
)

import git

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tcr")
sys.path.append(SRC_DIR)
import data_loader as dl
import featurization as ft
import model_utils
import utils

MODEL_DIR = os.path.join(SRC_DIR, "models")
assert os.path.isdir(MODEL_DIR)
sys.path.append(MODEL_DIR)
from transformer_custom import BertForSequenceClassificationMulti

logging.basicConfig(level=logging.INFO)

METRICS = []  # Global tracker for metrics states


def sigmoid(x):
    """Sigmoid"""
    return 1 / (1 + np.exp(-x))


def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """Compute binary metrics to report"""
    labels = pred.label_ids
    labels_expanded = np.zeros(pred.predictions.shape)
    labels_expanded[np.arange(labels.size), labels.squeeze()] += 1
    preds = pred.predictions.argmax(-1)
    preds_probs = softmax(pred.predictions, axis=1)
    if preds_probs.shape[1] == 2:
        preds_probs = preds_probs[:, 1:]
        labels_expanded = labels_expanded[:, 1:]
    assert (
        labels_expanded.shape == preds_probs.shape
    ), f"Got differing shapes: {labels_expanded.shape} {preds_probs.shape}"
    acc = metrics.accuracy_score(labels, preds)

    # Compute averages per category
    auroc_values, auprc_values = [], []
    for i in range(preds_probs.shape[-1]):
        l = labels_expanded[:, i]
        p = preds_probs[:, i]
        if len(np.unique(l)) < 2:
            continue
        auroc_values.append(metrics.roc_auc_score(l, p))
        auprc_values.append(metrics.average_precision_score(l, p))

    # Update global metrics tracker
    global METRICS
    METRICS.append(
        {"auroc_per_label": auroc_values, "auprc_per_label": auprc_values,}
    )

    auroc = np.mean(auroc_values)
    auprc = np.mean(auprc_values)
    return {
        "accuracy": acc,
        "auroc": auroc,
        "auprc": auprc,
    }


def compute_metrics_multi(pred: EvalPrediction) -> Dict[str, float]:
    """Compute multi-label metrics"""
    labels = pred.label_ids
    preds = pred.predictions
    preds_sigmoid = sigmoid(preds)
    assert labels.shape == preds.shape == preds_sigmoid.shape
    # For both, this calculates per-class metric and averages across classes
    auroc_values, auprc_values, used_classes, pos_rate = [], [], [], []
    for j in range(labels.shape[1]):
        if len(set(labels[:, j])) == 1:  # These are undefined
            continue
        used_classes.append(j)
        auroc = metrics.roc_auc_score(
            labels[:, j], preds_sigmoid[:, j], average="macro"
        )
        auprc = metrics.average_precision_score(
            labels[:, j], preds_sigmoid[:, j], average="macro"
        )
        auroc_values.append(auroc)
        auprc_values.append(auprc)
        pos_rate.append(float(np.mean(labels[:, j])))
    METRICS.append(
        {
            "auroc_per_label": auroc_values,
            "auprc_per_label": auprc_values,
            "pos_per_class": pos_rate,
            "used_classes": used_classes,
        }
    )
    return {"auroc": np.mean(auroc_values), "auprc": np.mean(auprc_values)}


def load_data_single(
    keyword: Literal["lcmv", "vdjdb", "pird", "covid", "nsclc"],
    segment: Literal["TRB", "TRA"] = "TRB",
) -> List[Tuple[str, str]]:
    """
    Load the data
    Returns pairs of (antigen, seq) or more generally (label, seq)
    Comma (,) can be included in antigen/label string to indicate multi-label
    Antigen/label can also be empty string to indicate multi-label
    """
    logging.info(f"Loading {keyword} {segment} data")
    assert segment == "TRA" or segment == "TRB"
    if keyword.lower() == "lcmv":
        if segment != "TRB":
            raise NotImplementedError
        lcmv_data = dl.load_lcmv_table()
        lcmv_trb, lcmv_labels = dl.dedup_lcmv_table_trb_only(lcmv_data)
        lcmv_labels = np.array(
            ["TetPos" in t or "TetMid" in t for t in lcmv_data["tetramer"]]
        )
        lcmv_antigen = [
            lcmv_data.iloc[i]["antigen.sequence"] if lcmv_labels[i] else ""
            for i in range(len(lcmv_data))
        ]
        return list(zip(lcmv_antigen, lcmv_trb))
    elif keyword.lower() == "vdjdb":  # Evey sequence here has a corresponding epitope
        vdjdb_tab = dl.load_vdjdb(tra_trb_filter=[segment])
        vdjdb_antigens = vdjdb_tab["antigen.epitope"]
        vdjdb_aa = vdjdb_tab["cdr3"]
        return list(zip(vdjdb_antigens, vdjdb_aa))
    elif keyword.lower() == "pird":
        tcr_key = f"CDR3.{'beta' if segment == 'TRB' else 'alpha'}.aa"
        pird_tab = dl.load_pird(with_antigen_only=True)
        pird_tab_tcr_nonnull = pird_tab.loc[~pd.isnull(pird_tab[tcr_key])]
        pird_antigens = pird_tab_tcr_nonnull["Antigen.sequence"]
        pird_tcrs = pird_tab_tcr_nonnull[tcr_key]
        logging.info(
            f"Loaded PIRD {segment} dataset with {len(pird_tcrs)} antigen-TCR pairs"
        )
        return list(zip(pird_antigens, pird_tcrs))
    elif keyword.lower() == "covid":
        if segment == "TRA":
            raise NotImplementedError
        logging.info("Loading immuneACCESS SARS-CoV-2 TRBs with TCRdb background")
        covid_table = dl.load_immuneaccess_mira_covid()
        covid_trbs = list(covid_table["TCR_aa"])
        covid_antigens = list(covid_table["Amino Acids"])
        logging.info(f"Loaded {len(covid_trbs)} SARS-COV-2 TRBs")

        # Dedup and merge TRBs with multiple labels
        uniq_trbs, uniq_antigens = dl.dedup_and_merge_labels(covid_trbs, covid_antigens)

        # Sample negatives
        tcrdb = dl.load_tcrdb()
        random_tcr_sampler = np.random.default_rng(seed=12345)
        random_tcr_idx = random_tcr_sampler.choice(
            np.arange(len(tcrdb)), size=len(uniq_trbs), replace=False
        )
        random_tcr = [tcrdb["AASeq"][i] for i in random_tcr_idx]
        random_antigens = [""] * len(random_tcr)  # Blank or unknown
        logging.info(f"Sampled {len(random_antigens)} random TRBs")

        return list(zip(uniq_antigens + random_antigens, uniq_trbs + random_tcr))
    elif keyword.lower() == "nsclc":
        if segment == "TRA":
            raise NotImplementedError
        logging.info("Loading NSCLC TRBs")
        nsclc_table = dl.load_reuben_nsclc()
        return list(zip(nsclc_table["label"], nsclc_table["aminoAcid"]))
    elif os.path.isfile(keyword):
        df = pd.read_csv(keyword, sep="\t")
        tcrs = df[segment]
        labels = df["label"]
        return list(zip(labels, tcrs))
    else:
        raise ValueError(f"Unrecognized data keyword or file not found: {keyword}")


def load_data(
    keywords: Iterable[Literal["lcmv", "VDJdb", "PIRD"]],
    segments: Iterable[Literal["TRA", "TRB"]],
    blacklist: Optional[Iterable[str]] = None,
    use_multilabel: bool = False,
) -> Dataset:
    """Load and concatenate the data"""
    # First, load blacklist
    blacklist_seqs = set()
    if blacklist is not None:
        for blacklist_fname in blacklist:
            seqs = utils.read_newline_file(blacklist_fname)
            blacklist_seqs.update(seqs)
    if blacklist_seqs:
        logging.info(f"{len(blacklist_seqs)} blacklisted sequences: {blacklist_seqs}")

    # Load in data
    antigens_seqs, aa_seqs = [], []
    for kw in keywords:
        for seg in segments:
            d = load_data_single(kw, segment=seg)
            antigen, aa = zip(*d)
            antigens_seqs.extend(antigen)
            aa_seqs.extend(aa)
            assert len(antigens_seqs) == len(aa_seqs)
    # If we have multiple labels or cases with no labels, we are in multilabel setting
    if use_multilabel:
        is_multilabel = any(["," in a or not a for a in antigens_seqs])
        if not is_multilabel:
            raise ValueError("Failed to interpret labels as multi labels")
        multilabels = [tuple(a.split(",")) for a in antigens_seqs]

        # Check for overlaps
        uniq_combos = utils.dedup(multilabels)
        has_overlaps = False
        for i, j in itertools.product(uniq_combos, uniq_combos):
            i, j = set(i), set(j)
            if i != j and i.intersection(j):
                logging.debug(f"Found overlap: {i}, {j}")
                has_overlaps = True
        if not has_overlaps:
            logging.warning(
                "Labels are multi labels, but no overlaps between labels! Is multilabel necessary?"
            )

        unique_labels = [
            seq
            for seq in utils.dedup(itertools.chain.from_iterable(multilabels))
            if seq
        ]  # Excludes empty string
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        labels = np.zeros((len(multilabels), len(unique_labels)), dtype=np.float32)
        for i, label_set in enumerate(multilabels):
            idx = [label_to_idx[a] for a in label_set if a]
            labels[i, idx] = 1
    else:
        onehot = ft.one_hot(antigens_seqs, alphabet=None)
        labels = np.where(onehot)[0]  # Index encoding, e.g. [0, 4, 2, 3]
        unique_labels = utils.dedup(antigens_seqs)
    logging.info(f"Generated labels of shape {labels.shape}")

    # Create dataset to return
    dset = dl.TcrFineTuneSingleDataset(
        aa_seqs, labels, label_continuous=False, label_labels=unique_labels
    )
    return dset


def get_bert_classifier(
    path: str, labels: Sequence[str], problem_type: str = "single_label_classification"
) -> BertForSequenceClassification:
    """Get BERT classifier model form the path"""
    bert_class = (
        BertForSequenceClassification
        if problem_type == "single_label_classification"
        else BertForSequenceClassificationMulti
    )
    logging.info(f"Loading BERT classifier for {problem_type}: {bert_class}")
    if os.path.isdir(path) or path.startswith("wukevin/"):  # Poor heuristic
        logging.info(f"Loading BERT classifier from {path} with {len(labels)} labels")
        # Remake config with correct number of labels
        cfg = BertConfig.from_pretrained(
            path,
            num_labels=len(labels),
            id2label={str(i): l for i, l in enumerate(labels)},
            label2id={l: i for i, l in enumerate(labels)},
        )
        retval = bert_class.from_pretrained(path)
        # Manually define the classifier layer to avoid shape mismatches
        retval.config = cfg
        retval.num_labels = len(labels)
        retval.classifier = nn.Linear(
            in_features=retval.classifier.in_features, out_features=len(labels)
        )
    elif os.path.isfile(path) and path.split(".")[-1] == "json":
        logging.info(
            f"Loading newly initialized BERT from {path} with {len(labels)} labels"
        )
        json_args = utils.load_json_params(path)
        cfg = BertConfig(
            **json_args,
            vocab_size=len(ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL),
            pad_token_id=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX[ft.PAD],
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
        )
        retval = bert_class(cfg)
    else:
        raise ValueError(f"Unrecognized value: {path}")
    return retval


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-p", "--pretrained", type=str, required=True, help="Pretrained network"
    )
    parser.add_argument(
        "--data",
        type=str,
        nargs="*",
        help="Datasets to train, keywords or filenames (files should be formatted as csv)",
    )
    parser.add_argument(
        "-b",
        "--blacklist",
        type=str,
        nargs="*",
        required=False,
        help="File containing labels to ignore",
    )
    parser.add_argument(
        "--multilabel",
        action="store_true",
        help="Try to interpret labels as multilabel",
    )
    parser.add_argument(
        "-s",
        "--segment",
        type=str,
        choices=["TRA", "TRB"],
        required=True,
        nargs="*",
        help="TRA or TRB",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default=os.getcwd(), help="Directory to save model"
    )
    parser.add_argument(
        "-d", "--downsample", type=float, default=1.0, help="Downsample training data"
    )
    parser.add_argument(
        "-m",
        "--monitor",
        type=str,
        default="auprc",
        choices=["auroc", "auprc", "acc", "loss"],
        help="Metric to monitor for best model",
    )
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=25, help="Max num epochs")
    parser.add_argument(
        "-w", "--warmup", type=float, default=0.1, help="Proportion of steps to warmup"
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    assert (
        0.0 < args.downsample <= 1.0
    ), f"Invalid value for downsampling: {args.downsample}"
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Setup logging
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.outdir, "classifier_training.log"), "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log git status
    repo = git.Repo(
        path=os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True
    )
    sha = repo.head.object.hexsha
    logging.info(f"Git commit: {sha}")
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    with open(os.path.join(args.outdir, "params.json"), "w") as sink:
        json.dump(vars(args), sink, indent=4)

    full_dset = load_data(args.data, args.segment, args.blacklist, args.multilabel)
    if full_dset.label_labels is not None:
        with open(os.path.join(args.outdir, "trained_labels.txt"), "w") as sink:
            for l in full_dset.label_labels:
                sink.write(l + "\n")
    train_dataset = dl.DatasetSplit(full_dset, split="train")
    valid_dataset = dl.DatasetSplit(full_dset, split="valid")
    test_dataset = dl.DatasetSplit(full_dset, split="test")

    # Write datasets, or downsample
    if not args.debug:
        # DownsampledDataset doesn't support writing
        logging.info("Writing train/valid/test dataset to disk")
        train_dataset.to_file(os.path.join(args.outdir, "train_dataset.json"))
        valid_dataset.to_file(os.path.join(args.outdir, "valid_dataset.json"))
        test_dataset.to_file(os.path.join(args.outdir, "test_dataset.json"))
    else:
        train_dataset = dl.DownsampledDataset(train_dataset, 0.05)

    if args.downsample < 1.0:
        logging.info(f"Downsampling training set to {args.downsample}")
        train_dataset = dl.DownsampledDataset(train_dataset, downsample=args.downsample)

    classifier = get_bert_classifier(
        args.pretrained,
        labels=full_dset.label_labels,
        problem_type="multi_label_classification"
        if full_dset.is_multilabel
        else "single_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=args.outdir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.monitor}",
        no_cuda=args.debug,  # Useful for debugging
        skip_memory_metrics=True,
        disable_tqdm=True,
        logging_dir=os.path.join(args.outdir, "logs"),
    )

    # Early stop uses metric_for_best_model from above
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics_multi
        if full_dset.is_multilabel
        else compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()

    # Save the global metrics before additioanl eval calls
    with open(os.path.join(args.outdir, "custom_metrics.json"), "w") as sink:
        json.dump(METRICS, sink, indent=4)

    # Save model and perform final evaluation
    trainer.save_model(args.outdir)
    valid_eval_metrics = trainer.evaluate()
    for k in sorted(valid_eval_metrics.keys()):
        logging.info(f"{k}\t{valid_eval_metrics[k]:.4f}")
    logging.info("Test set:")
    test_eval_metrics = trainer.evaluate(test_dataset)
    for k in sorted(test_eval_metrics.keys()):
        logging.info(f"{k}\t{test_eval_metrics[k]:.4f}")


if __name__ == "__main__":
    main()
