"""
Script to train model for next sentence prediction
"""

import os, sys
import logging
import json
import collections
import itertools
import argparse
from typing import *

import numpy as np
from sklearn import metrics
from scipy.special import softmax

import torch
import torch.nn as nn

import skorch
import skorch.helper
import neptune

import transformers
from transformers import (
    DataCollatorForLanguageModeling,
    BertForNextSentencePrediction,
    BertForSequenceClassification,
    BertForPreTraining,
    BertTokenizer,
    BertConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EvalPrediction,
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
from transformer_custom import BertForThreewayNextSentencePrediction


def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """Compute binary metrics to report"""
    # When doing pretraining (MLM + NSP) we have shapes:
    # predictions: (eval_examples, seq_len, 26) (eval_examples, 2)
    # labels: (seq_len,) (seq_len,)
    labels = pred.label_ids.squeeze()  # These are indexes of the correct class
    preds = pred.predictions.argmax(-1)
    labels_expanded = np.zeros_like(pred.predictions)
    labels_expanded[np.arange(len(labels_expanded)), labels] = 1
    assert np.all(np.sum(labels_expanded, axis=1) == 1)
    is_multiclass = pred.predictions.shape[1] > 2
    if is_multiclass:
        preds_probs = softmax(pred.predictions, axis=1)
    else:
        preds_probs = softmax(pred.predictions, axis=1)[:, 1]
    # Macro calculates metrics for each label and finds the unweighted mean
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        labels,
        preds,
        average="binary" if not is_multiclass else "macro",
    )
    acc = metrics.accuracy_score(labels, preds)
    auroc = metrics.roc_auc_score(labels, preds_probs, multi_class="ovr")
    auprc = metrics.average_precision_score(
        labels if not is_multiclass else labels_expanded, preds_probs
    )
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "auprc": auprc,
    }


def load_dataset(args):
    """Load in the dataset"""
    # Load the blacklist sequences:
    blacklist_antigens = None
    if args.blacklist:
        blacklist_antigens = []
        for blacklist_fname in args.blacklist:
            seqs = utils.read_newline_file(blacklist_fname)
            blacklist_antigens.extend(seqs)
        blacklist_antigens = utils.dedup(blacklist_antigens)
        logging.info(
            f"Loaded in {len(args.blacklist)} blacklist files for {len(blacklist_antigens)} unique blacklisted seqs"
        )

    # Build dataset according to given keyword arg
    if args.mode == "AB":
        logging.info("Loading 10x TRA/TRB pair dataset")
        tab = dl.load_10x()
        tra_max_len = max([len(s) for s in tab["TRA_aa"]])
        trb_max_len = max([len(s) for s in tab["TRB_aa"]])
        full_dataset = dl.TcrNextSentenceDataset(
            list(tab["TRA_aa"]),
            list(tab["TRB_aa"]),
            neg_ratio=args.negatives,
            tra_blacklist=blacklist_antigens,
            mlm=0.15 if args.mlm else 0,
        )
    elif args.mode == "antigenB":
        # Antigen and TRB sequences (EXCLUDES TRA)
        logging.info("Loading VDJdb + PIRD antigen + TRB data")
        pird_tab = dl.load_pird(with_antigen_only=True)
        pird_antigens = pird_tab["Antigen.sequence"]
        pird_trbs = pird_tab["CDR3.beta.aa"]
        # Load only TRB sequences for VDJdb
        vdjdb_tab = dl.load_vdjdb(tra_trb_filter=["TRB"])
        vdjdb_antigens = vdjdb_tab["antigen.epitope"]
        vdjdb_trbs = vdjdb_tab["cdr3"]
        full_dataset = dl.TcrNextSentenceDataset(
            list(pird_antigens) + list(vdjdb_antigens),
            list(pird_trbs) + list(vdjdb_trbs),
            neg_ratio=args.negatives,
            tra_blacklist=blacklist_antigens,
            mlm=0.15 if args.mlm else 0,
        )
    elif args.mode == "antigenAB":
        logging.info("Loading VDJdb + PIRD antigen + TRA/TRB data")
        pird_tab = dl.load_pird(with_antigen_only=True)
        pird_antigens = pird_tab["Antigen.sequence"]
        pird_trbs = pird_tab["CDR3.beta.aa"]
        pird_tras = pird_tab["CDR3.alpha.aa"]
        vdjdb_tab = dl.load_vdjdb(tra_trb_filter=["TRA", "TRB"])
        vdjdb_antigens = vdjdb_tab["antigen.epitope"]
        vdjdb_trs = vdjdb_tab["cdr3"]
        full_dataset = dl.TcrNextSentenceDataset(
            list(pird_antigens) + list(pird_antigens) + list(vdjdb_antigens),
            list(pird_trbs) + list(pird_tras) + list(vdjdb_trs),
            neg_ratio=args.negatives,
            tra_blacklist=blacklist_antigens,
            mlm=0.15 if args.mlm else 0,
        )
    elif args.mode == "antigenLCMV":
        logging.info(
            "Loading LCMV dataset, primarily used for fine tuning, ignoring given negatives ratio"
        )
        assert not args.dynamic, f"Cannot use dynamic sampling with LCMV dataset"
        lcmv_tab = dl.load_lcmv_table()
        lcmv_trbs = lcmv_tab["TRB"]
        lcmv_antigen = lcmv_tab["antigen.sequence"]
        lcmv_labels = np.array(lcmv_tab["tetramer"] == "TetPos")
        full_dataset = dl.TcrNextSentenceDataset(
            lcmv_antigen,
            lcmv_trbs,
            neg_ratio=0.0,
            tra_blacklist=blacklist_antigens,
            labels=lcmv_labels,
            mlm=0.15 if args.mlm else 0,
            shuffle=False,  # Do not shuffle dataset since we will downsample
        )
    elif args.mode.startswith("abLCMV"):
        assert args.mode in ["abLCMV", "abLCMVnoMid", "abLCMVmulticlass", "abLCMVold"]
        if args.mode == "abLCMVmulticlass":
            raise NotImplementedError

        logging.info("Loading LCMV dataset for TRA/TRB pairs, ignoring negatives ratio")
        lcmv_tab = dl.load_lcmv_table()
        if args.mode == "abLCMVnoMid":
            orig_count = len(lcmv_tab)
            lcmv_tab = lcmv_tab.loc[lcmv_tab["tetramer"] != "TetMid"]
            logging.info(f"Excluded {orig_count - len(lcmv_tab)} TetMid examples")

        if args.mode == "abLCMVold":
            # The old version where we don't dedup the data
            logging.warning(
                "Using legacy LCMV loading scheme where only TetPos is positive and we do NOT dedup"
            )
            lcmv_tras = list(lcmv_tab["TRA"])
            lcmv_trbs = list(lcmv_tab["TRB"])
            lcmv_labels = [l == "TetPos" for l in lcmv_tab["tetramer"]]

        else:
            lcmv_dedup_ab, lcmv_dedup_labels = dl.dedup_lcmv_table(lcmv_tab)
            lcmv_tras, lcmv_trbs = zip(*lcmv_dedup_ab)
            # Pos and mid are both considered positive
            lcmv_labels = np.array(
                ["TetPos" in l or "TetMid" in l for l in lcmv_dedup_labels]
            )
            logging.info(
                f"Created binary labels for LCMV dataset with {np.mean(lcmv_labels)} positive rate"
            )
        assert lcmv_labels is not None, "Failed to generate labels for LCMV"
        full_dataset = dl.TcrNextSentenceDataset(
            lcmv_tras,
            lcmv_trbs,
            neg_ratio=0.0,
            tra_blacklist=blacklist_antigens,
            labels=lcmv_labels,
            mlm=0.15 if args.mlm else 0,
            shuffle=False,  # Do not shuffle, train/valid/test split will shuffle
        )
    else:
        raise ValueError(f"Unrecognized mode: {args.mode}")
    return full_dataset


def get_model(path: str, mlm: bool = False, multiclass: bool = False):
    """Get a NSP model from the given pretrained path"""
    if mlm:
        if os.path.isdir(path):
            raise NotImplementedError("Cannot initialize MLM+NSP model from pretrained")
        elif os.path.isfile(path) and utils.is_json_file(path):
            logging.info(f"Loading BertForPretraining from scratch: {path}")
            params = utils.load_json_params(path)
            cfg = BertConfig(
                **params,
                vocab_size=len(ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL),
                pad_token_id=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX[ft.PAD],
            )
            model = BertForPreTraining(cfg)
        else:
            raise ValueError(f"Cannot initialize model from: {path}")
    else:
        if os.path.isdir(path):
            logging.info(
                f"Loading BertForNextSentencePrediction from pretrained: {path}"
            )
            if multiclass:
                model = BertForThreewayNextSentencePrediction.from_pretrained(path)
            else:
                model = BertForNextSentencePrediction.from_pretrained(path)
        elif os.path.isfile(path) and utils.is_json_file(path):
            params = utils.load_json_params(path)
            cfg = BertConfig(
                **params,
                vocab_size=len(ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL),
                pad_token_id=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX[ft.PAD],
            )
            model = BertForNextSentencePrediction(cfg)
        else:
            raise ValueError(f"Cannot initialize model from: {path}")
        logging.info("Loading BertForNextSentencePrediction")
    return model


def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to pretrained dir or config json"
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default=os.getcwd(), help="Output directory"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="antigenB",
        choices=[
            "AB",
            "antigenB",
            "antigenAB",
            "antigenLCMV",
            "abLCMV",
            "abLCMVnoMid",
            "abLCMVmulticlass",
            "abLCMVold",
        ],
        help="Pairing mode - TRA/TRB or Antigen/(TRA|TRB) or antigenLCMV (looks at TRB)",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="random",
        choices=["random", "antigen"],
        help="Method for determining data split",
    )
    parser.add_argument(
        "--blacklist",
        type=str,
        nargs="*",
        required=False,
        help="File containing ANTIGEN sequences to ignore. Useful for excluding certain antigens from training.",
    )
    parser.add_argument(
        "-n",
        "--negatives",
        type=float,
        default=1.0,
        help="Ratio of negatives to positives",
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="Downsample training to given proportion",
    )
    parser.add_argument(
        "-d",
        "--dynamic",
        action="store_true",
        help="Dynamically generate negative pairs in training",
    )
    parser.add_argument("--mlm", action="store_true", help="MLM in addition to NSP")
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Epochs to train")
    parser.add_argument(
        "-w", "--warmup", type=float, default=0.1, help="Proportion of steps to warmup"
    )
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--noneptune", action="store_true", help="Disable neptune integration"
    )
    return parser


def main():
    """Run training"""
    args = build_parser().parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Setup logging
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.outdir, "nsp_training.log"), "w")
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

    full_dataset = load_dataset(args)

    if "LCMV" not in args.mode:
        # No validation dataset
        logging.info(f"Training NSP in full, using only train/test splits")
        if args.split == "random":
            train_dataset = dl.DatasetSplit(
                full_dataset,
                split="train",
                valid=0,
                test=0.1,
                dynamic_training=args.dynamic,
            )
            eval_dataset = dl.DatasetSplit(
                full_dataset, split="test", valid=0, test=0.1
            )
        elif args.split == "antigen":
            # We split by antigen so that the evaluation and training sets are using
            # different antigens. This better captures whether or not we are learning
            # a general antigen-tcr interaction or not
            antigen_getter = lambda x: [pair[0] for pair in x.all_pairs]
            train_dataset = dl.DatasetSplitByAttribute(
                full_dataset,
                antigen_getter,
                split="train",
                dynamic_training=args.dynamic,
                valid=0,
                test=0.1,
            )
            eval_dataset = dl.DatasetSplitByAttribute(
                full_dataset,
                antigen_getter,
                split="test",
                dynamic_training=args.dynamic,
                valid=0,
                test=0.1,
            )
        else:
            raise ValueError(f"Unrecognized value for split method: {args.split}")

    else:
        logging.info(
            f"Got LCMV fine tuning dataset, using full train/valid/test split with no dynamic training"
        )
        assert not args.dynamic
        assert (
            args.split == "random"
        ), f"Only random split allowed for LCMV data with only one epitope"
        train_dataset = dl.DatasetSplit(
            full_dataset, split="train", dynamic_training=False
        )
        eval_dataset = dl.DatasetSplit(
            full_dataset, split="valid", dynamic_training=False
        )
        test_dataset = dl.DatasetSplit(
            full_dataset, split="test", dynamic_training=False
        )

    assert (
        0.0 <= args.downsample <= 1.0
    ), f"Invalid downsampling value {args.downsample}"
    if args.downsample < 1.0:
        logging.info(f"Downsampling training set to {args.downsample}")
        train_dataset = dl.DownsampledDataset(train_dataset, downsample=args.downsample)

    # Setting metric for best model auto sets greater_is_better as well
    training_args = TrainingArguments(
        output_dir=args.outdir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss" if args.mlm else "eval_auprc",
        no_cuda=False,
        skip_memory_metrics=True,
        disable_tqdm=not args.progress,
        logging_dir=os.path.join(args.outdir, "logging"),
    )

    model = get_model(args.model, mlm=args.mlm, multiclass="multiclass" in args.mode)

    # Set up neptune logging if specified
    neptune_logger = None
    neptune_tags = ["nsp", args.mode]
    if args.mlm:
        neptune_tags.append("mlm")
    if not utils.is_json_file(args.model):
        neptune_tags.append("pretrained")
    neptune_params = {
        "epochs": args.epochs,
        "batch_size": args.bs,
        "lr": args.lr,
        "warmup_ratio": args.warmup,
        "nsp_neg_ratio": args.negatives,
        "dynamic_training": args.dynamic,
        "data_split_method": args.split,
        "downsample": args.downsample,
        "blacklist": args.blacklist,
        "outdir": os.path.abspath(args.outdir),
    }
    if utils.is_json_file(args.model):
        neptune_params.update(utils.load_json_params(args.model))
    else:  # Is pretrained
        neptune_params["pretrained"] = os.path.abspath(args.model)

    if not args.noneptune:
        neptune.init(project_qualified_name="wukevin/tcr")
        experiment = neptune.create_experiment(
            name=f"nsp-{args.mode}" if not args.mlm else f"nsp-mlm-{args.mode}",
            params=neptune_params,
            tags=neptune_tags,
        )
        neptune_logger = model_utils.NeptuneHuggingFaceCallback(experiment)

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if not args.mlm else None,
        callbacks=[neptune_logger] if neptune_logger is not None else None,
        tokenizer=full_dataset.tok,
    )
    trainer.train()
    trainer.save_model(args.outdir)

    # Evaluate test set performance
    test_ab = test_dataset.all_sequences()  # Tuples of TRA/TRB
    test_labels = test_dataset.all_labels()
    test_nsp_preds = model_utils.get_transformer_nsp_preds(
        args.outdir,
        test_ab,
        inject_negatives=0,
        device=0,
    )[1][:, 1]
    test_auroc = metrics.roc_auc_score(test_labels, test_nsp_preds)
    test_auprc = metrics.average_precision_score(test_labels, test_nsp_preds)
    logging.info(f"Test set AUROC: {test_auroc:.4f}")
    logging.info(f"Test set AUPRC: {test_auprc:.4f}")

    with open(os.path.join(args.outdir, "test_perf.json"), "w") as sink:
        perf_dict = {
            "auroc": test_auroc,
            "auprc": test_auprc,
            "downsample": args.downsample,
        }
        logging.info(f"Writing test set performance metrics to {sink.name}")
        json.dump(perf_dict, sink, indent=4)


if __name__ == "__main__":
    main()
