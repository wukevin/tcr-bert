"""
Finetune the transformer on both TRA and TRB by creating and tuning two copies
of 
"""

import os, sys
import logging
import json
import itertools
import argparse
from typing import *

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import Subset

import skorch
import skorch.helper
import neptune

import transformers
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
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
import transformer_custom


logging.basicConfig(level=logging.INFO)

MODEL_KEYWORD_TO_HUB_PATH = {
    "ESM": ("facebookresearch/esm", "esm1b_t33_650M_UR50S"),
}


def get_lr_scheduler_callback(kw: str, max_lr: float, min_lr: float, num_epochs: int):
    """
    Get the learning scheduler callback
    """
    if not kw:  # If no kw is specified, then do not use a scheduler
        return None
    elif kw == "plateau":
        return skorch.callbacks.LRScheduler(
            policy="ReduceLROnPlateau", patience=5, factor=0.5, threshold=min_lr
        )
    elif kw == "linear":
        min_ratio = min_lr / max_lr
        assert min_ratio <= 1.0
        # This returns the multiplier for the LR as a function of epoch
        l = lambda e: min_ratio + (1.0 - min_ratio) * (num_epochs - e) / num_epochs
        assert np.isclose(l(0), 1.0), f"Expected max ratio 1.0 but got {l(0)}"
        assert np.isclose(
            l(num_epochs), min_ratio
        ), f"Expected min ratio {min_ratio} but got {l(num_epochs)}"
        return skorch.callbacks.LRScheduler(
            policy=torch.optim.lr_scheduler.LambdaLR, lr_lambda=l
        )
    elif kw == "exponen":
        return skorch.callbacks.LRScheduler(
            policy=torch.optim.lr_scheduler.ExponentialLR, gamma=0.2
        )
    logging.warning(f"Urecognized lr policy: {kw}")
    return None


def build_parser():
    """CLI Parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-r",
        "--regression",
        type=str,
        choices=["Prop"],
        help="Regression instead of classification",
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="Downsample training to given proportion",
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        type=str,
        default="wukevin/tcr-bert",
        help="Pretrained network as a path or keyword",
    )
    parser.add_argument("--freeze", action="store_true", help="Freeze encoder")
    parser.add_argument(
        "--sharedencoder",
        action="store_true",
        help="Use the same encoder for TRA and TRB",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["max", "mean", "cls", "pool"],
        default="cls",
        help="Pooling strategy",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout in classifier"
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default=os.getcwd(), help="Output directory"
    )
    parser.add_argument("--bs", type=int, default=128, help="Batch size")  # Tuned
    parser.add_argument(
        "-e", "--epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Initial learning rate")
    parser.add_argument(
        "--minlr", type=float, default=1e-7, help="Minimum learning rate"
    )
    parser.add_argument(
        "--lrsched",
        type=str,
        choices=["plateau", "linear", "exponen"],
        default="linear",
        help="Learning rate scheduling strategy",
    )
    parser.add_argument(
        "--monitor",
        choices=["loss", "auroc", "auprc", "auto"],
        default="auto",
        help="Metric to monitor for checkpointing and early stopping",
    )
    parser.add_argument(
        "--min-edit",
        dest="min_edit",
        type=int,
        default=3,
        help="Minimum (inclusive) edit distance between each item in training TRA/B pairs and a test TRA/B pair",
    )
    parser.add_argument("--device", type=int, default=0, help="Device to train on")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU for debugging")
    parser.add_argument(
        "--noneptune", action="store_true", help="Do not log to neptune"
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    # Post process some args
    if not os.path.exists(args.pretrained) and not args.pretrained.startswith(
        "wukevin/"
    ):
        assert (
            args.pretrained in MODEL_KEYWORD_TO_HUB_PATH
        ), f"Unrecognized pretrained model spec (must be a path or keyword): {args.pretrained}"
        args.pretrained = MODEL_KEYWORD_TO_HUB_PATH[args.pretrained]
    if args.monitor == "auto":
        if not args.regression:
            logging.info(f"Defaulting to monitoring AUPRC for classification")
            args.monitor = "auprc"
        else:
            logging.info(f"Defaulting to monitoring loss for regression")
            args.monitor = "loss"

    if not os.path.exists(args.outdir):
        logging.info(f"Creating output directory: {args.outdir}")
        os.makedirs(args.outdir)

    # Check some args
    assert args.lr >= args.minlr

    # Neptune
    neptune_logger = None
    if not args.noneptune:
        neptune.init(project_qualified_name="wukevin/tcr")
        experiment = neptune.create_experiment(
            name="finetuning",
            params={
                "downsample": args.downsample,
                "lr": args.lr,
                "min_lr": args.minlr,
                "lrsched": args.lrsched,
                "monitor": args.monitor,
                "pretrained": os.path.abspath(args.pretrained),
                "batch_size": args.bs,
                "max_epochs": args.epochs,
                "dropout": args.dropout,
                "freeze_encoders": args.freeze,
                "pooling": args.pooling,
                "sharedencoder": args.sharedencoder,
            },
            tags=[
                "bert",
                "finetuned",
                "lcmv_class" if not args.regression else "lcmv_regress",
                "ab",
            ],
        )
        neptune_logger = skorch.callbacks.NeptuneLogger(
            experiment, close_after_train=False
        )

    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.outdir, "training.log"), "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log git status
    repo = git.Repo(
        path=os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True
    )
    sha = repo.head.object.hexsha
    logging.info(f"Git commit: {sha}")

    # Log torch version and params
    with open(os.path.join(args.outdir, "params.json"), "w") as sink:
        json.dump(vars(args), sink, indent=4)
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    # Load data
    tcr_table = dl.load_lcmv_table()
    lcmv_ab, lcmv_labels = dl.dedup_lcmv_table(tcr_table)
    lcmv_tra, lcmv_trb = zip(*lcmv_ab)
    if not args.regression:  # Default classifcation task
        # Consider mid/pos as positive labels
        tcr_labels = np.array(
            ["TetPos" in l or "TetMid" in l for l in lcmv_labels], dtype=np.float32,
        )
    else:
        # Use the given argument to choose column in table for regression
        tcr_labels = np.array(tcr_table[args.regression], dtype=np.float32)

    # Create datasets
    if isinstance(args.pretrained, str):
        dataset = dl.TcrFineTuneDataset(
            lcmv_tra,
            lcmv_trb,
            tcr_labels,
            skorch_mode=True,
            idx_encode=True,
            label_continuous=args.regression is not None,
        )
    else:  # Expect a tuple
        logging.info("Loading ESM alphabet")
        _m, alphabet = torch.hub.load(*args.pretrained)
        dataset = dl.TcrFineTuneDataset(
            lcmv_tra,
            lcmv_trb,
            tcr_labels,
            tokenizer=alphabet.get_batch_converter(),
            skorch_mode=True,
            idx_encode=True,
            label_continuous=args.regression is not None,
        )
    train_dset = dl.DatasetSplit(dataset, "train")
    valid_dset = dl.DatasetSplit(dataset, "valid")
    test_dset = dl.DatasetSplit(dataset, "test")

    # Filter out training examples too similar to test sequences
    if args.min_edit > 0:
        train_pairs = train_dset.all_sequences()
        test_pairs = test_dset.all_sequences()
        train_dists = dl.min_dist_train_test_pairs(train_pairs, test_pairs)

        # Combined A + B must pass cutoff
        train_accept_idx = np.where(train_dists >= args.min_edit)[0]
        logging.info(
            f"Subset to {len(train_accept_idx)}/{len(train_dset)} sequences with >= {args.min_edit} edit distance"
        )
        train_dset = Subset(train_dset, indices=train_accept_idx)

    # Downsample training
    if args.downsample < 1.0:
        assert args.downsample > 0.0, "Cannot downsample to 0 or less"
        logging.info(f"Downsampling training set to {args.downsample}")
        train_dset = dl.DownsampledDataset(train_dset, downsample=args.downsample)

    if not args.regression:
        module_class = (
            transformer_custom.TwoPartBertClassifier
            if isinstance(args.pretrained, str)
            else transformer_custom.TwoPartEsmClassifier
        )
        criterion = nn.CrossEntropyLoss  # logsoftmax + NLL
    else:
        if not isinstance(args.pretrained, str):
            raise NotImplementedError
        module_class = transformer_custom.TwoPartBertRegressor
        criterion = nn.MSELoss

    ### Build callbacks
    callbacks = [
        skorch.callbacks.GradientNormClipping(
            gradient_clip_value=1.0
        ),  # from huggingface
        get_lr_scheduler_callback(args.lrsched, args.lr, args.minlr, args.epochs),
        skorch.callbacks.EpochTimer(),
    ]
    if not args.regression:  # Classification metrics
        callbacks.extend(
            [
                skorch.callbacks.EpochScoring(
                    "roc_auc", lower_is_better=False, on_train=False, name="valid_auroc"
                ),
                skorch.callbacks.EpochScoring(
                    "average_precision",
                    lower_is_better=False,
                    on_train=False,
                    name="valid_auprc",
                ),
            ]
        )
    else:  # Regression metrics
        pass
    callbacks.extend(
        [
            skorch.callbacks.EarlyStopping(
                patience=10,
                monitor=f"valid_{args.monitor}",
                lower_is_better=args.monitor == "loss",
            ),
            skorch.callbacks.Checkpoint(  # Seems to cause errors if placed before scoring
                dirname=args.outdir,
                fn_prefix="net_",
                monitor=f"valid_{args.monitor}_best",
            ),
            neptune_logger,
        ]
    )

    ### skorch
    net = skorch.NeuralNet(
        module=module_class,
        module__pretrained=args.pretrained,
        module__freeze_encoder=args.freeze,
        module__dropout=args.dropout,
        module__separate_encoders=not args.sharedencoder,
        module__seq_pooling=args.pooling,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        lr=args.lr,  # Default from huggingface transformers
        optimizer__lr=args.lr,  # Try to see if we can override
        max_epochs=args.epochs,
        batch_size=args.bs,
        train_split=skorch.helper.predefined_split(valid_dset),
        device=utils.get_device(args.device),
        iterator_train__shuffle=True,
        iterator_train__num_workers=8,
        iterator_valid__num_workers=8,
        callbacks=callbacks,
    )
    if not args.regression:
        net.classes_ = np.unique(dataset.labels)  # sklearn compatibility
    net.fit(train_dset)
    net.history.to_file(os.path.join(args.outdir, "history.json"))

    # Compute various validation stats
    cp = skorch.callbacks.Checkpoint(dirname=args.outdir, fn_prefix="net_")
    net.load_params(checkpoint=cp)

    if not args.regression:
        valid_preds = net.predict_proba(valid_dset)[:, 1]
        valid_truth = valid_dset.all_labels(idx_encode=False)[:, 1]
    else:
        valid_preds = net.predict(valid_dset)
        valid_truth = valid_dset.all_labels()
    np.savetxt(os.path.join(args.outdir, "valid_preds.txt"), valid_preds)
    np.savetxt(os.path.join(args.outdir, "valid_truth.txt"), valid_truth)

    auroc = metrics.roc_auc_score(valid_truth, valid_preds)
    auprc = metrics.average_precision_score(valid_truth, valid_preds)
    acc = metrics.accuracy_score(valid_truth, np.round(valid_preds))
    valid_perf_dict = {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "downsample": args.downsample,
    }
    for k, v in valid_perf_dict.items():
        logging.info(f"Valid {k}:\t{v:.4f}")
        if not args.noneptune:
            neptune_logger.experiment.log_metric(k, v)
    with open(os.path.join(args.outdir, "valid_perf.json"), "w") as sink:
        logging.info(f"Writing valid set performance metrics to {sink.name}")
        json.dump(valid_perf_dict, sink, indent=4)

    # Compute various test set stats
    test_preds = net.predict_proba(test_dset)[:, 1]
    test_truth = test_dset.all_labels()
    np.savetxt(os.path.join(args.outdir, "test_preds.txt"), test_preds)
    np.savetxt(os.path.join(args.outdir, "test_truth.txt"), test_truth)
    test_auroc = metrics.roc_auc_score(test_truth, test_preds)
    test_auprc = metrics.average_precision_score(test_truth, test_preds)
    test_acc = metrics.accuracy_score(test_truth, np.round(test_preds))
    test_perf_dict = {
        "auroc": test_auroc,
        "auprc": test_auprc,
        "accuracy": test_acc,
        "downsample": args.downsample,
    }
    for k, v in test_perf_dict.items():
        logging.info(f"Test {k}:\t{v:.4f}")
    with open(os.path.join(args.outdir, "test_perf.json"), "w") as sink:
        logging.info(f"Writing test set performance metrics to {sink.name}")
        json.dump(test_perf_dict, sink, indent=4)


if __name__ == "__main__":
    main()
