"""
Train a supervised convolutional network to perform classification of antigen binding
based on TCR sequences. This forms a baseline for other methods that we explore.
"""
import os, sys
import json
import logging
import argparse

import numpy as np
from sklearn import metrics

import torch
from torch.utils.data import Subset
import skorch
import skorch.helper
import neptune

import git

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tcr")
sys.path.append(SRC_DIR)
import data_loader as dl
import utils

sys.path.append(os.path.join(SRC_DIR, "models"))
import conv

logging.basicConfig(level=logging.INFO)
torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)  # Causes errors on GPU

REDUCE_LR_ON_PLATEAU_PARAMS = {
    "mode": "min",
    "factor": 0.1,
    "patience": 10,
    "min_lr": 1e-6,
}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="Downsample training to given proportion",
    )
    parser.add_argument(
        "-o", "--outdir", default=os.getcwd(), type=str, help="Output directory"
    )
    parser.add_argument(
        "--onehot", action="store_true", help="Use one-hot instead of embedding",
    )
    parser.add_argument("--batchsize", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "--maxepochs", type=int, default=500, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--earlystop", type=int, default=25, help="Early stopping patience"
    )
    parser.add_argument(
        "--monitor",
        choices=["loss", "auroc", "auprc"],
        default="auprc",
        help="Metric to monitor for checkpointing and early stopping",
    )
    parser.add_argument(
        "--min-edit",
        dest="min_edit",
        type=int,
        default=3,
        help="Minimum (inclusive) edit distance between each item in training TRA/B pairs and a test TRA/B pair",
    )
    parser.add_argument(
        "--noneptune", action="store_true", help="Disable neptune logging"
    )
    parser.add_argument("--device", type=int, default=0, help="GPU to train on")
    parser.add_argument("--seed", type=int, default=2394, help="Random seed")
    return parser


def main():
    """Load data and run training"""
    parser = build_parser()
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        logging.info(f"Creating output directory: {args.outdir}")
        os.makedirs(args.outdir)

    # Neptune
    neptune_logger = None
    if not args.noneptune:
        neptune.init(project_qualified_name="wukevin/tcr")
        experiment = neptune.create_experiment(
            name="supervised-cnn",
            params={
                "downsample": args.downsample,
                "onehot": args.onehot,
                "lr": args.lr,
                "early_stop_patience": args.earlystop,
                "batch_size": args.batchsize,
                "max_epochs": args.maxepochs,
            },
            tags=["cnn", "supervised"],
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

    # Lot torch version and params
    with open(os.path.join(args.outdir, "params.json"), "w") as sink:
        json.dump(vars(args), sink, indent=4)
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    # Load data
    tcr_table = dl.load_lcmv_table()
    if not args.onehot:
        tcr_dset = dl.TcrABSupervisedIdxDataset(tcr_table, idx_encode=True)
    else:
        tcr_dset = dl.TcrABSupervisedOneHotDataset(tcr_table, idx_encode=True)
    train_dset = dl.DatasetSplit(tcr_dset, "train")
    valid_dset = dl.DatasetSplit(tcr_dset, "valid")
    test_dset = dl.DatasetSplit(tcr_dset, "test")

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

    if args.downsample < 1.0:
        assert args.downsample > 0.0
        logging.info(f"Downsampling training set to {args.downsample}")
        train_dset = dl.DownsampledDataset(train_dset, downsample=args.downsample)

    utils.seed_all(args.seed)
    # Build model
    module_kwargs = {
        "module": conv.TwoPartConvNet,
        "module__use_embedding": not args.onehot,
    }
    net = skorch.NeuralNet(
        **module_kwargs,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=args.lr,
        max_epochs=args.maxepochs,
        batch_size=args.batchsize,
        train_split=skorch.helper.predefined_split(valid_dset),
        device=utils.get_device(args.device),
        callbacks=[
            skorch.callbacks.LRScheduler(
                policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                **REDUCE_LR_ON_PLATEAU_PARAMS,
            ),
            skorch.callbacks.GradientNormClipping(gradient_clip_value=5),
            skorch.callbacks.EpochScoring(
                "roc_auc", lower_is_better=False, on_train=False, name="valid_auroc"
            ),
            skorch.callbacks.EpochScoring(
                "average_precision",
                lower_is_better=False,
                on_train=False,
                name="valid_auprc",
            ),
            skorch.callbacks.EarlyStopping(
                patience=args.earlystop,
                monitor=f"valid_{args.monitor}",
                lower_is_better=args.monitor == "loss",
            ),
            skorch.callbacks.Checkpoint(  # Seems to cause errors if placed before scoring
                dirname=args.outdir,
                fn_prefix="net_",
                monitor=f"valid_{args.monitor}_best",
            ),
            neptune_logger,
        ],
    )
    net.classes_ = np.unique(valid_dset.all_labels())
    net.fit(train_dset)
    net.history.to_file(os.path.join(args.outdir, "history.json"))
    logging.info(f"Trained network:\n{net}")

    # Compute various validation stats
    cp = skorch.callbacks.Checkpoint(dirname=args.outdir, fn_prefix="net_")
    net.load_params(checkpoint=cp)
    valid_preds = net.predict_proba(valid_dset)[:, 1]
    valid_truth = valid_dset.all_labels(idx_encode=False)[:, 1]
    auroc = metrics.roc_auc_score(valid_truth, valid_preds)
    auprc = metrics.average_precision_score(valid_truth, valid_preds)
    acc = metrics.accuracy_score(valid_truth, np.round(valid_preds))
    logging.info(f"Valid AUROC: {auroc:.4f}")
    logging.info(f"Valid AUPRC: {auprc:.4f}")
    logging.info(f"Valid Acc:   {acc:.4f}")
    if neptune_logger is not None:
        neptune_logger.experiment.log_metric("auroc", auroc)
        neptune_logger.experiment.log_metric("auprc", auprc)
        neptune_logger.experiment.log_metric("accuracy", acc)

    # Compute test set stats
    test_preds = net.predict_proba(test_dset)[:, 1]
    test_truth = test_dset.all_labels(idx_encode=False)[:, 1]
    test_auroc = metrics.roc_auc_score(test_truth, test_preds)
    test_auprc = metrics.average_precision_score(test_truth, test_preds)
    test_acc = metrics.accuracy_score(test_truth, np.round(test_preds))
    logging.info(f"Test AUROC: {test_auroc:.4f}")
    logging.info(f"Test AUPRC: {test_auprc:.4f}")
    logging.info(f"Test acc:   {test_acc:.4f}")
    if neptune_logger is not None:
        neptune_logger.experiment.log_metric("test_auroc", auroc)
        neptune_logger.experiment.log_metric("test_auprc", auprc)
        neptune_logger.experiment.log_metric("test_accuracy", acc)
    # Write test set performance
    perf_dict = {
        "auroc": test_auroc,
        "auprc": test_auprc,
        "accuracy": test_acc,
        "downsample": args.downsample,
    }
    with open(os.path.join(args.outdir, "test_perf.json"), "w") as sink:
        logging.info(f"Writing test set performance to {sink.name}")
        json.dump(perf_dict, sink, indent=4)
    # Write test set predictions
    np.savetxt(os.path.join(args.outdir, "test_preds.txt"), test_preds)
    np.savetxt(os.path.join(args.outdir, "test_truth.txt"), test_truth)


if __name__ == "__main__":
    main()
