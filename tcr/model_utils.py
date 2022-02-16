"""
Various model utils
"""

import os, sys
import tempfile
import subprocess
import json
import logging
from itertools import zip_longest
from typing import *

import numpy as np
import pandas as pd
from scipy.special import softmax

import torch
import torch.nn as nn
import skorch

from transformers import (
    AutoModel,
    BertModel,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForSequenceClassification,
    ConvBertForMaskedLM,
    FillMaskPipeline,
    FeatureExtractionPipeline,
    TextClassificationPipeline,
    Pipeline,
    TrainerCallback,
    TrainerControl,
)

from neptune.experiments import Experiment
from neptune.api_exceptions import ChannelsValuesSendBatchError
from transformers.utils.dummy_pt_objects import AutoModelForMaskedLM

import gdown

import data_loader as dl
import featurization as ft
import utils

SRC_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(SRC_DIR, "models")
assert os.path.isdir(MODEL_DIR)
sys.path.append(MODEL_DIR)
from transformer_custom import (
    BertForSequenceClassificationMulti,
    BertForThreewayNextSentencePrediction,
    TwoPartBertClassifier,
)


# https://drive.google.com/u/1/uc?id=1VZ1qyNmeYu7mTdDmSH1i00lKIBY26Qoo&export=download
FINETUNED_DUAL_MODEL_ID = "1VZ1qyNmeYu7mTdDmSH1i00lKIBY26Qoo"
FINETUNED_DUAL_MODEL_BASENAME = "tcrbert_lcmv_finetuned_1.0.tar.gz"
FINETUNED_DUAL_MODEL_URL = f"https://drive.google.com/uc?id={FINETUNED_DUAL_MODEL_ID}"
FINETUNED_DUAL_MODEL_MD5 = "e51d8ae58974c2e02d37fd4b51d448ee"
FINETUNED_MODEL_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache/gdown/tcrbert"
)


class NeptuneHuggingFaceCallback(TrainerCallback):
    """
    Add Neptune support for HuggingFace transformers

    Reference:
    https://huggingface.co/transformers/_modules/transformers/integrations.html#WandbCallback
    """

    def __init__(
        self,
        experiment: Experiment,
        epoch_index: bool = True,
        blacklist_keys: Iterable[str] = [
            "train_runtime",
            "train_samples_per_second",
            "epoch",
            "total_flos",
            "eval_runtime",
            "eval_samples_per_second",
        ],
    ):
        self.experiment = experiment
        self.epoch_index = epoch_index
        self.blacklist_keys = set(blacklist_keys)

    def on_log(
        self, args, state, control: TrainerControl, model=None, logs=None, **kwargs
    ):
        """Log relevant values"""
        # Log only if main process
        if not state.is_world_process_zero:
            return

        for k, v in logs.items():
            if k not in self.blacklist_keys:
                # https://docs-legacy.neptune.ai/api-reference/neptune/experiments/index.html
                i = state.global_step if not self.epoch_index else logs["epoch"]
                try:
                    self.experiment.log_metric(k, i, v)
                except ChannelsValuesSendBatchError:
                    logging.warning(
                        f"Error sending index-value pair {k}:{v} to neptune (expected for end of transformers training)"
                    )


class TextMultiClassificationPipeline(Pipeline):
    """
    Multi class classification, applies sigmoid activation instead of softmax
    https://github.com/huggingface/transformers/blob/v4.8.2/src/transformers/pipelines/text_classification.py
    """

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.return_all_scores = return_all_scores

    def __call__(self, *args, **kwargs):
        """Perform classification"""
        outputs = super().__call__(*args, **kwargs)

        scores = 1.0 / (1.0 + np.exp(-outputs))  # Sigmoid
        if self.return_all_scores:
            return [
                [
                    {"label": self.model.config.id2label[i], "score": score.item()}
                    for i, score in enumerate(item)
                ]
                for item in scores
            ]
        else:
            return [
                {
                    "label": self.model.config.id2label[item.argmax()],
                    "score": item.max().item(),
                }
                for item in scores
            ]


class TextClassificationLogitsPipeline(Pipeline):
    """
    Return logits with no activation
    https://github.com/huggingface/transformers/blob/v4.8.2/src/transformers/pipelines/text_classification.py
    """

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.return_all_scores = return_all_scores

    def __call__(self, *args, **kwargs):
        """Perform classification"""
        scores = super().__call__(*args, **kwargs)
        if self.return_all_scores:
            return [
                [
                    {"label": self.model.config.id2label[i], "score": score.item()}
                    for i, score in enumerate(item)
                ]
                for item in scores
            ]
        else:
            return [
                {
                    "label": self.model.config.id2label[item.argmax()],
                    "score": item.max().item(),
                }
                for item in scores
            ]


def get_bert_variant_from_path(dirname: str) -> str:
    """Try to automatically parse BERT variant from path"""
    config_path = os.path.join(dirname, "config.json")
    assert os.path.isfile(config_path)
    with open(config_path) as source:
        config = json.load(source)
    assert "architectures" in config
    arches = config["architectures"]
    assert len(arches) == 1
    arch = arches.pop()
    if arch.startswith("ConvBert"):
        return "convbert"
    elif arch.startswith("Bert"):
        return "bert"
    else:
        raise ValueError(f"Cannot determine bert variant for: {arch}")


def get_pretrained_model_string(model_path: str) -> str:
    """
    Get the pretrained model summary string
    """
    model_path = os.path.abspath(model_path)
    model_path_chunks = model_path.split(os.sep)
    # huggingface transformers use this naming strategy
    checkpoint_idx = [
        i for i, tok in enumerate(model_path_chunks) if tok.startswith("checkpoint-")
    ]
    assert checkpoint_idx, "Could not find any checkpoint folders"
    return model_path_chunks[max(checkpoint_idx) - 1]


def load_two_part_bert_classifier(
    model_dir: Optional[str] = None, device: int = -1
) -> skorch.NeuralNet:
    """
    Load the model as a skorch neuralnet wrapping a TwoPartBertClassifier
    If model_dir is not given, defaults to a cache dir and automatically download model
    """
    if model_dir is None:
        dl_path = gdown.cached_download(
            FINETUNED_DUAL_MODEL_URL,
            path=os.path.join(FINETUNED_MODEL_CACHE_DIR, FINETUNED_DUAL_MODEL_BASENAME),
            md5=FINETUNED_DUAL_MODEL_MD5,
            postprocess=gdown.extractall,
            quiet=False,
        )
        logging.info(f"Model tarball at: {dl_path}")
        model_dir = os.path.join(
            FINETUNED_MODEL_CACHE_DIR,
            "lcmv_ab_finetune_cls_pooling_False_sharedencoder_0.2_dropout_25_epochs_3e-05_lr_linear_lrsched",
        )

    assert os.path.isdir(model_dir), f"Cannot find path: {model_dir}"

    # Read in json of params
    with open(os.path.join(model_dir, "params.json")) as source:
        model_params = json.load(source)
    net = skorch.NeuralNet(
        module=TwoPartBertClassifier,
        module__pretrained="wukevin/tcr-bert-mlm-only",  # Doesn't really matter, will be overwritten
        module__freeze_encoder=model_params["freeze"],
        module__dropout=model_params["dropout"],
        module__separate_encoders=not model_params["sharedencoder"],
        module__seq_pooling=model_params["pooling"],
        criterion=nn.CrossEntropyLoss,
        device=utils.get_device(device),
    )

    cp = skorch.callbacks.Checkpoint(dirname=model_dir, fn_prefix="net_")
    net.load_params(checkpoint=cp)
    return net


def load_fill_mask_pipeline(model_dir: str, device: int = -1):
    """
    Load the pipeline object that does mask filling
    model_dir can either be a directory to the model or repo on huggingface models
    https://huggingface.co/transformers/main_classes/pipelines.html#transformers.FillMaskPipeline
    """
    model = None
    if os.path.isdir(model_dir):
        bert_variant = get_bert_variant_from_path(model_dir)
        if bert_variant == "bert":
            model = BertForMaskedLM.from_pretrained(model_dir)
        elif bert_variant == "convbert":
            model = ConvBertForMaskedLM.from_pretrained(model_dir)
        else:
            raise ValueError(f"Unrecognized BERT variant: {bert_variant}")
    else:
        logging.debug("Model not found locally, attempting to find online")
        model = BertForMaskedLM.from_pretrained(model_dir)
    assert model is not None, f"Could not load MLM model"
    tok = ft.get_pretrained_bert_tokenizer(model_dir)
    pipeline = FillMaskPipeline(model, tok, device=device)
    return pipeline


def load_embed_pipeline(model_dir: str, device: int):
    """
    Load the pipeline object that gets embeddings
    """
    model = AutoModel.from_pretrained(model_dir)
    tok = ft.get_pretrained_bert_tokenizer(model_dir)
    pipeline = FeatureExtractionPipeline(model, tok, device=device)
    return pipeline


def load_classification_pipeline(
    model_dir: str = "wukevin/tcr-bert", multilabel: bool = False, device: int = 0
) -> TextClassificationPipeline:
    """
    Load the pipeline object that does classification
    """
    try:
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
    except OSError:
        tok = ft.get_aa_bert_tokenizer(64)

    if multilabel:
        model = BertForSequenceClassificationMulti.from_pretrained(model_dir)
        pipeline = TextMultiClassificationPipeline(
            model=model,
            tokenizer=tok,
            device=device,
            framework="pt",
            task="mulitlabel_classification",
            return_all_scores=True,
        )
    else:
        model = BertForSequenceClassification.from_pretrained(model_dir)
        pipeline = TextClassificationPipeline(
            model=model, tokenizer=tok, return_all_scores=True, device=device
        )
    return pipeline


def load_classification_logits_pipeline(model_dir: str, device: int = 0):
    """
    Load the pipeline object that returns logits
    """
    try:
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
    except OSError:
        tok = ft.get_aa_bert_tokenizer(64)
    # The exact model doesn't really matter
    model = BertForSequenceClassification.from_pretrained(model_dir)
    pipeline = TextClassificationLogitsPipeline(
        model=model, tokenizer=tok, return_all_scores=True, device=device
    )
    return pipeline


def get_transformer_attentions(
    model_dir: Union[str, nn.Module],
    seqs: Iterable[str],
    seq_pair: Optional[Iterable[str]] = None,
    *,
    layer: int = 0,
    batch_size: int = 256,
    device: int = 0,
) -> List[np.ndarray]:
    """Return a list of attentions"""
    device = utils.get_device(device)
    seqs = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]
    if isinstance(model_dir, str):
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
        model = BertModel.from_pretrained(
            model_dir, add_pooling_layer=False, output_attentions=True,
        ).to(device)
    elif isinstance(model_dir, nn.Module):
        tok = ft.get_aa_bert_tokenizer(64)
        model = model_dir.to(device)
    else:
        raise TypeError(f"Unhandled type for model_dir: {type(model_dir)}")

    chunks = dl.chunkify(seqs, batch_size)
    # This defaults to [None] to zip correctly
    chunks_pair = [None]
    if seq_pair is not None:
        assert len(seq_pair) == len(seqs)
        chunks_pair = dl.chunkify(
            [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seq_pair],
            batch_size,
        )
    # For single input, we get (list of seq, None) items
    # for a duo input, we get (list of seq1, list of seq2)
    chunks_zipped = list(zip_longest(chunks, chunks_pair))
    attentions = []
    with torch.no_grad():
        for seq_chunk in chunks_zipped:
            encoded = tok(
                *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
            )
            input_mask = encoded["attention_mask"].numpy()
            encoded = {k: v.to(device) for k, v in encoded.items()}
            # encoded contains input attention mask of (batch, seq_len)
            x = model.forward(
                **encoded, output_hidden_states=True, output_attentions=True
            )
            for i in range(len(seq_chunk[0])):
                seq_len = np.sum(input_mask[i])
                # The attentions tuple has length num_layers
                # Each entry in the attentions tuple is of shape (batch, num_attn_heads, 64, 64)
                a = x.attentions[layer][i].cpu().numpy()  # (num_attn_heads, 64, 64)
                # Nonzero entries are (num_attn_heads, 64, seq_len)
                # We subset to (num_attn_heads, seq_len, seq_len)
                # as it appears that's what bertviz does
                attentions.append(a[:, :seq_len, :seq_len])
    return attentions


def get_transformer_embeddings(
    model_dir: str,
    seqs: Iterable[str],
    seq_pair: Optional[Iterable[str]] = None,
    *,
    layers: List[int] = [-1],
    method: Literal["mean", "max", "attn_mean", "cls", "pool"] = "mean",
    batch_size: int = 256,
    device: int = 0,
) -> np.ndarray:
    """
    Get the embeddings for the given sequences from the given layers
    Layers should be given as negative integers, where -1 indicates the last
    representation, -2 second to last, etc.
    Returns a matrix of num_seqs x (hidden_dim * len(layers))
    Methods:
    - cls:  value of initial CLS token
    - mean: average of sequence length, excluding initial CLS token
    - max:  maximum over sequence length, excluding initial CLS token
    - attn_mean: mean over sequenced weighted by attention, excluding initial CLS token
    - pool: pooling layer
    If multiple layers are given, applies the given method to each layers
    and concatenate across layers
    """
    device = utils.get_device(device)
    seqs = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]
    try:
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
    except OSError:
        logging.warning("Could not load saved tokenizer, loading fresh instance")
        tok = ft.get_aa_bert_tokenizer(64)
    model = BertModel.from_pretrained(model_dir, add_pooling_layer=method == "pool").to(
        device
    )

    chunks = dl.chunkify(seqs, batch_size)
    # This defaults to [None] to zip correctly
    chunks_pair = [None]
    if seq_pair is not None:
        assert len(seq_pair) == len(seqs)
        chunks_pair = dl.chunkify(
            [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seq_pair],
            batch_size,
        )
    # For single input, we get (list of seq, None) items
    # for a duo input, we get (list of seq1, list of seq2)
    chunks_zipped = list(zip_longest(chunks, chunks_pair))
    embeddings = []
    with torch.no_grad():
        for seq_chunk in chunks_zipped:
            encoded = tok(
                *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
            )
            # manually calculated mask lengths
            # temp = [sum([len(p.split()) for p in pair]) + 3 for pair in zip(*seq_chunk)]
            input_mask = encoded["attention_mask"].numpy()
            encoded = {k: v.to(device) for k, v in encoded.items()}
            # encoded contains input attention mask of (batch, seq_len)
            x = model.forward(
                **encoded, output_hidden_states=True, output_attentions=True
            )
            if method == "pool":
                embeddings.append(x.pooler_output.cpu().numpy().astype(np.float64))
                continue
            # x.hidden_states contains hidden states, num_hidden_layers + 1 (e.g. 13)
            # Each hidden state is (batch, seq_len, hidden_size)
            # x.hidden_states[-1] == x.last_hidden_state
            # x.attentions contains attention, num_hidden_layers
            # Each attention is (batch, attn_heads, seq_len, seq_len)

            for i in range(len(seq_chunk[0])):
                e = []
                for l in layers:
                    # Select the l-th hidden layer for the i-th example
                    h = (
                        x.hidden_states[l][i].cpu().numpy().astype(np.float64)
                    )  # seq_len, hidden
                    # initial 'cls' token
                    if method == "cls":
                        e.append(h[0])
                        continue
                    # Consider rest of sequence
                    if seq_chunk[1] is None:
                        seq_len = len(seq_chunk[0][i].split())  # 'R K D E S' = 5
                    else:
                        seq_len = (
                            len(seq_chunk[0][i].split())
                            + len(seq_chunk[1][i].split())
                            + 1  # For the sep token
                        )
                    seq_hidden = h[1 : 1 + seq_len]  # seq_len * hidden
                    assert len(seq_hidden.shape) == 2
                    if method == "mean":
                        e.append(seq_hidden.mean(axis=0))
                    elif method == "max":
                        e.append(seq_hidden.max(axis=0))
                    elif method == "attn_mean":
                        # (attn_heads, seq_len, seq_len)
                        # columns past seq_len + 2 are all 0
                        # summation over last seq_len dim = 1 (as expected after softmax)
                        attn = x.attentions[l][i, :, :, : seq_len + 2]
                        # print(attn.shape)
                        print(attn.sum(axis=-1))
                        raise NotImplementedError
                    else:
                        raise ValueError(f"Unrecognized method: {method}")
                e = np.hstack(e)
                assert len(e.shape) == 1
                embeddings.append(e)
    if len(embeddings[0].shape) == 1:
        embeddings = np.stack(embeddings)
    else:
        embeddings = np.vstack(embeddings)
    del x
    del model
    torch.cuda.empty_cache()
    return embeddings


def get_transformer_nsp_preds(
    model_dir: str,
    seq_pairs: Iterable[Tuple[str, str]],
    inject_negatives: float = 0.0,
    as_probs: bool = True,
    device: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the NSP model at model_dir and use it to generate predictions
    if as_probs is true, we apply softmax to model output before returning
    otherwise, return logits
    """
    assert inject_negatives >= 0

    # Read in the model name from config json
    model_cls = BertForNextSentencePrediction
    if os.path.isdir(model_dir):
        with open(os.path.join(model_dir, "config.json")) as source:
            config_dict = json.load(source)
        model_names = config_dict["architectures"]
        if "BertForThreewayNextSentencePrediction" in model_names:
            model_cls = BertForThreewayNextSentencePrediction
        elif "BertForNextSentencePrediction" in model_names:
            model_cls = BertForNextSentencePrediction
        else:
            raise ValueError(f"Cannot recognize NSP model in: {model_names}")

    # Load the model
    device_id = utils.get_device(device)
    model = model_cls.from_pretrained(model_dir).to(device_id)
    dset = dl.TcrNextSentenceDataset(
        *zip(*seq_pairs), neg_ratio=inject_negatives, shuffle=False
    )
    all_encoded = dset.get_all_items()
    labels = all_encoded["labels"].numpy().squeeze()

    outputs = []
    with torch.no_grad():
        for batch in dl.chunkify_dict(dset.get_all_items()):
            batch = {k: v.to(device_id) for k, v in batch.items()}
            preds = model(**batch).logits.cpu().numpy()
            if as_probs:
                preds = softmax(preds, axis=1)
            outputs.append(preds)
    return labels, np.vstack(outputs)


def get_esm_embedding(
    seqs: Iterable[str],
    model_key: str = "esm1b_t33_650M_UR50S",
    batch_size: int = 1,
    device: Optional[int] = 0,
) -> np.ndarray:
    """
    Get the embedding from ESM for each of the given sequences

    Resources:
    - https://doi.org/10.1073/pnas.2016239118
    - https://github.com/facebookresearch/esm
    - https://github.com/facebookresearch/esm/blob/master/examples/variant_prediction.ipynb
    """
    esm_device = utils.get_device(device)
    esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm", model_key)
    esm_model = esm_model.to(esm_device)
    batch_converter = esm_alphabet.get_batch_converter()

    seqs_with_faux_labels = list(enumerate(seqs))
    labels, seqs, tokens = batch_converter(seqs_with_faux_labels)
    # Get per-base representations
    reps = []
    with torch.no_grad():
        for batch in dl.chunkify(tokens, chunk_size=batch_size):
            batch = batch.to(device)
            rep = (
                esm_model(batch, repr_layers=[33], return_contacts=True)[
                    "representations"
                ][33]
                .cpu()
                .numpy()
            )
            reps.append(rep)
            del batch  # Try to save some GPU memory
    reps = np.concatenate(reps, axis=0)

    # Get the overall sequence representations
    averaged = []
    for i, (_, seq) in enumerate(seqs_with_faux_labels):
        averaged.append(reps[i, 1 : len(seq) + 1].mean(axis=0))
    averaged = np.vstack(averaged)
    return averaged


def get_tape_embedding(
    seqs: Sequence[str], *, device: int = 3, tape_seed: int = 1234
) -> np.ndarray:
    """
    Get the TAPE unirep embedding of the sequences. We do this by calling the executable
    and using the averaged embedding

    https://github.com/songlab-cal/tape
    """
    # Create a temporary directory to write outputs to and save
    with tempfile.TemporaryDirectory() as tempdir:
        # Write the input file
        in_fasta = os.path.join(tempdir, "input.fasta")
        sequence_ids = []
        with open(in_fasta, "w") as sink:
            for i, seq in enumerate(seqs):
                header = f"sequence-{i}"
                sequence_ids.append(header)
                sink.write(f">{header}\n")
                sink.write(seq + "\n")
        # Create tape cmd and call
        out_npz = os.path.join(tempdir, "output.npz")
        tape_cmd = f"CUDA_VISIBLE_DEVICES={device} tape-embed unirep {in_fasta} {out_npz} babbler-1900 --tokenizer unirep --seed {tape_seed}"
        logging.info(f"TAPE command: {tape_cmd}")

        with open(os.path.join(tempdir, "tape.stdout"), "w") as outfile, open(
            os.path.join(tempdir, "tape.stderr"), "w"
        ) as errfile:
            retcode = subprocess.call(
                tape_cmd, shell=True, stdout=outfile, stderr=errfile
            )
        with open(os.path.join(tempdir, "tape.stdout")) as source:
            tape_stdout = source.readlines()
            logging.debug("".join(tape_stdout))
        with open(os.path.join(tempdir, "tape.stderr")) as source:
            tape_stderr = source.readlines()
            logging.debug("".join(tape_stderr))

        assert retcode == 0
        tape_embeddings_all = np.load(out_npz, allow_pickle=True)
        tape_embeddings = np.array(
            [tape_embeddings_all[h].item()["avg"].copy() for h in sequence_ids]
        )
    assert len(tape_embeddings) == len(seqs)
    return tape_embeddings


def reformat_classification_pipeline_preds(
    x: List[List[Dict[str, Union[float, str]]]]
) -> pd.DataFrame:
    """
    Helper function to take the output of a HuggingFace text classification pipeline
    and reformat it to a easier to parse format -- a DatFrame where each row corresponds
    to a single input and columns correspond to predicted labels

    The huggingface text classification pipeline outputs, for each input:
    [{'label': "foobar", "score": 0.29391}]
    """
    converted = []
    for l in x:
        l_dict = {d["label"]: d["score"] for d in l}
        s = pd.Series(l_dict)
        converted.append(s)
    return pd.DataFrame(converted)


def main():
    """On the fly testing"""
    import data_loader as dl

    net = load_two_part_bert_classifier()
    tra_sequences = ["CAALYGNEKITF"]  # We show only one sequence for simplicity
    trb_sequences = ["CASSDAGGRNTLYF"]
    my_dset = dl.TcrFineTuneDataset(tra_sequences, trb_sequences, skorch_mode=True)
    preds = net.predict_proba(my_dset)[:, 1]
    print(preds)


if __name__ == "__main__":
    main()
