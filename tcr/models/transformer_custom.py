"""
Transformer networks

Useful resources:
https://github.com/pytorch/examples/blob/master/word_language_model/model.py
"""
import os, sys
import json
import logging
from typing import *

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import (
    AutoModel,
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    NextSentencePredictorOutput,
)
from transformers.models.bert.configuration_bert import BertConfig

import fc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import featurization as ft
import utils


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


class TwoPartClassLogitsHead(nn.Module):
    """
    Classifier head that takes the encoded representation of TRA and TRB
    of shape (batch, hidden_dim), projects each through its own separate
    fully connected layer, before concatenating and projecting to final
    output
    """

    def __init__(
        self, a_enc_dim: int, b_enc_dim: int, n_out: int = 2, dropout: float = 0.1
    ):
        # BertForSequenceClassification adds, on top of the pooler (dense, tanh),
        # dropout layer with p=0.1
        # classifier linear layer
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.fc_a = fc.FullyConnectedLayer(a_enc_dim, 32, activation=nn.ReLU())
        self.fc_b = fc.FullyConnectedLayer(b_enc_dim, 32, activation=nn.ReLU())
        self.final_fc = fc.FullyConnectedLayer(64, n_out, None)

    def forward(self, a_enc, b_enc) -> torch.Tensor:
        a_enc = self.fc_a(self.dropout(a_enc))
        b_enc = self.fc_b(self.dropout(b_enc))
        enc = torch.cat([a_enc, b_enc], dim=-1)
        retval = self.final_fc(enc)
        return retval


class TwoPartRegressHead(nn.Module):
    """
    Regression head that takes the encoded representation of TRA and TRB
    of shape (batch, hidden_dim), prjoects each to a separate fully connected
    layer, before concatenatenating and projecting to output. Sigmoid activation
    to predict the cell proportion
    """

    def __init__(self, a_enc_dim: int, b_enc_dim: int, n_out: int = 1):
        super().__init__()
        self.fc_a = fc.FullyConnectedLayer(a_enc_dim, 32, activation=nn.ReLU())
        self.fc_b = fc.FullyConnectedLayer(b_enc_dim, 32, activation=nn.ReLU())
        self.final_fc = fc.FullyConnectedLayer(64, n_out, nn.Sigmoid())

    def forward(self, a_enc, b_enc) -> torch.Tensor:
        a_enc = self.fc_a(a_enc)
        b_enc = self.fc_b(b_enc)
        enc = torch.cat([a_enc, b_enc], dim=-1)
        retval = self.final_fc(enc).squeeze()
        return retval


class TwoPartBertClassifier(nn.Module):
    """
    Two part BERT model, one part each for tcr a/b
    """

    def __init__(
        self,
        pretrained: str,
        n_output: int = 2,
        freeze_encoder: bool = False,
        separate_encoders: bool = True,
        dropout: float = 0.2,
        seq_pooling: Literal["max", "mean", "cls", "pool"] = "cls",
    ):
        super().__init__()
        self.pooling_strat = seq_pooling
        self.pretrained = pretrained
        # BertPooler takes the hidden state corresponding to the first token
        # and applies:
        # - linear from hidden_size -> hidden_size
        # - Tanh activation elementwise
        # This is somewhat misleading since it really only looks at first token
        # while the "pooling" name implies looking at the whole sequence
        # See BertPooler class in https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
        # Do not add pooler to model unless we are using the pool
        self.trans_a = self.__initialize_transformer()
        if separate_encoders:
            self.trans_b = self.__initialize_transformer()
        else:
            self.trans_b = self.trans_a
        self.bert_variant = self.trans_a.base_model_prefix

        self.trans_a.train()
        self.trans_b.train()
        if freeze_encoder:
            for param in self.trans_a.parameters():
                param.requires_grad = False
            for param in self.trans_b.parameters():
                param.requires_grad = False

        self.head = TwoPartClassLogitsHead(
            self.trans_a.encoder.layer[-1].output.dense.out_features,
            self.trans_b.encoder.layer[-1].output.dense.out_features,
            n_out=n_output,
            dropout=dropout,
        )

    def __initialize_transformer(self) -> BertModel:
        """
        Centralized helper function to initialize transformer for
        encoding input sequences
        """
        if os.path.isfile(self.pretrained) and self.pretrained.endswith(".json"):
            logging.info(
                "Got json config as 'pretrained' - initializing BERT model from scratch"
            )
            params = utils.load_json_params(self.pretrained)
            config = BertConfig(
                **params,
                vocab_size=len(ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL),
                pad_token_id=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX[ft.PAD],
            )
            retval = BertModel(config)
        else:
            retval = AutoModel.from_pretrained(
                self.pretrained, add_pooling_layer=self.pooling_strat == "pool"
            )
        return retval

    def forward(self, tcr_a, tcr_b):
        bs = tcr_a.shape[0]
        # input of (batch, seq_len) (batch, 20)
        # last_hidden_state (batch, seq_len, hidden_size) (batch, 20, 144)
        # pooler_output (batch_size, hiddensize)
        if self.bert_variant == "bert":
            a_enc = self.trans_a(tcr_a)
            b_enc = self.trans_a(tcr_b)
        else:
            # a_enc = self.trans_a(tcr_a)[0]
            # b_enc = self.trans_b(tcr_b)[0]
            raise NotImplementedError

        # Perform seq pooling
        if self.pooling_strat == "mean":
            a_enc = a_enc.last_hidden_state.mean(dim=1)
            b_enc = b_enc.last_hidden_state.mean(dim=1)
        elif self.pooling_strat == "max":
            a_enc = a_enc.last_hidden_state.max(dim=1)[0]
            b_enc = b_enc.last_hidden_state.max(dim=1)[0]
        elif self.pooling_strat == "cls":
            a_enc = a_enc.last_hidden_state[:, 0, :]
            b_enc = b_enc.last_hidden_state[:, 0, :]
        elif self.pooling_strat == "pool":
            a_enc = a_enc.pooler_output
            b_enc = b_enc.pooler_output
        else:
            raise ValueError(f"Unrecognized pooling strategy: {self.pooling_strat}")

        retval = self.head(a_enc, b_enc)
        return retval


class TwoPartBertRegressor(TwoPartBertClassifier):
    """Same as above, but with a regression head instead"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = TwoPartRegressHead(
            self.trans_a.encoder.layer[-1].output.dense.out_features,
            self.trans_b.encoder.layer[-1].output.dense.out_features,
        )


class BertForSequenceClassificationMulti(BertPreTrainedModel):
    """
    Heavily inspired by the code introduced in 4.6.0 to do multi-label
    Actual implementation here was done using 4.7.0 as a reference

    References:
    https://github.com/huggingface/transformers/releases/tag/v4.6.0
    https://github.com/huggingface/transformers/blob/v4.7.0/src/transformers/models/bert/modeling_bert.py#L1462
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertMulticlassNSPHead(nn.Module):
    """
    Head for doing NSP across multiple classes
    Reference: https://huggingface.co/transformers/_modules/transformers/modeling_bert.html > BertOnlyNSPHead
    """

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 3)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertForThreewayNextSentencePrediction(BertPreTrainedModel):
    """
    Vanilla BERT NSP only does binary classification. Extend to do multi-class here.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertMulticlassNSPHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> NextSentencePredictorOutput:
        # Reference: https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForNextSentencePrediction
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # LogSoftMax + NLLLoss
            next_sentence_loss = loss_fct(
                seq_relationship_scores.view(-1, 3), labels.view(-1)
            )

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return (
                ((next_sentence_loss,) + output)
                if next_sentence_loss is not None
                else output
            )

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TwoPartEsmClassifier(nn.Module):
    """
    Two part ESM model, one part each for TCR A/B
    pretrained is given as a keyword NOT a path
    """

    def __init__(
        self,
        pretrained: Tuple[str, str] = ("facebookresearch/esm", "esm1b_t33_650M_UR50S"),
        n_output: int = 2,
        freeze_encoder: bool = False,
        separate_encoders: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trans_a, self.alphabet = torch.hub.load(*pretrained)
        if separate_encoders:
            self.trans_b, _ = torch.hub.load(*pretrained)
        else:
            logging.info("Using single encoder for both TRA/TRB")
            self.trans_b = self.trans_a
        self.trans_a.train()
        self.trans_b.train()
        if freeze_encoder:
            for param in self.trans_a.parameters():
                param.requires_grad = False
            for param in self.trans_b.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        in_shape = (
            self.trans_a.layers[-1].fc2.out_features
            + self.trans_b.layers[-1].fc2.out_features
        )
        logging.info(f"Input encoded shape to classifier: {in_shape}")
        self.final_fc = fc.FullyConnectedLayer(in_shape, n_output)

    def forward(self, tcr_a, tcr_b):
        # Input is (batch, seq_len)
        assert (
            len(tcr_a.shape) == len(tcr_b.shape) == 2
        ), f"Unexpected shapes: {tcr_a.shape} {tcr_b.shape}"
        bs = tcr_a.shape[0]
        # First token in input is always the '0'-th token
        tcr_a_keep_mask = tcr_a != self.alphabet.padding_idx
        tcr_a_keep_mask[:, 0] = False
        tcr_b_keep_mask = tcr_b != self.alphabet.padding_idx
        tcr_b_keep_mask[:, 0] = False

        # Shape of (batch, seq_len, hidden)
        # Average the non-padding entries per sequence in the batch
        a_enc = self.trans_a(tcr_a, repr_layers=[33])["representations"][33]
        a_enc = torch.mul(
            a_enc, tcr_a_keep_mask.unsqueeze(axis=-1).expand(a_enc.shape)
        ).sum(axis=1)
        a_enc /= tcr_a_keep_mask.sum(axis=1, keepdim=True)

        b_enc = self.trans_b(tcr_b, repr_layers=[33])["representations"][33]
        b_enc = torch.mul(
            b_enc, tcr_b_keep_mask.unsqueeze(axis=-1).expand(b_enc.shape)
        ).sum(axis=1)
        b_enc /= tcr_b_keep_mask.sum(axis=1, keepdim=True)

        enc = self.dropout(torch.cat([a_enc, b_enc], dim=-1))
        retval = self.final_fc(enc)
        return retval


def main():
    """On the fly testing"""
    pass


if __name__ == "__main__":
    main()
