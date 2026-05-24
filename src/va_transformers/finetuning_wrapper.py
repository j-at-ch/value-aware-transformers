import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    def __init__(self, num_in, num_out, dropout_in):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.dropout_in = dropout_in
        self.net = nn.Sequential(
            nn.Dropout(p=dropout_in),
            nn.Linear(num_in, num_out, bias=True)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, dropout_hidden):
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.dropout_hidden = dropout_hidden
        self.net = nn.Sequential(
            nn.Linear(num_in, num_hidden, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_hidden),
            nn.Linear(num_hidden, num_out, bias=True)
        )

    def forward(self, x):
        return self.net(x)


class FinetuningWrapper(nn.Module):
    def __init__(self, net, seq_len, num_classes, clf_or_reg='clf', hidden_dim=100, state_dict=None, weight=None,
                 load_from='pretrained', clf_style='on_EOS', clf_dropout=0., clf_depth=2):
        super().__init__()
        self.num_classes = num_classes
        self.clf_or_reg = clf_or_reg
        self.weight = weight.to(torch.float) if weight is not None else weight
        self.net = copy.deepcopy(net)
        self.max_seq_len = net.max_seq_len
        self.va_transformer = net.va_transformer
        self.logit_head = net.logit_head
        self.seq_len = seq_len
        self.load_from = load_from
        self.with_values = net.with_values
        self.clf_style = clf_style
        self.clf_dropout = clf_dropout
        self.clf_depth = clf_depth

        # quick check on clf versus reg mode

        if self.clf_or_reg == 'reg':
            assert self.num_classes == 1, "if in regression mode, num_classes must be 1"

        # initialise net hparams from pretrained if specified

        if self.load_from == 'pretrained' and state_dict is not None:
            self.load_state_dict(state_dict)

        # define classifier head layers

        if self.logit_head == "hierarchical":
            feature_emb_dim = net.project_out.out_features
        else:
            feature_emb_dim = net.attn_layers.dim

        if clf_style == 'flatten':
            num_features = feature_emb_dim * self.seq_len
        elif clf_style in ['on_sample_start', 'on_sample_end', 'sum', 'on_EOS']:
            num_features = feature_emb_dim
        elif clf_style == 'on_EOS-2':
            num_features = 2 * feature_emb_dim
        else:
            raise Exception(f"clf_style option {clf_style} is not implemented!")

        self.clf = Classifier(num_features, hidden_dim, num_classes, clf_dropout) if clf_depth == 2 \
            else SimpleClassifier(num_features, num_classes, clf_dropout)

        if self.load_from == 'finetuned' and state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x, predict=False, **kwargs):
        if self.with_values:
            x, quants, targets = x
            out, quants_out = self.net(x, quants=quants, return_embeddings=True, **kwargs)
            if self.logit_head == "separate":
                clf_in = torch.cat([out, quants_out], dim=2)
            elif self.logit_head == "hierarchical":
                clf_in = torch.cat([out, quants_out[:, :, :-1]], dim=2)
            else:
                clf_in = quants_out
        else:
            x, targets = x
            out = self.net(x, return_embeddings=True, **kwargs)
            clf_in = out

        b = out.size(0)  # batch_size
        targets = targets.long() if self.clf_or_reg == 'clf' else targets.float()

        if self.clf_style == 'on_EOS':
            eos_indices = torch.sum(x != 0, dim=1) - 1  # relies on pad_token being 0
            clf_in = clf_in[np.arange(b), eos_indices, :]
        elif self.clf_style == 'flatten':
            clf_in = torch.flatten(clf_in, start_dim=1)  # first dim is batch
        elif self.clf_style == 'sum':
            clf_in = torch.sum(clf_in, dim=1)
        elif self.clf_style == 'on_SOS':
            clf_in = clf_in[:, 0, :]
        elif self.clf_style == 'on_sample_end':
            clf_in = clf_in[:, -1, :]
        elif self.clf_style == 'on_EOS-2':
            eos_indices = torch.sum(x != 0, dim=1) - 1  # relies on pad_token being 0
            clf_in = torch.cat([clf_in[np.arange(b), eos_indices, :], clf_in[np.arange(b), eos_indices - 1, :]], dim=1)
        else:
            raise Exception(f"clf_style option {self.clf_style} is not implemented!")

        if self.clf_or_reg == 'reg':
            pre_act = torch.squeeze(self.clf(clf_in))
            preds = F.softplus(pre_act)
            loss = F.mse_loss(preds, targets)
            return preds if predict else loss

        logits = self.clf(clf_in)
        loss = F.cross_entropy(logits, targets, weight=self.weight)
        return logits if predict else loss
