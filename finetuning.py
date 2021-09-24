import os
import sys

import numpy as np
import pandas as pd
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import model_methods
from utils.data_utils import *
from utils.arguments import Arguments
from utils.mappings import Mappings, Labellers
from utils.samplers import VgSamplerDataset
from vg_transformers.vg_transformers import TransformerWrapper, Decoder
from vg_transformers.finetuning_wrapper import FinetuningWrapper


def finetune(args):
    print('*' * 17, 'chart-transformer summoned for finetuning with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    d_items_path = os.path.join(args.data_root, "D_LABITEMS.csv")
    train_path = os.path.join(args.data_root, "train_data.pkl")
    val_path = os.path.join(args.data_root, "val_data.pkl")
    mapping_path = os.path.join(args.data_root, "mappings.pkl")
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)

    train_tgt_path = os.path.join(args.data_root, "train_targets.pkl")
    val_tgt_path = os.path.join(args.data_root, "val_targets.pkl")
    params_path = os.path.join(args.model_root, args.pretrained_model)

    # device

    device = torch.device(args.device)

    # fetch mappings

    mappings_dict = fetch_mappings(mapping_path)

    pad_token = args.pad_token
    pad_guide_token = args.pad_quant_token
    sos_token = sos_guide_token = eos_token = eos_guide_token = None
    if args.specials == 'SOS':
        sos_token, sos_guide_token = len(mappings_dict['itemid2token']), 7
    elif args.specials == 'EOS':
        eos_token, eos_guide_token = len(mappings_dict['itemid2token']), 7
    elif args.specials == 'both':
        sos_token, sos_guide_token = len(mappings_dict['itemid2token']), 7
        eos_token, eos_guide_token = len(mappings_dict['itemid2token']) + 1, 8

    mappings = Mappings(mappings_dict,
                        pad_token=pad_token,
                        sos_token=sos_token,
                        eos_token=eos_token,
                        pad_quant_token=pad_guide_token,
                        sos_quant_token=sos_guide_token,
                        eos_quant_token=eos_guide_token
                        )

    print(f"[PAD] token is {mappings.pad_token}",
          f"[SOS] token is {mappings.sos_token}",
          f"[EOS] token is {mappings.eos_token}",
          f"[PAD] guide token is {mappings.pad_quant_token}",
          f"[SOS] guide token is {mappings.sos_quant_token}",
          f"[EOS] guide token is {mappings.eos_quant_token}",
          sep="\n")

    # labellers

    d_items_df = pd.read_csv(d_items_path, index_col='ITEMID', dtype={'ITEMID': str})
    labeller = Labellers(mappings, d_items_df)

    # fetch targets

    with open(train_tgt_path, 'rb') as f:
        x = pickle.load(f)
        train_targets = {k: v[args.targets] for k, v in x['train_targets'].items()}
        del x

    with open(val_tgt_path, 'rb') as f:
        x = pickle.load(f)
        val_targets = {k: v[args.targets] for k, v in x['val_targets'].items()}
        del x

    # get tokens

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    # get quantiles

    if args.quant_guides is None:
        quantiles_train = None
        quantiles_val = None
    else:
        quantiles_train = fetch_data_as_torch(train_path, 'train_quantiles')
        quantiles_val = fetch_data_as_torch(val_path, 'val_quantiles')

    train_dataset = VgSamplerDataset(data_train, args.seq_len, mappings, device,
                                     quants=quantiles_train, targets=train_targets,
                                     specials=args.specials,
                                     align_sample_at=args.align_sample_at
                                     )
    val_dataset = VgSamplerDataset(data_val, args.seq_len, mappings, device,
                                   quants=quantiles_val, targets=val_targets,
                                   specials=args.specials,
                                   align_sample_at=args.align_sample_at
                                   )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for quick test run

    if bool(args.test_run):
        train_loader = make_toy_loader(train_loader)
        val_loader = make_toy_loader(val_loader)

    # weighting with target propensities if necessary

    weights = None

    if args.clf_or_reg == 'clf':
        def propensity_from(targets_dict):
            return sum(targets_dict.values()) / len(targets_dict)

        p = propensity_from(train_targets)
        print(f"Train set positive class propensity is {p}")

        if bool(args.weighted_loss):
            weights = torch.tensor([p, 1 - p]).to(device)

    # fetch model params

    pretrained_ckpt = torch.load(params_path, map_location=device)
    states = pretrained_ckpt['model_state_dict']

    # initialisation of model

    model = TransformerWrapper(
        num_tokens=mappings.num_tokens,
        num_guide_tokens=mappings.num_quant_tokens,
        max_seq_len=args.seq_len,
        attn_layers=Decoder(
            dim=args.attn_dim,
            depth=args.attn_depth,
            heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            value_guides=args.quant_guides,
            use_rezero=bool(args.use_rezero),
            rotary_pos_emb=bool(args.rotary_pos_emb)
        ),
        use_guide_pos_emb=bool(args.use_quant_pos_emb)
    )

    # wrap model for finetuning

    fit_model = FinetuningWrapper(model, seq_len=args.seq_len, clf_or_reg=args.clf_or_reg, num_classes=args.num_classes,
                                  state_dict=states, weight=weights, load_from=args.load_from,
                                  value_guides=args.quant_guides, clf_style=args.clf_style,
                                  clf_dropout=args.clf_dropout, clf_depth=args.clf_depth)
    fit_model.to(device)

    # for name, param in states.named_parameters():
    #    print(name, param.requires_grad)

    print("base transformer specification:", fit_model.net, sep="\n")

    print("clf specification:", fit_model.clf,
          "clf style:", fit_model.clf_style,
          sep="\n")

    if bool(args.freeze_base):
        print("Freezing base transformer parameters...")
        for name, param in fit_model.named_parameters():
            if 'clf' not in name:
                param.requires_grad = False
    else:
        print("Base transformer parameters remaining unfrozen...")

    # initialise optimiser

    optimizer = torch.optim.Adam(fit_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
    writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
    training = model_methods.FinetuningMethods(fit_model, writer)

    # write initial embeddings

    if bool(args.write_initial_embeddings):
        training.write_embeddings(-1, mappings, labeller, args.seq_len, device)

    # training loop
    best_val_loss = np.inf
    early_stopping_counter = 0
    for epoch in range(args.num_epochs):

        # training and evaluation

        training.train(train_loader, optimizer, epoch, grad_accum_every=args.grad_accum_every)
        val_loss = training.evaluate(val_loader, epoch)

        # tracking model classification metrics

        if bool(args.predict_on_train):
            training.predict(train_loader, epoch, device, prefix="train")

        if args.clf_or_reg == 'clf':
            _, _, metrics = training.predict(val_loader, epoch, device, prefix="val", clf_or_reg=args.clf_or_reg)
        elif args.clf_or_reg == 'reg':
            _, _, metrics = training.predict(val_loader, epoch, device, prefix="val", clf_or_reg=args.clf_or_reg)

        # whether to checkpoint model

        if val_loss < best_val_loss:
            print("Saving checkpoint because best val_loss attained...")
            torch.save({
                'epoch': epoch,
                'val_loss': val_loss,
                'metrics': metrics,
                'args': vars(args),
                'model_state_dict': fit_model.state_dict(),
                'optim_state_dict': optimizer.state_dict()
            }, ckpt_path)

            # track checkpoint's embeddings

            if bool(args.write_best_val_embeddings):
                training.write_embeddings(epoch, mappings, labeller, args.seq_len, device)

            print("Checkpoint saved!\n")
            best_val_loss = min(val_loss, best_val_loss)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter == args.early_stopping_threshold:
            print('early stopping threshold hit! ending training...')
            break

        scheduler.step()

        # flushing writer

        print(f'epoch {epoch} completed!', '\n')
        print('flushing writer...')
        writer.flush()

    # write final embeddings

    if bool(args.write_final_embeddings):
        training.write_embeddings(args.num_epochs, mappings, labeller, args.seq_len, device)

    writer.close()
    print("training finished and writer closed!")


def evaluate(args):
    print('*' * 17, 'finetuned chart-transformer summoned for evaluation with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    d_items_path = os.path.join(args.data_root, "D_LABITEMS.csv")
    train_path = os.path.join(args.data_root, "train_data.pkl")
    val_path = os.path.join(args.data_root, "val_data.pkl")
    mapping_path = os.path.join(args.data_root, "mappings.pkl")
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)

    train_lbl_path = os.path.join(args.data_root, "train_targets.pkl")
    val_lbl_path = os.path.join(args.data_root, "val_targets.pkl")
    params_path = os.path.join(args.model_root, args.pretrained_model)

    # device

    device = torch.device(args.device)

    # fetch mappings

    mappings_dict = fetch_mappings(mapping_path)
    mappings = Mappings(mappings_dict)

    # labellers

    d_items_df = pd.read_csv(d_items_path, index_col='ITEMID', dtype={'ITEMID': str})
    labeller = Labellers(mappings_dict, d_items_df)

    # fetch labels

    with open(train_lbl_path, 'rb') as f:
        X = pickle.load(f)
        train_targets = {k: v[args.targets] for k, v in X['train_targets'].items()}
        del X

    with open(val_lbl_path, 'rb') as f:
        X = pickle.load(f)
        val_targets = {k: v[args.targets] for k, v in X['val_targets'].items()}
        del X

    # get tokens

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    # get quantiles

    if args.quant_guides is None:
        quantiles_train = None
        quantiles_val = None
    else:
        quantiles_train = fetch_data_as_torch(train_path, 'train_quantiles')
        quantiles_val = fetch_data_as_torch(val_path, 'val_quantiles')

    train_dataset = VgSamplerDataset(data_train, args.seq_len, device,
                                     quants=quantiles_train, targets=train_targets)
    val_dataset = VgSamplerDataset(data_val, args.seq_len, device,
                                   quants=quantiles_val, targets=val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for quick test run

    if bool(args.test_run):
        train_loader = [X for i, X in enumerate(train_loader) if i < 2]
        val_loader = [X for i, X in enumerate(val_loader) if i < 2]

    # unconditional label propensities

    def propensity_from(targets_dict):
        return sum(targets_dict.values()) / len(targets_dict)

    p = propensity_from(train_targets)
    print(f"Train set positive class propensity is {p}")

    if bool(args.weighted_loss):
        weights = torch.tensor([p, 1 - p]).to(device)
    else:
        weights = None

    # fetch model params

    pretrained_ckpt = torch.load(params_path, map_location=device)
    states = pretrained_ckpt['model_state_dict']

    pprint(states.keys())

    # initialisation of model

    model = TransformerWrapper(
        num_tokens=mappings.num_tokens,
        max_seq_len=args.seq_len,  # NOTE: max_seq_len necessary for the absolute positional embeddings.
        attn_layers=Decoder(
            dim=args.attn_dim,
            depth=args.attn_depth,
            heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            value_guides=args.quant_guides,
            use_rezero=bool(args.use_rezero),
            rotary_pos_emb=bool(args.rotary_pos_emb)
        )
    )

    # wrap model for finetuning

    fit_model = FinetuningWrapper(model, seq_len=args.seq_len, num_classes=2, hidden_dim=args.clf_hidden_dim,
                                  state_dict=states, weight=weights, load_from=args.load_from,
                                  value_guides=args.quant_guides, clf_style=args.clf_style,
                                  clf_dropout=args.clf_dropout)
    fit_model.to(device)

    # for name, param in states.named_parameters():
    #    print(name, param.requires_grad)

    print("model specification:", fit_model.net, sep="\n")

    print("clf specification:", fit_model.clf, "embedding reduction:", fit_model.clf_style, sep="\n")

    if bool(args.freeze_base):
        print("Freezing base transformer parameters...")
        for name, param in fit_model.named_parameters():
            if 'net.' in name:
                param.requires_grad = False
    else:
        print("Base transformer parameters remaining unfrozen...")

    evaluating = model_methods.FinetuningMethods(fit_model, None)
    train_out = evaluating.predict(train_loader, 'eval', device, prefix="train")
    val_out = evaluating.predict(val_loader, 'eval', device, prefix="val")


if __name__ == "__main__":
    arguments = Arguments(mode='finetuning').parse()

    # check output roots exist; if not, create...

    if not os.path.exists(arguments.save_root):
        os.mkdir(arguments.save_root)
    if not os.path.exists(arguments.logs_root):
        os.mkdir(arguments.logs_root)

    # run finetuning
    if arguments.mode == 'training':
        finetune(arguments)
    elif arguments.mode == 'evaluation':
        evaluate(arguments)
