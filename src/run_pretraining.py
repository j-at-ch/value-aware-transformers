import os

import numpy as np
import pandas as pd
from pprint import pprint
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from va_transformers.utils.utils import read_yaml
from va_transformers.utils import model_methods
from va_transformers.utils.data_utils import fetch_mappings, fetch_data_as_torch, make_toy_loader
from va_transformers.utils.arguments import Arguments
from va_transformers.utils.mappings import Mappings, Labellers
from va_transformers.utils.samplers import SeqSamplerDataset
from va_transformers.va_transformers import Decoder, TransformerWrapper
from va_transformers.autoregressive_wrapper import AutoregressiveWrapper


def main(args):
    print('*' * 17, f'va-transformer summoned for {args.mode} with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths
    train_path = os.path.join(args.data_root, "train_data.pkl")
    val_path = os.path.join(args.data_root, "val_data.pkl")
    d_items_path = os.path.join(args.data_root, "d_labitems.csv")
    mapping_path = os.path.join(args.data_root, "mappings.pkl")
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)

    # device
    device = torch.device(args.device)

    # mappings
    mappings_dict = fetch_mappings(mapping_path)
    len_t_dict = len(mappings_dict['itemid2token'])
    len_q_dict = len(mappings_dict['qname2qtoken'])

    pad_token = args.pad_token
    pad_quant_token = args.pad_quant_token if args.with_values else None
    sos_token = sos_quant_token = eos_token = eos_quant_token = None

    if args.specials == 'SOS':
        sos_token = len_t_dict
        sos_quant_token = len_q_dict if args.with_values else None
    elif args.specials == 'EOS':
        eos_token = len_t_dict
        eos_quant_token = len_q_dict if args.with_values else None
    elif args.specials == 'both':
        sos_token = len_t_dict
        sos_quant_token = len_q_dict if args.with_values else None
        eos_token = len_t_dict + 1
        eos_quant_token = (len_q_dict + 1) if args.with_values else None

    mappings = Mappings(
        mappings_dict,
        pad_token=pad_token,
        sos_token=sos_token,
        eos_token=eos_token,
        pad_quant_token=pad_quant_token,
        sos_quant_token=sos_quant_token,
        eos_quant_token=eos_quant_token
    )

    print(
        f"[PAD] token is {mappings.pad_token}",
        f"[SOS] token is {mappings.sos_token}",
        f"[EOS] token is {mappings.eos_token}",
        f"[PAD] quant token is {mappings.pad_quant_token}",
        f"[SOS] quant token is {mappings.sos_quant_token}",
        f"[EOS] quant token is {mappings.eos_quant_token}",
        sep="\n"
    )

    # labellers
    d_items_df = pd.read_csv(d_items_path, index_col='ITEMID', dtype={'ITEMID': str})
    labeller = Labellers(mappings, d_items_df)

    # get tokens from processed data
    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    # get quants
    if bool(args.with_values):
        quants_train = fetch_data_as_torch(train_path, 'train_quants')
        quants_val = fetch_data_as_torch(val_path, 'val_quants')
    else:
        quants_train = None
        quants_val = None

    # load data for pretraining based on arguments

    train_dataset = SeqSamplerDataset(
        tokens=data_train,
        seq_len=args.seq_len,
        mappings=mappings,
        device=device,
        quants=quants_train,
        specials=args.specials,
        align_sample_at=args.align_sample_at
    )

    val_dataset = SeqSamplerDataset(
        tokens=data_val,
        seq_len=args.seq_len,
        mappings=mappings,
        device=device,
        quants=quants_val,
        specials=args.specials,
        align_sample_at=args.align_sample_at
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    if bool(args.toy_run):
        train_loader = make_toy_loader(train_loader)
        val_loader = make_toy_loader(val_loader)

    # instantiate GPT-like decoder architecture
    model = TransformerWrapper(
        num_tokens=mappings.num_tokens,
        with_values=bool(args.with_values),
        num_quant_tokens=mappings.num_quant_tokens,
        max_seq_len=args.seq_len,
        attn_layers=Decoder(
            dim=args.attn_dim,
            depth=args.attn_depth,
            heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            use_rezero=bool(args.use_rezero),
            rotary_pos_emb=bool(args.rotary_pos_emb)
        ),
        token_emb_dim=args.token_emb_dim,
        quant_emb_dim=args.quant_emb_dim,
        logit_head=args.logit_head,
        va_transformer=bool(args.va_transformer)
    )

    # wrap model for pretraining
    pre_model = AutoregressiveWrapper(
        net=model,
        ignore_index=args.ignore_index,
        ignore_quant_index=args.ignore_quant_index
    )

    if args.load_from_checkpoint_at is not None:
        params_path = os.path.join(args.model_root, args.load_from_checkpoint_at)
        checkpoint = torch.load(params_path, map_location=device)
        states = checkpoint['model_state_dict']
        pre_model.load_state_dict(states)
    else:
        pass

    pre_model.to(device)

    print("model specification:", pre_model.net, sep="\n")

    if args.mode == 'pretraining':
        writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)

        optimizer = torch.optim.Adam(pre_model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
        training = model_methods.PretrainingMethods(pre_model, writer)

        tokens_to_write = list(mappings.token2itemid)

        # write initial embeddings
        if bool(args.write_initial_embeddings):
            training.write_token_emb(-1, tokens_to_write, labeller, args.seq_len, device)

        # training loop
        best_val_loss = np.inf
        early_stopping_counter = 0
        for epoch in range(args.num_epochs):
            training.train(
                train_loader,
                optimizer,
                epoch,
                grad_accum_every=args.grad_accum_every,
                gamma=args.gamma
            )

            val_losses = training.evaluate(
                val_loader,
                epoch,
                gamma=args.gamma
            )

            if val_losses.loss < best_val_loss:
                print("Saving checkpoint...")
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': pre_model.state_dict(),
                        'args': vars(args),
                        'optim_state_dict': optimizer.state_dict(),
                        'val_loss': val_losses.loss,
                        'token_loss': val_losses.token_loss,
                        'quant_loss': val_losses.quant_loss
                    },
                    ckpt_path
                )

                # track checkpoint's embeddings
                if bool(args.write_best_val_embeddings):
                    training.write_token_emb(epoch, tokens_to_write, labeller, args.seq_len, device)

                print("Checkpoint saved!\n")
                best_val_loss = val_losses.loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            print(f'epoch {epoch} completed!')
            print('flushing writer...')
            writer.flush()

            if early_stopping_counter == args.early_stopping_threshold:
                print('early stopping threshold hit! ending training...')
                break

            scheduler.step()

        # write final embeddings
        if bool(args.write_final_embeddings):
            training.write_token_emb(epoch, tokens_to_write, labeller, args.seq_len, device)

        writer.close()
        print("training finished and writer closed!")

    if bool(args.WARNING_TESTING):
        print("\nWARNING TEST set in use!\n")

        # load test set data

        test_path = os.path.join(args.data_root, "test_data.pkl")
        data_test = fetch_data_as_torch(test_path, 'test_tokens')
        if bool(args.with_values):
            quants_test = fetch_data_as_torch(test_path, 'test_quants')
        else:
            quants_test = None

        test_dataset = SeqSamplerDataset(data_test, args.seq_len, mappings, device,
                                         quants=quants_test,
                                         specials=args.specials,
                                         align_sample_at=args.align_sample_at)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size_tr, shuffle=True)

        if bool(args.toy_run):
            test_loader = make_toy_loader(test_loader)

        # test the model at the checkpoint
        params_path = os.path.join(args.model_root, args.model_name + '.pt')

        print(f"loading state dict from checkpoint at {params_path}...")

        checkpoint = torch.load(params_path, map_location=device)
        states = checkpoint['model_state_dict']
        pre_model.load_state_dict(states)

        print(f"checkpoint loaded!")

        writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)

        # test the model
        testing = model_methods.PretrainingMethods(pre_model, writer=writer)
        testing.evaluate(val_loader, epoch=0, gamma=args.gamma, prefix='re-val')
        testing.evaluate(test_loader, epoch=0, gamma=args.gamma, prefix='test')

        print("testing finished!")


if __name__ == "__main__":
    args = Arguments(mode='pretraining').parse()
    config = read_yaml(args.config)

    args.data_root = config.get('data_path')
    args.model_root = config.get('model_path')
    args.save_root = config.get('save_path')
    args.logs_root = config.get('logs_path')

    # check output roots exist: if not, create...
    if not os.path.exists(args.save_root):
        os.mkdir(args.save_root)
    if not os.path.exists(args.logs_root):
        os.mkdir(args.logs_root)

    # check that arguments are well-specified
    assert (args.gamma >= 0) and (args.gamma <= 1), "--gamma should satisfy 0 <= gamma <= 1."
    assert args.va_transformer == args.with_values, "with_values is only to be used with va_transformer"
    if args.with_values == 0:
        assert args.logit_head is None, "cannot use logit_head without values!"

    # run pretraining

    print(f"mode is {args.mode}")
    main(args)
