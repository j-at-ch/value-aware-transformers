import os

import numpy as np
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.va_transformers.utils import model_methods
from src.va_transformers.utils.data_utils import *
from src.va_transformers.utils.arguments import Arguments
from src.va_transformers.utils.mappings import Mappings, Labellers
from src.va_transformers.utils.samplers import SeqSamplerDataset
from src.va_transformers import TransformerWrapper, Decoder
from src.va_transformers import FinetuningWrapper


def main(args):
    print('*' * 17, f'va-transformer summoned for {args.mode} with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    d_items_path = os.path.join(args.data_root, "D_LABITEMS.csv")
    train_path = os.path.join(args.data_root, "train_data_sc.pkl")
    val_path = os.path.join(args.data_root, "val_data_sc.pkl")
    mapping_path = os.path.join(args.data_root, "mappings_sc.pkl")
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)

    train_tgt_path = os.path.join(args.data_root, "train_targets.pkl")
    val_tgt_path = os.path.join(args.data_root, "val_targets.pkl")
    params_path = os.path.join(args.model_root, args.pretrained_model)

    # device

    device = torch.device(args.device)

    # fetch mappings

    mappings_dict = fetch_mappings(mapping_path)
    len_t_dict = len(mappings_dict['itemid2token'])
    len_q_dict = len(mappings_dict['qname2qtoken']) if bool(args.with_values) else None

    pad_token = args.pad_token
    pad_quant_token = args.pad_quant_token if args.with_values else None
    sos_token = sos_quant_token = eos_token = eos_quant_token = None

    if args.specials == 'SOS':
        sos_token = len_t_dict
        sos_quant_token = len_q_dict if bool(args.with_values) else None
    elif args.specials == 'EOS':
        eos_token = len_t_dict
        eos_quant_token = len_q_dict if bool(args.with_values) else None
    elif args.specials == 'both':
        sos_token = len_t_dict
        sos_quant_token = len_q_dict if bool(args.with_values) else None
        eos_token = len_t_dict + 1
        eos_quant_token = (len_q_dict + 1) if bool(args.with_values) else None

    mappings = Mappings(mappings_dict,
                        pad_token=pad_token,
                        sos_token=sos_token,
                        eos_token=eos_token,
                        pad_quant_token=pad_quant_token,
                        sos_quant_token=sos_quant_token,
                        eos_quant_token=eos_quant_token
                        )

    print(f"[PAD] token is {mappings.pad_token}",
          f"[SOS] token is {mappings.sos_token}",
          f"[EOS] token is {mappings.eos_token}",
          f"[PAD] quant token is {mappings.pad_quant_token}",
          f"[SOS] quant token is {mappings.sos_quant_token}",
          f"[EOS] quant token is {mappings.eos_quant_token}",
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

    # get quants

    if bool(args.with_values):
        quants_train = fetch_data_as_torch(train_path, 'train_quants')
        quants_val = fetch_data_as_torch(val_path, 'val_quants')
    else:
        quants_train = None
        quants_val = None

    train_dataset = SeqSamplerDataset(data_train, args.seq_len, mappings, device,
                                      quants=quants_train, targets=train_targets,
                                      specials=args.specials,
                                      align_sample_at=args.align_sample_at
                                      )
    val_dataset = SeqSamplerDataset(data_val, args.seq_len, mappings, device,
                                    quants=quants_val, targets=val_targets,
                                    specials=args.specials,
                                    align_sample_at=args.align_sample_at
                                    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for quick test run

    if bool(args.toy_run):
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
    state_dict = pretrained_ckpt['model_state_dict']

    # initialisation of model

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

    # wrap model for finetuning

    fit_model = FinetuningWrapper(model,
                                  seq_len=args.seq_len,
                                  load_from=args.load_from,
                                  state_dict=state_dict,
                                  clf_or_reg=args.clf_or_reg,
                                  num_classes=args.num_classes,
                                  clf_style=args.clf_style,
                                  clf_dropout=args.clf_dropout,
                                  clf_depth=args.clf_depth,
                                  weight=weights)
    fit_model.to(device)

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

    if args.mode == "finetuning":

        # initialise optimiser

        optimizer = torch.optim.Adam(fit_model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
        writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
        training = model_methods.FinetuningMethods(fit_model, writer, clf_or_reg=args.clf_or_reg)

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

            _, _, metrics = training.predict(val_loader, epoch, device, prefix="val")

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

    if bool(args.WARNING_TESTING):
        print("\nWARNING TEST set in use!\n")

        # load test set data
        test_path = os.path.join(args.data_root, "test_data_sc.pkl")
        test_tgt_path = os.path.join(args.data_root, "test_targets.pkl")

        data_test = fetch_data_as_torch(test_path, 'test_tokens')
        if bool(args.with_values):
            quants_test = fetch_data_as_torch(test_path, 'test_quants')
        else:
            quants_test = None
        with open(test_tgt_path, 'rb') as f:
            x = pickle.load(f)
            test_targets = {k: v[args.targets] for k, v in x['test_targets'].items()}
            del x

        test_dataset = SeqSamplerDataset(data_test, args.seq_len, mappings, device,
                                         targets=test_targets,
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
        fit_model.load_state_dict(states)
        print(f"checkpoint loaded!")

        writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
        testing = model_methods.FinetuningMethods(fit_model, writer=writer, clf_or_reg=args.clf_or_reg)

        # test the model...

        val_losses = testing.evaluate(val_loader, epoch=0, prefix='re-val')
        _, _, val_metrics = testing.predict(val_loader, epoch=0, device=device, prefix="re-val")
        test_losses = testing.evaluate(test_loader, epoch=0, prefix='test')
        _, _, test_metrics = testing.predict(test_loader, epoch=0, device=device, prefix="test")

        # write results to auxiliary logs file for convenience

        print("writing finetuning logs to central csv for convenience!")
        central_logs_name = f'finetuningsc_{args.targets}_logs.csv'
        central_logs_path = os.path.join(args.logs_root, central_logs_name)
        if not os.path.isfile(central_logs_path):
            with open(central_logs_path, 'w') as f:
                if args.clf_or_reg == "clf":
                    f.write(f"model_name,pretrained_model,"
                            f"val_loss,test_loss,bal_acc_val,bal_acc_tst,roc_val,roc_tst\n")
                else:
                    f.write(f"model_name,pretrained_model,"
                            f"val_loss,test_loss,mse_val,mse_tst,r2_val,r2_tst\n")
        with open(central_logs_path, 'a') as f:
            if args.clf_or_reg == "clf":
                f.write(f"{args.model_name},{args.pretrained_model},{val_losses:.4f},{test_losses:.4f}"
                        f",{val_metrics['bal_acc']:.4f},{test_metrics['bal_acc']:.4f}"
                        f",{val_metrics['roc_auc']:.4f},{test_metrics['roc_auc']:.4f}\n")
            else:
                f.write(f"{args.model_name},{args.pretrained_model},{val_losses:.4f},{test_losses:.4f},"
                        f"{val_metrics['mse']:.4f},{test_metrics['mse']:.4f},"
                        f"{val_metrics['r2']:.4f},{test_metrics['r2']:.4f}\n")
        print(f"metrics written to {central_logs_path}")


if __name__ == "__main__":
    arguments = Arguments(mode='finetuning').parse()

    # check output roots exist; if not, create...

    if not os.path.exists(arguments.save_root):
        os.mkdir(arguments.save_root)
    if not os.path.exists(arguments.logs_root):
        os.mkdir(arguments.logs_root)

    # check that arguments are well-specified

    if arguments.clf_or_reg == 'reg':
        assert arguments.num_classes == 1, "if doing regression, num_classes for the clf_head must be 1!"

    # run finetuning
    print(f"mode is {arguments.mode}")
    main(arguments)
