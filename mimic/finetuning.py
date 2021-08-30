import pandas as pd
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

import methods
from data_utils import *
from arguments import Arguments
from models import FinetuningWrapper
from v_transformers.vtransformers import TransformerWrapper, Decoder


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
        train_targets = {k: v[args.label_set] for k, v in X['train_targets'].items()}
        del X

    with open(val_lbl_path, 'rb') as f:
        X = pickle.load(f)
        val_targets = {k: v[args.label_set] for k, v in X['val_targets'].items()}
        del X

    # get tokens

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    # get quantiles

    if args.value_guided == 'plain':
        quantiles_train = None
        quantiles_val = None
    else:
        quantiles_train = fetch_data_as_torch(train_path, 'train_quantiles')
        quantiles_val = fetch_data_as_torch(val_path, 'val_quantiles')

    #ft_train_dataset = ClsSamplerDataset(data_train, args.seq_len, device, labels=train_targets)
    #ft_val_dataset = ClsSamplerDataset(data_val, args.seq_len, device, labels=val_targets)
    #ft_train_loader = DataLoader(ft_train_dataset, batch_size=args.ft_batch_size, shuffle=True)
    #ft_val_loader = DataLoader(ft_val_dataset, batch_size=args.ft_batch_size, shuffle=True)

    train_dataset = VgSamplerDataset(data_train, args.seq_len, device, quantiles=quantiles_train, labels=train_targets)
    val_dataset = VgSamplerDataset(data_val, args.seq_len, device, quantiles=quantiles_val, labels=val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for quick test run

    if bool(args.test_run):
        train_loader = [X for i, X in enumerate(train_loader) if i < 2]
        val_loader = [X for i, X in enumerate(val_loader) if i < 2]

    print('*'*100)

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

    # base_states = {k[len('net.'):] if k[:len('net.')] == 'net.' else k: v for k, v in states.items()}

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
            value_guided=args.value_guided,
            use_rezero=bool(args.use_rezero),
            rotary_pos_emb=bool(args.rotary_pos_emb)
        )
    )

    # wrap model for finetuning

    fit_model = FinetuningWrapper(model,
                                  num_classes=2,
                                  seq_len=args.seq_len,  # doesn't model know this?
                                  state_dict=states,
                                  load_from_pretrained=True,
                                  weight=weights,
                                  value_guided=args.value_guided)
    fit_model.to(device)

    print("model specification:", fit_model.net, sep="\n")

    # initialise optimiser

    optimizer = torch.optim.Adam(fit_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
    writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
    training = methods.FinetuningMethods(fit_model, writer)

    # write initial embeddings

    if bool(args.write_initial_embeddings):
        training.write_embeddings(-1, mappings, labeller, args.seq_len, device)

    # training loop

    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        ________ = training.train(train_loader, optimizer, epoch)
        val_loss = training.evaluate(val_loader, epoch)

        # whether to checkpoint model

        if val_loss < best_val_loss:
            print("Saving checkpoint...")
            torch.save({
                'train_epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
                'model_state_dict': fit_model.state_dict(),
                'optim_state_dict': optimizer.state_dict()
            }, ckpt_path)

            # track checkpoint's embeddings

            if bool(args.write_best_val_embeddings):
                training.write_embeddings(epoch, mappings, labeller, args.seq_len, device)

            print("Checkpoint saved!\n")
            best_val_loss = val_loss

        scheduler.step()

        # tracking value_guided parameters

        if args.value_guided != 'plain':
            training.write_g_histograms(epoch)  # TODO add method to methods.FinetuningMethods

        # tracking model classification metrics for val set

        ________ = training.predict(train_loader, epoch, device, prefix="train")
        ________ = training.predict(val_loader, epoch, device, prefix="val")

        # flushing writer

        print(f'epoch {epoch} completed!')
        print('flushing writer...')
        writer.flush()

    # write final embeddings

    if bool(args.write_final_embeddings):
        training.write_embeddings(args.num_epochs, mappings, labeller, args.seq_len, device)

    writer.close()
    print("training finished and writer closed!")


if __name__ == "__main__":
    arguments = Arguments(mode='finetuning').parse()

    # check output roots exist; if not, create...

    if not os.path.exists(arguments.save_root):
        os.mkdir(arguments.save_root)
    if not os.path.exists(arguments.logs_root):
        os.mkdir(arguments.logs_root)

    # run finetuning

    finetune(arguments)
