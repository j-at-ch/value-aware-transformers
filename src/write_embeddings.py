"""
Loads a pretrained checkpoint and writes token embeddings to TensorBoard.
View with: tensorboard --logdir <logs_root>
"""

import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from va_transformers.utils.data_utils import fetch_mappings
from va_transformers.utils.mappings import Mappings
from va_transformers.utils.utils import read_yaml


def main(args):
    device = torch.device(args.device)

    config = read_yaml(args.config)
    data_root = config['data_path']
    logs_root = config['logs_path']

    mappings_dict = fetch_mappings(os.path.join(data_root, "mappings.pkl"))
    mappings = Mappings(mappings_dict, pad_token=0, eos_token=len(mappings_dict['itemid2token']))

    d_items = pd.read_csv(
        os.path.join(data_root, "d_labitems.csv"),
        index_col='ITEMID',
        dtype={'ITEMID': str}
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    token_emb_weights = checkpoint['model_state_dict']['net.token_emb.weight']
    quant_emb_weights = checkpoint['model_state_dict']['net.quant_emb.weight']

    # token embeddings: all tokens except PAD and EOS
    tokens = list(mappings_dict['token2itemid'].keys())
    token_embs = token_emb_weights[tokens].to(device)

    labels, categories, fluids, counts = [], [], [], []
    for tok in tokens:
        itemid = str(mappings.token2itemid[tok])
        count = mappings_dict['token2trcount'].get(tok, 0)
        if itemid in d_items.index:
            row = d_items.loc[itemid]
            label = row['LABEL']
            category = row['CATEGORY']
            fluid = row['FLUID']
            labels.append(f"{label} ({fluid})")
            categories.append(category)
            fluids.append(fluid)
        else:
            labels.append(f"itemid:{itemid}")
            categories.append('Unknown')
            fluids.append('Unknown')
        counts.append(count)

    metadata = list(zip(labels, categories, fluids, counts))
    metadata_header = ['label', 'category', 'fluid', 'train_count']

    tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    logs_path = os.path.join(logs_root, f'embeddings_{tag}')
    writer = SummaryWriter(log_dir=logs_path)

    writer.add_embedding(
        token_embs,
        metadata=metadata,
        metadata_header=metadata_header,
        global_step=checkpoint.get('epoch', 0),
        tag='token_embeddings'
    )

    # quant embeddings
    quant_tokens = list(mappings_dict['qtoken2qname'].keys())
    quant_embs = quant_emb_weights[quant_tokens].to(device)
    quant_labels = [mappings_dict['qtoken2qname'][q] for q in quant_tokens]

    writer.add_embedding(
        quant_embs,
        metadata=quant_labels,
        global_step=checkpoint.get('epoch', 0),
        tag='quant_embeddings'
    )

    writer.close()
    print(f"Embeddings written to {logs_path}")
    print(f"View with: tensorboard --logdir {logs_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
