import pickle
import torch
import pandas as pd


def fetch_mappings(mapping_path):
    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
    return mappings


def fetch_data_as_numpy(path, var_key):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    di = data[var_key]
    return di


def fetch_data_as_torch(path, var_key):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    di = data[var_key]
    return {k: torch.from_numpy(v) for k, v in di.items()}


def make_toy_loader(loader, size=2):
    toy_loader = []
    for i, X in enumerate(loader):
        if i == size - 1:
            break
        else:
            toy_loader.append(X)
    return toy_loader


def ts_to_posix(time):
    return pd.Timestamp(time, unit='s').timestamp()
