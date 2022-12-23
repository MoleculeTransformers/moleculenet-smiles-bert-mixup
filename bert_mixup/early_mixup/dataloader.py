from deepchem.molnet import load_bace_classification, load_bbbp
import numpy as np

from simcse import SimCSE
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from args_parser import parse_args
import sys
import pandas as pd


_datasets = {"bace": load_bace_classification, "bbbp": load_bbbp}


def embed_smiles(model, smiles):
    embeddings = model.encode(smiles)
    return embeddings


def get_dataloaders(args):
    model = SimCSE(args.model_name_or_path)

    _, datasets, _ = _datasets.get(args.dataset_name)(reload=False)
    (train_dataset, valid_dataset, test_dataset) = datasets

    train_indices = []
    train_labels = [y[0] for y in train_dataset.y]
    label_df = pd.DataFrame(train_labels, columns=["labels"])
    if args.samples_per_class > 0:
        np.random.seed()
        tp = np.random.choice(
            list(label_df[label_df["labels"] == 1].index),
            args.samples_per_class,
            replace=False,
        )
        tn = np.random.choice(
            list(label_df[label_df["labels"] == 0].index),
            args.samples_per_class,
            replace=False,
        )
        train_indices = list(tp) + list(tn)

    np.random.seed()

    train_smiles = train_dataset.ids[train_indices]
    train_embeddings = embed_smiles(model, smiles=list(train_smiles))
    train_labels = np.array([y[0] for y in train_dataset.y[train_indices]])

    val_smiles = valid_dataset.ids
    val_embeddings = embed_smiles(model, smiles=list(val_smiles))
    val_labels = np.array([y[0] for y in valid_dataset.y])

    test_smiles = test_dataset.ids
    test_embeddings = embed_smiles(model, smiles=list(test_smiles))
    test_labels = np.array([y[0] for y in test_dataset.y])

    train_data = TensorDataset(train_embeddings, torch.Tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.batch_size
    )

    val_data = TensorDataset(val_embeddings, torch.Tensor(val_labels))
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=len(val_data))

    test_data = TensorDataset(test_embeddings, torch.Tensor(test_labels))
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=len(test_data)
    )

    return train_dataloader, val_dataloader, test_dataloader
