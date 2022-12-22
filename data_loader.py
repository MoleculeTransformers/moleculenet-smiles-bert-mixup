from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from data import MoleculeData
import pandas as pd
import numpy as np
import torch


class MoleculeDataLoader:
    def __init__(
        self,
        dataset_name,
        batch_size=8,
        debug=0,
        n_augment=0,
        samples_per_class=-1,
        model_name_or_path="shahrukhx01/smole-bert",
    ):
        self.molecule_data = MoleculeData(
            dataset_name,
            debug=debug,
            n_augment=n_augment,
            samples_per_class=samples_per_class,
            model_name_or_path=model_name_or_path,
        )
        self.batch_size = batch_size

    def create_supervised_loaders(self, samples_per_class=-1):
        """
        Create Torch dataloaders for data splits
        """

        self.molecule_data.text_to_tensors()
        print("creating dataloaders")
        train_data = None
        print(f"SAMPLES per class: {samples_per_class}")
        print(
            f"Train samples before augmentation: {len(self.molecule_data.train_inputs[self.molecule_data.indices])}"
        )

        train_data = TensorDataset(
            torch.cat(
                (
                    self.molecule_data.train_inputs[self.molecule_data.indices],
                    self.molecule_data.train_inputs[
                        self.molecule_data.augmented_data_index :
                    ],
                ),
                0,
            ),
            torch.cat(
                (
                    self.molecule_data.train_masks[self.molecule_data.indices],
                    self.molecule_data.train_masks[
                        self.molecule_data.augmented_data_index :
                    ],
                ),
                0,
            ),
            torch.cat(
                (
                    self.molecule_data.train_labels[self.molecule_data.indices],
                    self.molecule_data.train_labels[
                        self.molecule_data.augmented_data_index :
                    ],
                ),
                0,
            ),
        )
        print(f"Train samples after augmentation: {len(train_data)}")
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.batch_size
        )

        validation_data = TensorDataset(
            self.molecule_data.validation_inputs,
            self.molecule_data.validation_masks,
            self.molecule_data.validation_labels,
        )
        validation_sampler = RandomSampler(validation_data)
        self.validation_dataloader = DataLoader(
            validation_data,
            sampler=validation_sampler,
            batch_size=len(validation_data),
        )

        test_data = TensorDataset(
            self.molecule_data.test_inputs,
            self.molecule_data.test_masks,
            self.molecule_data.test_labels,
        )
        test_sampler = RandomSampler(test_data)
        self.test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=len(test_data)
        )
        print("finished creating dataloaders")

    def create_semi_supervised_loaders(self, samples_per_class=100, n_augmentations=0):
        """
        Create Torch dataloaders for data splits
        """
        self.molecule_data.text_to_tensors()
        print("creating dataloaders")
        train_data = None
        print(f"SAMPLES per class: {samples_per_class}")
        print(
            f"Train samples before augmentation: {len(self.molecule_data.train_inputs[self.molecule_data.indices])}"
        )

        train_data = TensorDataset(
            torch.cat(
                (
                    self.molecule_data.train_inputs[self.molecule_data.indices],
                    self.molecule_data.train_inputs[
                        self.molecule_data.augmented_data_index :
                    ],
                ),
                0,
            ),
            torch.cat(
                (
                    self.molecule_data.train_masks[self.molecule_data.indices],
                    self.molecule_data.train_masks[
                        self.molecule_data.augmented_data_index :
                    ],
                ),
                0,
            ),
            torch.cat(
                (
                    self.molecule_data.train_labels[self.molecule_data.indices],
                    self.molecule_data.train_labels[
                        self.molecule_data.augmented_data_index :
                    ],
                ),
                0,
            ),
        )
        print(f"Train samples after augmentation: {len(train_data)}")
        labelled_sampler = RandomSampler(train_data)
        self.labelled_dataloader = DataLoader(
            train_data, sampler=labelled_sampler, batch_size=self.batch_size
        )

        ## create unlabelled dataloader

        unlabelled_indices = list(
            self.molecule_data.label_df.drop(self.molecule_data.indices, axis=0).index
        )
        unlabelled_data = TensorDataset(
            self.molecule_data.train_inputs[unlabelled_indices],
            self.molecule_data.train_masks[unlabelled_indices],
            self.molecule_data.train_labels[unlabelled_indices],
        )
        unlabelled_sampler = RandomSampler(unlabelled_data)
        self.unlabelled_dataloader = DataLoader(
            unlabelled_data, sampler=unlabelled_sampler, batch_size=self.batch_size
        )

        print(
            f"total data {len(self.molecule_data.label_df)} \
                labelled data {len(train_data)} \
                unlabelled data {len(unlabelled_data)}"
        )

        validation_data = TensorDataset(
            self.molecule_data.validation_inputs,
            self.molecule_data.validation_masks,
            self.molecule_data.validation_labels,
        )
        validation_sampler = RandomSampler(validation_data)
        self.validation_dataloader = DataLoader(
            validation_data, sampler=validation_sampler, batch_size=len(validation_data)
        )

        test_data = TensorDataset(
            self.molecule_data.test_inputs,
            self.molecule_data.test_masks,
            self.molecule_data.test_labels,
        )
        test_sampler = RandomSampler(test_data)
        self.test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=len(test_data)
        )
        print("finished creating dataloaders")


if __name__ == "__main__":
    spam_loader = MoleculeDataLoader(dataset_name="bace")
    spam_loader.create_loaders()
