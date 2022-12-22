import pandas as pd
import torch
from enumeration import SmilesEnumerator
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from tqdm import tqdm
import logging
from deepchem.molnet import load_bbbp, load_bace_classification
import numpy as np
from tqdm import tqdm

## setting the threshold of logger to INFO
logging.basicConfig(filename="data_loader.log", level=logging.INFO)

## creating an object
logger = logging.getLogger()


MOLECULE_NET_DATASETS = {"bbbp": load_bbbp, "bace": load_bace_classification}


class MoleculeData:
    def __init__(
        self,
        dataset_name,
        max_sequence_length=512,
        debug=0,
        n_augment=0,
        samples_per_class=-1,
        model_name_or_path="shahrukhx01/smole-bert",
    ):
        """
        Load dataset and bert tokenizer
        """
        self.debug = debug
        ## load data into memory
        tasks, datasets, transformers = MOLECULE_NET_DATASETS[dataset_name](
            reload=False
        )
        self.train_dataset, self.valid_dataset, self.test_dataset = datasets

        ## set max sequence length for model
        self.max_sequence_length = max_sequence_length
        ## get bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=True
        )
        self.enumerator = SmilesEnumerator()
        self.n_augment = n_augment
        self.samples_per_class = samples_per_class

    def train_val_test_split(self):
        """
        Separate out labels and texts
        """
        num_samples = 1_000_000
        if self.debug:
            print("Debug mode is enabled")
            num_samples = 100
        train_molecules = self.train_dataset.ids[:num_samples]
        train_labels = np.array(
            [int(label[0]) for label in self.train_dataset.y][:num_samples]
        )

        self.indices = []
        tp, tn = [], []
        self.augmented_data_index = len(train_molecules)
        self.label_df = pd.DataFrame(train_labels, columns=["labels"])
        if self.samples_per_class > 0:
            np.random.seed()
            tp = np.random.choice(
                list(self.label_df[self.label_df["labels"] == 1].index),
                self.samples_per_class,
                replace=False,
            )
            tn = np.random.choice(
                list(self.label_df[self.label_df["labels"] == 0].index),
                self.samples_per_class,
                replace=False,
            )
            self.indices = list(tp) + list(tn)
        aug_molecules, aug_labels = [], []
        if self.n_augment:
            for train_smiles, train_label in tqdm(
                zip(train_molecules[self.indices], train_labels[self.indices])
            ):

                molecules_augmented = self.enumerator.smiles_enumeration(
                    input_smiles=train_smiles, n_augment=self.n_augment
                )
                if len(molecules_augmented):
                    train_augmented_labels = [train_label] * len(molecules_augmented)
                    aug_molecules += molecules_augmented
                    aug_labels += train_augmented_labels
        if len(aug_molecules) and len(aug_molecules) == len(aug_labels):
            train_molecules, train_labels = list(train_molecules), list(train_labels)
            train_molecules += aug_molecules
            train_labels += aug_labels

        val_molecules = self.valid_dataset.ids
        val_labels = np.array([int(label[0]) for label in self.valid_dataset.y])

        test_molecules = self.test_dataset.ids
        test_labels = np.array([int(label[0]) for label in self.test_dataset.y])

        return (
            train_molecules,
            val_molecules,
            test_molecules,
            train_labels,
            val_labels,
            test_labels,
        )

    def preprocess(self, texts):
        """
        Add bert token (CLS and SEP) tokens to each sequence pre-tokenization
        """
        ## separate labels and texts before preprocessing
        # Adding CLS and SEP tokens at the beginning and end of each sequence for BERT
        texts_processed = ["[CLS] " + str(sequence) + " [SEP]" for sequence in texts]
        return texts_processed

    def tokenize(self, texts):
        """
        Use bert tokenizer to tokenize each sequence and post-process
        by padding or truncating to a fixed length
        """
        ## tokenize sequence
        tokenized_molecules = [self.tokenizer.tokenize(text) for text in tqdm(texts)]

        ## convert tokens to ids
        print("convert tokens to ids")
        text_ids = [
            self.tokenizer.convert_tokens_to_ids(x) for x in tqdm(tokenized_molecules)
        ]

        ## pad our text tokens for each sequence
        print("pad our text tokens for each sequence")
        text_ids_post_processed = pad_sequences(
            text_ids,
            maxlen=self.max_sequence_length,
            dtype="long",
            truncating="post",
            padding="post",
        )
        return text_ids_post_processed

    def create_attention_mask(self, text_ids):
        """
        Add attention mask for padding tokens
        """
        attention_masks = []
        # create a mask of 1s for each token followed by 0s for padding
        for seq in tqdm(text_ids):
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        return attention_masks

    def process_molecules(self):
        """
        Apply preprocessing and tokenization pipeline of texts
        """
        ## perform the split
        (
            train_molecules,
            val_molecules,
            test_molecules,
            train_labels,
            val_labels,
            test_labels,
        ) = self.train_val_test_split()

        print("preprocessing texts")
        ## preprocess train, val, test texts
        train_molecules_processed = self.preprocess(train_molecules)
        val_molecules_processed = self.preprocess(val_molecules)
        test_molecules_processed = self.preprocess(test_molecules)

        del train_molecules
        del val_molecules
        del test_molecules

        ## preprocess train, val, test texts
        print("tokenizing train texts")
        train_ids = self.tokenize(train_molecules_processed)
        print("tokenizing val texts")
        val_ids = self.tokenize(val_molecules_processed)
        print("tokenizing test texts")
        test_ids = self.tokenize(test_molecules_processed)

        del train_molecules_processed
        del val_molecules_processed
        del test_molecules_processed

        ## create masks for train, val, test texts
        print("creating train attention masks for texts")
        train_masks = self.create_attention_mask(train_ids)
        print("creating val attention masks for texts")
        val_masks = self.create_attention_mask(val_ids)
        print("creating test attention masks for texts")
        test_masks = self.create_attention_mask(test_ids)
        return (
            train_ids,
            val_ids,
            test_ids,
            train_masks,
            val_masks,
            test_masks,
            train_labels,
            val_labels,
            test_labels,
        )

    def text_to_tensors(self):
        """
        Converting all the data into torch tensors
        """
        (
            train_ids,
            val_ids,
            test_ids,
            train_masks,
            val_masks,
            test_masks,
            train_labels,
            val_labels,
            test_labels,
        ) = self.process_molecules()

        print("converting all variables to tensors")
        ## convert inputs, masks and labels to torch tensors
        self.train_inputs = torch.tensor(train_ids)

        train_values = np.max(train_labels) + 1

        self.train_labels = torch.tensor(
            train_labels,
            dtype=torch.long,
        )
        self.train_masks = torch.tensor(train_masks)

        self.validation_inputs = torch.tensor(val_ids)
        self.validation_labels = torch.tensor(
            val_labels,
            dtype=torch.long,
        )
        self.validation_masks = torch.tensor(val_masks)

        self.test_inputs = torch.tensor(test_ids)
        self.test_labels = torch.tensor(
            test_labels,
            dtype=torch.long,
        )
        self.test_masks = torch.tensor(test_masks)


if __name__ == "__main__":
    dataset_name = "bbbp"
    MoleculeData(dataset_name=dataset_name)
