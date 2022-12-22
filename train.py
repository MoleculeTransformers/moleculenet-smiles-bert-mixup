import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

from models.text_bert import TextBERT
from data_loader import MoleculeDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Mixup for text classification")
    parser.add_argument(
        "--name", default="cnn-text-fine-tune", type=str, help="name of the experiment"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        metavar="L",
        help="number of labels of the train dataset (default: 2)",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="shahrukhx01/smole-bert",
        metavar="M",
        help="name of the pre-trained transformer model from hf hub",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="bace",
        metavar="D",
        help="name of the molecule net dataset (default: bace) all: bace, bbbp",
    )
    parser.add_argument(
        "--cuda",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="use cuda if available",
    )
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.5, type=float, help="dropout rate")
    parser.add_argument("--decay", default=0.0, type=float, help="weight decay")
    parser.add_argument(
        "--model", default="TextCNN", type=str, help="model type (default: TextCNN)"
    )
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument(
        "--batch-size", default=50, type=int, help="batch size (default: 128)"
    )
    parser.add_argument(
        "--epoch", default=50, type=int, help="total epochs (default: 200)"
    )
    parser.add_argument(
        "--fine-tune",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="whether to fine-tune embedding or not",
    )
    parser.add_argument(
        "--method",
        default="embed",
        type=str,
        help="which mixing method to use (default: none)",
    )
    parser.add_argument(
        "--alpha",
        default=1.0,
        type=float,
        help="mixup interpolation coefficient (default: 1)",
    )
    parser.add_argument(
        "--save-path", default="out", type=str, help="output log/result directory"
    )
    parser.add_argument("--num-runs", default=10, type=int, help="number of runs")
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        metavar="DB",
        help="flag to enable debug mode for dev (default: 0)",
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=-1,
        metavar="SPC",
        help="no. of samples per class label to sample for SSL (default: 250)",
    )
    parser.add_argument(
        "--n-augment",
        type=int,
        default=0,
        metavar="NAUG",
        help="number of enumeration augmentations",
    )
    parser.add_argument(
        "--eval-after",
        type=int,
        default=10,
        metavar="EA",
        help="number of epochs after which model is evaluated on test set (default: 10)",
    )
    args = parser.parse_args()
    return args


def mixup_criterion_cross_entropy(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Classification:
    def __init__(self, args):
        self.args = args

        self.use_cuda = args.cuda and torch.cuda.is_available()

        # for reproducibility
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

        # data loaders
        data_loaders = MoleculeDataLoader(
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            debug=args.debug,
            n_augment=args.n_augment,
            samples_per_class=args.samples_per_class,
            model_name_or_path=args.model_name_or_path,
        )
        data_loaders.create_supervised_loaders(samples_per_class=args.samples_per_class)
        # model

        self.model = TextBERT(
            pretrained_model=args.model_name_or_path,
            num_class=args.num_labels,
            fine_tune=args.fine_tune,
            dropout=args.dropout,
        )
        self.device = torch.device(
            "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
        )
        self.model.to(self.device)

        # logs
        os.makedirs(args.save_path, exist_ok=True)
        self.model_save_path = os.path.join(args.save_path, args.name + "_weights.pt")
        self.log_path = os.path.join(args.save_path, args.name + "_logs.csv")
        print(str(args))
        with open(self.log_path, "a") as f:
            f.write(str(args) + "\n")
        with open(self.log_path, "a", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["mode", "epoch", "step", "loss", "acc"])

        # optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.decay
        )

        # for early stopping
        self.best_val_acc = 0
        self.early_stop = False
        self.val_patience = (
            0  # successive iteration when validation acc did not improve
        )

        self.iteration_number = 0

    def get_perm(self, x):
        """get random permutation"""
        batch_size = x.size()[0]
        if self.use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        return index

    def test(self, iterator):
        self.model.eval()
        test_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            # for _, batch in tqdm(enumerate(iterator), total=len(iterator), desc='test'):
            for _, batch in enumerate(iterator):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                y_pred = self.model(b_input_ids, b_input_mask)
                loss = self.criterion(y_pred, b_labels)
                test_loss += loss.item() * b_labels.shape[0]
                total += b_labels.shape[0]
                correct += torch.sum(torch.argmax(y_pred, dim=1) == b_labels).item()

        avg_loss = test_loss / total
        acc = 100.0 * correct / total
        return avg_loss, acc

    def train_mixup(self, epoch):
        self.model.train()
        train_loss = 0
        total = 0
        correct = 0
        for _, batch in enumerate(self.train_iterator):
            batch = tuple(t.to(self.device) for t in batch)
            lam = np.random.beta(self.args.alpha, self.args.alpha)
            b_input_ids, b_input_mask, b_labels = batch
            index = self.get_perm(b_input_ids)
            b_input_ids1 = b_input_ids1[:, index]
            b_input_mask1 = b_input_mask[:, index]
            b_labels1 = b_labels[:, index]

            if self.args.method == "embed":
                y_pred = self.model.forward_mix_embed(
                    b_input_ids, b_input_mask, b_input_ids1, b_input_mask1, lam
                )
            elif self.args.method == "sent":
                y_pred = self.model.forward_mix_sent(
                    b_input_ids, b_input_mask, b_input_ids1, b_input_mask1, lam
                )
            elif self.args.method == "encoder":
                y_pred = self.model.forward_mix_encoder(
                    b_input_ids, b_input_mask, b_input_ids1, b_input_mask1, lam
                )
            else:
                raise ValueError("invalid method name")

            loss = mixup_criterion_cross_entropy(
                self.criterion, y_pred, b_labels, b_labels1, lam
            )
            train_loss += loss.item() * b_labels.shape[0]
            total += b_labels.shape[0]
            _, predicted = torch.max(y_pred.data, 1)
            correct += (
                (
                    lam * predicted.eq(b_labels.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(b_labels1.data).cpu().sum().float()
                )
            ).item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # eval
            self.iteration_number += 1
            if self.iteration_number % self.args.eval_after == 0:
                avg_loss = train_loss / total
                acc = 100.0 * correct / total
                # print('Train loss: {}, Train acc: {}'.format(avg_loss, acc))
                train_loss = 0
                total = 0
                correct = 0

                val_loss, val_acc = self.test(iterator=self.val_iterator)
                # print('Val loss: {}, Val acc: {}'.format(val_loss, val_acc))
                if val_acc > self.best_val_acc:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    self.best_val_acc = val_acc
                    self.val_patience = 0
                else:
                    self.val_patience += 1
                    if self.val_patience == self.config.patience:
                        self.early_stop = True
                        return
                with open(self.log_path, "a", newline="") as out:
                    writer = csv.writer(out)
                    writer.writerow(
                        ["train", epoch, self.iteration_number, avg_loss, acc]
                    )
                    writer.writerow(
                        ["val", epoch, self.iteration_number, val_loss, val_acc]
                    )
                self.model.train()

    def run(self):
        for epoch in range(self.args.epoch):
            print(
                "------------------------------------- Epoch {} -------------------------------------".format(
                    epoch
                )
            )
            if self.args.method == "none":
                self.train(epoch)
            else:
                self.train_mixup(epoch)
            if self.early_stop:
                break
        print("Training complete!")
        print("Best Validation Acc: ", self.best_val_acc)

        self.model.load_state_dict(torch.load(self.model_save_path))
        # train_loss, train_acc = self.test(self.train_iterator)
        val_loss, val_acc = self.test(self.val_iterator)
        test_loss, test_acc = self.test(self.test_iterator)

        with open(self.log_path, "a", newline="") as out:
            writer = csv.writer(out)
            # writer.writerow(['train', -1, -1, train_loss, train_acc])
            writer.writerow(["val", -1, -1, val_loss, val_acc])
            writer.writerow(["test", -1, -1, test_loss, test_acc])

        # print('Train loss: {}, Train acc: {}'.format(train_loss, train_acc))
        print("Val loss: {}, Val acc: {}".format(val_loss, val_acc))
        print("Test loss: {}, Test acc: {}".format(test_loss, test_acc))

        return val_acc, test_acc


if __name__ == "__main__":
    args = parse_args()
    num_runs = args.num_runs

    test_acc = []
    val_acc = []

    for i in range(num_runs):
        cls = Classification(args)
        val, test = cls.run()
        val_acc.append(val)
        test_acc.append(test)
        args.seed += 1

    with open(os.path.join(args.save_path, args.name + "_result.txt", "a")) as f:
        f.write(str(args))
        f.write("val acc:" + str(val_acc) + "\n")
        f.write("test acc:" + str(test_acc) + "\n")
        f.write("mean val acc:" + str(np.mean(val_acc)) + "\n")
        f.write("std val acc:" + str(np.std(val_acc, ddof=1)) + "\n")
        f.write("mean test acc:" + str(np.mean(test_acc)) + "\n")
        f.write("std test acc:" + str(np.std(test_acc, ddof=1)) + "\n\n\n")
