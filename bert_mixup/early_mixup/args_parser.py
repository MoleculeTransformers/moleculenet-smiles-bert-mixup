import argparse

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
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument(
        "--batch-size", default=50, type=int, help="batch size (default: 128)"
    )
    parser.add_argument(
        "--epoch", default=20, type=int, help="total epochs (default: 20)"
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
    parser.add_argument("--num-runs", default=1, type=int, help="number of runs")
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
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        metavar="PAT",
        help="Patience epochs when doing model selection if the model does not improve",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="eval_result.csv",
        metavar="OF",
        help="outpul file for logging metrics",
    )
    args = parser.parse_args()
    return args