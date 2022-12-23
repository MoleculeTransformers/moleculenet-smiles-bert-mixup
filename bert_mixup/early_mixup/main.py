import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from args_parser import parse_args
from model import MolNet
from dataloader import get_dataloaders
from train_bert import train_bert
from eval import evaluate_model


if __name__ == "__main__":
    args = parse_args()
    input_dim = 512
    output_dim = args.num_labels
    set_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args)
    model_mlp = MolNet(input_dim=input_dim, output_dim=output_dim).to(set_device)
    criterion = nn.CrossEntropyLoss().to(set_device)
    optimizer = getattr(optim, "Adam")(model_mlp.parameters(), lr=args.lr)

    ## train model
    best_model = train_bert(
        train_dataloader,
        val_dataloader,
        model_mlp,
        args,
        set_device,
        optimizer,
        criterion,
    )

    ## evaluate the model
    evaluate_model(args, best_model, test_dataloader, criterion, set_device)
