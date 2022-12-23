import torch
import numpy as np
from utils import (
    mixup_augment,
    mixup_criterion_cross_entropy,
    get_perm,
    flat_auroc_score,
)


def train_bert(
    train_dataloader, val_dataloader, model_mlp, args, set_device, optimizer, criterion
):
    best_model = None

    best_accuracy = 0.0
    train_loss_history, recall_train_history = [], []
    validation_loss_history, recall_validation_history = list(), list()
    for epoch in range(0, args.epoch):
        model_mlp.train()
        train_loss_scores = []
        training_acc_scores = []
        y_pred, y_true = list(), list()
        predictions = []
        for x, y in train_dataloader:
            ## perform forward pass
            x = x.type(torch.FloatTensor).to(set_device)
            y = y.type(torch.LongTensor).to(set_device)
            for i in range(args.n_augment):
                lam = np.random.beta(args.alpha, args.alpha)
                indices_permuted = get_perm(x, args)
                x2 = x[indices_permuted, :]
                y2 = y[indices_permuted]
                mixup_x, _ = mixup_augment(
                    embedding1=x, embedding2=x2, label1=y, label2=y2, lam=lam
                )

                pred = model_mlp(mixup_x)

                preds = torch.max(pred, 1)[1]

                ## accumulate predictions per batch for the epoch
                """y_pred += list([x.item() for x in preds.detach().cpu().numpy()])
                    targets = torch.LongTensor([x.item() for x in list(targets)])
                    y_true +=  list([x.item() for x in targets.detach().cpu().numpy()])"""

                ## compute loss and perform backward pass
                loss = mixup_criterion_cross_entropy(
                    criterion, pred, y, y2, lam=lam
                )  ## compute loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predictions.append(pred)

                ## accumulate train loss
                train_loss_scores.append(loss.item())

        ## accumulate loss, recall, f1, precision per epoch
        train_loss_history.append((sum(train_loss_scores) / len(train_loss_scores)))
        # recall = flat_auroc_score(predictions, y_true)
        # recall_train_history.append(recall)
        print(f"Training =>  Epoch : {epoch+1} | Loss : {train_loss_history[-1]}")
        # | AUROC score: {recall_train_history[-1]}')

        model_mlp.eval()
        predictions = None
        with torch.no_grad():
            validation_loss_scores = list()
            y_true_val, y_pred_val = list(), list()

            ## perform validation pass
            for batch, targets in val_dataloader:
                ## perform forward pass
                batch = batch.type(torch.FloatTensor).to(set_device)
                pred = model_mlp(batch)
                predictions = pred
                preds = torch.max(pred, 1)[1]

                ## accumulate predictions per batch for the epoch
                y_pred_val += list([x.item() for x in preds.detach().cpu().numpy()])
                targets = torch.LongTensor([x.item() for x in list(targets)])
                y_true_val += list([x.item() for x in targets.detach().cpu().numpy()])

                ## computing validate loss
                loss = criterion(
                    pred.to(set_device), targets.to(set_device)
                )  ## compute loss

                ## accumulate validate loss
                validation_loss_scores.append(loss.item())

            ## accumulate loss, recall, f1, precision per epoch
            validation_loss_history.append(
                (sum(validation_loss_scores) / len(validation_loss_scores))
            )
            recall = flat_auroc_score(predictions, y_true_val)
            recall_validation_history.append(recall)

            print(
                f"Validation =>  Epoch : {epoch+1} | Loss : {validation_loss_history[-1]} | AUROC score: {recall_validation_history[-1]} "
            )

            if recall_validation_history[-1] > best_accuracy:
                best_accuracy = recall_validation_history[-1]
                print("Selecting the model...")
                best_model = model_mlp

    return best_model
