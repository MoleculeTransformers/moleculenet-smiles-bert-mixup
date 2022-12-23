import torch
from utils import flat_auroc_score


def evaluate_model(args, model_mlp, test_dataloader, criterion, set_device):
    test_loss_history, auroc_test_history = list(), list()
    model_mlp.eval()
    predictions = None
    with torch.no_grad():
        test_loss_scores = list()
        y_true_val, y_pred_val = list(), list()

        ## perform test pass
        for batch, targets in test_dataloader:
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
            test_loss_scores.append(loss.item())

        ## accumulate loss, auroc, f1, precision per epoch
        test_loss_history.append((sum(test_loss_scores) / len(test_loss_scores)))
        auroc = flat_auroc_score(predictions, y_true_val)
        auroc_test_history.append(auroc)

        print(f"Test => AUROC score: {auroc_test_history[-1]} ")
        with open(args.out_file, "a+") as f:
            f.write(
                f"{args.dataset_name}, {args.samples_per_class}, {args.n_augment}, {auroc_test_history[-1]}\n"
            )
