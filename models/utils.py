import os
import sys
import errno
import json
import time
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn import metrics

import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader

from simuclr.utils import paint


def makedir(path):
    """
    Creates a directory if not already exists.

    :param str path: The path which is to be created.
    :return: None
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.exists:
        print(f"[+] Created directory in {path}")


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name, fmt=":4f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            makedir(os.path.dirname(fpath))
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def init_weights(model, method):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network)

    :param model: network of which weights are to be initialised
    :param str method: Method to initialise weights
    :return: network with initialised weights
    """
    for m in model.modules():
        if isinstance(m, nn.Linear) or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            if method == "normal":
                torch.nn.init.normal_(m.weight)
            elif method == "orthogonal":
                torch.nn.init.orthogonal_(m.weight)
            elif method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            elif method == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            elif method == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            elif method == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            # LSTM initialisation
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    if method == "normal":
                        torch.nn.init.normal_(param.data)
                    elif method == "orthogonal":
                        torch.nn.init.orthogonal_(param.data)
                    elif method == "xavier_uniform":
                        torch.nn.init.xavier_uniform_(param.data)
                    elif method == "xavier_normal":
                        torch.nn.init.xavier_normal_(param.data)
                    elif method == "kaiming_uniform":
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif method == "kaiming_normal":
                        torch.nn.init.kaiming_normal_(param.data)
                elif "weight_hh" in name:
                    if method == "normal":
                        torch.nn.init.normal_(param.data)
                    elif method == "orthogonal":
                        torch.nn.init.orthogonal_(param.data)
                    elif method == "xavier_uniform":
                        torch.nn.init.xavier_uniform_(param.data)
                    elif method == "xavier_normal":
                        torch.nn.init.xavier_normal_(param.data)
                    elif method == "kaiming_uniform":
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif method == "kaiming_normal":
                        torch.nn.init.kaiming_normal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0.0)
    return model


def run_train_analysis(train_results):
    """
    Runs an average and subject-wise analysis of saved train results.

    :param train_results: the train result dataframe returned by the cross_validate function.
    :return: None
    """
    # average analysis
    avg_t_loss, avg_t_acc, avg_t_fm, avg_t_fw = [], [], [], []
    avg_v_loss, avg_v_acc, avg_v_fm, avg_v_fw = [], [], [], []

    # average analysis
    print(paint("AVERAGE RESULTS"))
    for i, row in train_results.iterrows():
        if i == 0:
            avg_t_loss = np.asarray(row["t_loss"])
            avg_t_acc = np.asarray(row["t_acc"])
            avg_t_fm = np.asarray(row["t_fm"])
            avg_t_fw = np.asarray(row["t_fw"])
            avg_v_loss = np.asarray(row["v_loss"])
            avg_v_acc = np.asarray(row["v_acc"])
            avg_v_fm = np.asarray(row["v_fm"])
            avg_v_fw = np.asarray(row["v_fw"])
        else:
            avg_t_loss = np.add(avg_t_loss, row["t_loss"])
            avg_t_acc = np.add(avg_t_acc, row["t_acc"])
            avg_t_fm = np.add(avg_t_fm, row["t_fm"])
            avg_t_fw = np.add(avg_t_fw, row["t_fw"])
            avg_v_loss = np.add(avg_v_loss, row["v_loss"])
            avg_v_acc = np.add(avg_v_acc, row["v_acc"])
            avg_v_fm = np.add(avg_v_fm, row["v_fm"])
            avg_v_fw = np.add(avg_v_fw, row["v_fw"])

    avg_t_loss /= len(train_results)
    avg_t_acc /= len(train_results)
    avg_t_fm /= len(train_results)
    avg_t_fw /= len(train_results)
    avg_v_loss /= len(train_results)
    avg_v_acc /= len(train_results)
    avg_v_fm /= len(train_results)
    avg_v_fw /= len(train_results)

    print("\nAverage Train results (last epoch):")
    print(
        "Loss: {:.4f} - Accuracy: {:.4f} - F1-score (macro): {:.4f} - F1-score (weighted): {:.4f}".format(
            avg_t_loss[-1], avg_t_acc[-1], avg_t_fm[-1], avg_t_fw[-1]
        )
    )
    print("\nAverage Validation results (last epoch):")
    print(
        "Loss: {:.4f} - Accuracy: {:.4f} - F1-score (macro): {:.4f} - F1-score (weighted): {:.4f}".format(
            avg_v_loss[-1], avg_v_acc[-1], avg_v_fm[-1], avg_v_fw[-1]
        )
    )


def run_test_analysis(test_results):
    """
    Runs an average analysis of saved test results.

    :param test_results: the test result dataframe returned by the cross_validate function.
    :return: None
    """
    if test_results is not None:
        avg_t_loss, avg_t_acc, avg_t_fm, avg_t_fw = 0.0, 0.0, 0.0, 0.0
        t_loss_list, t_acc_list, t_fm_list, t_fw_list = [], [], [], []
        # average analysis
        for i, row in test_results.iterrows():
            if i == 0:
                avg_t_loss = np.asarray(row["test_loss"])
                avg_t_acc = np.asarray(row["test_acc"])
                avg_t_fm = np.asarray(row["test_fm"])
                avg_t_fw = np.asarray(row["test_fw"])

                t_loss_list.append(row["test_loss"])
                t_acc_list.append(row["test_acc"])
                t_fm_list.append(row["test_fm"])
                t_fw_list.append(row["test_fw"])
            else:
                avg_t_loss = np.add(avg_t_loss, row["test_loss"])
                avg_t_acc = np.add(avg_t_acc, row["test_acc"])
                avg_t_fm = np.add(avg_t_fm, row["test_fm"])
                avg_t_fw = np.add(avg_t_fw, row["test_fw"])

                t_loss_list.append(row["test_loss"])
                t_acc_list.append(row["test_acc"])
                t_fm_list.append(row["test_fm"])
                t_fw_list.append(row["test_fw"])

        if len(t_loss_list) > 0:
            print(
                f"Loss: {np.mean(t_loss_list):.4f}\u00B1{np.std(t_loss_list):.4f}",
                end="",
            )
            print(
                f"- Accuracy: {np.mean(t_acc_list):.4f}\u00B1{np.std(t_acc_list):.4f}",
                end="",
            )
            print(
                f"- F1-score (macro): {np.mean(t_fm_list):.4f}\u00B1{np.std(t_fm_list):.4f}",
                end="",
            )
            print(
                f"- F1-score: {np.mean(t_fw_list):.4f}\u00B1{np.std(t_fw_list):.4f}",
                end="",
            )


def rerun_analysis(log_directory):
    """
    Method used to rerun an analysis by loading up saved train and (if applicable) test results.

    :param log_directory: directory where results were saved to (e.g. 20211205/225740)
    :return: None
    """
    train_results_df = pd.read_csv(
        os.path.join("../logs", log_directory, "train_results.csv"), index_col=None
    )
    train_results_df[
        ["t_loss", "t_acc", "t_fm", "t_fw", "v_loss", "v_acc", "v_fm", "v_fw"]
    ] = train_results_df[
        ["t_loss", "t_acc", "t_fm", "t_fw", "v_loss", "v_acc", "v_fm", "v_fw"]
    ].apply(
        lambda x: list(map(json.loads, x))
    )
    run_train_analysis(train_results_df)
    if os.path.isfile(os.path.join("../logs", log_directory, "test_results.csv")):
        test_results_df = pd.read_csv(
            os.path.join("../logs", log_directory, "test_results.csv"), index_col=None
        )
        run_test_analysis(test_results_df)


# def wandb_logging(train_results, test_results, config, epoch_name="epochs"):
#     num_epochs = len(train_results["t_loss"])
#
#     t_loss, t_acc, t_fw, t_fm = (
#         np.zeros(num_epochs),
#         np.zeros(num_epochs),
#         np.zeros(num_epochs),
#         np.zeros(num_epochs),
#     )
#     v_loss, v_acc, v_fw, v_fm = (
#         np.zeros(num_epochs),
#         np.zeros(num_epochs),
#         np.zeros(num_epochs),
#         np.zeros(num_epochs),
#     )
#
#     for i in range(len(train_results)):
#         t_loss = np.add(t_loss, train_results["t_loss"][i])
#         t_acc = np.add(t_acc, train_results["t_acc"][i])
#         t_fw = np.add(t_fw, train_results["t_fw"][i])
#         t_fm = np.add(t_fm, train_results["t_fm"][i])
#
#         v_loss = np.add(v_loss, train_results["v_loss"][i])
#         v_acc = np.add(v_acc, train_results["v_acc"][i])
#         v_fw = np.add(v_fw, train_results["v_fw"][i])
#         v_fm = np.add(v_fm, train_results["v_fm"][i])
#
#     table = wandb.Table(
#         data=[
#             [a, b, c, d, e, f, g, h, i]
#             for (a, b, c, d, e, f, g, h, i) in zip(
#                 list(range(num_epochs)),
#                 t_loss / len(train_results),
#                 t_acc / len(train_results),
#                 t_fw / len(train_results),
#                 t_fm / len(train_results),
#                 v_loss / len(train_results),
#                 v_acc / len(train_results),
#                 v_fw / len(train_results),
#                 v_fm / len(train_results),
#             )
#         ],
#         columns=[
#             "epochs",
#             "t_loss",
#             "t_acc",
#             "t_fw",
#             "t_fm",
#             "v_loss",
#             "v_acc",
#             "v_fw",
#             "v_fm",
#         ],
#     )
#
#     wandb.log(
#         {
#             "train_loss": wandb.plot.line(
#                 table, "epochs", "t_loss", title="Train Loss"
#             ),
#             "train_acc": wandb.plot.line(
#                 table, "epochs", "t_acc", title="Train Accuracy"
#             ),
#             "train_fm": wandb.plot.line(
#                 table, "epochs", "t_fm", title="Train F1-macro"
#             ),
#             "train_fw": wandb.plot.line(
#                 table, "epochs", "t_fw", title="Train F1-weighted"
#             ),
#             "val_loss": wandb.plot.line(table, "epochs", "v_loss", title="Valid Loss"),
#             "val_acc": wandb.plot.line(
#                 table, "epochs", "v_acc", title="Valid Accuracy"
#             ),
#             "val_fm": wandb.plot.line(table, "epochs", "v_fm", title="Valid F1-macro"),
#             "val_fw": wandb.plot.line(
#                 table, "epochs", "v_fw", title="Valid F1-weigthed"
#             ),
#         }
#     )
#
#     if test_results is not None:
#         avg_test_loss = test_results["test_loss"].mean()
#         avg_test_acc = test_results["test_acc"].mean()
#         avg_test_fm = test_results["test_fm"].mean()
#         avg_test_fw = test_results["test_fw"].mean()
#
#         # Average and standard deviation for validation results
#         std_test_loss = test_results["test_loss"].std()
#         std_test_acc = test_results["test_acc"].std()
#         std_test_fm = test_results["test_fm"].std()
#         std_test_fw = test_results["test_fw"].std()
#
#         wandb.log(
#             {
#                 "test_loss": avg_test_loss,
#                 "test_acc": avg_test_acc,
#                 "test_fm": avg_test_fm,
#                 "test_fw": avg_test_fw,
#                 "_test_loss": f"{avg_test_loss:.4f} ± {std_test_loss:.4f}",
#                 "_test_acc": f"{avg_test_acc:.4f} ± {std_test_acc:.4f}",
#                 "_test_fm": f"{avg_test_fm:.4f} ± {std_test_fm:.4f}",
#                 "_test_fw": f"{avg_test_fw:.4f} ± {std_test_fw:.4f}",
#             }
#         )


def wandb_logging(train_results, test_results, config, epoch_name="epochs", suffix=""):
    # Determine max_epochs (longest run across all seeds)
    max_epochs = max(len(seed_result) for seed_result in train_results["t_loss"])

    # Initialize arrays with NaNs (so shorter runs don't interfere)
    t_loss, t_acc, t_fw, t_fm = (
        np.full((len(train_results["t_loss"]), max_epochs), np.nan),
        np.full((len(train_results["t_loss"]), max_epochs), np.nan),
        np.full((len(train_results["t_loss"]), max_epochs), np.nan),
        np.full((len(train_results["t_loss"]), max_epochs), np.nan),
    )
    v_loss, v_acc, v_fw, v_fm = (
        np.full((len(train_results["v_loss"]), max_epochs), np.nan),
        np.full((len(train_results["v_loss"]), max_epochs), np.nan),
        np.full((len(train_results["v_loss"]), max_epochs), np.nan),
        np.full((len(train_results["v_loss"]), max_epochs), np.nan),
    )

    # Store results while handling different lengths per seed
    for seed_idx, seed_length in enumerate(map(len, train_results["t_loss"])):
        t_loss[seed_idx, :seed_length] = train_results["t_loss"][seed_idx]
        t_acc[seed_idx, :seed_length] = train_results["t_acc"][seed_idx]
        t_fw[seed_idx, :seed_length] = train_results["t_fw"][seed_idx]
        t_fm[seed_idx, :seed_length] = train_results["t_fm"][seed_idx]

        v_loss[seed_idx, :seed_length] = train_results["v_loss"][seed_idx]
        v_acc[seed_idx, :seed_length] = train_results["v_acc"][seed_idx]
        v_fw[seed_idx, :seed_length] = train_results["v_fw"][seed_idx]
        v_fm[seed_idx, :seed_length] = train_results["v_fm"][seed_idx]

    # Compute mean across seeds, ignoring NaNs
    avg_t_loss = np.nanmean(t_loss, axis=0)
    avg_t_acc = np.nanmean(t_acc, axis=0)
    avg_t_fw = np.nanmean(t_fw, axis=0)
    avg_t_fm = np.nanmean(t_fm, axis=0)

    avg_v_loss = np.nanmean(v_loss, axis=0)
    avg_v_acc = np.nanmean(v_acc, axis=0)
    avg_v_fw = np.nanmean(v_fw, axis=0)
    avg_v_fm = np.nanmean(v_fm, axis=0)

    # Prepare WandB table
    table = wandb.Table(
        data=[
            [epoch, a, b, c, d, e, f, g, h]
            for epoch, (a, b, c, d, e, f, g, h) in enumerate(
                zip(
                    avg_t_loss,
                    avg_t_acc,
                    avg_t_fw,
                    avg_t_fm,
                    avg_v_loss,
                    avg_v_acc,
                    avg_v_fw,
                    avg_v_fm,
                )
            )
        ],
        columns=[
            "epochs",
            "t_loss",
            "t_acc",
            "t_fw",
            "t_fm",
            "v_loss",
            "v_acc",
            "v_fw",
            "v_fm",
        ],
    )

    # Log metrics
    wandb.log(
        {
            f"train_loss{suffix}": wandb.plot.line(
                table, "epochs", "t_loss", title="Train Loss"
            ),
            f"train_acc{suffix}": wandb.plot.line(
                table, "epochs", "t_acc", title="Train Accuracy"
            ),
            f"train_fm{suffix}": wandb.plot.line(
                table, "epochs", "t_fm", title="Train F1-macro"
            ),
            f"train_fw{suffix}": wandb.plot.line(
                table, "epochs", "t_fw", title="Train F1-weighted"
            ),
            f"val_loss{suffix}": wandb.plot.line(
                table, "epochs", "v_loss", title="Valid Loss"
            ),
            f"val_acc{suffix}": wandb.plot.line(
                table, "epochs", "v_acc", title="Valid Accuracy"
            ),
            f"val_fm{suffix}": wandb.plot.line(
                table, "epochs", "v_fm", title="Valid F1-macro"
            ),
            f"val_fw{suffix}": wandb.plot.line(
                table, "epochs", "v_fw", title="Valid F1-weighted"
            ),
        }
    )

    # Handle test results
    if test_results is not None:
        avg_test_loss = np.mean(test_results["test_loss"])
        avg_test_acc = np.mean(test_results["test_acc"])
        avg_test_fm = np.mean(test_results["test_fm"])
        avg_test_fw = np.mean(test_results["test_fw"])

        std_test_loss = np.std(test_results["test_loss"])
        std_test_acc = np.std(test_results["test_acc"])
        std_test_fm = np.std(test_results["test_fm"])
        std_test_fw = np.std(test_results["test_fw"])

        wandb.log(
            {
                f"test_loss{suffix}": avg_test_loss,
                f"test_acc{suffix}": avg_test_acc,
                f"test_fm{suffix}": avg_test_fm,
                f"test_fw{suffix}": avg_test_fw,
                f"_test_loss{suffix}": f"{avg_test_loss:.4f} ± {std_test_loss:.4f}",
                f"_test_acc{suffix}": f"{avg_test_acc:.4f} ± {std_test_acc:.4f}",
                f"_test_fm{suffix}": f"{avg_test_fm:.4f} ± {std_test_fm:.4f}",
                f"_test_fw{suffix}": f"{avg_test_fw:.4f} ± {std_test_fw:.4f}",
            }
        )


# Match the size of the data loaders
def cycle_loader(loader):
    # itertools.cycle() cannot be used here as it will not run collate_fn after the 1st iteration
    while True:
        for batch in loader:
            # Each iteration will apply collate_fn
            yield batch


def calc_epsilon(grads, lower_bound=1e-10, upper_bound=1e-2):
    grad_norm = torch.sqrt(sum(torch.norm(grad) ** 2 for grad in grads))
    epsilon = 0.01 / grad_norm
    epsilon = torch.clamp(epsilon, min=lower_bound, max=upper_bound)
    return epsilon


def set_model_params(model, params):
    """
    Set model parameters to the given values

    :param model:
    :param params:
    :return:
    """
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), params):
            param.copy_(new_param)


def save_model(model, epoch, config, best=False, filename=None):
    """
    Saves the model checkpoint.

    :param model: Model being trained.
    :param args: Arguments/configurations used for training.
    :param epoch: Current epoch number.
    :param best: Whether the model is the best model.
    :param filename: Custom filename for the checkpoint.
    """
    if not hasattr(model, "path_checkpoints"):
        raise ValueError(
            "Model must inherit from BaseModel to use save_model function."
        )

    save_path = model.path_checkpoints

    # Default filename if none provided
    if filename is None:
        if best:
            filename = "model_best.pth"
        else:
            filename = f"model_epoch_{epoch}.pth"

    model_filepath = os.path.join(save_path, filename)

    # Save the model state and additional metadata
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config if config else None,
        },
        model_filepath,
    )

    print(f"Model saved to {model_filepath}")
    return model_filepath


def train_one_epoch(model, loader, criterion, optimizer, verbose=True, print_freq=100):
    losses = AverageMeter("Loss")
    model.train()
    for batch_idx, (data, target, idx) in enumerate(loader):
        data = data.cuda()
        target = target.view(-1).cuda()
        # print(data.device, target.device)
        z, logits = model(data)
        loss = criterion(logits, target)
        losses.update(loss.item(), data.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            if batch_idx % print_freq == 0:
                print(f"[-] Batch {batch_idx + 1}/{len(loader)}\t Loss: {str(losses)}")


def eval_model(model, eval_data, criterion=None, batch_size=256, seed=1):
    """
    Evaluate trained model.

    :param model: A trained model which is to be evaluated.
    :param eval_data: A SensorDataset containing the data to be used for evaluating the model.
    :param criterion: Citerion object which was used during training of model.
    :param batch_size: Batch size to use during evaluation.
    :param seed: Random seed which is employed.

    :return: loss, accuracy, f1 weighted and macro for evaluation data; if return_results, also predictions
    """

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    print(paint("Running HAR evaluation loop ..."))

    loader_test = DataLoader(
        eval_data,
        batch_size,
        False,
        pin_memory=False,
        worker_init_fn=np.random.seed(int(seed)),
    )

    print("[-] Loading checkpoint ...")

    path_checkpoint = os.path.join(model.path_checkpoints, "best.pth")

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    start_time = time.time()
    loss_test, acc_test, fm_test, fw_test = eval_one_epoch(
        model, loader_test, criterion
    )

    print(
        paint(
            f"[-] Test loss: {loss_test:.2f}"
            f"\tacc: {100 * acc_test:.2f}(%)\tfm: {100 * fm_test:.2f}(%)\tfw: {100 * fw_test:.2f}(%)"
        )
    )

    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    print(paint(f"Finished HAR evaluation loop (h:m:s): {elapsed}"))

    return loss_test, acc_test, fm_test, fw_test, elapsed


def eval_one_epoch(model, loader, criterion):
    """
    Eval model for a one of epoch.

    :param model: A trained model which is to be evaluated.
    :param loader: A DataLoader object containing the data to be used for evaluating the model.
    :param criterion: The loss object.
    :return: loss, accuracy, f1 weighted and macro for evaluation data
    """

    losses = AverageMeter("Loss")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(loader):
            data = data.cuda()
            target = target.cuda()

            # print(data.device, target.device, model.device)

            z, logits = model(data)
            loss = criterion(logits, target.view(-1))
            losses.update(loss.item(), data.shape[0])
            probabilities = torch.nn.Softmax(dim=1)(logits)
            _, predictions = torch.max(probabilities, 1)

            y_pred.append(predictions.cpu().numpy().reshape(-1))
            y_true.append(target.cpu().numpy().reshape(-1))

    # append invalid samples at the beginning of the test sequence
    # if loader.dataset.prefix == "test":
    #     ws = data.shape[1] - 1
    #     samples_invalid = [y_true[0][0]] * ws
    #     y_true.append(samples_invalid)
    #     y_pred.append(samples_invalid)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    acc = metrics.accuracy_score(y_true, y_pred)
    fm = metrics.f1_score(y_true, y_pred, average="macro")
    fw = metrics.f1_score(y_true, y_pred, average="weighted")

    return losses.avg, acc, fm, fw


def train_model(
    model,
    train_data,
    val_data,
    test_data,
    batch_size_train,
    seed=1,
    class_weights=None,
    verbose=True,
    n_epochs=100,
    lr=1e-3,
    lr_step=10,
    lr_decay=0.9,
    weights_init="orthogonal",
    batch_size_test=256,
    custom_sampler=None,
    early_stopping_patience=100,
):
    if verbose:
        print(
            paint(
                f"================= Running HAR training loop with seed {seed} ================="
            )
        )
    if custom_sampler is not None:
        loader_train = DataLoader(
            train_data,
            batch_size=batch_size_train,
            worker_init_fn=np.random.seed(int(seed)),
            sampler=custom_sampler,
        )
    else:
        loader_train = DataLoader(
            train_data,
            batch_size=batch_size_train,
            shuffle=True,
            worker_init_fn=np.random.seed(int(seed)),
        )
    loader_val = DataLoader(
        val_data,
        batch_size=batch_size_test,
        shuffle=False,
        worker_init_fn=np.random.seed(int(seed)),
    )
    loader_test = DataLoader(
        test_data,
        batch_size=batch_size_test,
        shuffle=False,
        worker_init_fn=np.random.seed(int(seed)),
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_decay
    )

    if weights_init is not None:
        init_weights(model, weights_init)
    else:
        print(
            "[-] No weight initialization method provided. Skipping weight initialization..."
        )
    metric_best = 0.0
    early_stop_counter = 0  # Counter for early stopping

    # Store the training and validation metrics
    t_loss, t_acc, t_fm, t_fw = [], [], [], []
    v_loss, v_acc, v_fm, v_fw = [], [], [], []

    for epoch in range(n_epochs):
        if verbose:
            print(
                f"------------------------------ Epoch {epoch + 1}/{n_epochs} ------------------------------\n"
                f"Learning rate: {optimizer.param_groups[0]['lr']}"
            )
        train_one_epoch(model, loader_train, criterion, optimizer)

        loss, acc, fm, fw = eval_one_epoch(model, loader_train, criterion)
        loss_val, acc_val, fm_val, fw_val = eval_one_epoch(model, loader_val, criterion)

        # Check performance on the test dataset
        loss_test, acc_test, fm_test, fw_test = eval_one_epoch(
            model, loader_test, criterion
        )

        # Store the metrics
        t_loss.append(loss)
        t_acc.append(acc)
        t_fm.append(fm)
        t_fw.append(fw)
        v_loss.append(loss_val)
        v_acc.append(acc_val)
        v_fm.append(fm_val)
        v_fw.append(fw_val)

        if verbose:
            print(
                paint(
                    f"\tTrain loss: {loss:.2f} \tacc: {100 * acc:.2f}(%)\tfm: {100 * fm:.2f}(%)\tfw: {100 * fw:.2f}"
                    f"(%)\t"
                )
            )

            print(
                paint(
                    f"\tVal loss:   {loss_val:.2f} \tacc: {100 * acc_val:.2f}(%)\tfm: {100 * fm_val:.2f}(%)"
                    f"\tfw: {100 * fw_val:.2f}(%)"
                )
            )

            print(
                paint(
                    f"\tTest loss:  {loss_test:.2f} \tacc: {100 * acc_test:.2f}(%)\tfm: {100 * fm_test:.2f}(%)"
                    f"\tfw: {100 * fw_test:.2f}(%)"
                )
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }

        metric = fm_val
        if metric >= metric_best:  # Ignore first 5 epochs
            if verbose:
                print(
                    paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue")
                )
            # Don't update the best metric for the first 5 epochs as it's unstable
            metric_best = metric
            early_stop_counter = 0  # Reset early stopping counter
            torch.save(checkpoint, os.path.join(model.path_checkpoints, "best.pth"))
        else:
            early_stop_counter += 1

        # Check early stopping condition
        if early_stop_counter >= early_stopping_patience:
            if verbose:
                print(
                    paint(
                        f"[!] Early stopping triggered. No improvement for {early_stopping_patience} consecutive epochs.",
                        "red",
                    )
                )
            break  # Stop training

        if lr_step > 0:
            scheduler.step()

    return t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, criterion
