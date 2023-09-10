import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_predictions(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    title: Optional[str] = None,
    plot_all: bool = False,
    save_path: Optional[str] = None,
):
    """Plot the predictions and targets

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader for the data
        model (torch.nn.Module): Trained model
        title (str, optional): Title of the plot. Defaults to None.
        plot_all (bool, optional): Plots the line for the predictions and targets against the input. Defaults to False.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    x_tensor = torch.tensor(dataloader.dataset.x, dtype=torch.float32).squeeze()
    y_tensor = torch.tensor(dataloader.dataset.y, dtype=torch.float32).squeeze()
    if title is None:
        title1 = f"Predictions and targets against input"
        title2 = f"Predicted vs Actual"
    else:
        title1 = f"Predictions and targets against input for {title}"
        title2 = f"Predicted vs Actual for {title}"

    with torch.no_grad():
        # Plot over all the data
        preds = model(x_tensor)

        # Plot actual vs predicted
        plt.scatter(y_tensor, preds, color="blue")
        # Add a diagonal line
        plt.plot(y_tensor, y_tensor, color="red")
        plt.title(title2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        if save_path is not None:
            plt.savefig(f"{save_path}_preds_vs_actual.png")

        plt.show()
        plt.close()

        if plot_all:
            # Plot actual and predicted against x
            plt.scatter(x_tensor, y_tensor, color="red", label="Actual")
            plt.scatter(x_tensor, preds, color="blue", label="Predicted")
            plt.title(title1)
            plt.xlabel("input")
            plt.ylabel("output")
            plt.legend()
            if save_path is not None:
                plt.savefig(f"{save_path}_preds_and_targets.png")
            plt.show()
            plt.close()

        print(f"Test MSE: {nn.MSELoss()(preds, y_tensor): .4f}")


def plot_losses(
    train_loss: List[float],
    val_loss: List[float],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot the losses over the epochs

    Args:
        train_loss (List[float]): Training loss over the epochs
        val_loss (List[float]): Validation loss over the epochs
        title (str, optional): Title of the plot. Defaults to None.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    if title is None:
        title = f"Train and Validation losses"
    else:
        title = f"Train and Validation losses for {title}"
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, label="Train loss")
    plt.plot(range(epochs), val_loss, label="Val loss")
    plt.title(title)
    plt.legend()
    if save_path is not None:
        plt.savefig(f"{save_path}_losses.png")
    plt.show()
    plt.close()
