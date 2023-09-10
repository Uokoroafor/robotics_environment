import random
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import Tuple, List, Optional, Union
import sys
from utils.logging_utils import TrainingLogger
from utils.plot_utils import plot_losses, plot_predictions
from utils.file_utils import create_training_folder, save_losses


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        loss_fn: torch.nn.modules.loss._Loss,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """Constructor class for Trainer used to train a transformer model for language modelling and text generation
        Args:
            model (nn.Module): Model to train
            optimiser (torch.optim.Optimizer): Optimiser to use for training
            loss_fn (torch.nn.modules.loss._Loss): Loss function to use for training
        """
        self.train_data = None
        self.val_data = None
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.best_model_dict = None

        # Preallocate variables defined in set_training_hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a folder to save the model and training losses
        self.path = create_training_folder()

        # Move the model to the device
        self.model.to(self.device)

        # Save the model architecture as a txt file
        with open(f"{self.path}/model.txt", "w") as f:
            f.write(str(self.model))

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int,
        eval_every: int = 1,
        save_model: bool = True,
        save_model_path: Optional[str] = None,
        plotting: bool = True,
        verbose: bool = True,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
    ):
        """Train the model
        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
            epochs (int): Number of epochs to train for
            eval_every (int, optional): Evaluate the model every eval_every epochs. Defaults to 1.
            save_model (bool, optional): Whether to save the model(s) and save the best model. Defaults to True.
            save_model_path (Optional[str], optional): Path to save the model. Defaults to None.
            plotting (bool, optional): Whether to plot the losses. Defaults to True.
            verbose (Optional[bool], optional): Whether to print the progress of training. Defaults to True.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            early_stopping_patience (int, optional): Number of iterations to wait before stopping early. Defaults to 10.

        """

        train_losses = []
        val_losses = []
        lowest_val_loss = float("inf")
        logger = TrainingLogger(
            self.path + "/training_logs/training_log.txt",
            name="training_log",
            verbose=verbose,
        )

        logger.log_info(f"Training {type(self.model).__name__} for {epochs} iterations")
        count = 0

        try:
            for i in range(epochs):
                # Running for one extra epoch to get the final validation loss
                if i % eval_every == 0:
                    train_loss, val_loss = self.train_loop(
                        train_dataloader, val_dataloader
                    )

                    logger.log_info(
                        f"At Iteration: {max(1, i)}/{epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"
                    )

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                    # Update the best model state dict and lowest validation loss
                    lowest_val_loss, count = self.update_best_model_dict(
                        val_loss, lowest_val_loss, count
                    )

                    if early_stopping and count >= early_stopping_patience:
                        logger.log_info(f"Stopping early after {i} iterations")
                        break

                else:
                    train_loss, _ = self.train_loop(train_dataloader)
                    train_losses.append(train_loss)

            if save_model:
                # Load and save the best model
                self.model.load_state_dict(self.best_model_dict)
                save_model_path = self.save_best_model(save_model_path)
                logger.log_info(f"Saved best model at: {save_model_path}")

                # Save the losses
                save_losses(train_losses, val_losses, self.path)
                logger.log_info(
                    f"Saved losses at: {self.path}/training_logs/losses.csv"
                )

            else:
                # If we are not saving the model, load the best model
                self.model.load_state_dict(self.best_model_dict)

            if plotting:
                plot_save_path = (
                    f"{self.path}/training_logs/{type(self.model).__name__}_losses.png"
                    if save_model
                    else None
                )

                plot_losses(
                    train_losses,
                    val_losses,
                    title=type(self.model).__name__,
                    save_path=plot_save_path,
                )
        except Exception as e:
            logger.log_error(f"Error while training: {str(e)}")
            raise e

        except KeyboardInterrupt:
            logger.log_info("Training interrupted by the user")

        return self.model, train_losses, val_losses

    def save_model(self, model_path: str):
        """Save the model
        Args:
            model_path (str): Path to save the model
        """
        torch.save(self.model, model_path)

    def save_best_model(self, best_model_path: Optional[str]):
        """Save the best model
        Args:
            best_model_path (Optional[str]): Path to save the best model
        """
        if best_model_path is None:
            best_model_path = (
                f"{self.path}/saved_models/{type(self.model).__name__}_best.pt"
            )
        self.save_model(best_model_path)
        return best_model_path

    def update_best_model_dict(
        self, loss_val: float, lowest_val_loss: float, count: int
    ) -> Tuple[float, int]:
        """Update the best model dictionary if the validation loss is the lowest so far
        Args:
            loss_val (float): Dictionary containing the training and validation losses
            lowest_val_loss (float): Lowest validation loss so far
            count (int): Number of iterations since the lowest validation loss was updated
        """
        if loss_val < lowest_val_loss:
            # Update the lowest validation loss
            lowest_val_loss = loss_val
            # Save the model state dict
            self.best_model_dict = self.model.state_dict()
            # Reset the count
            count = 0
        else:
            # Increment the count
            count += 1
        return lowest_val_loss, count

    def train_loop(
        self,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tuple[float, Optional[float]]:
        """Train the model for one epoch and return the train and validation loss.

        Args:
            dataloader_train: The training data
            dataloader_val: The validation data it is optional because we might not want to validate after every epoch

        Returns:
            Tuple[float, Optional[float]]: The train and validation loss if validate is True, otherwise just the train loss

        """
        train_loss = 0

        for batch, (X_train, y_train) in enumerate(dataloader_train):
            train_pred = self.model(X_train).squeeze(-1)
            loss = self.loss_fn(train_pred, y_train)

            # Backpropagation
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss += loss.item()
        train_loss /= len(dataloader_train.dataset)
        if dataloader_val is not None:
            with torch.no_grad():
                val_loss = 0
                for batch, (X_val, y_val) in enumerate(dataloader_val):
                    val_pred = self.model(X_val).squeeze(-1)
                    loss = self.loss_fn(val_pred, y_val)
                    val_loss += loss.item()
                val_loss /= len(dataloader_val.dataset)
        else:
            val_loss = None

        return train_loss, val_loss

    def evaluate(
        self, testdata: Union[torch.utils.data.DataLoader, torch.Tensor]
    ) -> float:
        """Test the model and return the test loss and accuracy.

        Args:
            testdata: The data to test on
            loss_fn: The loss function to use

        Returns:
            The average test loss and accuracy
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            if isinstance(testdata, torch.Tensor):
                test_pred = self.model(testdata).squeeze(-1)
                loss = self.loss_fn(test_pred, testdata)
                test_loss += loss.item()
            else:
                for batch, (X_test, y_test) in enumerate(testdata):
                    test_pred = self.model(X_test).squeeze(-1)
                    loss = self.loss_fn(test_pred, y_test)
                    test_loss += loss.item()
        test_loss /= len(testdata.dataset)
        return test_loss


def set_seed(seed: Optional[int] = 0):
    """Set the random seed for reproducibility
    Args:
        seed (Optional[int], optional): Random seed. Defaults to 0.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if "numpy" in sys.modules:
        sys.modules["numpy"].random.seed(seed)

    if "random" in sys.modules:
        sys.modules["random"].seed(seed)


# def train_model(model: nn.Module, dataloader_train: torch.utils.data.DataLoader, dataloader_val: torch.utils.data.DataLoader,
#                 loss_fn: nn.Module, optimiser: torch.optim.Optimizer, epochs: int = 10) -> Tuple[List[float], List[float]]:
#     """Train the model for the specified number of epochs and return the train and validation loss.
#
#     Args:
#         model: The model to train
#         dataloader_train: The training data
#         dataloader_val: The validation data
#         loss_fn: The loss function to use
#         optimiser: The optimiser to use
#         epochs: The number of epochs to train for (default 10)
#
#     Returns:
#         The average training loss and validation loss for each epoch
#
#     """
#
#     val_losses = []
#     train_losses = []
#     best_val_loss = float('inf')
#     best_state_dict = None
#     for t in range(epochs):
#         # print(f"Epoch {t + 1}\n-------------------------------")
#
#         train_loss, val_loss = train_loop(model, dataloader_train, dataloader_val, loss_fn, optimiser)
#
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_state_dict = model.state_dict()
#
#         if t % 10 == 0:
#             print(f"Epoch {t}\n-------------------------------")
#             print(f"Train_loss: {train_loss:.4f}, Val_loss: {val_loss:.4f}")
#
#         # Check for early stopping
#         if t > 50:
#             if round(val_losses[-1], 4) >= round(val_losses[-2], 4) >= round(val_losses[-3], 4) >= \
#                     round(val_losses[-4], 4) >= round(val_losses[-5], 4):
#                 print(f"Stopping early at epoch {t}")
#                 break
#
#         model.load_state_dict(best_state_dict)
#         # Save best model
#         torch.save(model, f"{data_folder + model_name}_model.pt")
#         print("Done!")
#
#         # Plot the losses
#         plot_losses(train_losses, val_losses, model_name)
#
#         # Evaluate the model
#         plot_predictions(test_dataloader, model, model_name)
#         # Print time in hours, minutes and seconds
#         print(
#             f"Time taken: {(time.time() - start_time) / 3600:.0f} hours"
#             f", {((time.time() - start_time) % 3600) / 60:.0f} minutes, {((time.time() - start_time) % 3600) % 60:.0f} "
#             f"seconds\n\n")
