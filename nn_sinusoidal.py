import random
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np


# Create a neural network to predict the output of the sinusoidal function

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze()


class SinusoidalDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.x = df[['x']].values
        self.y = df[['y']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]

        batch_x = []
        batch_y = []

        for i in idx:
            batch_x.append(self.x[i])
            batch_y.append(self.y[i])
        # For this dataset we only want a 1D tensor for the output and input

        # Convert batch_x and batch_y to numpy arrays first
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        return torch.tensor(batch_x, dtype=torch.float32).squeeze(), torch.tensor(batch_y,
                                                                                  dtype=torch.float32).squeeze()


# Load the datasets from the saved csv files
data_folder = 'sinusoidal_functions/'
train_files = ['train_sin.csv', 'train_cos.csv', 'train_sin+cos.csv', 'train_xsinx.csv', 'train_xcosx.csv',
               'train_xsinx+xcosx.csv']
test_files = ['test_sin.csv', 'test_cos.csv', 'test_sin+cos.csv', 'test_xsinx.csv', 'test_xcosx.csv',
              'test_xsinx+xcosx.csv']

model_names = ['sin', 'cos', 'sin+cos', 'xsinx', 'xcosx', 'xsinx+xcosx']

num_test_examples = 1000
num_train_examples = 10000

# Create validation set from training set
num_val_examples = 1000

indices = list(range(num_train_examples))
random.seed(6_345_789)
random.shuffle(indices)

train_indices = indices[:-num_val_examples]
val_indices = indices[-num_val_examples:]

# save train and val indices to csv
df_train_indices = pd.DataFrame(train_indices, columns=['indices'])
df_val_indices = pd.DataFrame(val_indices, columns=['indices'])
df_train_indices.to_csv(data_folder + 'train_indices.csv', index=False)
df_val_indices.to_csv(data_folder + 'val_indices.csv', index=False)

# Now create training and test loop for each function
def train_loop(dataloader_train, model, loss_fn, optimizer, dataloader_val):
    train_loss = 0

    for batch, (X_val, y_val) in enumerate(dataloader_train):
        val_pred = model(X_val)
        loss = loss_fn(val_pred, y_val)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(dataloader_train.dataset)

    with torch.no_grad():
        val_loss = 0
        for batch, (X_val, y_val) in enumerate(dataloader_val):
            val_pred = model(X_val)
            loss = loss_fn(val_pred, y_val)
            val_loss += loss.item()
        val_loss /= len(dataloader_val.dataset)

    return train_loss, val_loss, model


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.round() == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def plot_predictions(dataloader, model, name):
    x_tensor = torch.tensor(dataloader.dataset.x, dtype=torch.float32).squeeze()
    y_tensor = torch.tensor(dataloader.dataset.y, dtype=torch.float32).squeeze()
    with torch.no_grad():
        # Plot over all the data
        preds = model(x_tensor)
        # print('Pred.shape: ', preds.shape)
        # print('y_tensor.shape: ', y_tensor.shape)
        # print('preds: ', preds[:10])
        # print('y_tensor: ', y_tensor[:10])
        # plot a scatter of predictions vs actual
        # plt.scatter(y_tensor, preds)
        # plt.plot(y_tensor, y_tensor, color='red', label='Actual')

        # Plot actual and predicted against x
        plt.scatter(x_tensor, y_tensor, color='red', label='Actual')
        plt.scatter(x_tensor, preds, color='blue', label='Predicted')
        plt.title(f'Plot for {name} function')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.show()


def plot_losses(train_loss, val_loss, name):
    # plot over the epochs
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, label='Train loss')
    plt.plot(range(epochs), val_loss, label='Val loss')
    plt.title(f'Losses for {name} function')
    plt.legend()
    plt.show()


def main():
    start_time = time.time()
    for train_file, test_file, model_name in zip(train_files, test_files, model_names):
        # Load the datasets from the saved csv files
        df = pd.read_csv(data_folder + train_file)
        df_test = pd.read_csv(data_folder + test_file)

        # get train and val datasets
        df_train = df.iloc[train_indices]
        df_val = df.iloc[val_indices]

        # Create the datasets
        train_dataset = SinusoidalDataset(df_train)
        val_dataset = SinusoidalDataset(df_val)
        test_dataset = SinusoidalDataset(df_test)

        # Create the dataloaders
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

        # Create the model
        model = Net()

        # Create the loss function and optimiser
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # Train the model
        epochs = 100
        loss_history = []

        val_losses = []
        train_losses = []
        best_val_loss = float('inf')
        best_state_dict = None
        for t in range(epochs):
            # print(f"Epoch {t + 1}\n-------------------------------")

            train_loss, val_loss, model = train_loop(train_dataloader, model, loss_fn, optimizer, val_dataloader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model.state_dict()

            if t % 10 == 0:
                print(f"Epoch {t}\n-------------------------------")
                print(f"{model_name} train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

            # Check for early stopping
            if t > 50:
                if round(val_losses[-1], 4) >= round(val_losses[-2], 4) >= round(val_losses[-3], 4) >= \
                        round(val_losses[-4], 4) >= round(val_losses[-5], 4):
                    print(f"Stopping early at epoch {t}")
                    break

        model.load_state_dict(best_state_dict)
        # Save best model
        torch.save(model, f"{data_folder + model_name}_model.pt")
        print("Done!")

        # Plot the losses
        plot_losses(train_losses, val_losses, model_name)

        # Evaluate the model
        plot_predictions(test_dataloader, model, model_name)
        # Print time in hours, minutes and seconds
        print(
            f"Time taken: {(time.time() - start_time) / 3600:.0f} hours"
            f", {((time.time() - start_time) % 3600) / 60:.0f} minutes, {((time.time() - start_time) % 3600) % 60:.0f} "
            f"seconds\n\n")


if __name__ == '__main__':
    main()
