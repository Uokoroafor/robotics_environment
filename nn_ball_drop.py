import random
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

# create a neural network to classify if ball_1 or ball_2 drops first or if they drop at the same time
# input: x1, y1, x2, y2, dx1, dy1, dx2, dy2

# Seed all random number generators for reproducibility
seed = 6_345_789
random.seed(seed)
torch.manual_seed(seed)


class NetBall(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


class BallDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.x = df[["x1", "y1", "x2", "y2", "dx1", "dy1", "dx2", "dy2"]].values
        self.y = df[["ans"]].values
        # convert y to one-hot encoding 'same' = [1, 0, 0], 'ball_1' = [0, 1, 0], 'ball_2' = [0, 0, 1]
        self.y = [
            [1, 0, 0] if y == "same" else [0, 1, 0] if y == "ball_1" else [0, 0, 1]
            for y in self.y
        ]

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

        return torch.tensor(batch_x), torch.tensor(batch_y)


net_ball = NetBall()

loss_fn = nn.CrossEntropyLoss()

optim_same = torch.optim.Adam(net_ball.parameters(), lr=0.0001)

df = pd.read_csv("data/ball_drop/ball_drop.csv")

indices = list(range(len(df)))
random.shuffle(indices)
train_indices = indices[: int(0.8 * len(indices))]
val_indices = indices[int(0.8 * len(indices)) : int(0.9 * len(indices))]
test_indices = indices[int(0.9 * len(indices)) :]

# save train, val, test indices to csvs
pd.DataFrame(train_indices).to_csv("data/indices/10k/train_indices.csv", index=False)
pd.DataFrame(val_indices).to_csv("data/indices/10k/val_indices.csv", index=False)
pd.DataFrame(test_indices).to_csv("data/indices/10k/test_indices.csv", index=False)

train_dataset = BallDataset(df.iloc[train_indices])
val_dataset = BallDataset(df.iloc[val_indices])
test_dataset = BallDataset(df.iloc[test_indices])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)


def train(net, optim, loss_fn, train_loader, val_loader, epochs=10):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_dict = None
    counter = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optim.zero_grad()
            batch_y_hat = net(batch_x.float()).float()

            # Apply softmax to get probabilities
            # remove the superfluous dimensions for both dim(1)
            batch_y_hat = batch_y_hat.squeeze(1)
            batch_y = batch_y.squeeze(1)

            batch_y = batch_y.squeeze()
            loss = loss_fn(batch_y_hat, batch_y.float())  # .squeeze())
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        # Append average loss for this epoch
        train_losses.append(epoch_loss / len(train_loader))

        net.eval()
        epoch_val_loss = 0
        for batch_x, batch_y in val_loader:
            batch_y_hat = net(batch_x.float())
            # squeeze the superfluous dimensions for both dim(1)
            batch_y_hat = batch_y_hat.squeeze(1)
            batch_y = batch_y.squeeze(1)
            loss = loss_fn(batch_y_hat, batch_y.float())  # .squeeze()
            epoch_val_loss += loss.item()
        # Append average loss for this epoch
        val_losses.append(epoch_val_loss / len(val_loader))

        # print losses to 4 decimal places
        print(
            f"Epoch {epoch} : train loss = {train_losses[-1]:.4f}, val loss = {val_losses[-1]:.4f}"
        )
        # Save model if val loss is lower than the previous lowest val loss
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_dict = net.state_dict()

        # Early stopping
        if epoch > 5 and val_losses[-1] > val_losses[-2]:
            counter += 1
            # stop if the last 5 val losses are all higher than the previous
            # or the last 5 are within 0.001 of each other

            if counter > 5 or min(val_losses[-5:]) > best_val_loss:
                print("Early stopping at epoch :", epoch)
                break

        else:
            counter = 0

    # Save best model
    net.load_state_dict(best_model_dict)
    # save the best model at the end of training
    torch.save(net, "data/ball_drop/best_model_ball.pt")

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.show()


train(net_ball, optim_same, loss_fn, train_loader, val_loader, epochs=1000)


def test(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(3, 3)
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_y_hat = net(batch_x.float()).squeeze(1)
            batch_y = batch_y.squeeze(1)
            correct += (
                torch.eq(torch.argmax(batch_y_hat, dim=1), torch.argmax(batch_y, dim=1))
                .sum()
                .item()
            )
            total += len(batch_y)
            # update confusion matrix
            for t, p in zip(
                torch.argmax(batch_y, dim=1), torch.argmax(batch_y_hat, dim=1)
            ):
                confusion_matrix[t.long(), p.long()] += 1
            # print(confusion_matrix)

            # calculate accuracy per class
            # for i in range(3):

    print(f"Test accuracy: {correct / total}")
    print(confusion_matrix / confusion_matrix.sum(1).view(-1, 1))


test(net_ball, test_loader)
