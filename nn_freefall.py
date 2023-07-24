import random
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt


# Create a neural network to predict the vertical velocity and position
# of a falling object at time t given its initial x, y position and x, y velocity
# Input: x_0, y_0, dx_0, dy_0, t_1
class Netdiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


class Netsame(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class FreeFallDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.x = df[['x_0', 'y_0', 'dx_0', 'dy_0', 't_1']].values
        self.y = df[['y_1']].values

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


net_diff = Netdiff()
net_same = Netsame()

loss_fn_same = nn.MSELoss()
loss_fn_diff = nn.MSELoss()

optim_same = torch.optim.Adam(net_same.parameters(), lr=0.001)
optim_diff = torch.optim.Adam(net_diff.parameters(), lr=0.001)

df_same = pd.read_csv('freefall/examples_same_steps.csv')
indices = list(range(len(df_same)))
random.shuffle(indices)
train_indices = indices[:int(0.8 * len(indices))]
val_indices = indices[int(0.8 * len(indices)):int(0.9 * len(indices))]
test_indices = indices[int(0.9 * len(indices)):]

train_dataset_same = FreeFallDataset(df_same.iloc[train_indices])
val_dataset_same = FreeFallDataset(df_same.iloc[val_indices])
test_dataset_same = FreeFallDataset(df_same.iloc[test_indices])

batch_size = 32
data_loader_train_same = torch.utils.data.DataLoader(train_dataset_same, batch_size=batch_size, shuffle=True)
data_loader_val_same = torch.utils.data.DataLoader(val_dataset_same, batch_size=batch_size, shuffle=True)
data_loader_test_same = torch.utils.data.DataLoader(test_dataset_same, batch_size=batch_size, shuffle=True)

df_diff = pd.read_csv('freefall/examples_diff_steps.csv')

train_dataset_diff = FreeFallDataset(df_diff.iloc[train_indices])
val_dataset_diff = FreeFallDataset(df_diff.iloc[val_indices])
test_dataset_diff = FreeFallDataset(df_diff.iloc[test_indices])

data_loader_train_diff = torch.utils.data.DataLoader(train_dataset_diff, batch_size=batch_size, shuffle=True)
data_loader_val_diff = torch.utils.data.DataLoader(val_dataset_diff, batch_size=batch_size, shuffle=True)
data_loader_test_diff = torch.utils.data.DataLoader(test_dataset_diff, batch_size=batch_size, shuffle=True)


def evaluate(net, dataloader, loss_fn):
    with torch.no_grad():
        loss = 0
        for x, y in dataloader:
            y_pred = net(x.float())
            loss += loss_fn(y_pred, y.float())
        return loss / len(dataloader.dataset)


def train(net, optim, loss_fn, train_dataloader, val_dataloader, epochs=100):
    best_val_loss = float('inf')
    best_state_dict = None
    count = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = 0
        for x, y in train_dataloader:
            optim.zero_grad()
            y_pred = net(x.float())
            loss = loss_fn(y_pred, y.float())
            loss.backward()
            optim.step()
            train_loss += loss.item()

        val_loss = evaluate(net, val_dataloader, loss_fn)
        train_losses.append(train_loss / len(train_dataloader))
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = net.state_dict()
            count = 0
        else:
            count += 1
        if count > 5:
            print(f'Early stopping at epoch {epoch}')
            break

        if epoch % 10 == 0:
            print(f'Epoch {epoch} training loss: {train_loss / len(train_dataloader):.4f}')
            print(f'Validation loss: {val_loss:.4f}')

    net.load_state_dict(best_state_dict)

    # print(f'Test loss: {evaluate(net, data_loader_test, loss_fn):.4f}')

    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title(f'Losses for {net.__class__.__name__} NN model')
    plt.legend()
    plt.show()
    return net


print('Creating same network')
same_args = {'net': net_same, 'optim': optim_same, 'loss_fn': loss_fn_same, 'train_dataloader': data_loader_train_same,
             'val_dataloader': data_loader_val_same, 'epochs': 100}

print('Creating different network')
diff_args = {'net': net_diff, 'optim': optim_diff, 'loss_fn': loss_fn_diff, 'train_dataloader': data_loader_train_diff,
             'val_dataloader': data_loader_val_diff, 'epochs': 100}

print('\nTraining same network')
net_same = train(**same_args)

print('\nTraining different network')
net_diff = train(**diff_args)

torch.save(net_same, 'freefall/freefall_same.pt')
torch.save(net_diff, 'freefall/freefall_diff.pt')

# Now load examples_test.csv and run the networks on it
df_test = pd.read_csv('freefall/examples_tests.csv')
test_dataset = FreeFallDataset(df_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('\nEvaluating same network')
print(f'Test loss: {evaluate(net_same, test_dataloader, loss_fn_same):.4f}')

print('\nEvaluating different network')
print(f'Test loss: {evaluate(net_diff, test_dataloader, loss_fn_diff):.4f}')

# Now run the networks on the test data and plot predicted vs actual
with torch.no_grad():
    for x, y in test_dataloader:
        y_pred_same = net_same(x.float())
        y_pred_diff = net_diff(x.float())
        break

plt.figure()
plt.scatter(y, y_pred_same, label='same')
plt.scatter(y, y_pred_diff, label='diff')
plt.plot([0, 1], [0, 1], color='black')
# plot a diagonal line to show perfect correlation
plt.plot(y[:, 0, 0], y[:, 0, 0], color='black')
plt.title('Predicted vs actual for test data set using same and different networks')

plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.legend()
plt.show()
