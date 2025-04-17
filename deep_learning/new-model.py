import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import csv


class PipesDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        curr_dir = os.getcwd()
        data_path = os.path.join(curr_dir, path)
        self.df = pd.read_csv(data_path, dtype=str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        all = self.df.iloc[idx]

        state = all.iloc[0]
        action = all.iloc[1]

        # Create a list, where each entry in the list is an int
        # the list as a whole represents the state of the board
        state_int_list = [int(x) for x in state]
        state_tensor = torch.tensor(state_int_list)

        action_int_list = [int(x) for x in action]
        action_tensor = torch.tensor(action_int_list)

        return (state_tensor, action_tensor)


train_pipes = DataLoader(PipesDataset("data/train.csv"), batch_size=64, shuffle=True)
test_pipes = DataLoader(PipesDataset("data/test.csv"), batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_pipes))
test_features, test_labels = next(iter(test_pipes))


class PipesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

n = 4

learning_rate = 1e-3
batch_size = 64

# Initialize the loss function
loss_fn = nn.BCEWithLogitsLoss()

model = PipesPredictor(input_size=4 * (n**2), hidden_size=128, output_size=n**2).to(
    device
)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    total_correct = 0  # total correct label predictions
    total_labels = 0  # total number of labels across all samples

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).float()

            logits = model(X)
            loss = loss_fn(logits, y)
            test_loss += loss.item()

            # Convert logits to probabilities and then to a 0/1 mask.
            probs = torch.sigmoid(logits)
            predicted_mask = (probs >= 0.5).float()

            # Count all correct label predictions (element-wise comparison)
            total_correct += (predicted_mask == y).float().sum().item()
            total_labels += y.numel()  # count of all individual labels

    avg_loss = test_loss / num_batches
    # Calculate average per-label accuracy as a percentage.
    label_accuracy = (total_correct / total_labels) * 100
    print(
        f"Test Avg loss: {avg_loss:.4f}, Average Label Accuracy: {label_accuracy:.2f}%"
    )


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_pipes, model, loss_fn, optimizer)
    test_loop(test_pipes, model, loss_fn)
print("Done!")
model_file_name = "model.pth"
curr_dir = os.getcwd()
model_file_path = os.path.join(curr_dir, model_file_name)
torch.save(model.state_dict(), model_file_path)
