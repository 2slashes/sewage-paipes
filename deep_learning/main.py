import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd
import os


# Convert the dataset into torch tensor format
class PipesDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        curr_dir = os.path.dirname(__file__)
        data_path = os.path.join(curr_dir, path)
        self.df = pd.read_csv(data_path)  # dataframe

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

        action_tensor = torch.tensor(action)

        return (state_tensor, action_tensor)


# split the data into training and testing data
train_data = PipesDataset("data/train.csv")
test_data = PipesDataset("data/test.csv")

# prepare the dataset for training with DataLoaders
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# get the header from the training data
train_features, train_labels = next(iter(train_dataloader))
test_features, test_labels = next(iter(test_dataloader))

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


class PipesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


n = 4
model = PipesPredictor(n**2 * 4, 64, n**2).to(device)
# state = torch.rand((1, n**2 * 4), device=device)
# output = model(state)
# print(output)
# move = torch.argmax(output).item()
# print(f"Suggested move: Pipe {move}")

learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")