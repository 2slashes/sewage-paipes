import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class StateActionDataset(Dataset):
    def __init__(self, csv_file):
        # Read CSV
        self.df = pd.read_csv(csv_file)

        # Convert the string of bits (e.g. '0110...') into a list of ints
        self.df["state"] = self.df["state"].apply(lambda s: [int(ch) for ch in s])

        # Convert to torch tensors
        self.X = torch.tensor(self.df["state"].tolist(), dtype=torch.float32)
        self.y = torch.tensor(self.df["action"].values, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_size=64, hidden_size=32, num_classes=16):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),  # 16 possible classes (0-15)
        )

    def forward(self, x):
        return self.network(x)


def train(model, dataloader, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for states, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def main():
    # Point to your CSV file
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "data/demo.csv")

    # Create dataset & dataloader
    dataset = StateActionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Check input_size automatically from one of the samples
    input_size = len(dataset[0][0])
    print(f"Detected input_size = {input_size}")

    # Create MLP model
    model = MLP(input_size=input_size, hidden_size=32, num_classes=16)

    # Train
    train(model, dataloader, epochs=10, lr=1e-3)


if __name__ == "__main__":
    main()
