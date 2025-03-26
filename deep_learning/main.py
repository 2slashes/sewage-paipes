import torch
from torch.utils.data import DataLoader
import os
from pipes_nn_classes import PipesDataset, PipesPredictor, PipesLoss


# split the data into training and testing data
train_data = PipesDataset("data/train.csv")
test_data = PipesDataset("data/test.csv")

# prepare the dataset for training with DataLoaders
batch_size = 64
train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

# get the header from the training data
train_features, train_labels = next(iter(train_dataloader))
test_features, test_labels = next(iter(test_dataloader))

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

n = 4

model = PipesPredictor(n**2 * 4, 64, n**2).to(device)

learning_rate = 1e-3
epochs = 5
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = PipesLoss()


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
            for i in range(pred.shape[0]):  # Iterate through each sample in the batch
                if y[i][
                    pred.argmax(1)[i].item()
                ]:  # Check if predicted action is in the list of valid actions
                    correct += 1

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

model_file_name = "model.pth"
curr_dir = os.path.dirname(__file__)
model_file_path = os.path.join(curr_dir, model_file_name)
torch.save(model.state_dict(), model_file_path)
