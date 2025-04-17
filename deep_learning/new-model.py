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


def pick_move(state, visited):
    # convert the state to integers
    state_int_list = [int(x) for x in state]
    # convert the state to a tensor
    state_tensor = torch.tensor(state_int_list).to(device).float()
    # get the predicted move from the neural network
    prob = model(state_tensor)
    results = torch.topk(prob, 16).indices

    # Initialize the visited state if we haven't seen it before
    if state not in visited:
        visited[state] = set()

    # Try each of the top moves
    result = None
    for r in results:
        move = int(r)
        # Skip if we've already tried this move for this state
        if not move in visited[state]:
            result = move
            break

    if result is None:
        raise Exception("No valid moves available")
    return result


def pipe_rotate_binary(board: str, pipe: int):
    """
    Takes a binary representation of a board of pipes as a string, and a pipe to rotate. Outputs a binary representation of the board after rotating the pipe.

    :params pipe: The pipe to rotate
    :params board: Binary representation of the board as a string

    """
    # each pipe has 4 values associated to it, so pipe n starts at index 4 * n
    start_index = 4 * pipe
    up = board[start_index]
    right = board[start_index + 1]
    down = board[start_index + 2]
    left = board[start_index + 3]

    # rotate clockwise
    new_board = (
        board[:start_index] + left + up + right + down + board[start_index + 4 :]
    )

    return new_board


def get_bad_pipes(state, goal):
    bad = []
    # Compare each 4-bit chunk (representing a pipe)
    for i in range(0, len(state), 4):
        if state[i : i + 4] != goal[i : i + 4]:
            bad.append(i // 4)
    return bad


model.eval()

# Load puzzles from `puzzles.csv`
initials: list[str] = []
goals: list[str] = []
prev_moves: list[int] = []
puzzle_path = os.path.join(curr_dir, "data/puzzles.csv")
with open(puzzle_path, newline="") as csvfile:
    reader = csv.reader(csvfile)
    # skip the header
    next(reader)
    for row in reader:
        initials.append(row[0])
        goals.append(row[1])
        prev_moves.append(int(row[2]))

outlier_threshold = 70

corrects = 0
bad_pipes_when_incorrect = []
all_moves = []
extra_moves = []
bad_pipes_on_moves = []
normal_initials = []
normal_goals = []

outlier_corrects = 0
outlier_bad_pipes_when_incorrect = []
outlier_all_moves = []
outlier_extra_moves = []
outlier_bad_pipes_on_moves = []

outlier_initials = []
outlier_goals = []

for i in range(len(initials)):
    initial = initials[i]
    goal = goals[i]
    visited = {}

    state = initial
    visited[initial] = set()
    cur_puzzle_moves = 0
    cur_puzzle_bad_pipes_when_incorrect = []
    cur_puzzle_corrects = 0

    cur_puzzle_bad_pipes_on_moves = []

    while state != goal:
        move = pick_move(state, visited)
        bad_pipes = get_bad_pipes(state, goal)
        cur_puzzle_bad_pipes_on_moves.append(len(bad_pipes))
        if move in bad_pipes:
            cur_puzzle_corrects += 1
        else:
            cur_puzzle_bad_pipes_when_incorrect.append(len(bad_pipes))
        visited[state].add(move)
        state = pipe_rotate_binary(state, move)
        cur_puzzle_moves += 1

    if len(all_moves) + len(outlier_all_moves) > 0:
        percentage_outliers = round(
            len(outlier_all_moves) / (len(all_moves) + len(outlier_all_moves)) * 100, 2
        )
    else:
        percentage_outliers = 0
    print(f"Solution {i+1}: {cur_puzzle_moves} moves. {percentage_outliers}% outliers")

    if cur_puzzle_moves - prev_moves[i] < 0:
        raise Exception(
            f"Solution {i+1} has {cur_puzzle_moves} moves, but the minimum number of moves is {prev_moves[i]}"
        )

    if cur_puzzle_moves < outlier_threshold:
        corrects += cur_puzzle_corrects

        all_moves.append(cur_puzzle_moves)
        extra_moves.append(cur_puzzle_moves - prev_moves[i])

        bad_pipes_on_moves.extend(cur_puzzle_bad_pipes_on_moves)
        bad_pipes_when_incorrect.extend(cur_puzzle_bad_pipes_when_incorrect)

        normal_initials.append(initial)
        normal_goals.append(goal)
    else:
        outlier_corrects += cur_puzzle_corrects

        outlier_all_moves.append(cur_puzzle_moves)
        outlier_extra_moves.append(cur_puzzle_moves - prev_moves[i])

        outlier_bad_pipes_when_incorrect.extend(cur_puzzle_bad_pipes_when_incorrect)
        outlier_bad_pipes_on_moves.extend(cur_puzzle_bad_pipes_on_moves)

        outlier_initials.append(initial)
        outlier_goals.append(goal)
