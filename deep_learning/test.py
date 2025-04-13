import csv
import os
from torch import nn
import torch


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


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

n = 4
model = PipesPredictor(input_size=4 * (n**2), hidden_size=128, output_size=n**2).to(
    device
)

model_file_name = "model.pth"
curr_dir = os.getcwd()
model_file_path = os.path.join(curr_dir, model_file_name)
model.load_state_dict(torch.load(model_file_path, weights_only=True))
model.eval()


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


initials: list[str] = []
goals: list[str] = []

puzzle_path = os.path.join(curr_dir, "data/puzzles.csv")
with open(puzzle_path, newline="") as csvfile:
    reader = csv.reader(csvfile)
    # skip the header
    next(reader)
    for row in reader:
        initials.append(row[0])
        goals.append(row[1])

total_moves = 0
corrects = 0
bad_pipes_when_incorrect = []
all_moves = []

for i in range(len(initials)):
    initial = initials[i]
    goal = goals[i]
    visited = {}

    state = initial
    visited[initial] = set()
    cur_puzzle_moves = 0
    while state != goal:
        move = pick_move(state, visited)
        bad_pipes = get_bad_pipes(state, goal)
        if move in bad_pipes:
            corrects += 1
        else:
            bad_pipes_when_incorrect.append(len(bad_pipes))
        visited[state].add(move)
        state = pipe_rotate_binary(state, move)
        cur_puzzle_moves += 1
        total_moves += 1
        # print(f"Accuracy: {corrects / total_moves}")
        # if len(bad_pipes_when_incorrect) > 0:
        #     print(
        #         f"Average bad pipes when incorrect: {sum(bad_pipes_when_incorrect) / len(bad_pipes_when_incorrect)}"
        #     )
    print(f"Solution {i+1}: {cur_puzzle_moves} moves")
    all_moves.append(cur_puzzle_moves)

print("-----------")
print(f"Accuracy: {corrects / total_moves}")
print(
    f"Average bad pipes when incorrect: {sum(bad_pipes_when_incorrect) / len(bad_pipes_when_incorrect)}"
)
print(f"Average moves: {total_moves / len(initials)}")

print(f"Outliers: {[x for x in all_moves if x > 100]}")
