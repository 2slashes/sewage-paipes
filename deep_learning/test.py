from pipes_nn_classes import PipesPredictor
import torch
from torch import nn
import os

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

n = 4
model = PipesPredictor(n**2 * 4, 64, n**2).to(device)

model_file_name = "model.pth"
curr_dir = os.path.dirname(__file__)
model_file_path = os.path.join(curr_dir, model_file_name)
model.load_state_dict(torch.load(model_file_path, weights_only=True))
model.eval()

global cache
cache = []
global visited
visited = set()


# initial = "0010100001001001101011001010011101011000010001111001110101011100"
# goal = "0010001001000011101011000101101110100010001010111100110101011001"
initial = "0110101010110100010101101110000110101010001110111000001101000100"
goal = "0110010101110001101001101011001010101010110010111000110000011000"
# state_int_list = [int(x) for x in state]
# state_tensor = torch.tensor(state_int_list).to(device).float()
# result = model(state_tensor)
# print(torch.softmax(result, dim=0))
# print(torch.argmax(result))


def pick_move(state):
    global cache
    global visited
    if len(cache) > 3:
        cache = cache[-3:]
    # convert the state to integers
    state_int_list = [int(x) for x in state]
    # convert the state to a tensor
    state_tensor = torch.tensor(state_int_list).to(device).float()
    # get the predicted move from the neural network
    prob = model(state_tensor)
    # result = torch.argmax(prob).item()
    results = torch.topk(prob, 16).indices
    # print(results)
    # output the result
    # print(torch.softmax(prob, dim=0))
    # print(result)
    # cache.append(result)

    # apply the rotation to the state and return the new state
    new_state = state
    result = 0
    for r in results:
        result = int(r)
        new_state = pipe_rotate_binary(result, state)
        if new_state not in visited:
            visited.add(new_state)
            break
    if new_state == state:
        raise Exception("chyme")
    return new_state, result


def pipe_rotate_binary(pipe: int, board: str):
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


state = initial
visited.add(initial)
moves = 0
while state != goal:
    state, result = pick_move(state)
    print(state, result)
    visited.add(state)
    moves += 1
print(f"moves: {moves}")

# 0010010101110011001010101101101000100011010010100100110101011001
# 0010010101110011001001011101101000100011010010100100110101011001
# 0100010101110011010001011011101001000011100010100100110101011001
