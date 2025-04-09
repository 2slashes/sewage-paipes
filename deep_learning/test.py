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


initial = "1000010110110110100010101110010100011001101010100010111000110010"
goal = "0100010101110011010001011011101001000011101010101000110110011000"
# state_int_list = [int(x) for x in state]
# state_tensor = torch.tensor(state_int_list).to(device).float()
# result = model(state_tensor)
# print(torch.softmax(result, dim=0))
# print(torch.argmax(result))


def pick_move(state):
    global cache
    if len(cache) > 3:
        cache = cache[-3:]
    # convert the state to integers
    state_int_list = [int(x) for x in state]
    # convert the state to a tensor
    state_tensor = torch.tensor(state_int_list).to(device).float()
    # get the predicted move from the neural network
    prob = model(state_tensor)
    result = torch.argmax(prob).item()
    if len(cache) == 3 and cache[0] == cache[1] == cache[2] == result:
        result = int(torch.topk(prob, 2).values[1].item())
    # output the result
    # print(torch.softmax(prob, dim=0))
    # print(result)
    cache.append(result)

    # apply the rotation to the state and return the new state
    return pipe_rotate_binary(result, state), result


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
while state != goal:
    state, result = pick_move(state)
    print(state, result)

# 0010010101110011001010101101101000100011010010100100110101011001
# 0010010101110011001001011101101000100011010010100100110101011001
# 0100010101110011010001011011101001000011100010100100110101011001
