from pipe_typings import PipeType, Assignment
import random
import os
import csv

def clockwise_rotate(pipe: PipeType, n: int) -> PipeType:
    """
    Rotate a PipeType clockwise by 90*n degrees.

    :params pipe: The pipe to be rotated
    :params n: number of 90 degree rotations to perform. 0 = no rotation, 1 = 90 degree rotation, 2 = 180 degrtee rotation etc.
    """
    top = pipe[(0 - n) % 4]
    right = pipe[(1 - n) % 4]
    bottom = pipe[(2 - n) % 4]
    left = pipe[(3 - n) % 4]

    new_pipe: PipeType = (top, right, bottom, left)
    return new_pipe

def random_rotate_board(board: Assignment, num_rotations: int) -> list[Assignment]:
    """
    Rotates every pipe on a pipes board by either 0, 90, 180, or 270 degrees, at random.

    :params board: list of PipeTypes that represents the board to be randomly rotated
    :params num_rotations: number of times to rotate the full board, also the number of new boards that will be returned.
    :returns: list of boards after random rotation.  
    """
    new_boards: list[Assignment] = []
    pipe_rotations: list[dict[int, int]] = []
    while len(new_boards) < num_rotations:
        new_board: Assignment = []
        pipe_rotations.append({})
        # generate a random rotation of each pipe in the board
        for i, pipe in enumerate(board):
            num_rotations_pipe = random.randint(0, 3)
            new_pipe = clockwise_rotate(pipe, num_rotations_pipe)
            new_board.append(new_pipe)
            # The number of rotations to return the pipe to its original position is 4, so the number of rotations in the solution is 4 minus the number that have already been done
            # if no rotations were done, then none should be done in the solution
            if num_rotations_pipe:
                pipe_rotations[-1][i] = (4 - num_rotations_pipe) % 4
        # ensure that the random rotation of the board has not been generated already
        if new_board not in new_boards:
            new_boards.append(new_board)
    generate_solutions(new_boards, board, pipe_rotations)
    return new_boards

def generate_solutions(boards: list[Assignment], solution: Assignment, rotation_maps: list[dict[int, int]]):
    curr_dir = os.path.dirname(__file__)
    csv_dir = os.path.join(curr_dir, "../deep_learning/data/")
    csv_file_name = os.path.join(curr_dir, "../deep_learning/data/test.csv")

    # create the directory for the csv if it doesn't exist
    try:
        os.makedirs(csv_dir)
    except PermissionError:
        print(f"Permission denied: Unable to create '{csv_dir}'.")
    except FileExistsError:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    
    goal_state = generate_one_state_str(solution)
    initial_states = []
    for board in boards:
        initial_states.append(generate_one_state_str(board))
    for n, rotations in enumerate(rotation_maps):
        output_rotations(initial_states[n], goal_state, rotations, csv_file_name)
        
def output_rotations(initial_state: str, goal_state: str, rotations: dict[int, int], file_path: str):
    include_header = True
    if os.path.exists(file_path) and os.path.getsize(file_path) != 0:
        # Check if the csv file exists and is not empty
        include_header = False
    with open(file_path, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # write the header if the csv file is empty or if it didn't exist
        if include_header:
            writer.writerow(["state", "actions"])
        cur_state = initial_state
        # for key in rotations:
        #     for n in range(rotations[key]):
        #         # create a binary string representing the pipes that need to be rotated
        #         rotation_str: str = ""
        #         for i in range(len(initial_state)//4):
        #             if i in rotations and rotations[i]:
        #                 rotation_str += "1"
        #             else:
        #                 rotation_str += "0"
        #         writer.writerow([cur_state, rotation_str])
        #         rotations[key] -= 1
        #         cur_state = pipe_rotate_binary(key, cur_state)
        
        while rotations:
            # select a random key in rotations
            key = list(rotations.keys())[0]
            # create a binary string representing the pipes that need to be rotated
            # find the first index that must be rotated as part of the solution
            rotation_str: str = str(key)
            # for i in range(len(initial_state)//4):
            #     if i in rotations and rotations[i]:
            #         rotation_str += "1"
            #     else:
            #         rotation_str += "0"
            
            writer.writerow([cur_state, rotation_str])
            rotations[key] -= 1
            if rotations[key] <= 0:
                del rotations[key]
            cur_state = pipe_rotate_binary(key, cur_state)

    # check if the final state matches the goal state
    if goal_state != cur_state:
        print(goal_state)
        print(cur_state)
        raise Exception("goal is wrong")
    

def generate_one_state_str(state: Assignment):
    output = ""
    for pipe in state:
        for dir in range(4):
            if pipe[dir]:
                output += "1"
            else:
                output += "0"
    return output

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