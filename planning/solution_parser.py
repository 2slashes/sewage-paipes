import os
import re
import csv


def parse_all():
    """
    Parses all solution files from pddl/solutions/ and the corresponding state files from states/. Associates the current state with a rotation of 90 degrees clockwise on a specified pipe, then applies the rotation to get the next state. This state/rotation pair is outputted to a csv in ../deep-learning/data/, and this is repeated for all rotations in all solution files.
    """
    # define the action names in the solution file
    action_names: list[str] = [
        "rotate_pipe_up",
        "rotate_pipe_right",
        "rotate_pipe_down",
        "rotate_pipe_left",
        "rotate_pipe_up_right",
        "rotate_pipe_down_right",
        "rotate_pipe_down_left",
        "rotate_pipe_up_left",
        "rotate_pipe_up_down",
        "rotate_pipe_left_right",
        "rotate_pipe_up_right_down",
        "rotate_pipe_right_down_left",
        "rotate_pipe_down_left_up",
        "rotate_pipe_left_up_right",
    ]
    curr_dir = os.path.dirname(__file__)
    csv_dir = os.path.join(curr_dir, "../deep-learning/data/")
    csv_file_name = os.path.join(curr_dir, "../deep-learning/data/out.csv")

    # create the directory for the csv if it doesn't exist
    try:
        os.makedirs(csv_dir)
    except PermissionError:
        print(f"Permission denied: Unable to create '{csv_dir}'.")
    except FileExistsError:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")

    # pattern that matches a solution file: solution{i} where i is an int
    solution_file_pattern = re.compile(r"^solution(\d+)$")
    # pattern that matches an action in the solution file: ({action} {object})
    action_pattern = re.compile(r"^\((\w+) (\w+)\)$")
    # iterate through all the solution and state directories and parse all the solutions
    solutions_dir: str = "planning/pddl/solutions"
    solutions_dir: str = os.path.join(curr_dir, "pddl/solutions")
    states_dir: str = os.path.join(curr_dir, "states")
    include_header = True
    if os.path.exists(csv_file_name) and os.path.getsize(csv_file_name) != 0:
        # Check if the csv file exists and is not empty
        include_header = False
    with open(csv_file_name, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # write the header if the csv file is empty or if it didn't exist
        if include_header:
            writer.writerow(["state", "action"])

        # iterate through directories in the solution directory
        for solution_dir in os.scandir(solutions_dir):
            # ensure that the directory name is a digit - this digit represents the dimension of the puzzle
            if solution_dir.is_dir() and solution_dir.name.isdigit():
                # define object names: number of objects is the directory name squared, since the directory name is the dimension of the puzzle
                object_names: list[str] = []
                n: int = int(solution_dir.name)
                n_squared: int = n * n
                for i in range(n_squared):
                    # each object name is p{i}, where i is the number of the object
                    object_names.append(f"p{i}")
                state_dir = f"{states_dir}/{n}"
                # iterate through each file, find lines with the pattern (action_name object_name)
                for solution_file in os.listdir(solution_dir):
                    # ensure that the file matches a solution file
                    match = re.match(solution_file_pattern, solution_file)
                    if match:
                        # extract the int from the solution file name and use this to get the state file corresponding to it
                        problem_num = int(match.group(1))
                        state_file_path = os.path.join(state_dir, f"state{problem_num}")

                        solution_file_path = os.path.join(solution_dir, solution_file)

                        # open the solution and state files
                        with open(
                            solution_file_path, "r", encoding="utf-8"
                        ) as solution_file, open(
                            state_file_path, "r", encoding="utf-8"
                        ) as state_file:
                            # get the initial state from the state file
                            state: str = state_file.readline().strip()

                            # read each line in the solution file
                            for line in solution_file:
                                line = line.strip()
                                # make sure that the line contains an action
                                match = action_pattern.match(line)
                                if match:
                                    # validate the action and object names
                                    action, obj = match.groups()
                                    if action in action_names and obj in object_names:
                                        # object names are of the form p{int}
                                        # remove the leading p and get the int
                                        object_num = int(obj[1:])
                                        # write the state followed by the object to the csv
                                        writer.writerow([state, object_num])
                                        # compute the state for the next iteration
                                        state = pipe_rotate_binary(object_num, state)
                                    else:
                                        raise Exception(
                                            "action or object doesn't match"
                                        )
                            # ensure that the solution is correct by checking if the goal state matches the state after all rotations have been applied
                            goal_state: str = state_file.readline().strip()
                            if goal_state != state:
                                raise Exception("goal is wrong")

                        # delete the solution and state files after they have been completely parsed
                        try:
                            os.remove(solution_file_path)
                        except FileNotFoundError:
                            print(f"Error: {solution_file_path} not found.")
                        try:
                            os.remove(state_file_path)
                        except FileNotFoundError:
                            print(f"Error: {state_file_path} not found.")
                # remove the solution and state subdirectories after all solution/state files have been emptied
                os.rmdir(solution_dir)
                os.rmdir(state_dir)
    # remove the solution and state directories after they have been fully cleared
    os.rmdir(solutions_dir)
    os.rmdir(states_dir)
    print("parsed data successfully")


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


parse_all()
