import os
import re
import csv

def parse_all():
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
    csv_dir = os.path.join(curr_dir, "../deep_learning/data/")
    csv_file_name = os.path.join(curr_dir, "../deep_learning/data/out.csv")

    try:
        os.makedirs(csv_dir)
    except PermissionError:
        print(f"Permission denied: Unable to create '{csv_dir}'.")
    except FileExistsError:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    
    solution_file_pattern = re.compile(r"^solution(\d+)$")
    action_pattern = re.compile(r"^\((\w+) (\w+)\)$")
    # iterate through all the solution directories and parse all the solutions
    solutions_dir: str = "planning/pddl/solutions"
    solutions_dir: str = os.path.join(curr_dir, "pddl/solutions")
    states_dir: str = os.path.join(curr_dir, "states")
    include_header = True
    if os.path.exists(csv_file_name) and os.path.getsize(csv_file_name) != 0:
        # Check if the file is empty
        include_header = False
    with open(csv_file_name, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if include_header:
            writer.writerow(["state", "action"])

        for dir_name in os.scandir(solutions_dir):
            if dir_name.is_dir() and dir_name.name.isdigit():
                # define object names: number of objects is the directory name squared, since the directory name is n
                object_names: list[str] = []
                n: int = int(dir_name.name)
                n_squared: int = n * n
                for i in range(n_squared):
                    object_names.append(f"p{i}")
                state_dir = f"{states_dir}/{n}"
                # iterate through each file, find lines with the pattern (action_name object_name)
                for solution_file in os.listdir(dir_name):
                    match = re.match(solution_file_pattern, solution_file)
                    if match:
                        problem_num = int(match.group(1))
                        solution_file_path = os.path.join(dir_name, solution_file)
                        state_file_path = os.path.join(state_dir, f"state{problem_num}")
                        with open(solution_file_path, "r", encoding="utf-8") as solution_file, open(state_file_path, "r", encoding="utf-8") as state_file:
                            # open a writer for the csv file and label columns
                            
                            # get the initial state from file2
                            state: str = state_file.readline().strip()
                            for line in solution_file:
                                line = line.strip()  # Remove leading/trailing whitespace
                                match = action_pattern.match(line)
                                
                                if match:
                                    action, obj = match.groups()
                                    if action in action_names and obj in object_names:
                                        # object names are of the form p{int}
                                        # remove the leading p and get the int
                                        object_num = int(obj[1:])
                                        # write the state followed by the object to the csv
                                        writer.writerow([state, object_num])
                                        # apply the rotation to the state to get the next state
                                        state = pipe_rotate_binary(object_num, state)
                                    else:
                                        raise Exception("action or object doesn't match")
                            goal_state: str = state_file.readline().strip()
                            if goal_state != state:
                                raise Exception("goal is wrong")
                            
                        try:
                            os.remove(solution_file_path)
                        except FileNotFoundError:
                            print(f"Error: {solution_file_path} not found.")
                        try:
                            os.remove(state_file_path)
                        except FileNotFoundError:
                            print(f"Error: {state_file_path} not found.")
                os.rmdir(dir_name)
                os.rmdir(state_dir)
    os.rmdir(solutions_dir)
    os.rmdir(states_dir)
    print("parsed data successfully")
            
                                
                                
def pipe_rotate_binary(pipe: int, board: str):
    """
    Takes a binary representation of a board of pipes, and a pipe to rotate. Outputs a binary representation of the board after rotating the pipe.

    :params pipe: The pipe to rotate
    :params board: Binary representation of the board
    
    """
    # each pipe has 4 values associated to it, so pipe n starts at index 4 * n
    start_index = 4 * pipe
    up = board[start_index]
    right = board[start_index + 1]
    down = board[start_index + 2]
    left = board[start_index + 3]

    # rotate clockwise
    new_board = board[:start_index] + left + up + right + down + board[start_index + 4:]

    return new_board


parse_all()