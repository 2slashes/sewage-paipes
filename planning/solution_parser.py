import os
import re

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

    solution_file_pattern = re.compile(r"^solution\d+\.1$")
    action_pattern = re.compile(r"^\((\w+) (\w+)\)$")
    # iterate through all the solution directories and parse all the solutions
    root_dir: str = "planning/pddl/solutions"
    for dir_name in os.scandir(root_dir):
        if dir_name.is_dir() and dir_name.name.isdigit():
            # define object names: number of objects is the directory name squared, since the directory name is n
            object_names: list[str] = []
            n_squared: int = int(dir_name.name) * int(dir_name.name)
            for i in range(n_squared):
                object_names.append(f"p{i}")
            
            # iterate through each file, find lines with the pattern (action_name object_name)
            for solution_file in os.listdir(dir_name):
                if solution_file_pattern.match(solution_file):
                    file_path = os.path.join(dir_name, solution_file)
                    with open(file_path, "r", encoding="utf-8") as file:
                        for line in file:
                            line = line.strip()  # Remove leading/trailing whitespace
                            match = action_pattern.match(line)
                            
                            if match:
                                action, obj = match.groups()
                                if action in action_names and obj in object_names:
                                    print(f"  ✅ Valid action found: {line}")
                                else:
                                    raise Exception("action or object doesn't match")
parse_all()