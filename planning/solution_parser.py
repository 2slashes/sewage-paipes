import os

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

root_dir: str = "/planning/pddl/solutions"

for dir_name in os.listdir(root_dir):
    path = os.path.join(root_dir, dir_name)
    if os.path.isdir(path):
        if dir_name.isdigit():
            n: int = int(dir_name) * int(dir_name)