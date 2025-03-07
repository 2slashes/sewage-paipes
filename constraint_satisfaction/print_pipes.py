from math import sqrt
from typing import Literal
from pipe_typings import PipeType

# mapping of PipeTypes to a character that represents them visually.
PIPE_CHAR: dict[PipeType, str] = {
    (True, False, False, False): "╵",  # Open at the top
    (False, True, False, False): "╶",  # Open at the right
    (False, False, True, False): "╷",  # Open at the bottom
    (False, False, False, True): "╴",  # Open at the left
    (True, True, False, False): "└",  # Elbow (bottom-left)
    (True, False, True, False): "│",  # Vertical pipe
    (True, False, False, True): "┘",  # Elbow (bottom-right)
    (False, True, True, False): "┌",  # Elbow (top-left)
    (False, True, False, True): "─",  # Horizontal pipe
    (False, False, True, True): "┐",  # Elbow (top-right)
    (True, True, True, False): "├",  # T-junction (left, down, up)
    (True, True, False, True): "┴",  # T-junction (left, right, down)
    (True, False, True, True): "┤",  # T-junction (right, down, up)
    (False, True, True, True): "┬",  # T-junction (left, right, up)
}


def print_pipes_grid(pipes: list[PipeType]) -> None:
    """
    Prints a visual representation of a grid of pipes

    :params pipes: grid to be visualized
    """
    n = int(sqrt(len(pipes)))
    for i in range(len(pipes)):
        # print the pipe character corresponding to current pipe
        print(PIPE_CHAR[pipes[i]], end="")
        # print a new line after each row of pipes
        if i % n == n - 1:
            print()
