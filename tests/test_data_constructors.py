from math import sqrt
from pipes_utils import gridify
from typing import Literal, Optional
from csp import DomainGenerator, PartialAssignment, PipeType, Variable

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
PipeName = Literal[
    "Up",
    "Right",
    "Down",
    "Left",
    "UpRight",
    "UpDown",
    "UpLeft",
    "RightDown",
    "RightLeft",
    "DownLeft",
    "UpRightDown",
    "UpRightLeft",
    "UpDownLeft",
    "RightDownLeft",
]
PIPE: dict[PipeName, PipeType] = {
    "Up": (True, False, False, False),
    "Right": (False, True, False, False),
    "Down": (False, False, True, False),
    "Left": (False, False, False, True),
    "UpRight": (True, True, False, False),
    "UpDown": (True, False, True, False),
    "UpLeft": (True, False, False, True),
    "RightDown": (False, True, True, False),
    "RightLeft": (False, True, False, True),
    "DownLeft": (False, False, True, True),
    "UpRightDown": (True, True, True, False),
    "UpRightLeft": (True, True, False, True),
    "UpDownLeft": (True, False, True, True),
    "RightDownLeft": (False, True, True, True),
}


def partial_assignment_to_variables(
    partial_assignment: PartialAssignment,
) -> list[Variable]:
    variables: list[Variable] = []
    n = len(partial_assignment)
    side_length = int(sqrt(n))
    for i in range(len(partial_assignment)):
        top = i == 0
        bottom = i in range(n - side_length, n)
        left = i % side_length == 0
        right = i % side_length == side_length - 1
        var = Variable(
            location=i,
            domain=DomainGenerator.generate_domain(top, right, bottom, left),
        )
        variables.append(var)
    return variables


def print2DGrid(pipes: list[list[Optional[PipeType]]]) -> None:
    row_num = 1
    for row in pipes:
        print(row_num, " ", end="")
        for pipe in row:
            if pipe is None:
                print("•", end="")
            else:
                print(PIPE_CHAR[pipe], end="")
        row_num += 1
        print()


def print1DGrid(pipes: list[Optional[PipeType]]) -> None:
    print2DGrid(gridify(pipes))
