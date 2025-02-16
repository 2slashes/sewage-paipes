from pipe_typings import *
from csp import Variable
from pipes_utils import *
from math import sqrt

def validator(
    pipes: list[PipeType]
) -> bool:
    """
    Ensures that everything is connected
    """
    stack: list[tuple[PipeType, int]] = [(pipes[0], 0)]
    visited: list[int] = []
    while len(stack) > 0:
        cur, loc = stack.pop()
        visited.append(loc)
        adj_vals: tuple[int, int, int, int] = find_adj(loc, int(sqrt(len(pipes))))
        top_val: Optional[PipeType] = None
        if adj_vals[0] != -1:
            top_val = pipes[adj_vals[0]]
        right_val: Optional[PipeType] = None
        if adj_vals[1] != -1:
            right_val = pipes[adj_vals[1]]
        bottom_val: Optional[PipeType] = None
        if adj_vals[2] != -1:
            bottom_val = pipes[adj_vals[2]]
        left_val: Optional[PipeType] = None
        if adj_vals[3] != -1:
            left_val = pipes[adj_vals[3]]
        connections: tuple[bool, bool, bool, bool] = check_connections(cur, (top_val, right_val, bottom_val, left_val))
        for i in range(4):
            if connections[i] and adj_vals[i] not in visited:
                stack.append((pipes[adj_vals[i]], adj_vals[i]))
    if len(visited) == len(pipes):
        return True
    return False

def pruner(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    