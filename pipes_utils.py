import math
from csp import *


def find_adj(center: int, n: int) -> tuple[int, int, int, int]:
    """
    :param center: Index of the pipe to compute the adjacent indices from
    :param n: dimension of the grid of the pipes puzzle
    :return: A tuple of four ints, the first representing the index of the value above, going clockwise from there. A value of -1 as one of the elements of the tuple indicates that there is no adjacent pipe in that direction (i.e. The center pipe is on an edge in that direction)
    """
    above = center - n
    right = center + 1
    below = center + n
    left = center - 1

    if above < 0:
        above = -1

    if right % n == 0:
        right = -1

    if below >= (n * n):
        below = -1

    if left % n == n - 1:
        left = -1

    return (above, right, below, left)


def check_connections(
    center: PipeType,
    adj: tuple[
        Optional[PipeType], Optional[PipeType], Optional[PipeType], Optional[PipeType]
    ],
) -> tuple[bool, bool, bool, bool]:
    """
    :param main: the "center" pipe
    :param adj: holds pipes adjacent to the main variable, with adj[0] is above main, and going clockwise from there
    :return connections: a tuple holding the connection directions from the main variable, following the same direction format as the adj parameter
    """
    connections: list[bool] = [False] * 4
    for i in range(len(center)):
        # iterate through the surrounding pipes
        if center[i]:
            adj_pipe = adj[i]
            # check if the current adjacent pipe has an opening facing towards the center pipe
            if adj_pipe is not None and adj_pipe[(i + 2) % 4]:
                connections[i] = True

    connected_up = connections[0]
    connected_right = connections[1]
    connected_down = connections[2]
    connected_left = connections[3]

    return (connected_up, connected_right, connected_down, connected_left)


def flatten(pipes: list[list[Optional[PipeType]]]) -> list[Optional[PipeType]]:
    return [pipe for row in pipes for pipe in row]


def gridify(pipes: list[Optional[PipeType]]) -> list[list[Optional[PipeType]]]:
    n = math.sqrt(len(pipes))
    if n % 1 != 0:
        raise ValueError("The length of the list must be a perfect square")
    n = int(n)
    twoDGrid: list[list[Optional[PipeType]]] = []
    for i in range(0, len(pipes), n):
        twoDGrid.append(pipes[i : i + n])
    return twoDGrid
