from typing import Optional
from pipe_typings import PipeType
from csp import Assignment, Variable, print1DGrid
from math import sqrt
from pipes_utils import find_adj, check_connections


def validator(pipes: list[PipeType]) -> bool:
    """
    Ensures that everything is connected
    """
    visited: list[int] = []
    dft(pipes, 0, visited)
    if len(visited) == len(pipes):
        return True
    return False


def dft(pipes: list[PipeType], loc: int, visited: list[int]) -> None:
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
    connections: tuple[bool, bool, bool, bool] = check_connections(
        pipes[loc], (top_val, right_val, bottom_val, left_val)
    )
    for i in range(4):
        if connections[i] and adj_vals[i] not in visited:
            dft(pipes, adj_vals[i], visited)


def pseudo_assign(variables: list[Variable]) -> list[PipeType]:
    """
    Creates a "pseudo assignment" of pipes. A pseudo assignment is created by taking the active domains of assigned variables and creating a PipeType containing all the possible directions that the unassigned variable could point in.

    :params variables: All the variables for the csp.
    :returns: List containing pseudo-assigned values for the variables
    """
    pseudo_assignment: list[PipeType] = []
    for var in variables:
        assignment = var.get_assignment()
        if assignment is not None:
            pseudo_assignment.append(assignment)
        else:
            pseudo_pipe: list[bool] = [False, False, False, False]
            for active_domain in var.get_active_domain():
                all_true = False
                for direction in range(4):
                    if active_domain[direction]:
                        pseudo_pipe[direction] = True
                        if sum(pseudo_pipe) == 4:
                            all_true = True
                            break
                if all_true:
                    break
            pseudo_assignment.append(
                (pseudo_pipe[0], pseudo_pipe[1], pseudo_pipe[2], pseudo_pipe[3])
            )

    return pseudo_assignment


def optimistic_psuedo_assign(variables: list[Variable]) -> list[PipeType]:
    """
    Same as pseudo_assign, but assumes (True, True, True, True) if unassigned

    :params variables: All the variables for the csp.
    :returns: List containing pseudo-assigned values for the variables
    """
    pseudo_assignment: list[PipeType] = []
    for var in variables:
        assignment = var.get_assignment()
        if assignment is not None:
            pseudo_assignment.append(assignment)
        else:
            pseudo_assignment.append((True, True, True, True))
    return pseudo_assignment


def pruner(variables: list[Variable]) -> dict[Variable, list[PipeType]]:
    pseudo_assignment = optimistic_psuedo_assign(variables)
    time = -1
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    articulation_points: set[int] = set()

    find_articulation_points(
        assignment=pseudo_assignment,
        loc=0,
        time=time,
        disc=disc,
        low=low,
        articulation_points=articulation_points,
    )
    result: dict[Variable, list[PipeType]] = {}
    for point in articulation_points:
        var_to_prune = variables[point]
        for potential_assignment in var_to_prune.get_active_domain():
            assignment = pseudo_assignment.copy()
            assignment[point] = potential_assignment
            if not validator(assignment):
                if var_to_prune not in result:
                    result[var_to_prune] = [potential_assignment]
                else:
                    result[var_to_prune].append(potential_assignment)
                var_to_prune.prune([potential_assignment])
    return result


def find_articulation_points(
    assignment: Assignment,
    loc: int,
    time: int,
    disc: dict[int, int],
    low: dict[int, int],
    articulation_points: set[int],
    parent: Optional[int] = None,
):
    """
    Uses Tarjan's algorithm to find articulation points in the graph
    """
    time += 1
    disc[loc] = time
    low[loc] = time

    adj_indices = find_adj(loc, int(sqrt(len(assignment))))

    top_val: Optional[PipeType] = None
    if adj_indices[0] != -1:
        top_val = assignment[adj_indices[0]]
    right_val: Optional[PipeType] = None
    if adj_indices[1] != -1:
        right_val = assignment[adj_indices[1]]
    bottom_val: Optional[PipeType] = None
    if adj_indices[2] != -1:
        bottom_val = assignment[adj_indices[2]]
    left_val: Optional[PipeType] = None
    if adj_indices[3] != -1:
        left_val = assignment[adj_indices[3]]
    connections: tuple[bool, bool, bool, bool] = check_connections(
        assignment[loc], (top_val, right_val, bottom_val, left_val)
    )

    child_count = 0
    for direction in range(4):
        curr_neighbor = adj_indices[direction]
        if connections[direction] and curr_neighbor != parent:
            if curr_neighbor not in disc:
                child_count += 1
                time = find_articulation_points(
                    assignment,
                    curr_neighbor,
                    time,
                    disc,
                    low,
                    articulation_points,
                    parent=loc,
                )
                low[loc] = min(low[loc], low[curr_neighbor])
                if low[curr_neighbor] >= disc[loc]:
                    articulation_points.add(loc)
                if child_count > 1 and parent is None:
                    articulation_points.add(loc)
            else:
                low[loc] = min(low[loc], disc[curr_neighbor])
    return time
