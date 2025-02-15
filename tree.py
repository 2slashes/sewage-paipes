from typing import Optional
from csp import PipeType, Variable, PartialAssignment, Assignment
from math import sqrt
from pipes_utils import check_connections, find_adj


def assignment_has_cycle(
    curr: int,
    assignment: Assignment,
    prev: Optional[int] = None,
    visited: set[int] = set(),
) -> bool:
    if curr in visited:
        return True
    visited.add(curr)

    adj_indexes = find_adj(curr, int(sqrt(len(assignment))))

    center_pipe = assignment[curr]
    top_pipe, right_pipe, bottom_pipe, left_pipe = [
        assignment[i] if i != -1 else None for i in adj_indexes
    ]

    pipe_tuple = (top_pipe, right_pipe, bottom_pipe, left_pipe)

    adj_connections = check_connections(center_pipe, pipe_tuple)

    for adj_i, adj_is_connected in zip(adj_indexes, adj_connections):
        if adj_is_connected and adj_i != prev:
            if assignment_has_cycle(adj_i, assignment, curr, visited):
                return True
    return False


def validator(assignment: Assignment) -> bool:
    return assignment_has_cycle(0, assignment)


def get_duplicated_touched(
    curr: int,
    assignment: PartialAssignment,
    prev: Optional[int] = None,
    touched: set[int] = set(),
) -> Optional[int]:
    """
    Creates a tree from assignment, checks if any squares are touched twice.
    A square is "touched" if a pipe's opening is pointing towards it.

    :param parent: the current node
    :param assignment: the assignment of the pipes
    :param prev: the previous node
    :param touched: the set of nodes that have been touched

    :return: the node that has been touched twice
    """
    center_pipe = assignment[curr]
    if center_pipe is None:
        raise Exception("Traversed to an unassigned pipe")

    adj_indexes = find_adj(curr, int(sqrt(len(assignment))))

    for i, adj_i in enumerate(adj_indexes):
        if center_pipe[i]:
            if adj_i == -1:
                raise Exception(
                    f"Pipe pointing to edge of grid in the direction of {i}"
                )

            if adj_i != prev and adj_i in touched:
                return adj_i

            touched.add(adj_i)

    top_pipe, right_pipe, bottom_pipe, left_pipe = [
        assignment[i] if i != -1 else None for i in adj_indexes
    ]
    pipe_tuple = (top_pipe, right_pipe, bottom_pipe, left_pipe)

    adj_connections = check_connections(center_pipe, pipe_tuple)

    for adj_i, adj_is_connected in zip(adj_indexes, adj_connections):
        if adj_is_connected and adj_i != prev:
            duplicate_touch = get_duplicated_touched(adj_i, assignment, curr, touched)
            if duplicate_touch:
                return duplicate_touch
    return None


def pruner(variables: list[Variable]) -> dict[Variable, list[PipeType]]:
    assignment: PartialAssignment = [var.get_assignment() for var in variables]
    if all(x is None for x in assignment):
        return {}

    # get index of first non-none value
    seed_index = next((i for i, x in enumerate(assignment) if x is not None), -1)
    duplicate_touch = get_duplicated_touched(seed_index, assignment)

    if duplicate_touch is None:
        return {}

    variable_to_prune = next(
        (var for var in variables if var.location == duplicate_touch),
    )

    pruned_values = {variable_to_prune: variable_to_prune.active_domain.copy()}
    variable_to_prune.active_domain.clear()

    return pruned_values
