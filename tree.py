from typing import Literal, Optional, Union

from csp import PipeType, Variable
from math import sqrt

from pipes_utils import check_connections, find_adj

PartialAssignment = list[Optional[PipeType]]
Assignment = list[PipeType]


class Node:
    def __init__(self, location: int) -> None:
        self.location = location
        self.children: list[Node] = []

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.location == other.location
        return False

    def __hash__(self):
        return hash(self.location)

    def has_cycle(self) -> bool:
        visited: set[Node] = set()
        stack: list[Node] = [self]
        while stack:
            current = stack.pop()
            if current in visited:
                return True
            visited.add(current)
            stack.extend(current.children)
        return False


def recurse_generate_graph_from_pipetypes_false_if_cycle(
    parent: Node,
    assignment: Assignment,
    prev: Optional[Node] = None,
    visited: set[Node] = set(),
) -> Union[Node, Literal[False]]:
    if parent in visited:
        return False
    # row = parent.location // int(sqrt(len(assignment)))
    # col = parent.location % int(sqrt(len(assignment)))

    (top, right, bottom, left) = find_adj(parent.location, int(sqrt(len(assignment))))
    center_pipe = assignment[parent.location]

    top_pipe = assignment[top] if top != -1 else None
    right_pipe = assignment[right] if right != -1 else None
    bottom_pipe = assignment[bottom] if bottom != -1 else None
    left_pipe = assignment[left] if left != -1 else None

    (top_connect, right_connect, bottom_connect, left_connect) = check_connections(
        center_pipe, (top_pipe, right_pipe, bottom_pipe, left_pipe)
    )

    if top_connect:
        top_pipe_node = Node(top)
        if top_pipe_node != prev:
            parent.children.append(top_pipe_node)
            result = recurse_generate_graph_from_pipetypes_false_if_cycle(
                top_pipe_node, assignment, parent, visited | {parent}
            )
            if result == False:
                return False

    if right_connect:
        right_pipe_node = Node(right)
        if right_pipe_node != prev:
            parent.children.append(right_pipe_node)
            result = recurse_generate_graph_from_pipetypes_false_if_cycle(
                right_pipe_node, assignment, parent, visited | {parent}
            )
            if result == False:
                return False

    if bottom_connect:
        bottom_pipe_node = Node(bottom)
        if bottom_pipe_node != prev:
            parent.children.append(bottom_pipe_node)
            result = recurse_generate_graph_from_pipetypes_false_if_cycle(
                bottom_pipe_node, assignment, parent, visited | {parent}
            )
            if result == False:
                return False

    if left_connect:
        left_pipe_node = Node(left)
        if left_pipe_node != prev:
            parent.children.append(left_pipe_node)
            result = recurse_generate_graph_from_pipetypes_false_if_cycle(
                left_pipe_node, assignment, parent, visited | {parent}
            )
            if result == False:
                return False

    return parent


def validator(assignment: Assignment) -> bool:
    root = Node(0)
    return (
        recurse_generate_graph_from_pipetypes_false_if_cycle(root, assignment) != False
    )


def get_duplicated_touched(
    parent: Node,
    assignment: PartialAssignment,
    prev: Optional[Node] = None,
    touched: set[Node] = set(),
) -> Optional[Node]:
    """
    Creates a tree from assignment, checks if any squares are touched twice.
    A square is "touched" if a pipe's opening is pointing towards it.

    :param parent: the current node
    :param assignment: the assignment of the pipes
    :param prev: the previous node
    :param touched: the set of nodes that have been touched

    :return: the node that has been touched twice
    """
    center_pipe = assignment[parent.location]
    if center_pipe is None:
        raise Exception("Traversed to an unassigned pipe")

    adj_indexes = find_adj(parent.location, int(sqrt(len(assignment))))

    for i, adj_i in enumerate(adj_indexes):
        if center_pipe[i]:
            if adj_i == -1:
                raise Exception(
                    f"Pipe pointing to edge of grid in the direction of {i}"
                )

            adj_node = Node(adj_i)
            if adj_node != prev and adj_node in touched:
                return adj_node
            touched.add(adj_node)

    top_pipe, right_pipe, bottom_pipe, left_pipe = [
        assignment[i] if i != -1 else None for i in adj_indexes
    ]
    pipe_tuple = (top_pipe, right_pipe, bottom_pipe, left_pipe)

    adj_connections = check_connections(center_pipe, pipe_tuple)

    for adj_i, adj_is_connected in zip(adj_indexes, adj_connections):
        if adj_is_connected:
            next_node = Node(adj_i)
            if next_node != prev:
                parent.children.append(next_node)
                duplicate_touch = get_duplicated_touched(
                    next_node, assignment, parent, touched
                )
                if duplicate_touch:
                    return duplicate_touch
    return None


def pruner(variables: list[Variable]) -> dict[Variable, list[PipeType]]:
    assignment: PartialAssignment = [var.get_assignment() for var in variables]
    if all(x is None for x in assignment):
        return {}

    # get index of first non-none value
    seed_index = next((i for i, x in enumerate(assignment) if x is not None), -1)
    root = Node(seed_index)
    duplicate_touch = get_duplicated_touched(root, assignment)

    if duplicate_touch is None:
        return {}

    n = int(sqrt(len(assignment)))
    x = duplicate_touch.location // n
    y = duplicate_touch.location % n

    variable_to_prune = next(
        (var for var in variables if var.location[0] == x and var.location[1] == y),
    )

    pruned_values = {variable_to_prune: variable_to_prune.active_domain.copy()}
    variable_to_prune.active_domain.clear()

    return pruned_values
