from typing import Literal, Optional, Union
from csp import PipeType
from math import sqrt

from pipes_utils import check_connections, find_adj

Assignment = list[Optional[PipeType]]


class Node:
    def __init__(self, location: int):
        self.location = location
        self.children: list[Node] = []

    def __eq__(self, other):
        return self.location == other.value

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


# def generate_graph_from_pipetypes(assignment: Assignment) -> Node:
#     pre_root = Node((float("inf"), float("inf")))

#     root = Node((0, 0))

#     stack: list[tuple[Node, Node]] = [(pre_root, root)]
#     while stack:
#         (parent, curr) = stack.pop()
#         parent_is_left = parent.location[1] == curr.location[1] - 1
#         parent_is_top = parent.location[0] == curr.location[0] - 1
#         parent_is_right = parent.location[1] == curr.location[1] + 1
#         parent_is_bottom = parent.location[0] == curr.location[0] + 1

#         parent.children.append(curr)
#         (has_top, has_right, has_bottom, has_left) = find_connection(
#             assignment, curr.location
#         )
#         if has_top and not parent_is_top:
#             stack.append((curr, Node((curr.location[0] - 1, curr.location[1]))))
#         if has_right and not parent_is_right:
#             stack.append((curr, Node((curr.location[0], curr.location[1] + 1))))
#         if has_bottom and not parent_is_bottom:
#             stack.append((curr, Node((curr.location[0] + 1, curr.location[1]))))
#         if has_left and not parent_is_left:
#             stack.append((curr, Node((curr.location[0], curr.location[1] - 1))))

#     return root


def recurse_generate_graph_from_pipetypes_false_if_cycle(
    parent: Node, assignment: Assignment, prev: Optional[Node] = None
) -> Union[Node, Literal[False]]:

    (top, right, bottom, left) = find_adj(parent.location, int(sqrt(len(assignment))))
    center_pipe = assignment[parent.location]
    if center_pipe is None:
        raise Exception("Center pipe is None")

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
            recurse_generate_graph_from_pipetypes_false_if_cycle(
                top_pipe_node, assignment, parent
            )
    if parent.has_cycle():
        return False

    if right_connect:
        right_pipe_node = Node(right)
        if right_pipe_node != prev:
            parent.children.append(right_pipe_node)
            recurse_generate_graph_from_pipetypes_false_if_cycle(
                right_pipe_node, assignment, parent
            )
    if parent.has_cycle():
        return False

    if bottom_connect:
        bottom_pipe_node = Node(bottom)
        if bottom_pipe_node != prev:
            parent.children.append(bottom_pipe_node)
            recurse_generate_graph_from_pipetypes_false_if_cycle(
                bottom_pipe_node, assignment, parent
            )
    if parent.has_cycle():
        return False

    if left_connect:
        left_pipe_node = Node(left)
        if left_pipe_node != prev:
            parent.children.append(left_pipe_node)
            recurse_generate_graph_from_pipetypes_false_if_cycle(
                left_pipe_node, assignment, parent
            )
    if parent.has_cycle():
        return False

    return parent


def tree_sat(assignment: Assignment) -> bool:
    root = Node(0)
    return (
        recurse_generate_graph_from_pipetypes_false_if_cycle(root, assignment) != False
    )
