from typing import Optional
from pipe_typings import PipeType
from csp import Variable
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


def pruner(variables: list[Variable]) -> dict[Variable, list[PipeType]]:
    pruned: dict[Variable, list[PipeType]] = {}
    pseudo_assignment = pseudo_assign(variables)
    pruned: dict[Variable, list[PipeType]] = {}

    can_be_connected = validator(pseudo_assignment)
    if not can_be_connected:
        for var in variables:
            if var.get_assignment() is None:
                pruned = {var: var.get_active_domain()}
                var.prune(var.get_active_domain())
                break
    else:
        for i in range(len(pseudo_assignment)):
            find_isolated_path(variables, pseudo_assignment, i, -1, pruned)
    return pruned


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


def find_isolated_path(
    variables: list[Variable],
    pseudo_assignment: list[PipeType],
    i: int,
    last_dir: int,
    pruned: dict[Variable, list[PipeType]],
):
    main_pipe = pseudo_assignment[i]
    main_var = variables[i]
    adj_index = find_adj(i, int(sqrt(len(pseudo_assignment))))
    to_prune: list[PipeType] = []
    # holds adjacent PipeTypes, not including the pipe that came before in the path
    adj_pipe_list: list[Optional[PipeType]] = [None, None, None, None]
    for i in range(4):
        if adj_index[i] != -1 and i != last_dir:
            adj_pipe_list[i] = pseudo_assignment[adj_index[i]]
    adj_pipes = (adj_pipe_list[0], adj_pipe_list[1], adj_pipe_list[2], adj_pipe_list[3])
    connections = check_connections(main_pipe, adj_pipes)
    num_connections = 0
    cur_dir = 0
    for i in range(4):
        if connections[i]:
            num_connections += 1
            cur_dir = i
    if num_connections == 1:
        # the path continues, prune from current variable
        path_dir = 0
        for i in range(4):
            if connections[i] and i != last_dir:
                path_dir = i
                break
        if main_var.get_assignment() is None:
            active_domain = main_var.get_active_domain()
            for assignment in active_domain:
                if not assignment[cur_dir] or (
                    last_dir != -1 and not assignment[last_dir]
                ):
                    to_prune.append(assignment)
                    if main_var in pruned:
                        pruned[main_var].append(assignment)
                    else:
                        pruned[main_var] = [assignment]
            main_var.prune(to_prune)
        find_isolated_path(
            variables,
            pseudo_assignment,
            adj_index[path_dir],
            (path_dir + 2) % 4,
            pruned,
        )
