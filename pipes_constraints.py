from csp import *
from pipes_utils import *

def has_connection(
    pipes: list[PipeType]
) -> bool:
    """
    checks if a pipe has at least one connection with another pipe
    vars[0]: pipe above
    vars[1]: pipe to the left
    vars[2]: pipe below
    vars[3]: pipe to the right
    vars[4]: main pipe, the one that is being checked for constraints
    :param pipes: list of variables (or None) of length 5. The last variable in the list is the "main" variable, and the four other variables in the list are the variables that are adjacent to the "main" variable
    """
    adj = (pipes[0], pipes[1], pipes[2], pipes[3])
    connections = check_connections(pipes[4], adj)
    for c in connections:
        if c:
            return True
    return False


def not_blocked_h(
    pipes: list[PipeType]
) -> bool:
    """
    Ensures that two horizontally-adjacent pipes are not blocking each other

    :param pipes: a tuple of two pipes. pipes[0] is the one on the left, pipes[1] is the one on the right.
    """
    left = pipes[0]
    right = pipes[1]
    # check if the left pipe's right opening is the same as the right pipe's left opening
    if left[1] != right[3]:
        return False
    return True

def not_blocked_v(
    pipes: list[PipeType]
) -> bool:
    """
    Ensures that two vertically-adjacent pipes are not blocking each other

    :param pipes: a tuple of two pipes. pipes[0] is the one above, pipes[1] is the one below.
    """
    above = pipes[0]
    below = pipes[1]
    # check if the top pipe's bottom opening is the same as the bottom pipe's top opening
    if above[2] != below[0]:
        return False
    return True

def not_blocked_pruner_h(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 2 variables that would result in one of the pipes being blocked by an exit of another pipe

    :params pipes: tuple of two pipes where pipes[0] is to the left of pipes[1]
    :returns: A list of the active domains of the pipes prior to pruning
    """
    left = pipes[0]
    right = pipes[1]

    left_assignment = left.get_assignment()
    right_assignment = right.get_assignment()

    to_prune: dict[Variable, list[PipeType]] = {}
    if left_assignment is not None and right_assignment is None:
        for pipe_type in right.get_active_domain():
            # there is a path to the right pipe, prune all PipeTypes for the right pipe where the pipe doesn't connect with the left
            # or
            # there is no path to the right pipe, prune all PipeTypes for the right pipe where the pipe tries to connect with the left pipe
            if left_assignment[1] != pipe_type[3]:
                if right in to_prune:
                    if pipe_type not in to_prune[right]:
                        to_prune[right].append(pipe_type)
                else:
                    to_prune[right] = [pipe_type]

    elif right_assignment is not None and left_assignment is None:
        for pipe_type in left.get_active_domain():
            # there is a path to the left pipe, prune all PipeTypes for the left pipe where the pipe doesn't connect with the right
            # or
            # there is no path to the left pipe, prune all PipeTypes for the left pipe where the pipe tries to connect with the right pipe
            if right_assignment[3] != pipe_type[1]:
                if left in to_prune:
                    if pipe_type not in to_prune[left]:
                        to_prune[left].append(pipe_type)
                else:
                    to_prune[left] = [pipe_type]

    # if there are no assignments for either pipe, nothing should be pruned.
    # if both pipes are assigned, don't prune
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def not_blocked_pruner_v(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 2 variables that would result in one of the pipes being blocked by an exit of another pipe

    :params pipes: tuple of two pipes where pipes[0] is above pipes[1]
    :returns: A dict mapping the variables to the values to remove from their active domain
    """
    # above = pipes[0]
    # below = pipes[1]

    # top_assignment = above.get_assignment()
    # bottom_assignment = below.get_assignment()
    # to_prune: dict[Variable, list[PipeType]] = {}
    # if top_assignment is not None:
    #     if top_assignment[2]:
    #         # there is a path to the bottom pipe, prune all the assignments for the bottom pipe where the pipe doesn't connect with the top
    #         for pipe_type in below.get_active_domain():
    #             if not pipe_type[0]:
    #                 to_prune[below].append(pipe_type)
    #     else:
    #         # there is no path to the bottom pipe, prune all assignments for the bottom pipe where the pipe tries to connect with the top
    #         for pipe_type in below.get_active_domain():
    #             if pipe_type[0]:
    #                 to_prune[below].append(pipe_type)

    # elif bottom_assignment is not None:
    #     if bottom_assignment[0]:
    #         # there is a path to the top pipe, prune all the assignments for the top pipe where the pipe doesn't connect with the bottom
    #         for pipe_type in above.active_domain:
    #             if not pipe_type[2]:
    #                 to_prune[above].append(pipe_type)
    #     else:
    #         # there is no path to the top pipe, prune all the assignments for the top pipe where it tries to connect with the bottom pipe
    #         for pipe_type in above.get_active_domain():
    #             if pipe_type[2]:
    #                 to_prune[above].append(pipe_type)

    # # if there are no assignments for either pipe, nothing should be pruned.
    # # if both pipes are assigned, don't prune
    # return to_prune

    top = pipes[0]
    bottom = pipes[1]

    top_assignment = top.get_assignment()
    bottom_assignment = bottom.get_assignment()

    to_prune: dict[Variable, list[PipeType]] = {}
    if top_assignment is not None and bottom_assignment is None:
        for pipe_type in bottom.get_active_domain():
            # there is a path to the bottom pipe, prune all PipeTypes for the bottom pipe where the pipe doesn't connect with the top
            # or
            # there is no path to the bottom pipe, prune all PipeTypes for the bottom pipe where the pipe tries to connect with the top pipe
            if top_assignment[2] != pipe_type[0]:
                if bottom in to_prune:
                    if pipe_type not in to_prune[bottom]:
                        to_prune[bottom].append(pipe_type)
                else:
                    to_prune[bottom] = [pipe_type]

    elif bottom_assignment is not None and top_assignment is None:
        for pipe_type in top.get_active_domain():
            # there is a path to the top pipe, prune all PipeTypes for the top pipe where the pipe doesn't connect with the bottom
            # or
            # there is no path to the top pipe, prune all PipeTypes for the top pipe where the pipe tries to connect with the bottom pipe
            if bottom_assignment[0] != pipe_type[2]:
                if top in to_prune:
                    if pipe_type not in to_prune[top]:
                        to_prune[top].append(pipe_type)
                else:
                    to_prune[top] = [pipe_type]
                    
    # if there are no assignments for either pipe, nothing should be pruned.
    # if both pipes are assigned, don't prune
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def connectivity_pruner_mid(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 5 variables for which assigning this value prevents a valid solution to the constraint

    :params pipes: A list of 5 pipe variables, where the last one is the center pipe and the other four pipes are adjacent to the center, with the top one in index 0 and going clockwise from there.
    :returns: a dict mapping variables to the values to remove from their active domain
    """
    center = pipes[4]
    adj = pipes[:4]
    
    center_val = center.get_assignment()
    adj_vals = (
        adj[0].get_assignment(),
        adj[1].get_assignment(),
        adj[2].get_assignment(),
        adj[3].get_assignment()
    )
    # if the center one is unassigned, prune values that don't point to an empty pipe or to a pipe that has a half-connection with the center pipe
    to_prune: dict[Variable, list[PipeType]] = {}
    if center_val is None:
        # check which adjacent pipes are unassigned, or are assigned with a connection to the center pipe
        # the center has to have an outgoing connection to one of the pipes in that list
        # if the center has a pipetype in the active domain that doesn't connect to one of those pipes, remove it from the active domain
        at_least_one_connection: list[int] = []
        for i in range(4):
            this_adj_pipe_val = adj[i].get_assignment()
            if this_adj_pipe_val is None or this_adj_pipe_val[(i + 2) % 4]:
                at_least_one_connection.append(i)
        # remove pipetypes from center where there is no connection with any of the pipes in at_least_one_connection
        # in other words, if all the values in at_least_one_connection are false, prune the pipetype from the domain
        for pipe_type in center.get_active_domain():
            prune_this_type = True
            for dir in at_least_one_connection:
                if pipe_type[dir]:
                    prune_this_type = False
                    break
            if prune_this_type:
                if center in to_prune:
                    to_prune[center].append(pipe_type)
                else:
                    to_prune[center] = [pipe_type]
        # if the center is unassigned, and must connect to one specific unassigned edge, then prune values from that unassigned edge that don't connect to the center
        # in any other situation where the center is unassigned, nothing can be said about the active domains of the adjacent variables
        one_dir = adj[at_least_one_connection[0]]
        if len(at_least_one_connection) == 1 and one_dir.get_assignment() is None:
            dir = at_least_one_connection[0]
            for pipe_type in one_dir.get_active_domain():
                if not pipe_type[(dir + 2) % 4]:
                    if one_dir in to_prune:
                        to_prune[one_dir].append(pipe_type)
                    else:
                        to_prune[one_dir] = [pipe_type]
    else:
        # if an edge is unassigned but the center is assigned, has no connection, and can only connect with this unassigned edge (i.e. there are no other unassigned edges), prune values from this edge that don't connect with the center
        connections = check_connections(center_val, adj_vals)
        num_connections = sum(connections)
        if num_connections == 0:
            # there are no connections
            # check if there is one unassigned adjacent variable
            num_unassigned = 0
            dir = 0
            for cur_adj_dir in range(4):
                cur_adj_val = adj_vals[cur_adj_dir]
                if cur_adj_val is None:
                    num_unassigned += 1
                    dir = cur_adj_dir
            if num_unassigned == 1:
                # there is one unassigned adjacent variable, and it is in direction dir
                # prune all the values from this variable that don't connect with the center
                for pipe_type in adj[dir].get_active_domain():
                    if not pipe_type[(dir + 2) % 4]:
                        if adj[dir] in to_prune:
                            to_prune[adj[dir]].append(pipe_type)
                        else:
                            to_prune[adj[dir]] = [pipe_type]
        # if more than one edge is unassigned, nothing can be said about what should be removed from the active domain of the edges.
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def connectivity_pruner_top(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 4 variables for which assigning this value prevents a valid solution to the constraint

    :params pipes: A list of 4 pipe variables, where the last one is the center pipe and the other three pipes are adjacent to the center, with the right one in index 0 and going clockwise from there. The center variable is on the top row, meaning there is no variable above it.
    :returns: a dict mapping variables to the values to remove from their active domain
    """
    center = pipes[3]
    adj = [pipes[0]] + pipes[:3]
    
    center_val = center.get_assignment()
    # the top pipe is absent, so adj_vals[0] should not be a real assignment and should instead indicate that there is no variable here
    adj_vals = (
        (False, False, False, False),
        adj[0].get_assignment(),
        adj[1].get_assignment(),
        adj[2].get_assignment()
    )
    # if the center one is unassigned, prune values that don't point to an empty pipe or to a pipe that has a half-connection with the center pipe
    to_prune: dict[Variable, list[PipeType]] = {}
    if center_val is None:
        # check which adjacent pipes are unassigned, or are assigned with a connection to the center pipe
        # the center has to have an outgoing connection to one of the pipes in that list
        # if the center has a pipetype in the active domain that doesn't connect to one of those pipes, remove it from the active domain
        at_least_one_connection: list[int] = []
        for i in range(1, 4):
            this_adj_pipe_val = adj[i].get_assignment()
            if this_adj_pipe_val is None or this_adj_pipe_val[(i + 2) % 4]:
                at_least_one_connection.append(i)
        # remove pipetypes from center where there is no connection with any of the pipes in at_least_one_connection
        # in other words, if all the values in at_least_one_connection are false, prune the pipetype from the domain
        for pipe_type in center.get_active_domain():
            prune_this_type = True
            for dir in at_least_one_connection:
                if pipe_type[dir]:
                    prune_this_type = False
                    break
            if prune_this_type:
                if center in to_prune:
                    to_prune[center].append(pipe_type)
                else:
                    to_prune[center] = [pipe_type]
        # if the center is unassigned, and must connect to one specific unassigned edge, then prune values from that unassigned edge that don't connect to the center
        # in any other situation where the center is unassigned, nothing can be said about the active domains of the adjacent variables
        one_dir = adj[at_least_one_connection[0]]
        if len(at_least_one_connection) == 1 and one_dir.get_assignment() is None:
            dir = at_least_one_connection[0]
            for pipe_type in one_dir.get_active_domain():
                if not pipe_type[(dir + 2) % 4]:
                    if one_dir in to_prune:
                        to_prune[one_dir].append(pipe_type)
                    else:
                        to_prune[one_dir] = [pipe_type]
    else:
        # if an edge is unassigned but the center is assigned, has no connection, and can only connect with this unassigned edge (i.e. there are no other unassigned edges), prune values from this edge that don't connect with the center
        connections = check_connections(center_val, adj_vals)
        num_connections = sum(connections)
        if num_connections == 0:
            # there are no connections
            # check if there is one unassigned adjacent variable
            num_unassigned = 0
            dir = 0
            for cur_adj_dir in range(1, 4):
                cur_adj_val = adj_vals[cur_adj_dir]
                if cur_adj_val is None:
                    num_unassigned += 1
                    dir = cur_adj_dir
            if num_unassigned == 1:
                # there is one unassigned adjacent variable, and it is in direction dir
                # prune all the values from this variable that don't connect with the center
                for pipe_type in adj[dir].get_active_domain():
                    if not pipe_type[(dir + 2) % 4]:
                        if adj[dir] in to_prune:
                            to_prune[adj[dir]].append(pipe_type)
                        else:
                            to_prune[adj[dir]] = [pipe_type]
        # if more than one edge is unassigned, nothing can be said about what should be removed from the active domain of the edges.
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def connectivity_pruner_right(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 4 variables for which assigning this value prevents a valid solution to the constraint

    :params pipes: A list of 4 pipe variables, where the last one is the center pipe and the other three pipes are adjacent to the center, with the top one in index 0 and going clockwise from there. The center variable is on the right column, meaning there is no variable to the right of it.
    :returns: a dict mapping variables to the values to remove from their active domain
    """
    center = pipes[3]
    adj = [pipes[0]] + [pipes[0]] + pipes[1:3]
    
    center_val = center.get_assignment()
    # the right pipe is absent, so adj_vals[1] should not be a real assignment and should instead indicate that there is no variable here
    adj_vals = (
        adj[0].get_assignment(),
        (False, False, False, False),
        adj[1].get_assignment(),
        adj[2].get_assignment()
    )
    # if the center one is unassigned, prune values that don't point to an empty pipe or to a pipe that has a half-connection with the center pipe
    to_prune: dict[Variable, list[PipeType]] = {}
    if center_val is None:
        # check which adjacent pipes are unassigned, or are assigned with a connection to the center pipe
        # the center has to have an outgoing connection to one of the pipes in that list
        # if the center has a pipetype in the active domain that doesn't connect to one of those pipes, remove it from the active domain
        at_least_one_connection: list[int] = []
        for i in [0, 2, 3]:
            this_adj_pipe_val = adj[i].get_assignment()
            if this_adj_pipe_val is None or this_adj_pipe_val[(i + 2) % 4]:
                at_least_one_connection.append(i)
        # remove pipetypes from center where there is no connection with any of the pipes in at_least_one_connection
        # in other words, if all the values in at_least_one_connection are false, prune the pipetype from the domain
        for pipe_type in center.get_active_domain():
            prune_this_type = True
            for dir in at_least_one_connection:
                if pipe_type[dir]:
                    prune_this_type = False
                    break
            if prune_this_type:
                if center in to_prune:
                    to_prune[center].append(pipe_type)
                else:
                    to_prune[center] = [pipe_type]
        # if the center is unassigned, and must connect to one specific unassigned edge, then prune values from that unassigned edge that don't connect to the center
        # in any other situation where the center is unassigned, nothing can be said about the active domains of the adjacent variables
        one_dir = adj[at_least_one_connection[0]]
        if len(at_least_one_connection) == 1 and one_dir.get_assignment() is None:
            dir = at_least_one_connection[0]
            for pipe_type in one_dir.get_active_domain():
                if not pipe_type[(dir + 2) % 4]:
                    if one_dir in to_prune:
                        to_prune[one_dir].append(pipe_type)
                    else:
                        to_prune[one_dir] = [pipe_type]
    else:
        # if an edge is unassigned but the center is assigned, has no connection, and can only connect with this unassigned edge (i.e. there are no other unassigned edges), prune values from this edge that don't connect with the center
        connections = check_connections(center_val, adj_vals)
        num_connections = sum(connections)
        if num_connections == 0:
            # there are no connections
            # check if there is one unassigned adjacent variable
            num_unassigned = 0
            dir = 0
            for cur_adj_dir in [0, 2, 3]:
                cur_adj_val = adj_vals[cur_adj_dir]
                if cur_adj_val is None:
                    num_unassigned += 1
                    dir = cur_adj_dir
            if num_unassigned == 1:
                # there is one unassigned adjacent variable, and it is in direction dir
                # prune all the values from this variable that don't connect with the center
                for pipe_type in adj[dir].get_active_domain():
                    if not pipe_type[(dir + 2) % 4]:
                        if adj[dir] in to_prune:
                            to_prune[adj[dir]].append(pipe_type)
                        else:
                            to_prune[adj[dir]] = [pipe_type]
        # if more than one edge is unassigned, nothing can be said about what should be removed from the active domain of the edges.
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def connectivity_pruner_bottom(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 4 variables for which assigning this value prevents a valid solution to the constraint

    :params pipes: A list of 4 pipe variables, where the last one is the center pipe and the other three pipes are adjacent to the center, with the top one in index 0 and going clockwise from there. The center variable is on the bottom row, meaning there is no variable below it.
    :returns: a dict mapping variables to the values to remove from their active domain
    """
    center = pipes[3]
    adj = pipes[:2] + [pipes[0]] + [pipes[3]]
    
    center_val = center.get_assignment()
    # the bottom pipe is absent, so adj_vals[2] should not be a real assignment and should instead indicate that there is no variable here
    adj_vals = (
        adj[0].get_assignment(),
        adj[1].get_assignment(),
        (False, False, False, False),
        adj[2].get_assignment()
    )
    # if the center one is unassigned, prune values that don't point to an empty pipe or to a pipe that has a half-connection with the center pipe
    to_prune: dict[Variable, list[PipeType]] = {}
    if center_val is None:
        # check which adjacent pipes are unassigned, or are assigned with a connection to the center pipe
        # the center has to have an outgoing connection to one of the pipes in that list
        # if the center has a pipetype in the active domain that doesn't connect to one of those pipes, remove it from the active domain
        at_least_one_connection: list[int] = []
        for i in [0, 1, 3]:
            this_adj_pipe_val = adj[i].get_assignment()
            if this_adj_pipe_val is None or this_adj_pipe_val[(i + 2) % 4]:
                at_least_one_connection.append(i)
        # remove pipetypes from center where there is no connection with any of the pipes in at_least_one_connection
        # in other words, if all the values in at_least_one_connection are false, prune the pipetype from the domain
        for pipe_type in center.get_active_domain():
            prune_this_type = True
            for dir in at_least_one_connection:
                if pipe_type[dir]:
                    prune_this_type = False
                    break
            if prune_this_type:
                if center in to_prune:
                    to_prune[center].append(pipe_type)
                else:
                    to_prune[center] = [pipe_type]
        # if the center is unassigned, and must connect to one specific unassigned edge, then prune values from that unassigned edge that don't connect to the center
        # in any other situation where the center is unassigned, nothing can be said about the active domains of the adjacent variables
        one_dir = adj[at_least_one_connection[0]]
        if len(at_least_one_connection) == 1 and one_dir.get_assignment() is None:
            dir = at_least_one_connection[0]
            for pipe_type in one_dir.get_active_domain():
                if not pipe_type[(dir + 2) % 4]:
                    if one_dir in to_prune:
                        to_prune[one_dir].append(pipe_type)
                    else:
                        to_prune[one_dir] = [pipe_type]
    else:
        # if an edge is unassigned but the center is assigned, has no connection, and can only connect with this unassigned edge (i.e. there are no other unassigned edges), prune values from this edge that don't connect with the center
        connections = check_connections(center_val, adj_vals)
        num_connections = sum(connections)
        if num_connections == 0:
            # there are no connections
            # check if there is one unassigned adjacent variable
            num_unassigned = 0
            dir = 0
            for cur_adj_dir in [0, 1, 3]:
                cur_adj_val = adj_vals[cur_adj_dir]
                if cur_adj_val is None:
                    num_unassigned += 1
                    dir = cur_adj_dir
            if num_unassigned == 1:
                # there is one unassigned adjacent variable, and it is in direction dir
                # prune all the values from this variable that don't connect with the center
                for pipe_type in adj[dir].get_active_domain():
                    if not pipe_type[(dir + 2) % 4]:
                        if adj[dir] in to_prune:
                            to_prune[adj[dir]].append(pipe_type)
                        else:
                            to_prune[adj[dir]] = [pipe_type]
        # if more than one edge is unassigned, nothing can be said about what should be removed from the active domain of the edges.
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def connectivity_pruner_left(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 4 variables for which assigning this value prevents a valid solution to the constraint

    :params pipes: A list of 4 pipe variables, where the last one is the center pipe and the other three pipes are adjacent to the center, with the top one in index 0 and going clockwise from there. The center variable is on the left column, meaning there is no variable to the left of it.
    :returns: a dict mapping variables to the values to remove from their active domain
    """
    center = pipes[3]
    adj = pipes[:3] + [pipes[0]]
    
    center_val = center.get_assignment()
    # the left pipe is absent, so adj_vals[3] should not be a real assignment and should instead indicate that there is no variable here
    adj_vals = (
        adj[0].get_assignment(),
        adj[1].get_assignment(),
        adj[2].get_assignment(),
        (False, False, False, False)
    )
    # if the center one is unassigned, prune values that don't point to an empty pipe or to a pipe that has a half-connection with the center pipe
    to_prune: dict[Variable, list[PipeType]] = {}
    if center_val is None:
        # check which adjacent pipes are unassigned, or are assigned with a connection to the center pipe
        # the center has to have an outgoing connection to one of the pipes in that list
        # if the center has a pipetype in the active domain that doesn't connect to one of those pipes, remove it from the active domain
        at_least_one_connection: list[int] = []
        for i in range(3):
            this_adj_pipe_val = adj[i].get_assignment()
            if this_adj_pipe_val is None or this_adj_pipe_val[(i + 2) % 4]:
                at_least_one_connection.append(i)
        # remove pipetypes from center where there is no connection with any of the pipes in at_least_one_connection
        # in other words, if all the values in at_least_one_connection are false, prune the pipetype from the domain
        for pipe_type in center.get_active_domain():
            prune_this_type = True
            for dir in at_least_one_connection:
                if pipe_type[dir]:
                    prune_this_type = False
                    break
            if prune_this_type:
                if center in to_prune:
                    to_prune[center].append(pipe_type)
                else:
                    to_prune[center] = [pipe_type]
        # if the center is unassigned, and must connect to one specific unassigned edge, then prune values from that unassigned edge that don't connect to the center
        # in any other situation where the center is unassigned, nothing can be said about the active domains of the adjacent variables
        one_dir = adj[at_least_one_connection[0]]
        if len(at_least_one_connection) == 1 and one_dir.get_assignment() is None:
            dir = at_least_one_connection[0]
            for pipe_type in one_dir.get_active_domain():
                if not pipe_type[(dir + 2) % 4]:
                    if one_dir in to_prune:
                        to_prune[one_dir].append(pipe_type)
                    else:
                        to_prune[one_dir] = [pipe_type]
    else:
        # if an edge is unassigned but the center is assigned, has no connection, and can only connect with this unassigned edge (i.e. there are no other unassigned edges), prune values from this edge that don't connect with the center
        connections = check_connections(center_val, adj_vals)
        num_connections = sum(connections)
        if num_connections == 0:
            # there are no connections
            # check if there is one unassigned adjacent variable
            num_unassigned = 0
            dir = 0
            for cur_adj_dir in range(3):
                cur_adj_val = adj_vals[cur_adj_dir]
                if cur_adj_val is None:
                    num_unassigned += 1
                    dir = cur_adj_dir
            if num_unassigned == 1:
                # there is one unassigned adjacent variable, and it is in direction dir
                # prune all the values from this variable that don't connect with the center
                for pipe_type in adj[dir].get_active_domain():
                    if not pipe_type[(dir + 2) % 4]:
                        if adj[dir] in to_prune:
                            to_prune[adj[dir]].append(pipe_type)
                        else:
                            to_prune[adj[dir]] = [pipe_type]
        # if more than one edge is unassigned, nothing can be said about what should be removed from the active domain of the edges.
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def connectivity_pruner_top_right(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 3 variables for which assigning this value prevents a valid solution to the constraint

    :params pipes: A list of 3 pipe variables, where the last one is the center pipe and the other three pipes are adjacent to the center, with the bottom one in index 0 and going clockwise from there. The center variable is in the top right corner, meaning there is no variable above or to the right of it.
    :returns: a dict mapping variables to the values to remove from their active domain
    """
    center = pipes[2]
    adj = [pipes[0]] + [pipes[0]] + pipes[:2]
    
    center_val = center.get_assignment()
    # the top and right pipes are absent, so adj_vals[0, 1] should not be real assignments and should instead indicate that there is no variable here
    adj_vals = (
        (False, False, False, False),
        (False, False, False, False),
        adj[0].get_assignment(),
        adj[1].get_assignment()
    )
    # if the center one is unassigned, prune values that don't point to an empty pipe or to a pipe that has a half-connection with the center pipe
    to_prune: dict[Variable, list[PipeType]] = {}
    if center_val is None:
        # check which adjacent pipes are unassigned, or are assigned with a connection to the center pipe
        # the center has to have an outgoing connection to one of the pipes in that list
        # if the center has a pipetype in the active domain that doesn't connect to one of those pipes, remove it from the active domain
        at_least_one_connection: list[int] = []
        for i in [2, 3]:
            this_adj_pipe_val = adj[i].get_assignment()
            if this_adj_pipe_val is None or this_adj_pipe_val[(i + 2) % 4]:
                at_least_one_connection.append(i)
        # remove pipetypes from center where there is no connection with any of the pipes in at_least_one_connection
        # in other words, if all the values in at_least_one_connection are false, prune the pipetype from the domain
        for pipe_type in center.get_active_domain():
            prune_this_type = True
            for dir in at_least_one_connection:
                if pipe_type[dir]:
                    prune_this_type = False
                    break
            if prune_this_type:
                if center in to_prune:
                    to_prune[center].append(pipe_type)
                else:
                    to_prune[center] = [pipe_type]
        # if the center is unassigned, and must connect to one specific unassigned edge, then prune values from that unassigned edge that don't connect to the center
        # in any other situation where the center is unassigned, nothing can be said about the active domains of the adjacent variables
        one_dir = adj[at_least_one_connection[0]]
        if len(at_least_one_connection) == 1 and one_dir.get_assignment() is None:
            dir = at_least_one_connection[0]
            for pipe_type in one_dir.get_active_domain():
                if not pipe_type[(dir + 2) % 4]:
                    if one_dir in to_prune:
                        to_prune[one_dir].append(pipe_type)
                    else:
                        to_prune[one_dir] = [pipe_type]
    else:
        # if an edge is unassigned but the center is assigned, has no connection, and can only connect with this unassigned edge (i.e. there are no other unassigned edges), prune values from this edge that don't connect with the center
        connections = check_connections(center_val, adj_vals)
        num_connections = sum(connections)
        if num_connections == 0:
            # there are no connections
            # check if there is one unassigned adjacent variable
            num_unassigned = 0
            dir = 0
            for cur_adj_dir in [2, 3]:
                cur_adj_val = adj_vals[cur_adj_dir]
                if cur_adj_val is None:
                    num_unassigned += 1
                    dir = cur_adj_dir
            if num_unassigned == 1:
                # there is one unassigned adjacent variable, and it is in direction dir
                # prune all the values from this variable that don't connect with the center
                for pipe_type in adj[dir].get_active_domain():
                    if not pipe_type[(dir + 2) % 4]:
                        if adj[dir] in to_prune:
                            to_prune[adj[dir]].append(pipe_type)
                        else:
                            to_prune[adj[dir]] = [pipe_type]
        # if more than one edge is unassigned, nothing can be said about what should be removed from the active domain of the edges.
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def connectivity_pruner_bottom_right(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 3 variables for which assigning this value prevents a valid solution to the constraint

    :params pipes: A list of 3 pipe variables, where the last one is the center pipe and the other three pipes are adjacent to the center, with the top one in index 0 and going clockwise from there. The center variable is in the bottom right corner, meaning there is no variable below or to the right of it.
    :returns: a dict mapping variables to the values to remove from their active domain
    """
    center = pipes[2]
    adj = [pipes[0], pipes[0], pipes[0], pipes[1]]
    
    center_val = center.get_assignment()
    # the bottom and right pipes are absent, so adj_vals[1, 2] should not be real assignments and should instead indicate that there is no variable here
    adj_vals = (
        adj[0].get_assignment(),
        (False, False, False, False),
        (False, False, False, False),
        adj[1].get_assignment()
    )
    # if the center one is unassigned, prune values that don't point to an empty pipe or to a pipe that has a half-connection with the center pipe
    to_prune: dict[Variable, list[PipeType]] = {}
    if center_val is None:
        # check which adjacent pipes are unassigned, or are assigned with a connection to the center pipe
        # the center has to have an outgoing connection to one of the pipes in that list
        # if the center has a pipetype in the active domain that doesn't connect to one of those pipes, remove it from the active domain
        at_least_one_connection: list[int] = []
        for i in [0, 3]:
            this_adj_pipe_val = adj[i].get_assignment()
            if this_adj_pipe_val is None or this_adj_pipe_val[(i + 2) % 4]:
                at_least_one_connection.append(i)
        # remove pipetypes from center where there is no connection with any of the pipes in at_least_one_connection
        # in other words, if all the values in at_least_one_connection are false, prune the pipetype from the domain
        for pipe_type in center.get_active_domain():
            prune_this_type = True
            for dir in at_least_one_connection:
                if pipe_type[dir]:
                    prune_this_type = False
                    break
            if prune_this_type:
                if center in to_prune:
                    to_prune[center].append(pipe_type)
                else:
                    to_prune[center] = [pipe_type]
        # if the center is unassigned, and must connect to one specific unassigned edge, then prune values from that unassigned edge that don't connect to the center
        # in any other situation where the center is unassigned, nothing can be said about the active domains of the adjacent variables
        one_dir = adj[at_least_one_connection[0]]
        if len(at_least_one_connection) == 1 and one_dir.get_assignment() is None:
            dir = at_least_one_connection[0]
            for pipe_type in one_dir.get_active_domain():
                if not pipe_type[(dir + 2) % 4]:
                    if one_dir in to_prune:
                        to_prune[one_dir].append(pipe_type)
                    else:
                        to_prune[one_dir] = [pipe_type]
    else:
        # if an edge is unassigned but the center is assigned, has no connection, and can only connect with this unassigned edge (i.e. there are no other unassigned edges), prune values from this edge that don't connect with the center
        connections = check_connections(center_val, adj_vals)
        num_connections = sum(connections)
        if num_connections == 0:
            # there are no connections
            # check if there is one unassigned adjacent variable
            num_unassigned = 0
            dir = 0
            for cur_adj_dir in [0, 3]:
                cur_adj_val = adj_vals[cur_adj_dir]
                if cur_adj_val is None:
                    num_unassigned += 1
                    dir = cur_adj_dir
            if num_unassigned == 1:
                # there is one unassigned adjacent variable, and it is in direction dir
                # prune all the values from this variable that don't connect with the center
                for pipe_type in adj[dir].get_active_domain():
                    if not pipe_type[(dir + 2) % 4]:
                        if adj[dir] in to_prune:
                            to_prune[adj[dir]].append(pipe_type)
                        else:
                            to_prune[adj[dir]] = [pipe_type]
        # if more than one edge is unassigned, nothing can be said about what should be removed from the active domain of the edges.
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def connectivity_pruner_bottom_left(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 3 variables for which assigning this value prevents a valid solution to the constraint

    :params pipes: A list of 3 pipe variables, where the last one is the center pipe and the other three pipes are adjacent to the center, with the top one in index 0 and going clockwise from there. The center variable is in the bottom left corner, meaning there is no variable below or to the left of it.
    :returns: a dict mapping variables to the values to remove from their active domain
    """
    center = pipes[2]
    adj = pipes[:2] + [pipes[0]] + [pipes[0]]
    
    center_val = center.get_assignment()
    # the bottom and left pipes are absent, so adj_vals[2, 3] should not be real assignments and should instead indicate that there is no variable here
    adj_vals = (
        adj[0].get_assignment(),
        adj[1].get_assignment(),
        (False, False, False, False),
        (False, False, False, False)
    )
    # if the center one is unassigned, prune values that don't point to an empty pipe or to a pipe that has a half-connection with the center pipe
    to_prune: dict[Variable, list[PipeType]] = {}
    if center_val is None:
        # check which adjacent pipes are unassigned, or are assigned with a connection to the center pipe
        # the center has to have an outgoing connection to one of the pipes in that list
        # if the center has a pipetype in the active domain that doesn't connect to one of those pipes, remove it from the active domain
        at_least_one_connection: list[int] = []
        for i in [0, 1]:
            this_adj_pipe_val = adj[i].get_assignment()
            if this_adj_pipe_val is None or this_adj_pipe_val[(i + 2) % 4]:
                at_least_one_connection.append(i)
        # remove pipetypes from center where there is no connection with any of the pipes in at_least_one_connection
        # in other words, if all the values in at_least_one_connection are false, prune the pipetype from the domain
        for pipe_type in center.get_active_domain():
            prune_this_type = True
            for dir in at_least_one_connection:
                if pipe_type[dir]:
                    prune_this_type = False
                    break
            if prune_this_type:
                if center in to_prune:
                    to_prune[center].append(pipe_type)
                else:
                    to_prune[center] = [pipe_type]
        # if the center is unassigned, and must connect to one specific unassigned edge, then prune values from that unassigned edge that don't connect to the center
        # in any other situation where the center is unassigned, nothing can be said about the active domains of the adjacent variables
        one_dir = adj[at_least_one_connection[0]]
        if len(at_least_one_connection) == 1 and one_dir.get_assignment() is None:
            dir = at_least_one_connection[0]
            for pipe_type in one_dir.get_active_domain():
                if not pipe_type[(dir + 2) % 4]:
                    if one_dir in to_prune:
                        to_prune[one_dir].append(pipe_type)
                    else:
                        to_prune[one_dir] = [pipe_type]
    else:
        # if an edge is unassigned but the center is assigned, has no connection, and can only connect with this unassigned edge (i.e. there are no other unassigned edges), prune values from this edge that don't connect with the center
        connections = check_connections(center_val, adj_vals)
        num_connections = sum(connections)
        if num_connections == 0:
            # there are no connections
            # check if there is one unassigned adjacent variable
            num_unassigned = 0
            dir = 0
            for cur_adj_dir in [0, 1]:
                cur_adj_val = adj_vals[cur_adj_dir]
                if cur_adj_val is None:
                    num_unassigned += 1
                    dir = cur_adj_dir
            if num_unassigned == 1:
                # there is one unassigned adjacent variable, and it is in direction dir
                # prune all the values from this variable that don't connect with the center
                for pipe_type in adj[dir].get_active_domain():
                    if not pipe_type[(dir + 2) % 4]:
                        if adj[dir] in to_prune:
                            to_prune[adj[dir]].append(pipe_type)
                        else:
                            to_prune[adj[dir]] = [pipe_type]
        # if more than one edge is unassigned, nothing can be said about what should be removed from the active domain of the edges.
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune

def connectivity_pruner_top_left(
    pipes: list[Variable]
) -> dict[Variable, list[PipeType]]:
    """
    prunes values from 3 variables for which assigning this value prevents a valid solution to the constraint

    :params pipes: A list of 3 pipe variables, where the last one is the center pipe and the other three pipes are adjacent to the center, with the right one in index 0 and going clockwise from there. The center variable is in the top left corner, meaning there is no variable above or to the left of it.
    :returns: a dict mapping variables to the values to remove from their active domain
    """
    center = pipes[2]
    adj = [pipes[0]] + pipes[:2] + [pipes[0]]
    
    center_val = center.get_assignment()
    # the top and left pipes are absent, so adj_vals[0, 3] should not be real assignments and should instead indicate that there is no variable here
    adj_vals = (
        (False, False, False, False),
        adj[0].get_assignment(),
        adj[1].get_assignment(),
        (False, False, False, False)
    )
    # if the center one is unassigned, prune values that don't point to an empty pipe or to a pipe that has a half-connection with the center pipe
    to_prune: dict[Variable, list[PipeType]] = {}
    if center_val is None:
        # check which adjacent pipes are unassigned, or are assigned with a connection to the center pipe
        # the center has to have an outgoing connection to one of the pipes in that list
        # if the center has a pipetype in the active domain that doesn't connect to one of those pipes, remove it from the active domain
        at_least_one_connection: list[int] = []
        for i in [1, 2]:
            this_adj_pipe_val = adj[i].get_assignment()
            if this_adj_pipe_val is None or this_adj_pipe_val[(i + 2) % 4]:
                at_least_one_connection.append(i)
        # remove pipetypes from center where there is no connection with any of the pipes in at_least_one_connection
        # in other words, if all the values in at_least_one_connection are false, prune the pipetype from the domain
        for pipe_type in center.get_active_domain():
            prune_this_type = True
            for dir in at_least_one_connection:
                if pipe_type[dir]:
                    prune_this_type = False
                    break
            if prune_this_type:
                if center in to_prune:
                    to_prune[center].append(pipe_type)
                else:
                    to_prune[center] = [pipe_type]
        # if the center is unassigned, and must connect to one specific unassigned edge, then prune values from that unassigned edge that don't connect to the center
        # in any other situation where the center is unassigned, nothing can be said about the active domains of the adjacent variables
        one_dir = adj[at_least_one_connection[0]]
        if len(at_least_one_connection) == 1 and one_dir.get_assignment() is None:
            dir = at_least_one_connection[0]
            for pipe_type in one_dir.get_active_domain():
                if not pipe_type[(dir + 2) % 4]:
                    if one_dir in to_prune:
                        to_prune[one_dir].append(pipe_type)
                    else:
                        to_prune[one_dir] = [pipe_type]
    else:
        # if an edge is unassigned but the center is assigned, has no connection, and can only connect with this unassigned edge (i.e. there are no other unassigned edges), prune values from this edge that don't connect with the center
        connections = check_connections(center_val, adj_vals)
        num_connections = sum(connections)
        if num_connections == 0:
            # there are no connections
            # check if there is one unassigned adjacent variable
            num_unassigned = 0
            dir = 0
            for cur_adj_dir in [1, 2]:
                cur_adj_val = adj_vals[cur_adj_dir]
                if cur_adj_val is None:
                    num_unassigned += 1
                    dir = cur_adj_dir
            if num_unassigned == 1:
                # there is one unassigned adjacent variable, and it is in direction dir
                # prune all the values from this variable that don't connect with the center
                print(dir)
                for pipe_type in adj[dir].get_active_domain():
                    if not pipe_type[(dir + 2) % 4]:
                        if adj[dir] in to_prune:
                            to_prune[adj[dir]].append(pipe_type)
                        else:
                            to_prune[adj[dir]] = [pipe_type]
        # if more than one edge is unassigned, nothing can be said about what should be removed from the active domain of the edges.
    for var in to_prune:
        var.prune(to_prune[var])
    return to_prune