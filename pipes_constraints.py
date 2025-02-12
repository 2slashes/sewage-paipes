from csp import *
from pipes_utils import *

def has_connection(pipes: tuple[Optional[PipeType], Optional[PipeType], Optional[PipeType], Optional[PipeType], PipeType]) -> bool:
    '''
    checks if a pipe has at least one connection with another pipe
    vars[0]: pipe above
    vars[1]: pipe to the left
    vars[2]: pipe below
    vars[3]: pipe to the right
    vars[4]: main pipe, the one that is being checked for constraints
    :param pipes: list of variables (or None) of length 5. The last variable in the list is the "main" variable, and the four other variables in the list are the variables that are adjacent to the "main" variable
    '''
    # get the assignment of the main pipe
    # pipe = vars[4].get_assignment()
    # for i in range(len(pipe)):
    #     # iterate through the surrounding pipes
    #     if pipe[i]:
    #         if vars[i] and vars[i].assignment is not None:
    #             # get the type of the corresponding pipe
    #             adj_type = vars[i].get_assignment()
    #             # check if the adjacent pipe has a connection to the current pipe
    #             if adj_type[(i + 2) % 4]:
    #                 return True
    # return False
    adj = (pipes[0], pipes[1], pipes[2], pipes[3])
    connections = check_connections(pipes[4], adj)
    for c in connections:
        if c:
            return True
    return False

def not_blocked(pipes: tuple[Optional[PipeType], Optional[PipeType], Optional[PipeType], Optional[PipeType], PipeType]) -> bool:
    '''
    ensures that the main pipe is not blocked by another pipe
    vars[0]: pipe above
    vars[1]: pipe to the left
    vars[2]: pipe below
    vars[3]: pipe to the right
    vars[4]: main pipe, the one that is being checked
    '''
    # get the assignment of the main pipe
    main = pipes[4]
    for i in range(len(main)):
        # iterate through the surrounding pipes
        if main[i]:
            if pipes[i] is not None:
                # get the type of the corresponding pipe
                adj_type = pipes[i]
                # check if the adjacent pipe is blocking the current pipe
                if adj_type is not None and not adj_type[(i + 2) % 4]:
                    return False
    return True