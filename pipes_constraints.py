from csp import *

def has_connection(vars: list[Variable]) -> bool:
    '''
    checks if a pipe has at least one connection with another pipe
    vars[0]: pipe above
    vars[1]: pipe to the left
    vars[2]: pipe below
    vars[3]: pipe to the right
    vars[4]: main pipe, the one that is being checked for constraints
    '''
    # get the assignment of the main pipe
    type = vars[4].get_assignment()
    for i in range(len(type)):
        # iterate through the surrounding pipes
        if type[i]:
            if vars[i] and vars[i].assignment is not None:
                # get the type of the corresponding pipe
                adj_type = vars[i].get_assignment()
                # check if the adjacent pipe has a connection to the current pipe
                if adj_type[(i + 2) % 4]:
                    return True
    return False

def not_blocked(vars: list[Variable]) -> bool:
    '''
    ensures that the main pipe is not blocked by another pipe
    vars[0]: pipe above
    vars[1]: pipe to the left
    vars[2]: pipe below
    vars[3]: pipe to the right
    vars[4]: main pipe, the one that is being checked
    '''
    # get the assignment of the main pipe
    type = vars[4].get_assignment()
    for i in range(len(type)):
        # iterate through the surrounding pipes
        if type[i]:
            if vars[i] and vars[i].assignment is not None:
                # get the type of the corresponding pipe
                adj_type = vars[i].get_assignment()
                # check if the adjacent pipe is blocking the current pipe
                if not adj_type[(i + 2) % 4]:
                    return False
    return True