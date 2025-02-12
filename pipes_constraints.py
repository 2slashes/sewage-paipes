from csp import *

def has_connection(vars: list[Variable]) -> bool:
    '''
    main: the pipe that is being checked to see if there is a connection
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