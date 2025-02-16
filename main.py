from csp import Variable, Constraint, CSP, DomainGenerator, find_adj
from pipes_constraints import connectivity_pruner_bottom, connectivity_pruner_bottom_left, connectivity_pruner_bottom_right, connectivity_pruner_left, connectivity_pruner_mid, connectivity_pruner_right, connectivity_pruner_top, connectivity_pruner_top_left, connectivity_pruner_top_right, has_connection_bottom, has_connection_bottom_left, has_connection_bottom_right, has_connection_left, has_connection_mid, has_connection_right, has_connection_top, has_connection_top_left, has_connection_top_right, not_blocked_h, not_blocked_pruner_h, not_blocked_pruner_v, not_blocked_v
from pipe_typings import Assignment
from tree import validator as tree_validator, pruner as tree_pruner
from connected import validator as connected_validator, pruner as connected_pruner


n = 4
variables: list[Variable] = []

# initialize variable objects
for i in range(n):
    row: list[Variable] = []
    for j in range(n):
        top = i == 0
        bottom = i == n - 1
        left = j == 0
        right = j == n - 1
        var = Variable(
            location=i * n + j,
            domain=DomainGenerator.generate_domain(top, right, bottom, left),
        )
        row.append(var)
    variables += row

all_cons: list[Constraint] = []
# create constraints for connectivity
connectivity_cons: list[Constraint] = []
# there is one connectivity constraint for each variable
for i in range(len(variables)):
    adj: list[int] = list(find_adj(i, n))
    # adj_variables holds the variables for the connectivity constraint
    adj_variables: list[Variable] = []
    for dir in range(4):
        if adj[dir] != -1:
            adj_variables.append(variables[adj[dir]])
    con_vars: list[Variable] = adj_variables + [variables[i]]
    name = f"connectivity {i}"
    if i == 0:
        # top left corner
        connectivity_cons.append(
            Constraint(
                name, has_connection_top_left, connectivity_pruner_top_left, con_vars
            )
        )
    elif i == n - 1:
        # top right corner
        connectivity_cons.append(
            Constraint(
                name, has_connection_top_right, connectivity_pruner_top_right, con_vars
            )
        )
    elif i == n * (n - 1):
        # bottom left corner
        connectivity_cons.append(
            Constraint(
                name,
                has_connection_bottom_left,
                connectivity_pruner_bottom_left,
                con_vars,
            )
        )
    elif i == n * n - 1:
        # bottom right corner
        connectivity_cons.append(
            Constraint(
                name,
                has_connection_bottom_right,
                connectivity_pruner_bottom_right,
                con_vars,
            )
        )
    elif i < n - 1:
        # top row
        connectivity_cons.append(
            Constraint(name, has_connection_top, connectivity_pruner_top, con_vars)
        )
    elif i % n == 0:
        # left column
        connectivity_cons.append(
            Constraint(name, has_connection_left, connectivity_pruner_left, con_vars)
        )
    elif i % n == n - 1:
        # right column
        connectivity_cons.append(
            Constraint(name, has_connection_right, connectivity_pruner_right, con_vars)
        )
    elif i > n * (n - 1):
        # bottom row
        connectivity_cons.append(
            Constraint(
                name, has_connection_bottom, connectivity_pruner_bottom, con_vars
            )
        )
    else:
        # middle
        connectivity_cons.append(
            Constraint(name, has_connection_mid, connectivity_pruner_mid, con_vars)
        )

# create binary constraints for no blocking
no_blocking_cons: list[Constraint] = []
# start with horizontal cons
# there are (n - 1)*n constraints, each row has n-1 pairs of adjacent variables and there are n rows.
no_blocking_h: list[Constraint] = []
for i in range(n):
    # i represents a row
    for j in range(n - 1):
        # j represents a column
        # get the variable and its right neighbour
        left = variables[i * n + j]
        right = variables[i * n + j + 1]
        scope = [left, right]
        name = f"no blocking horizontal {i * n + j, i * n + j + 1}"
        no_blocking_h.append(
            Constraint(name, not_blocked_h, not_blocked_pruner_h, scope)
        )

# vertical cons
no_blocking_v: list[Constraint] = []
for i in range(n - 1):
    for j in range(n):
        above = variables[i * n + j]
        below = variables[(i + 1) * n + j]
        scope = [above, below]
        name = f"no blocking vertical {i * n + j, (i + 1) * n + j}"
        no_blocking_v.append(
            Constraint(name, not_blocked_v, not_blocked_pruner_v, scope)
        )

# add cons
no_blocking_cons += no_blocking_h + no_blocking_v

# create tree constraint
tree_con: Constraint = Constraint("tree", tree_validator, tree_pruner, variables)

connected_con: Constraint = Constraint(
    "connected", connected_validator, connected_pruner, variables
)
all_cons = connectivity_cons + no_blocking_cons + [tree_con, connected_con]


# create csp
csp = CSP("Sewage pAIpes", variables, all_cons)
solutions: list[Assignment] = []
csp.gac_all(solutions)