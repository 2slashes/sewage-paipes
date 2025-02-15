from csp import *
from pipes_constraints import *
from pipes_utils import *
from tree import *

n = 5
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
            location=(i, j),
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
    con_vars: list[Variable] = adj_variables + [variables[adj[i]]]
    name = f"connectivity {i}"
    connectivity_cons.append(Constraint(name, has_connection, connectivity_pruner, con_vars))

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
        no_blocking_h.append(Constraint(name, not_blocked_h, not_blocked_pruner_h, scope))

# vertical cons
no_blocking_v: list[Constraint] = []
for i in range(n-1):
    for j in range(n):
        above = variables[i * n + j]
        below = variables[(i + 1) * n + j]
        scope = [above, below]
        name = f"no blocking vertical {i * n + j, (i + 1) * n + j}"
        no_blocking_v.append(Constraint(name, not_blocked_v, not_blocked_pruner_v, scope))

# add cons
no_blocking_cons += no_blocking_h + no_blocking_v

# create tree constraint
tree_con: Constraint = Constraint("tree", validator, pruner, variables)
all_cons = connectivity_cons + no_blocking_cons + [tree_con]

# create csp
csp = CSP("Sewage pAIpes", variables, all_cons)