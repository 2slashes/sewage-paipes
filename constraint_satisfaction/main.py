import time
from csp import Variable, Constraint, CSP, DomainGenerator
from pipes_constraints import (
    not_blocked_h,
    not_blocked_pruner_h,
    not_blocked_pruner_v,
    not_blocked_v,
)
from pipe_typings import Assignment
from tree import validator as tree_validator, pruner as tree_pruner
from connected import validator as connected_validator, pruner as connected_pruner
from random_rotation import random_rotate_board
from pddl_output import generate_pddl


n = 3
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
all_cons = no_blocking_cons + [tree_con, connected_con]


# create csp
csp = CSP("Sewage pAIpes", variables, all_cons)
solutions_gac: list[Assignment] = []
t0 = time.time()
csp.gac_all(solutions_gac)
t1 = time.time()
print(f"time: {t1 - t0}")
random_rotation = random_rotate_board(solutions_gac[0])

print(generate_pddl("chyme", "pipes", random_rotation, solutions_gac[0]))