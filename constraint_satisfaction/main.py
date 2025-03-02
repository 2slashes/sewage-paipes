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
import os
from math import ceil

n: int = 0
n_validated = False
while not n_validated:
    solutions_str = input("Enter the board dimension size (2-25): ")
    if not solutions_str.isdigit():
        print("Value entered is not a digit.")
        continue
    n = int(solutions_str)
    if n < 2 or n > 25:
        print("Value for n is not between 2 and 25.")
        continue
    print(f"Value of n validated as {n}\n")
    n_validated = True

generate_pddl_str = input("Would you like to generate PDDL output files for these solutions? Y/n: ").lower()
should_generate_pddl = True if generate_pddl_str == 'y' or generate_pddl_str == 'yes' else False
print()

num_rotations: int = 0
rotations_validated = False
if should_generate_pddl:
    # how many rotations
    while not rotations_validated:
        rotations_str = input("Enter the number of rotations for each solution: ")
        if not rotations_str.isdigit():
            print("Value entered is not a digit")
            continue
        num_rotations = int(rotations_str)
        if num_rotations < 1 or num_rotations > 100:
            print("Number of rotations is invalid, must be between 1 and 100.")
            continue
        print(f"Number of rotations validated as {num_rotations}\n")
        rotations_validated = True
# max
num_solutions = 0
solutions_validated = False
while not solutions_validated:
    solutions_str = input("Enter the maximum number of solutions (-1 for unbounded number of solutions): ")
    try:
        num_solutions = int(solutions_str)
    except ValueError:
        print("Value entered is not a digit.")
        continue
    if num_solutions < 1 and num_solutions != -1:
        print("Number of solutions must be at least 1.")
        continue
    if num_solutions == -1 and should_generate_pddl:
        warning_str = input("WARNING: This action could generate A LOT of PDDL files.\nAre you sure you want an unbounded number of solutions? Y/n: ").lower()
        cancel = False if warning_str == "y" or warning_str == "yes" else True
        if cancel:
            continue
    print(f"Number of solutions validated as {num_solutions}\n")
    solutions_validated = True

print_solutions_str = input("Would you like to print a visual representation for these solutions? Y/n: ").lower()
should_print_solutions = True if print_solutions_str == 'y' or print_solutions_str == 'yes' else False
print()

max_num_boards_generated = -1
if num_solutions != -1:
    if should_generate_pddl:
        max_num_boards_generated = ceil(num_solutions / num_rotations)
    else:
        max_num_boards_generated = num_solutions

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
csp.gac_all(solutions_gac, max_num_boards_generated, should_print_solutions)
t1 = time.time()
print(f"time: {t1 - t0}")

# generate the PDDL files for all of the problems
if should_generate_pddl:
    directory_name = f"planning/pddl/problems/{n}/"
    try:
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    count = 0
    for solution in solutions_gac:
        random_rotations = random_rotate_board(solution, num_rotations)
        for rotation in random_rotations:
            pddl: str = generate_pddl(f"pipes{count}", "pipes", rotation, solution)
            with open(f"{directory_name}problem{count}.pddl", "w") as file:
                file.write(pddl)
            count += 1