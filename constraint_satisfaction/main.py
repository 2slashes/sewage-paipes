import time
import argparse
from csp import Variable, Constraint, CSP, DomainGenerator
from constraints.no_half_connections import (
    validator_h as no_half_connections_validator_h,
    validator_v as no_half_connections_validator_v,
    pruner_h as no_half_connections_pruner_h,
    pruner_v as no_half_connections_pruner_v,
)
from pipe_typings import Assignment
from constraints.no_cycles import validator as tree_validator, pruner as tree_pruner
from constraints.connected import (
    validator as connected_validator,
    pruner as connected_pruner,
)
from pddl_output import generate_all_pddl_and_state_files
from math import ceil

from random_rotation import random_rotate_board, generate_solutions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate solutions for Sewage pAIpes puzzle"
    )
    parser.add_argument(
        "-n", "--size", type=int, required=True, help="Board dimension size (2-25)"
    )
    parser.add_argument(
        "--generate-csv",
        action="store_true",
        help="Generate CSV output files and initial/goal state files",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=1,
        help="Number of rotations for each solution (1-100)",
    )
    parser.add_argument(
        "--random-order",
        action="store_true",
        help="Randomize the order of generated solutions",
    )
    parser.add_argument(
        "--max-solutions",
        type=int,
        default=-1,
        help="Maximum number of solutions (-1 for unbounded)",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Do not print visual representation of solutions",
    )
    parser.add_argument(
        "--csv-format",
        choices=["actions", "goal"],
        default="actions",
        help='CSV output format: "actions" for state-to-actions mapping, "goal" for state-to-goal mapping',
    )

    args = parser.parse_args()

    # Validate arguments
    if args.size < 2 or args.size > 25:
        parser.error("Board size must be between 2 and 25")

    if args.generate_csv and (args.rotations < 1 or args.rotations > 100):
        parser.error("Number of rotations must be between 1 and 100")

    if args.max_solutions < 1 and args.max_solutions != -1:
        parser.error(
            "Maximum number of solutions must be at least 1 or -1 for unbounded"
        )

    return args


def main():
    args = parse_args()
    n = args.size
    should_generate_csv = args.generate_csv
    num_rotations = args.rotations
    randomize_order = args.random_order
    max_solutions = args.max_solutions
    should_print_solutions = not args.no_print
    csv_format = args.csv_format

    max_num_boards_generated = -1
    if max_solutions != -1:
        if should_generate_csv:
            max_num_boards_generated = ceil(max_solutions / num_rotations)
        else:
            max_num_boards_generated = max_solutions

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
                Constraint(
                    name,
                    no_half_connections_validator_h,
                    no_half_connections_pruner_h,
                    scope,
                )
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
                Constraint(
                    name,
                    no_half_connections_validator_v,
                    no_half_connections_pruner_v,
                    scope,
                )
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
    csp.gac_all(
        solutions_gac, max_num_boards_generated, should_print_solutions, randomize_order
    )
    t1 = time.time()
    print(f"time: {t1 - t0}")
    if should_generate_csv:
        for solution in solutions_gac:
            random_rotate_board(solution, num_rotations, csv_format)


if __name__ == "__main__":
    main()
