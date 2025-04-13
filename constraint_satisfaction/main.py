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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and print all solutions for Sewage pAIpes puzzle"
    )
    parser.add_argument(
        "-n", "--size", type=int, required=True, help="Board dimension size (2-25)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.size < 2 or args.size > 25:
        parser.error("Board size must be between 2 and 25")

    return args


def main():
    args = parse_args()
    n = args.size

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
    no_blocking_h: list[Constraint] = []
    for i in range(n):
        for j in range(n - 1):
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
        solutions_gac, -1, True, False
    )  # Always print solutions, no randomization
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")
    print(f"Total solutions found: {len(solutions_gac)}")


if __name__ == "__main__":
    main()
