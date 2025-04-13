import os
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

from random_rotation import write_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training and test data for Sewage pAIpes puzzle"
    )
    parser.add_argument(
        "-n", "--size", type=int, required=True, help="Board dimension size (2-25)"
    )
    parser.add_argument(
        "--solutions-train",
        type=int,
        required=True,
        help="Number of solutions for the training set",
    )
    parser.add_argument(
        "--puzzles-train",
        type=int,
        required=True,
        help="Number of puzzles per solution for training set",
    )
    parser.add_argument(
        "--solutions-test",
        type=int,
        required=True,
        help="Number of solutions for the test set",
    )
    parser.add_argument(
        "--puzzles-test",
        type=int,
        required=True,
        help="Number of puzzles per solution for test set",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.size < 2 or args.size > 25:
        parser.error("Board size must be between 2 and 25")

    if args.solutions_train < 1:
        parser.error("Number of training solutions must be at least 1")

    if args.puzzles_train < 1:
        parser.error("Number of training puzzles per solution must be at least 1")

    if args.solutions_test < 1:
        parser.error("Number of test solutions must be at least 1")

    if args.puzzles_test < 1:
        parser.error("Number of test puzzles per solution must be at least 1")

    return args


def main():
    args = parse_args()
    n = args.size
    num_solutions_train = args.solutions_train
    num_puzzles_train = args.puzzles_train
    num_solutions_test = args.solutions_test
    num_puzzles_test = args.puzzles_test

    # Initialize variables and constraints
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

    # Generate training solutions
    solutions: list[Assignment] = []
    t0 = time.time()
    csp.gac_all(
        solutions=solutions,
        max_solutions=num_solutions_train + num_solutions_test,
        print_solutions=False,
        randomize_order=True,
    )

    if (num_solutions_train + num_solutions_test) > len(solutions):
        raise ValueError(
            f"Not enough solutions found. Found {len(solutions)} solutions, but {num_solutions_train * num_puzzles_train + num_solutions_test * num_puzzles_test} are needed"
        )

    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "../deep_learning/data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    write_csv(
        solutions[:num_solutions_train],
        num_puzzles_train,
        os.path.join(data_dir, "train.csv"),
    )
    write_csv(
        solutions[num_solutions_train:],
        num_puzzles_test,
        os.path.join(data_dir, "test.csv"),
    )

    t1 = time.time()
    print(f"Time: {t1 - t0}")


if __name__ == "__main__":
    main()
