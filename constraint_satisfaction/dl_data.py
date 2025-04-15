import math
import os
import random
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
from random_rotation import (
    write_csv,
    write_puzzles_csv,
    create_puzzle,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training and test data for Sewage pAIpes puzzle"
    )
    parser.add_argument("-n", "--size", type=int, help="Board dimension size (2-25)")
    parser.add_argument(
        "--solutions-train",
        type=int,
        help="Number of solutions for the training set",
    )
    parser.add_argument(
        "--puzzles-train",
        type=int,
        help="Number of puzzles per solution for training set",
    )
    parser.add_argument(
        "--solutions-test",
        type=int,
        help="Number of solutions for the test set",
    )
    parser.add_argument(
        "--puzzles-test",
        type=int,
        help="Number of puzzles per solution for test set",
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        help="Number of puzzles to generate for the network to play",
    )
    parser.add_argument(
        "--gac-after-every",
        type=int,
        default=4,
        help="Number of solutions to generate before creating a new CSP instance",
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Run in augmentation mode using outliers.csv",
    )
    parser.add_argument(
        "--puzzles",
        type=int,
        help="Number of puzzles to generate per goal in augmentation mode",
    )

    args = parser.parse_args()

    if args.aug:
        if not args.puzzles:
            parser.error("--puzzles is required in augmentation mode")
        if args.puzzles < 1:
            parser.error("Number of puzzles must be at least 1")
    else:
        # Validate arguments for normal mode
        if args.size is None:
            parser.error("--size is required in normal mode")
        if args.size < 2 or args.size > 25:
            parser.error("Board size must be between 2 and 25")
        if args.solutions_train is None:
            parser.error("--solutions-train is required in normal mode")
        if args.puzzles_train is None:
            parser.error("--puzzles-train is required in normal mode")
        if args.solutions_test is None:
            parser.error("--solutions-test is required in normal mode")
        if args.puzzles_test is None:
            parser.error("--puzzles-test is required in normal mode")
        if args.num_puzzles is None:
            parser.error("--num-puzzles is required in normal mode")
        if args.gac_after_every < 1:
            parser.error("Number of solutions before new GAC must be at least 1")

        if args.solutions_train < 0:
            parser.error("Number of training solutions cannot be negative")
        if args.puzzles_train < 0:
            parser.error("Number of training puzzles per solution cannot be negative")
        if args.solutions_test < 0:
            parser.error("Number of test solutions cannot be negative")
        if args.puzzles_test < 0:
            parser.error("Number of test puzzles per solution cannot be negative")
        if args.num_puzzles < 0:
            parser.error("Number of puzzles to generate cannot be negative")

    return args


def augment_data(num_puzzles_per_goal: int):
    """
    Augment data from outliers.csv by generating puzzles for each goal state.
    For each goal, splits the generated puzzles 75/25 between train and test sets.
    """
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "../deep_learning/data")
    outliers_file = os.path.join(data_dir, "outliers.csv")

    if not os.path.exists(outliers_file):
        raise FileNotFoundError(f"outliers.csv not found in {data_dir}")

    # Read goals from outliers.csv
    goals = []
    with open(outliers_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            goal_str = line.strip()
            # Convert binary string to Assignment
            goal = []
            for i in range(0, len(goal_str), 4):
                pipe = (
                    goal_str[i] == "1",
                    goal_str[i + 1] == "1",
                    goal_str[i + 2] == "1",
                    goal_str[i + 3] == "1",
                )
                goal.append(pipe)
            goals.append(goal)

    # Calculate split points
    num_train_puzzles = int(num_puzzles_per_goal * 0.75)

    # Generate puzzles for each goal and split them
    train_data = []
    test_data = []

    for goal in goals:
        # Generate all puzzles for this goal
        low_goal_puzzles_labels: list[tuple[str, str]] = []
        high_goal_puzzles_labels: list[tuple[str, str]] = []
        low = 0.75
        num_low_puzzles = int(num_puzzles_per_goal * low)
        num_high_puzzles = num_puzzles_per_goal - num_low_puzzles
        for _ in range(num_low_puzzles):
            puzzle, label = create_puzzle(
                goal, random.randint(1, int(math.sqrt(len(goal))))
            )
            low_goal_puzzles_labels.append((puzzle, label))
        for _ in range(num_high_puzzles):
            puzzle, label = create_puzzle(goal, random.randint(1, len(goal)))
            high_goal_puzzles_labels.append((puzzle, label))

    # Append to existing CSV files
    train_file = os.path.join(data_dir, "train.csv")
    test_file = os.path.join(data_dir, "test.csv")

    # Write train data
    with open(train_file, "a") as f:
        for puzzle, label in train_data:
            f.write(f"{puzzle},{label}\n")

    # Write test data
    with open(test_file, "a") as f:
        for puzzle, label in test_data:
            f.write(f"{puzzle},{label}\n")


def main():
    args = parse_args()

    if args.aug:
        augment_data(args.puzzles)
        return

    n = args.size
    num_solutions_train = args.solutions_train
    num_puzzles_train = args.puzzles_train
    num_solutions_test = args.solutions_test
    num_puzzles_test = args.puzzles_test
    gac_after_every = args.gac_after_every

    total_solutions_needed = num_solutions_train + num_solutions_test + args.num_puzzles
    solutions: list[Assignment] = []
    t0 = time.time()

    gac_called = 0

    while len(solutions) < total_solutions_needed:
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
        tree_con: Constraint = Constraint(
            "tree", tree_validator, tree_pruner, variables
        )

        connected_con: Constraint = Constraint(
            "connected", connected_validator, connected_pruner, variables
        )
        all_cons = no_blocking_cons + [tree_con, connected_con]

        # create csp
        csp = CSP("Sewage pAIpes", variables, all_cons)

        # Generate solutions for this CSP instance
        current_solutions: list[Assignment] = []
        solutions_needed = min(gac_after_every, total_solutions_needed - len(solutions))
        csp.gac_all(
            solutions=current_solutions,
            max_solutions=solutions_needed,
            print_solutions=False,
            randomize_order=True,
        )
        gac_called += 1

        # Add only unique solutions
        for solution in current_solutions:
            if solution not in solutions:
                solutions.append(solution)
                if len(solutions) >= total_solutions_needed:
                    break

    if len(solutions) < total_solutions_needed:
        raise ValueError(
            f"Not enough unique solutions found. Found {len(solutions)} solutions, but {total_solutions_needed} are needed"
        )

    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "../deep_learning/data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Only write training CSV if there are training solutions
    if num_solutions_train > 0:
        write_csv(
            solutions[:num_solutions_train],
            num_puzzles_train,
            os.path.join(data_dir, "train.csv"),
        )

    # Only write test CSV if there are test solutions
    if num_solutions_test > 0:
        write_csv(
            solutions[num_solutions_train : num_solutions_train + num_solutions_test],
            num_puzzles_test,
            os.path.join(data_dir, "test.csv"),
        )

    # Only write puzzles CSV if there are puzzle solutions
    puzzle_solutions = solutions[num_solutions_train + num_solutions_test :]
    if len(puzzle_solutions) > 0:
        write_puzzles_csv(
            puzzle_solutions,
            os.path.join(data_dir, "puzzles.csv"),
        )

    t1 = time.time()
    print(f"Time: {t1 - t0}")
    print(f"GAC called: {gac_called}")


if __name__ == "__main__":
    main()
