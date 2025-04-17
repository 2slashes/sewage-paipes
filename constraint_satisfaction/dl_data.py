import math
import os
import random
import time
import argparse
from combined import create_pipes_csp
from pipes_utils import Assignment, PipeType
from puzzle_generation import (
    scramble_all,
    scramble_k,
)

# Generates data for the neural network to train from
# implementation details and information regarding how to use the file can be found in the dl_data.py section in the report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training and test data for Sewage pAIpes puzzle"
    )
    parser.add_argument("-n", "--size", type=int, help="Board dimension size (2-25)")
    parser.add_argument(
        "--train-solutions",
        type=int,
        default=0,
        help="Number of solutions for the training set (-1 to generate all possible solutions)",
    )
    parser.add_argument(
        "--train-variations",
        type=int,
        default=1,
        help="Number of puzzles per solution for training set",
    )
    parser.add_argument(
        "--test-solutions",
        type=int,
        default=0,
        help="Number of solutions for the test set (-1 to generate all possible solutions)",
    )
    parser.add_argument(
        "--test-variations",
        type=int,
        default=1,
        help="Number of puzzles per solution for test set",
    )
    parser.add_argument(
        "--puzzle-solutions",
        type=int,
        default=0,
        help="Number of puzzles to generate for the network to play (-1 to generate all possible solutions)",
    )
    parser.add_argument(
        "--puzzle-variations",
        type=int,
        default=1,
        help="Number of variations to generate per puzzle solution",
    )
    parser.add_argument(
        "--gac-after-every",
        type=int,
        default=1,
        help="Number of solutions to generate before creating a new CSP instance",
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Run in augmentation mode using outliers.csv",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print solutions as they are generated",
    )

    args = parser.parse_args()

    if not args.aug:
        # Validate board size
        if args.size is None:
            parser.error("--size is required in normal mode")
        if args.size < 2 or args.size > 25:
            parser.error("Board size must be between 2 and 25")

        # Check for at least one solution
        has_train = args.train_solutions > 0 or args.train_solutions == -1
        has_test = args.test_solutions > 0 or args.test_solutions == -1
        has_puzzle = args.puzzle_solutions > 0 or args.puzzle_solutions == -1

        if not (has_train or has_test or has_puzzle):
            parser.error(
                "At least one solution (train, test, or puzzle) must be provided"
            )

        # Validate solution counts
        if args.train_solutions < -1:
            parser.error("Number of training solutions cannot be less than -1")
        if args.test_solutions < -1:
            parser.error("Number of test solutions cannot be less than -1")
        if args.puzzle_solutions < -1:
            parser.error("Number of puzzle solutions cannot be less than -1")

        # Check for -1 in any solution option
        has_negative_one = (
            (args.train_solutions == -1)
            or (args.test_solutions == -1)
            or (args.puzzle_solutions == -1)
        )

        if has_negative_one:
            # If any solution is -1, ensure other solutions are 0
            if (
                (
                    args.train_solutions == -1
                    and (args.test_solutions != 0 or args.puzzle_solutions != 0)
                )
                or (
                    args.test_solutions == -1
                    and (args.train_solutions != 0 or args.puzzle_solutions != 0)
                )
                or (
                    args.puzzle_solutions == -1
                    and (args.train_solutions != 0 or args.test_solutions != 0)
                )
            ):
                parser.error(
                    "When using -1 for any solution option, other solutions must be 0"
                )

            # Disallow GAC after every if -1 is used
            if args.gac_after_every != 1:
                parser.error(
                    "--gac-after-every cannot be provided when using -1 for any solution option"
                )
        else:
            # Validate GAC after every for normal cases
            if args.gac_after_every < 1:
                parser.error("Number of solutions before new GAC must be at least 1")

    return args


def augment_data():
    """
    Augment data from outliers.csv by generating puzzles for each goal state.
    For each goal, generates sqrt(extra_moves) puzzles using the same logic as write_csv in random_rotation.py.
    Splits the generated puzzles 75/25 between train and test sets and writes them to train.csv and test.csv, respectively.
    """
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "../deep_learning/data")
    outliers_file = os.path.join(data_dir, "outliers.csv")

    if not os.path.exists(outliers_file):
        raise FileNotFoundError(f"outliers.csv not found in {data_dir}")

    # Read goals and extra_moves from outliers.csv
    goals = []
    extra_moves = []
    with open(outliers_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            # Split into initial_state, solution_state, extra_moves
            _, goal_str, _, moves_str = line.strip().split(",")
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
            extra_moves.append(int(moves_str))

    train_data: list[tuple[str, str]] = []
    test_data: list[tuple[str, str]] = []

    for goal, moves in zip(goals, extra_moves):
        puzzles_labels: list[tuple[str, str]] = []
        num_puzzles = max(
            1, round(math.log10(moves) * 10 + 300)
        )  # At least 1 puzzle per goal
        for _ in range(num_puzzles):
            puzzle, label = scramble_k(goal, max(1, round((2 * random.random()) ** 4)))
            puzzles_labels.append((generate_one_state_str(puzzle), label))
        # Split 75/25
        split_idx = int(len(puzzles_labels) * 0.75)
        train_data.extend(puzzles_labels[:split_idx])
        test_data.extend(puzzles_labels[split_idx:])

    train_file = os.path.join(data_dir, "train.csv")
    test_file = os.path.join(data_dir, "test.csv")

    with open(train_file, "a") as f:
        for puzzle, label in train_data:
            f.write(f"{puzzle},{label}\n")

    with open(test_file, "a") as f:
        for puzzle, label in test_data:
            f.write(f"{puzzle},{label}\n")


def write_csv(
    solutions: list[Assignment],
    num_puzzles_per_solution: int,
    file_path: str,
):
    """
    Write a CSV file where first column is the puzzle and second column represents
    which pipes to rotate to get to the solution.

    :params lows: The proportion of the puzzles per solution that would have sqrt(len(solution)) pipes to rotate or less
    """
    output: list[list[str]] = []
    for solution in solutions:
        for _ in range(num_puzzles_per_solution):
            puzzle, label = scramble_k(
                solution, max(1, round((2 * random.random()) ** 4))
            )
            output.append([generate_one_state_str(puzzle), label])
    with open(file_path, mode="w", newline="") as csv_file:
        # write the header "state,actions"
        csv_file.write("state,actions\n")
        for row in output:
            csv_file.write(f"{row[0]},{row[1]}\n")


def write_puzzles_csv(
    solutions: list[Assignment], file_path: str, variations_per_solution: int = 1
):
    """
    Write a CSV file where first column is the initial puzzle state and second column is the solution state.
    This uses the create_challenging_puzzle function to create the puzzle.

    Args:
        solutions: List of solution states
        file_path: Path to write the CSV file to
        variations_per_solution: Number of variations to generate per solution
    """
    output: list[list[str]] = []
    for solution in solutions:
        for _ in range(variations_per_solution):
            puzzle, _, min_moves_to_solve = scramble_all(solution)
            output.append(
                [
                    generate_one_state_str(puzzle),
                    generate_one_state_str(solution),
                    min_moves_to_solve,
                ]
            )

    with open(file_path, mode="w", newline="") as csv_file:
        # write the header "initial_state,solution_state"
        csv_file.write("initial_state,solution_state,min_moves_to_solve\n")
        for row in output:
            csv_file.write(f"{row[0]},{row[1]},{row[2]}\n")


def generate_one_state_str(state: Assignment):
    output = ""
    for pipe in state:
        for dir in range(4):
            if pipe[dir]:
                output += "1"
            else:
                output += "0"
    return output


def main():
    args = parse_args()

    if args.aug:
        augment_data()
        return

    n = args.size

    total_solutions_needed = (
        args.train_solutions + args.test_solutions + args.puzzle_solutions
    )
    solutions: set[tuple[PipeType, ...]] = set()
    t0 = time.time()

    gac_called = 0

    print("Generating Solutions...")
    if total_solutions_needed == -1:
        csp = create_pipes_csp(n)
        csp.gac_all(
            solutions=solutions,
            max_solutions=-1,
            print_solutions=args.print,
            random_start=True,
        )
    else:
        while len(solutions) < total_solutions_needed:
            csp = create_pipes_csp(n)

            # Generate solutions for this CSP instance
            current_solutions: set[tuple[PipeType, ...]] = set()
            solutions_needed = min(
                args.gac_after_every, total_solutions_needed - len(solutions)
            )
            csp.gac_all(
                solutions=current_solutions,
                max_solutions=solutions_needed,
                print_solutions=args.print,
                random_start=True,
            )
            gac_called += 1

            solutions |= current_solutions
            if len(solutions) == total_solutions_needed:
                break
            if len(solutions) > total_solutions_needed:
                raise ValueError(
                    f"Found {len(solutions)} solutions, but {total_solutions_needed} are needed"
                )

    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "../deep_learning/data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    solutions_list = [list(solution) for solution in solutions]

    # Only write training CSV if there are training solutions
    if args.train_solutions > 0:
        print("Writing training variations...")
        write_csv(
            solutions_list[: args.train_solutions],
            args.train_variations,
            os.path.join(data_dir, "train.csv"),
        )

    # Only write test CSV if there are test solutions
    if args.test_solutions > 0:
        print("Writing test variations...")
        write_csv(
            solutions_list[
                args.train_solutions : args.train_solutions + args.test_solutions
            ],
            args.test_variations,
            os.path.join(data_dir, "test.csv"),
        )

    # Only write puzzles CSV if there are puzzle solutions
    if args.puzzle_solutions > 0:
        print("Writing puzzle variations...")
        write_puzzles_csv(
            solutions_list[args.train_solutions + args.test_solutions :],
            os.path.join(data_dir, "puzzles.csv"),
            variations_per_solution=args.puzzle_variations,
        )

    t1 = time.time()
    print(f"Time: {t1 - t0}")
    print(f"GAC called: {gac_called}")


if __name__ == "__main__":
    main()
