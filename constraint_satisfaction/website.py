import json
import time
import argparse
from combined import create_pipes_csp
from pipes_utils import PipeType, Assignment

# Generates and outputs solutions for Sewage pAIpes puzzle in a JSON file
# implementation details and information regarding how to use the file can be found in the main.py section of the report


def generate_one_state_str(state: Assignment):
    output = ""
    for pipe in state:
        for dir in range(4):
            if pipe[dir]:
                output += "1"
            else:
                output += "0"
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and output solutions for Sewage pAIpes puzzle in a JSON file"
    )
    parser.add_argument(
        "-n", "--size", type=int, required=True, help="Board dimension size (2-25)"
    )
    parser.add_argument(
        "--solutions",
        type=int,
        required=True,
        help="Number of solutions to generate (-1 for all possible solutions)",
    )
    parser.add_argument(
        "--gac-after-every",
        type=int,
        default=1,
        help="Number of solutions to generate before creating a new CSP instance",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.size < 2 or args.size > 25:
        parser.error("Board size must be between 2 and 25")

    if args.solutions < -1:
        parser.error("Number of solutions must be at least 1 or -1 for all solutions")

    # Validate GAC after every
    if args.solutions == -1:
        if args.gac_after_every != 1:
            parser.error(
                "--gac-after-every cannot be provided when using -1 for solutions"
            )
    else:
        if args.gac_after_every < 1:
            parser.error("Number of solutions before new GAC must be at least 1")

    return args


def main():
    args = parse_args()
    n = args.size
    num_solutions = args.solutions
    solutions_gac: set[tuple[PipeType, ...]] = set()
    gac_called = 0

    t0 = time.time()

    if num_solutions == -1:
        # For -1, call GAC only once to find all solutions
        csp = create_pipes_csp(n)
        csp.gac_all(
            solutions=solutions_gac,
            max_solutions=-1,
            print_solutions=True,
            random_start=True,
        )
        gac_called += 1
    else:
        # For specific number of solutions, call GAC multiple times based on gac-after-every
        while len(solutions_gac) < num_solutions:
            csp = create_pipes_csp(n)
            current_solutions: set[tuple[PipeType, ...]] = set()
            solutions_needed = min(
                args.gac_after_every, num_solutions - len(solutions_gac)
            )

            csp.gac_all(
                solutions=current_solutions,
                max_solutions=solutions_needed,
                print_solutions=True,
                random_start=True,
            )
            gac_called += 1

            solutions_gac |= current_solutions
            if len(solutions_gac) == num_solutions:
                break
            if len(solutions_gac) > num_solutions:
                raise ValueError(
                    f"Found {len(solutions_gac)} solutions, but {num_solutions} are needed"
                )

    t1 = time.time()
    print(f"Finished generating solutions in {t1 - t0} seconds")
    print(f"GAC called: {gac_called} times")

    solutions_list = [
        generate_one_state_str(list(solution)) for solution in solutions_gac
    ]

    # output solutions to json file
    with open(f"../website/4.json", "w") as f:
        json.dump(solutions_list, f)

    print(f"Time taken: {t1 - t0} seconds")
    print(f"Total solutions found: {len(solutions_gac)}")


if __name__ == "__main__":
    main()
