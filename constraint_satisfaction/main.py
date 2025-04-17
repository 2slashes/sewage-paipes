import time
import argparse
from combined import create_pipes_csp
from pipes_utils import Assignment, PipeType

# Prints a visual representation of the Pipes solutions to the terminal
# implementation details and information regarding how to use the file can be found in the main.py section of the report


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

    # create csp
    csp = create_pipes_csp(n)
    solutions_gac: set[tuple[PipeType, ...]] = set()

    t0 = time.time()
    csp.gac_all(solutions=solutions_gac, max_solutions=-1, print_solutions=True)
    t1 = time.time()

    print(f"Time taken: {t1 - t0} seconds")
    print(f"Total solutions found: {len(solutions_gac)}")


if __name__ == "__main__":
    main()
