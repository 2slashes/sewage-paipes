import math
from pipe_typings import PipeType, Assignment
import random


def clockwise_rotate(pipe: PipeType, n: int) -> PipeType:
    """
    Rotate a PipeType clockwise by 90*n degrees.

    :params pipe: The pipe to be rotated
    :params n: number of 90 degree rotations to perform. 0 = no rotation, 1 = 90 degree rotation, 2 = 180 degrtee rotation etc.
    """
    top = pipe[(0 - n) % 4]
    right = pipe[(1 - n) % 4]
    bottom = pipe[(2 - n) % 4]
    left = pipe[(3 - n) % 4]

    new_pipe: PipeType = (top, right, bottom, left)
    return new_pipe


def create_puzzle(solution: Assignment) -> tuple[Assignment, list[int]]:
    """
    Create a puzzle from a solution by randomly selecting a number k
    between 1 and sqrt(len(solution)). Then, randomly select k pipes in the solution
    to rotate. When a pipe is chosen for rotation, the number of rotation is a random
    number between 1 and 3.
    :params solution: The solution to create a puzzle from
    :returns: A tuple of (puzzle, incorrect_pipes_indicies)
    """
    puzzle: Assignment = solution.copy()
    # get index of pipes to rotate
    pipes_to_rotate: list[int] = random.sample(
        [i for i in range(len(solution))],
        k=random.randint(1, int(math.sqrt(len(solution)))),
    )
    # rotate the pipes
    for index in pipes_to_rotate:
        puzzle[index] = clockwise_rotate(puzzle[index], random.randint(1, 3))
    return puzzle, pipes_to_rotate


def generate_one_state_str(state: Assignment):
    output = ""
    for pipe in state:
        for dir in range(4):
            if pipe[dir]:
                output += "1"
            else:
                output += "0"
    return output


def write_csv(
    solutions: list[Assignment], num_puzzles_per_solution: int, file_path: str
):
    """
    Write a CSV file where first column is the puzzle and second column represents
    which pipes to rotate to get to the solution.
    """
    output: list[list[str]] = []
    for solution in solutions:
        for _ in range(num_puzzles_per_solution):
            puzzle, pipes_to_rotate = create_puzzle(solution)
            puzzle_str = generate_one_state_str(puzzle)

            # create binary string of which pipes to rotate. 1 if the pipe is in the incorrect_pipes list, 0 otherwise
            label = ["0"] * len(solutions[0])
            for index in pipes_to_rotate:
                label[index] = "1"

            output.append([puzzle_str, "".join(label)])
    with open(file_path, mode="w", newline="") as csv_file:
        # write the header "state,actions"
        csv_file.write("state,actions\n")
        for row in output:
            csv_file.write(f"{row[0]},{row[1]}\n")
