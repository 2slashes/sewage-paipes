import math
from pipe_typings import PipeType, Assignment
import random


def random_rotate_board(
    board: list[PipeType], num_rotations: int
) -> list[list[PipeType]]:
    """
    Rotates every pipe on a pipes board by either 0, 90, 180, or 270 degrees, at random.

    :params board: list of PipeTypes that represents the board to be randomly rotated
    :params num_rotations: number of times to rotate the full board, also the number of new boards that will be returned.
    :returns: list of boards after random rotation.
    """
    new_boards: list[list[PipeType]] = []
    while len(new_boards) < num_rotations:
        new_board: list[PipeType] = []
        # generate a random rotation of each pipe in the board
        for pipe in board:
            num_rotations_pipe = random.randint(0, 3)
            new_pipe = clockwise_rotate(pipe, num_rotations_pipe)
            new_board.append(new_pipe)
        # ensure that the random rotation of the board has not been generated already
        if new_board not in new_boards:
            new_boards.append(new_board)
    return new_boards


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


def create_puzzle(solution: Assignment, k: int) -> tuple[str, str]:
    """
    Create a puzzle from a solution by randomly selecting k pipes
    to rotate. The number of rotations is a random number between 1 and 3.
    :params solution: The solution to create a puzzle from
    :params k: The number of pipes to rotate
    :returns: A tuple of (puzzle, label)
    """
    puzzle: Assignment = solution.copy()
    # get index of pipes to rotate
    pipes_to_rotate: list[int] = random.sample(
        [i for i in range(len(solution))],
        k=k,
    )
    # rotate the pipes
    for index in pipes_to_rotate:
        puzzle[index] = clockwise_rotate(puzzle[index], random.randint(1, 3))

    label: list[str] = ["0"] * len(solution)
    for index in pipes_to_rotate:
        label[index] = "1"
    return generate_one_state_str(puzzle), "".join(label)


def create_challenging_puzzle(solution: Assignment) -> tuple[str, str, str]:
    """
    Create a puzzle that the network will eventually play on. This is more scrambled.
    This is also used for data augmentation.
    Iterate through each pipe and rotate it 0, 1, 2, or 3 times (chosen randomly).
    :returns: A tuple of (puzzle, bad_pipe_indices, min_moves_to_solve)
    """
    puzzle: Assignment = solution.copy()
    bad_pipe_indices: list[str] = ["0"] * len(solution)
    min_moves_to_solve = 0
    for index in range(len(solution)):
        rotations = random.randint(0, 3)
        if rotations > 0:
            bad_pipe_indices[index] = "1"
        puzzle[index] = clockwise_rotate(puzzle[index], rotations)
        if puzzle[index] == (True, False, True, False) or puzzle[index] == (
            False,
            True,
            False,
            True,
        ):
            min_moves_to_solve += rotations % 2
        else:
            min_moves_to_solve += (4 - rotations) % 4
    return (
        generate_one_state_str(puzzle),
        "".join(bad_pipe_indices),
        str(min_moves_to_solve),
    )


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
            puzzle_str, label = create_puzzle(
                solution, max(1, round((2 * random.random()) ** 4))
            )
            output.append([puzzle_str, label])
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
            puzzle_str, _, min_moves_to_solve = create_challenging_puzzle(solution)
            output.append(
                [puzzle_str, generate_one_state_str(solution), min_moves_to_solve]
            )

    with open(file_path, mode="w", newline="") as csv_file:
        # write the header "initial_state,solution_state"
        csv_file.write("initial_state,solution_state,min_moves_to_solve\n")
        for row in output:
            csv_file.write(f"{row[0]},{row[1]},{row[2]}\n")
