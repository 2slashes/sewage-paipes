from pipes_utils import PipeType, Assignment
import random

# Contains functions that generate initial states from goal states
# Implementation details can be found in the puzzle_generation.py section of the report

def scramble_k(solution: Assignment, k: int) -> tuple[Assignment, str]:
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
    return puzzle, "".join(label)


def scramble_all(solution: Assignment) -> tuple[Assignment, str, str]:
    """
    Create a puzzle that from a solution.
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
        puzzle,
        "".join(bad_pipe_indices),
        str(min_moves_to_solve),
    )


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
