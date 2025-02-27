from pipe_typings import PipeType
from csp import print1DGrid
import random

def clockwise_rotate(pipe: PipeType, num_rotations: int) -> PipeType:
    """
    Rotate a PipeType clockwise by 90*n degrees.

    :params pipe: The pipe to be rotated
    :params n: number of 90 degree rotations to perform. 0 = no rotation, 1 = 90 degree rotation, 2 = 180 degrtee rotation etc.
    """
    top = pipe[(0 - num_rotations) % 4]
    right = pipe[(1 - num_rotations) % 4]
    bottom = pipe[(2 - num_rotations) % 4]
    left = pipe[(3 - num_rotations) % 4]

    new_pipe: PipeType = (top, right, bottom, left)
    return new_pipe

def random_rotate_board(board: list[PipeType]):
    new_board: list[PipeType] = []
    for pipe in board:
        num_rotations = random.randint(0, 3)
        new_pipe = clockwise_rotate(pipe, num_rotations)
        new_board.append(new_pipe)

    print1DGrid(new_board) # type: ignore
    return new_board