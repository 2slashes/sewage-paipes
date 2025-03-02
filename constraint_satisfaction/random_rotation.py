from pipe_typings import PipeType
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

def random_rotate_board(board: list[PipeType], num_rotations: int) -> list[list[PipeType]]:
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
            num_rotations = random.randint(0, 3)
            new_pipe = clockwise_rotate(pipe, num_rotations)
            new_board.append(new_pipe)
        # ensure that the random rotation of the board has not been generated already
        if new_board not in new_boards:
            new_boards.append(new_board)
    return new_boards