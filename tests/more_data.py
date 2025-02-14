from csp import PipeType
from typing import Optional
from visualizer import print2DGrid, PIPE

# 4×4 grid, partially filled
sampleGrid1WithLoopsPartiallyFilled: list[list[Optional[PipeType]]] = [
    [PIPE["RightDown"], PIPE["RightLeft"], PIPE["DownLeft"], None],
    [PIPE["UpDown"], None, PIPE["UpDown"], None],
    [PIPE["UpDown"], PIPE["UpLeft"], PIPE["RightDown"], None],
    [None, None, PIPE["UpLeft"], None],
]

sampleGrid2WithLoopsPartiallyFilled: list[list[Optional[PipeType]]] = [
    [PIPE["Down"], None, None, None],
    [PIPE["UpDownLeft"], PIPE["RightDown"], None, None],
    [None, PIPE["UpRight"], None, None],
    [None, None, None, None],
]

sampleGrid3WithLoopsCompletelyFilled: list[list[Optional[PipeType]]] = [
    [PIPE["RightDown"], PIPE["RightLeft"], PIPE["DownLeft"]],
    [PIPE["UpDown"], None, PIPE["UpDown"]],
    [PIPE["UpRight"], PIPE["RightLeft"], PIPE["UpLeft"]],
]

sampleGrid4NoLoopsPartiallyFilled: list[list[Optional[PipeType]]] = [
    [PIPE["Right"], None, None, None],
    [None, PIPE["Down"], None, None],
    [None, None, PIPE["Left"], None],
    [None, None, None, None],
]

sampleGrid5NoLoopsPartiallyFilled: list[list[Optional[PipeType]]] = [
    [None, PIPE["Down"], None, None],
    [PIPE["Up"], None, PIPE["Right"], None],
    [None, None, None, PIPE["Left"]],
    [None, None, None, None],
]

sampleGrid6NoLoopsFullyFilled: list[list[Optional[PipeType]]] = [
    # Row 0
    [PIPE["Right"], PIPE["RightLeft"], PIPE["DownLeft"]],
    # Row 1
    [PIPE["RightDown"], PIPE["Left"], PIPE["UpDown"]],
    # Row 2
    [PIPE["UpRight"], PIPE["RightLeft"], PIPE["UpLeft"]],
]

print("Sample Grid 1:")
print2DGrid(sampleGrid1WithLoopsPartiallyFilled)
print("\nSample Grid 2:")
print2DGrid(sampleGrid2WithLoopsPartiallyFilled)
print("\nSample Grid 3:")
print2DGrid(sampleGrid3WithLoopsCompletelyFilled)
print("\nSample Grid 4:")
print2DGrid(sampleGrid4NoLoopsPartiallyFilled)
print("\nSample Grid 5:")
print2DGrid(sampleGrid5NoLoopsPartiallyFilled)
print("\nSample Grid 6:")
print2DGrid(sampleGrid6NoLoopsFullyFilled)
