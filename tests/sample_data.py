from csp import DomainGenerator, PipeType
from typing import Optional
from visualizer import print2DGrid, PIPE
from csp import Variable
from pipes_utils import flatten

almostLoopGrid1: list[list[Optional[PipeType]]] = [
    # Row 0
    [PIPE["RightDown"], PIPE["DownLeft"], None, None, None],
    # Row 1
    [
        PIPE["UpDown"],
        PIPE["UpRight"],
        PIPE["DownLeft"],
        None,
        None,
    ],
    # Row 2
    [
        None,
        PIPE["RightDown"],
        PIPE["UpLeft"],
        None,
        None,
    ],
    # Row 3
    [PIPE["UpRight"], PIPE["UpLeft"], None, None, None],
    # Row 4
    [None, None, None, None, None],
]

loopGrid2: list[list[Optional[PipeType]]] = [
    # Row 0
    [
        PIPE["RightDown"],
        PIPE["RightLeft"],
        PIPE["RightDownLeft"],
        PIPE["RightLeft"],
        PIPE["Left"],
    ],
    # Row 1
    [
        PIPE["UpRight"],
        PIPE["RightLeft"],
        PIPE["UpRightLeft"],
        PIPE["RightLeft"],
        PIPE["Left"],
    ],
    # Row 2
    [None, None, None, None, None],
    # Row 3
    [None, None, None, None, None],
    # Row 4
    [None, None, None, None, None],
]

noLoopGrid1: list[list[Optional[PipeType]]] = [
    # Row 0
    [PIPE["RightDown"], None, None, None, None],
    # Row 1
    [PIPE["UpDown"], None, None, None, None],
    # Row 2
    [PIPE["UpDown"], None, None, None, None],
    # Row 3
    [PIPE["UpDown"], None, None, None, None],
    # Row 4
    [
        PIPE["UpRight"],
        PIPE["RightLeft"],
        PIPE["RightLeft"],
        PIPE["RightLeft"],
        PIPE["UpLeft"],
    ],
]

noLoopGrid2: list[list[Optional[PipeType]]] = [
    # Row 0
    [
        PIPE["Right"],
        PIPE["RightLeft"],
        PIPE["DownLeft"],
        None,
        None,
    ],
    # Row 1
    [None, None, PIPE["UpDown"], None, None],
    # Row 2
    [None, None, PIPE["UpDown"], None, None],
    # Row 3
    [None, None, PIPE["UpDown"], None, None],
    # Row 4
    [
        None,
        None,
        PIPE["UpRight"],
        PIPE["RightLeft"],
        PIPE["UpLeft"],
    ],
]
flattenedLoopGrid1 = flatten(almostLoopGrid1)
# create a list of variables
loopGrid1Variables: list[Variable] = []
for i in range(len(flattenedLoopGrid1)):
    top = i < 5
    bottom = i >= 15
    left = i % 5 == 0
    right = i % 5 == 4
    var = Variable(
        location=(i // 5, i % 5),
        domain=DomainGenerator.generate_domain(top, right, bottom, left),
    )
    if flattenedLoopGrid1[i] is not None:
        pipe = flattenedLoopGrid1[i]
        assert pipe is not None
        var.assign(pipe)
    loopGrid1Variables.append(var)

if __name__ == "__main__":
    print("Loop Grid 1")
    print2DGrid(almostLoopGrid1)

    print("Loop Grid 2")
    print2DGrid(loopGrid2)

    print("No Loop Grid 1")
    print2DGrid(noLoopGrid1)

    print("No Loop Grid 2")
    print2DGrid(noLoopGrid2)
