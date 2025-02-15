# from test_data_constructors import PIPE, print1DGrid
from test_data.test_data_constructors import PIPE, print1DGrid
from pipes_utils import flatten

almostLoopGrid1 = flatten(
    [
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
)

loopGrid2 = flatten(
    [
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
)

noLoopGrid1 = flatten(
    [
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
)

noLoopGrid2 = flatten(
    [
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
)


if __name__ == "__main__":
    print("Loop Grid 1")
    print1DGrid(almostLoopGrid1)

    print("Loop Grid 2")
    print1DGrid(loopGrid2)

    print("No Loop Grid 1")
    print1DGrid(noLoopGrid1)

    print("No Loop Grid 2")
    print1DGrid(noLoopGrid2)
