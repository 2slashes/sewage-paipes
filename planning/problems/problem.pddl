(define
    (problem solvingpipes)
    (:domain pipes)
    (:objects
        p0 - pipe
        p1 - pipe
        p2 - pipe
        p3 - pipe
        p4 - pipe
        p5 - pipe
        p6 - pipe
        p7 - pipe
        p8 - pipe
    )

    (:init
        (open-right p0)
        (open-down p0)
        (open-right p1)
        (open-down p1)
        (open-left p1)
        (open-up p2)
        (open-left p2)
        (open-right p3)
        (open-left p3)
        (open-up p4)
        (open-down p4)
        (open-right p5)
        (open-left p5)
        (open-down p6)
        (open-right p7)
        (open-left p8)
    )
    (:goal
        (and
            (open-right p0)
            (open-down p0)
            (open-right p1)
            (open-down p1)
            (open-left p1)
            (open-down p2)
            (open-left p2)
            (open-up p3)
            (open-down p3)
            (open-up p4)
            (open-down p4)
            (open-up p5)
            (open-down p5)
            (open-up p6)
            (open-up p7)
            (open-up p8)
        )
    )
)
