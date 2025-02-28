(define
    (problem pipes148)
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
        (open-up p1)
        (open-right p1)
        (open-left p1)
        (open-down p2)
        (open-left p2)
        (open-right p3)
        (open-left p3)
        (open-right p4)
        (open-left p4)
        (open-up p5)
        (open-down p5)
        (open-right p6)
        (open-down p7)
        (open-right p8)

    )
    (:goal
        (and
            (open-down p0)
            (open-right p1)
            (open-down p2)
            (open-left p2)
            (open-up p3)
            (open-right p3)
            (open-down p4)
            (open-left p4)
            (open-up p5)
            (open-down p5)
            (open-right p6)
            (open-up p7)
            (open-right p7)
            (open-left p7)
            (open-up p8)
            (open-left p8)
        )
    )
)