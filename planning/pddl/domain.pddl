(define
    (domain pipes)
    (:requirements :strips :typing :negative-preconditions)
    (:types
        pipe - object
    )
    (:predicates
        (open-up ?p - pipe)
        (open-right ?p - pipe)
        (open-down ?p - pipe)
        (open-left ?p - pipe)
    )
    (:action ROTATE_PIPE_UP
        ; Rotate a pipe that only has an opening facing up 90 degrees clockwise, such that the opening is facing right.
        :parameters (?p - pipe)
        :precondition (and
            (open-up ?p)
            (not (open-right ?p))
            (not (open-down ?p))
            (not (open-left ?p))
        )
        :effect (and
            (not (open-up ?p))
            (open-right ?p)
        )
    )
    (:action ROTATE_PIPE_RIGHT
        ; Rotate a pipe that only has an opening facing right 90 degrees clockwise, such that the opening is facing down.
        :parameters (?p - pipe)
        :precondition (and
            (not (open-up ?p))
            (open-right ?p)
            (not (open-down ?p))
            (not (open-left ?p))
        )
        :effect (and
            (not (open-right ?p))
            (open-down ?p)
        )
    )
    (:action ROTATE_PIPE_DOWN
        ; Rotate a pipe that only has an opening facing down 90 degrees clockwise, such that the opening is facing left.
        :parameters (?p - pipe)
        :precondition (and
            (not (open-up ?p))
            (not (open-right ?p))
            (open-down ?p)
            (not (open-left ?p))
        )
        :effect (and
            (not (open-down ?p))
            (open-left ?p)
        )
    )
    (:action ROTATE_PIPE_LEFT
        ; Rotate a pipe that only has an opening facing left 90 degrees clockwise, such that the opening is facing up.
        :parameters (?p - pipe)
        :precondition (and
            (not (open-up ?p))
            (not (open-right ?p))
            (not (open-down ?p))
            (open-left ?p)
        )
        :effect (and
            (not (open-left ?p))
            (open-up ?p)
        )
    )
    (:action ROTATE_PIPE_UP_RIGHT
        ; Rotate a pipe that has openings facing up and right 90 degrees clockwise, such that the openings are facing right and down.
        :parameters (?p - pipe)
        :precondition (and
            (open-up ?p)
            (open-right ?p)
            (not (open-down ?p))
            (not (open-left ?p))
        )
        :effect (and
            (not (open-up ?p))
            (open-down ?p)
        )
    )
    (:action ROTATE_PIPE_DOWN_RIGHT
        ; Rotate a pipe that has openings facing down and right 90 degrees clockwise, such that the openings are facing left and down.
        :parameters (?p - pipe)
        :precondition (and
            (not (open-up ?p))
            (open-right ?p)
            (open-down ?p)
            (not (open-left ?p))
        )
        :effect (and
            (not (open-right ?p))
            (open-left ?p)
        )
    )
    (:action ROTATE_PIPE_DOWN_LEFT
        ; Rotate a pipe that has openings facing down and left 90 degrees clockwise, such that the openings are facing left and up.
        :parameters (?p - pipe)
        :precondition (and
            (not (open-up ?p))
            (not (open-right ?p))
            (open-down ?p)
            (open-left ?p)
        )
        :effect (and
            (not (open-down ?p))
            (open-up ?p)
        )
    )
    (:action ROTATE_PIPE_UP_LEFT
        ; Rotate a pipe that has openings facing up and left 90 degrees clockwise, such that the openings are facing right and up.
        :parameters (?p - pipe)
        :precondition (and
            (open-up ?p)
            (not (open-right ?p))
            (not (open-down ?p))
            (open-left ?p)
        )
        :effect (and
            (not (open-left ?p))
            (open-right ?p)
        )
    )
    (:action ROTATE_PIPE_UP_DOWN
        ; Rotate a pipe that has openings facing up and down 90 degrees clockwise, such that the openings are facing right and left.
        :parameters (?p - pipe)
        :precondition (and
            (open-up ?p)
            (not (open-right ?p))
            (open-down ?p)
            (not (open-left ?p))
        )
        :effect (and
            (open-left ?p)
            (open-right ?p)
            (not (open-up ?p))
            (not (open-down ?p))
        )
    )
    (:action ROTATE_PIPE_LEFT_RIGHT
        ; Rotate a pipe that has openings facing left and right 90 degrees clockwise, such that the openings are facing up and down.
        :parameters (?p - pipe)
        :precondition (and
            (not (open-up ?p))
            (open-right ?p)
            (not (open-down ?p))
            (open-left ?p)
        )
        :effect (and
            (not (open-left ?p))
            (not (open-right ?p))
            (open-up ?p)
            (open-down ?p)
        )
    )
    (:action ROTATE_PIPE_UP_RIGHT_DOWN
        ; Rotate a pipe that has openings facing up, right and down 90 degrees clockwise, such that the openings are facing right, down and left.
        :parameters (?p - pipe)
        :precondition (and
            (open-up ?p)
            (open-right ?p)
            (open-down ?p)
            (not (open-left ?p))
        )
        :effect (and
            (not (open-up ?p))
            (open-left ?p)
        )
    )
    (:action ROTATE_PIPE_RIGHT_DOWN_LEFT
        ; Rotate a pipe that has openings facing right, down and left 90 degrees clockwise, such that the openings are facing down, left and up.
        :parameters (?p - pipe)
        :precondition (and
            (not (open-up ?p))
            (open-right ?p)
            (open-down ?p)
            (open-left ?p)
        )
        :effect (and
            (not (open-right ?p))
            (open-up ?p)
        )
    )
    (:action ROTATE_PIPE_DOWN_LEFT_UP
        ; Rotate a pipe that has openings facing down, left and up 90 degrees clockwise, such that the openings are facing left, up and right.
        :parameters (?p - pipe)
        :precondition (and
            (open-up ?p)
            (not (open-right ?p))
            (open-down ?p)
            (open-left ?p)
        )
        :effect (and
            (not (open-down ?p))
            (open-right ?p)
        )
    )

    (:action ROTATE_PIPE_LEFT_UP_RIGHT
        ; Rotate a pipe that has openings facing left, up and right 90 degrees clockwise, such that the openings are facing up, right and down.
        :parameters (?p - pipe)
        :precondition (and
            (open-up ?p)
            (open-right ?p)
            (not (open-down ?p))
            (open-left ?p)
        )
        :effect (and
            (not (open-left ?p))
            (open-down ?p)
        )
    )
)