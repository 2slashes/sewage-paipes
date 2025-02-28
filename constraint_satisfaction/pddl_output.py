from pipe_typings import Assignment

def generate_pddl(problem_name: str, domain_name: str, initial_state: Assignment, goal_state: Assignment):
    return f"""(define
    (problem {problem_name})
    (:domain {domain_name})
    (:objects
{generate_pddl_pipes_objects(len(initial_state))}
    )
    (:init
{generate_pddl_pipes_init(initial_state)}
    )
    (:goal
{generate_pddl_pipes_goal(goal_state)}
    )
)"""

def generate_pddl_pipes_objects(n: int) -> str:
    return "        " + "\n        ".join(f"p{i} - pipe" for i in range(n))

def generate_pddl_pipes_init(initial_state: Assignment) -> str:
    output = ""
    for i, pipe in enumerate(initial_state):
        for dir in range(4):
            if pipe[dir]:
                if dir == 0:
                    output += f"        (open-up p{i})\n"
                if dir == 1:
                    output += f"        (open-right p{i})\n"
                if dir == 2:
                    output += f"        (open-down p{i})\n"
                if dir == 3:
                    output += f"        (open-left p{i})\n"
    return output

def generate_pddl_pipes_goal(goal_state: Assignment) -> str:
    output = "        (and\n"
    for i, pipe in enumerate(goal_state):
        for dir in range(4):
            if pipe[dir]:
                if dir == 0:
                    output += f"            (open-up p{i})\n"
                if dir == 1:
                    output += f"            (open-right p{i})\n"
                if dir == 2:
                    output += f"            (open-down p{i})\n"
                if dir == 3:
                    output += f"            (open-left p{i})\n"
    output += "        )"
    return output