from pipe_typings import Assignment
import os
from random_rotation import random_rotate_board

def generate_all_pddl_and_state_files(solutions: list[Assignment], num_rotations: int, n: int, num_solutions: int):
    curr_dir = os.path.dirname(__file__)
    pddl_dir = os.path.join(curr_dir, f"../planning/pddl/problems/{n}/")
    try:
        os.makedirs(pddl_dir)
        print(f"Directory '{pddl_dir}' created successfully.")
    except FileExistsError:
        pass
    except PermissionError:
        print(f"Permission denied: Unable to create '{pddl_dir}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    state_dir = os.path.join(curr_dir, f"../planning/states/{n}/")
    try:
        os.makedirs(state_dir)
        print(f"Directory '{state_dir}' created successfully.")
    except FileExistsError:
        pass
    except PermissionError:
        print(f"Permission denied: Unable to create '{state_dir}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    count = 0
    for solution in solutions:
        random_rotations = random_rotate_board(solution, num_rotations)
        for rotation in random_rotations:
            pddl: str = generate_pddl(f"pipes{count}", "pipes", rotation, solution)
            with open(f"{pddl_dir}problem{count}.pddl", "w") as file:
                file.write(pddl)
            state_str: str = generate_state_str(rotation, solution)
            with open(f"{state_dir}state{count}", "w") as file:
                file.write(state_str)
            count += 1
            if count >= num_solutions:
                return

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

def generate_state_str(initial_state: Assignment, goal_state: Assignment):
    output = ""
    for pipe in initial_state:
        for dir in range(4):
            if pipe[dir]:
                output += "1"
            else:
                output += "0"
    output += "\n"
    for pipe in goal_state:
        for dir in range(4):
            if pipe[dir]:
                output += "1"
            else:
                output += "0"

    return output