from pipe_typings import Assignment
import os
from random_rotation import random_rotate_board


def generate_all_pddl_and_state_files(
    solutions: list[Assignment], num_rotations: int, n: int, max_solutions: int
):
    """
    Generates a specified number of PDDL problem files and state files for a list of pipes solutions. Outputs PDDL files to ../planning/pddl/problems/{n}/, and state files to ../planning/states/{n}/, where n is the dimension (width and height) of the input board.

    :params solutions: list of solutions to the pipes puzzle
    :params num_rotations: number of times to rotate each solution, maintaining the same goal state but changing the initial state of the puzzle
    :params n: dimension (width and height) of the puzzle
    :params max_solutions: maximum number of files that will be generated
    """

    # create the directory for PDDL problem files if it doesn't exist
    curr_dir = os.path.dirname(__file__)
    pddl_dir = os.path.join(curr_dir, f"../planning/pddl/problems/{n}/")
    try:
        os.makedirs(pddl_dir)
    except FileExistsError:
        pass
    except PermissionError:
        print(f"Permission denied: Unable to create '{pddl_dir}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # create the directory for state files if it doesn't exist
    state_dir = os.path.join(curr_dir, f"../planning/states/{n}/")
    try:
        os.makedirs(state_dir)
    except FileExistsError:
        pass
    except PermissionError:
        print(f"Permission denied: Unable to create '{state_dir}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # track the number of files generated
    cur_num_files = 0
    # iterate through solutions. A solution represents the goal state of the puzzle.
    for solution in solutions:
        # rotate each solution the specified number of times
        # random rotations of the solution represent the initial state of the puzzle.
        random_rotations = random_rotate_board(solution, num_rotations)
        for rotation in random_rotations:
            # generate a PDDL problem file for each initial state
            pddl: str = generate_pddl(
                f"pipes{cur_num_files}", "pipes", rotation, solution
            )
            with open(f"{pddl_dir}problem{cur_num_files}.pddl", "w") as file:
                file.write(pddl)
            # generate a state file, which contains a binary representation of the initial state of the puzzle, for each initial state
            state_str: str = generate_state_str(rotation, solution)
            with open(f"{state_dir}state{cur_num_files}", "w") as file:
                file.write(state_str)
            cur_num_files += 1
            # check if enough solutions have been generated
            if cur_num_files >= max_solutions and max_solutions != -1:
                return


def generate_pddl(
    problem_name: str,
    domain_name: str,
    initial_state: Assignment,
    goal_state: Assignment,
):
    """
    Generates a string that matches a PDDL problem file for pipes puzzles, for the given initial and goal states

    :params problem_name: The problem name for this instance of the puzzle
    :params domain_name: Name of the domain file that corresponds to this problem file
    :params initial_state: The initial state of the puzzle
    :params goal_state: The goal state of the puzzle
    """
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
    """
    Generates a string that matches objects in the PDDL for a pipes problem, using the type "pipe" for a pipe and p{i} for the name, where i is the next pipe to add

    :params n: the size of the puzzle
    """
    return "        " + "\n        ".join(f"p{i} - pipe" for i in range(n))


def generate_pddl_pipes_init(initial_state: Assignment) -> str:
    """
    Generates a string that matches the initial state in the PDDL for a pipes problem.

    :params initial_state: The initial state to match
    """
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
    """
    Generates a string that matches the goal state in the PDDL for a pipes problem.

    :params goal_state: The goal state to match
    """
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
    """
    Generates a string that matches a binary representation of an initial state and goal state for a pipes problem.

    :params initial_state: The initial state to match
    :params goal_state: The goal state to match
    """
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
