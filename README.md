# Sewage pAIpes

This repository hosts the implementation of the 3 methods we used to solve various aspects of the _pipes_ puzzle. The approaches are:

1. Constraint Satisfaction (To generate solvable configurations)
2. Planning (To solve the puzzle knowing its final state)
3. Deep Learning (To solve the puzzle without knowing its final state)

## Constraint Satisfaction (Puzzle Generation)

Run `constraint_satisfaction/main.py` and follow the prompts to generate solvable puzzle configurations

## Planning

After generating solvable PDDL files, we use the [planutils](https://github.com/AI-Planning/planutils) docker image to
solve the puzzles. To do this, go into the planning directory, then run:

1. `docker run -it --privileged -v ./pddl:/root/pddl -w /root/pddl aiplanning/planutils:latest bash`
2. `planutils activate`
3. `./generate_solutions`

This generates solutions. `planning/solution_parser.py` parses the problem and solution in a CSV file, making it easier to work
with in neural networks.

## Deep Learning

In the `deep-learning/` directory, the dependencies are listed in `requirements.txt` and can be installed in a virtual environment or conda environment. This is still a work in progress, but the neural network resides in `main.ipynb`.
