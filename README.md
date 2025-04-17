# Sewage pAIpes

This repository hosts the implementation of the 3 methods we used to solve various aspects of the _pipes_ puzzle. The approaches are:

1. Constraint Satisfaction (To generate solvable configurations)
2. Planning (To solve the puzzle knowing its final state)
3. Deep Learning (To solve the puzzle without knowing its final state)

## Constraint Satisfaction (Puzzle Generation)

There are three files you can run directly. Run them to see the required options:

- `main.py`: Prints out all solutions given a specific nxn board dimension
- `dl_data.py`: Outputs training, testing, and puzzle data for deep learning. Output is located in `deep_learning/data/`. Also can be used to augment outliers
  - Training and testing data is used to train the model and test its accuracy per move
  - Puzzle data are puzzles that the network will eventually play on
  - User specifies number of solutions and number of puzzles (variations) to generate per solution
  - With the `--aug` option, it augments the data located in `deep_learning/data/outlier.csv` and appends them to `train.csv` and `test.csv`
  - With the `--print` option, it prints the solution as it generates them
- `planning_data.py`: Outputs PDDL files for the planner to solve. Output is located in `planning/pddl/problems/`

## Planning

After generating solvable PDDL files, we use the [planutils](https://github.com/AI-Planning/planutils) docker image to
solve the puzzles. To do this, go into the planning directory, then run:

```bash
docker run -it --privileged -v ./pddl:/root/pddl -w /root/pddl --rm aiplanning/planutils:latest bash -c "source ./generate_solutions"
```

## Deep Learning

In the `deep-learning/` directory, the dependencies are listed in `requirements.txt` and can be installed in a virtual environment or conda environment. The main entry point is `main.ipynb`, the notebook contains code and documentation for the training and solving process.
