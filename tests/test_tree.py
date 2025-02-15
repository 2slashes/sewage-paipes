from pipes_utils import flatten
from visualizer import print1DGrid
from tree import validator, pruner
from sample_data import (
    almostLoopGrid1,
    loopGrid2,
    noLoopGrid1,
    noLoopGrid2,
    loopGrid1Variables,
)


flattenGrid1 = flatten(almostLoopGrid1)
flattenGrid2 = flatten(loopGrid2)
flattenGrid3 = flatten(noLoopGrid1)
flattenGrid4 = flatten(noLoopGrid2)

print1DGrid(flattenGrid1)
print(validator(flattenGrid1))
print(pruner(loopGrid1Variables))

print1DGrid(flattenGrid2)
print(validator(flattenGrid2))

print1DGrid(flattenGrid3)
print(validator(flattenGrid3))

print1DGrid(flattenGrid4)
print(validator(flattenGrid4))
