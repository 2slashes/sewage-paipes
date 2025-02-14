from pipes_utils import flatten
from visualizer import print1DGrid
from tree import tree_sat
from sample_data import loopGrid1, loopGrid2, noLoopGrid1, noLoopGrid2


flattenGrid1 = flatten(loopGrid1)
flattenGrid2 = flatten(loopGrid2)
flattenGrid3 = flatten(noLoopGrid1)
flattenGrid4 = flatten(noLoopGrid2)

print1DGrid(flattenGrid1)
print(tree_sat(flattenGrid1))

print1DGrid(flattenGrid2)
print(tree_sat(flattenGrid2))

print1DGrid(flattenGrid3)
print(tree_sat(flattenGrid3))

print1DGrid(flattenGrid4)
print(tree_sat(flattenGrid4))
