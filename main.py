from csp import *

n = 5
variables: list[list[Variable]] = []

for i in range(n):
    row: list[Variable] = []
    for j in range(n):
        top = i == 0
        bottom = i == n - 1
        left = j == 0
        right = j == n - 1
        var = Variable(name=(i, j), domain=DomainGenerator.generate_domain(top, right, bottom, left))
        row.append(var)
    variables.append(row)

