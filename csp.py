from typing import Optional


class PipeType:
    """
    A pipe type :D
    Up, right, down, left
    """
    def __init__(self, arr: list):
        if len(arr) == 4:
            self.arr = arr
        else:
            raise Exception("Cannot make pipe type with array of length != 4")

class Variable:
    '''
    Variable for pipe (each location)
    '''
    name: tuple
    domain: list[PipeType]

    def __init__(self, name: tuple, domain: list[PipeType], assignment: Optional[PipeType]):
        self.name = name
        self.domain = domain
        self.active_domain = domain
        self.assignment = assignment

    def get_domain(self):
        return list(self.domain)
    
    def get_assignment(self):
        return PipeType(list(self.assignment.arr))
    
    def prune(self, to_remove: list[PipeType]):
        for pipe in to_remove:
            self.active_domain.remove(pipe)
        
    def assign(self, assignment: PipeType):
        self.assignment = assignment
    
class Constraint:
    '''
    Constraint for a CSP
    '''
    def __init__(self, name, sat, vars):
        self.name = name
        self.sat = sat
        self.vars = vars

    def check_tuple(self, t):
        return self.sat(t)
    
    def get_vars(self):
        return vars

# csp class
# 