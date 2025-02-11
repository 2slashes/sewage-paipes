from typing import Optional, Callable


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
    def __init__(self, name: str, sat: Callable[[list[Variable]], bool], scope: list[Variable]):
        self.name = name
        self.check_tuple = sat
        self.scope = scope
    
    def get_scope(self):
        return self.scope

    def check_domains(self):
        for var in self.vars:
            if not len(var.active_domain):
                return False
        return True

    def add_to_scope(self, var: Variable):
        self.scope.append(var)
    
    def remove_from_scope(self, var: Variable):
        self.scope.remove(var)

class csp:
    '''
    csp
    '''
    def __init__(self, vars: list[Variable], cons: list[Constraint]):
        self.vars = vars
        self.cons = cons