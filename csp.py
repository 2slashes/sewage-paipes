from typing import Optional, Callable

class PipeType:
    """
    A class representing a type of pipe with connections in four directions: up, right, down, and left.
    """
    def __init__(self, arr: list[bool]):
        """
        Initialize a PipeType with an array of four booleans representing openings
        
        :param arr: A list of four booleans indicating whether there is an openings in the order [up, right, down, left].
        :raises Exception: If the length of the array is not 4.
        """
        if len(arr) == 4:
            self.arr = arr
        else:
            raise Exception("Cannot make pipe type with array of length != 4")

class Variable:
    """
    A class representing a variable for a pipe at each location in the CSP.
    """

    def __init__(self, name: tuple, domain: list[PipeType], assignment: Optional[PipeType]=None):
        """
        Initialize a Variable with a name, domain, and an optional assignment.
        
        :param name: A tuple representing the name of the variable.
        :param domain: A list of PipeType objects representing the domain of the variable.
        :param assignment: An optional PipeType object representing the current assignment.
        """
        self.name = name
        self.domain = domain
        self.active_domain = domain
        self.assignment = assignment

    def get_domain(self):
        """
        Get the domain of the variable.
        
        :return: A list of PipeType objects representing the domain.
        """
        return list(self.domain)
    
    def get_assignment(self):
        """
        Get the current assignment of the variable.
        
        :return: A PipeType object representing the current assignment.
        """
        return PipeType(list(self.assignment.arr))
    
    def prune(self, to_remove: list[PipeType]):
        """
        Prune the active domain by removing specified PipeType objects.
        
        :param to_remove: A list of PipeType objects to be removed from the active domain.
        """
        for pipe in to_remove:
            self.active_domain.remove(pipe)
        
    def assign(self, assignment: PipeType):
        """
        Assign a PipeType object to the variable.
        
        :param assignment: A PipeType object to be assigned to the variable.
        """
        self.assignment = assignment
    
class Constraint:
    """
    A class representing a constraint for a CSP.
    """
    def __init__(self, name: str, sat: Callable[[list[Variable]], bool], scope: list[Variable]):
        """
        Initialize a Constraint with a name, satisfaction function, and scope.
        
        :param name: A string representing the name of the constraint.
        :param sat: A callable function that takes a list of Variables and returns a boolean indicating if the constraint is satisfied.
        :param scope: A list of Variable objects representing the scope of the constraint.
        """
        self.name = name
        self.check_tuple = sat
        self.scope = scope
    
    def get_scope(self):
        """
        Get the scope of the constraint.
        
        :return: A list of Variable objects representing the scope.
        """
        return list(self.scope)

    def check_domains(self):
        """
        Check if all variables in the scope have non-empty active domains.
        
        :return: True if all variables have non-empty active domains, False otherwise.
        """
        for var in self.vars:
            if not len(var.active_domain):
                return False
        return True

    def add_to_scope(self, var: Variable):
        """
        Add a variable to the scope of the constraint.
        
        :param var: A Variable object to be added to the scope.
        """
        self.scope.append(var)
    
    def remove_from_scope(self, var: Variable):
        """
        Remove a variable from the scope of the constraint.
        
        :param var: A Variable object to be removed from the scope.
        """
        self.scope.remove(var)

class csp:
    '''
    csp
    '''
    def __init__(self, vars: list[Variable], cons: list[Constraint]):
        self.vars = vars
        self.cons = cons