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
        
    def __eq__(self, other):
        """
        Compare two PipeType objects for equality.
        
        :param other: A PipeType object to compare to.
        :return: True if the two PipeType objects are equal, False otherwise.
        """
        return self.arr == other.arr
    
    def __repr__(self):
        """
        Return a string representation of the PipeType.
        
        :return: A string representing the PipeType.
        """
        return f"PipeType({self.arr})"
    
class DomainGenerator:
    all_domain = [
        PipeType([True, True, True, False]),
        PipeType([True, True, False, True]),
        PipeType([True, True, False, False]),
        PipeType([True, False, True, True]),
        PipeType([True, False, True, False]),
        PipeType([True, False, False, True]),
        PipeType([True, False, False, False]),
        PipeType([False, True, True, True]),
        PipeType([False, True, True, False]),
        PipeType([False, True, False, True]),
        PipeType([False, True, False, False]),
        PipeType([False, False, True, True]),
        PipeType([False, False, True, False]),
        PipeType([False, False, False, True]),
    ]

    @staticmethod
    def generate_domain(top: bool, right: bool, bottom: bool, left: bool):
        """
        Generate a domain based on the four boolean flags:
        if i == 0: 0 is false (top)
        if i == n-1: 2 is false (bottom)

        if j == 0: 3 is false (left)
        if j == n-1: 1 is false (right)

        :param top: A boolean indicating if the top of the pipe is blocked.
        :param right: A boolean indicating if the right of the pipe is blocked.
        :param bottom: A boolean indicating if the bottom of the pipe is blocked.
        :param left: A boolean indicating if the left of the pipe is blocked.
        :return: A list of PipeType objects representing the domain.
        
        """
        domain = DomainGenerator.all_domain.copy()
        if top:
            domain = [pipe for pipe in domain if not pipe.arr[0]]
        elif right:
            domain = [pipe for pipe in domain if not pipe.arr[1]]
        if bottom:
            domain = [pipe for pipe in domain if not pipe.arr[2]]
        elif left:
            domain = [pipe for pipe in domain if not pipe.arr[3]]
        return domain


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
    
    def unassign(self):
        """
        Unassign the variable by setting the assignment to None.
        """
        self.assignment = None
    
    def __repr__(self):
        """
        Return a string representation of the variable.
        
        :return: A string representing the variable.
        """
        ass = "Unassigned" if self.assignment is None else self.assignment
        return f"Variable {self.name}: {ass} in {self.active_domain}"
    
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