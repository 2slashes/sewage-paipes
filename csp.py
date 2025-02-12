from typing import Optional, Callable

PipeType = tuple[bool, bool, bool, bool]


class DomainGenerator:
    all_domain: list[PipeType] = [
        (True, True, True, False),
        (True, True, False, True),
        (True, True, False, False),
        (True, False, True, True),
        (True, False, True, False),
        (True, False, False, True),
        (True, False, False, False),
        (False, True, True, True),
        (False, True, True, False),
        (False, True, False, True),
        (False, True, False, False),
        (False, False, True, True),
        (False, False, True, False),
        (False, False, False, True),
    ]

    @staticmethod
    def generate_domain(
        top: bool, right: bool, bottom: bool, left: bool
    ) -> list[PipeType]:
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
            domain = [pipe for pipe in domain if not pipe[0]]
        elif right:
            domain = [pipe for pipe in domain if not pipe[1]]
        if bottom:
            domain = [pipe for pipe in domain if not pipe[2]]
        elif left:
            domain = [pipe for pipe in domain if not pipe[3]]
        return domain


class Variable:
    """
    A class representing a variable for a pipe at each location in the CSP.
    """

    def __init__(
        self,
        location: tuple[int, int],
        domain: list[PipeType],
        assignment: Optional[PipeType] = None,
    ):
        """
        Initialize a Variable with a name, domain, and an optional assignment.

        :param name: A tuple representing the name of the variable.
        :param domain: A list of PipeType objects representing the domain of the variable.
        :param assignment: An optional PipeType object representing the current assignment.
        """
        self.name = location
        self.domain = domain
        self.active_domain = domain
        if assignment is not None:
            self.assign(assignment)

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
        return self.assignment

    def prune(self, to_remove: list[PipeType]):
        """
        Prune the active domain by removing specified PipeType objects.

        :param to_remove: A list of PipeType to be removed from the active domain.
        """
        for pipe in to_remove:
            self.active_domain.remove(pipe)

    def assign(self, assignment: PipeType):
        """
        Assign a PipeType to the variable.

        :param assignment: A PipeType to be assigned to the variable.
        """

        if assignment not in self.domain:
            raise Exception("Attempted to assign variable to value not in domain")
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

    def __init__(
        self,
        name: str,
        sat: Callable[[list[Optional[PipeType]]], bool],
        scope: list[Variable],
    ):
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
        for var in self.scope:
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
    """
    csp
    """

    def __init__(self, name: str, vars: list[Variable], cons: list[Constraint]):
        self.name = name
        self.vars: list[Variable] = []
        self.cons = cons
        self.vars_to_cons: dict = {}

        for var in vars:
            self.add_var(var)

        for con in cons:
            self.add_con(con)

    def add_var(self, var):
        if type(var) is not Variable:
            raise Exception(
                "Tried to add a non-variable object as a variable in", self.name
            )
        if var not in self.vars:
            self.vars.append(var)
            self.vars.vars_to_cons[var] = []

    def add_con(self, con):
        if type(con) is not Constraint:
            raise Exception(
                "Tried to add a non-constraint object as a constraint in", self.name
            )
        if con not in self.cons:
            for var in con.scope:
                if var not in self.vars_to_cons:
                    raise Exception(
                        "Trying to add constraint with unknown variable to", self.name
                    )
                self.vars_to_cons[var].append(con)
            self.cons.append(con)

    def get_cons(self):
        return self.cons.copy()

    def get_vars(self):
        return self.vars.copy()

    def get_cons_with_var(self, var):
        return self.vars_to_cons[var].copy()

    def backtracking_search(self):
        pass
