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
        self.assignment: Optional[PipeType] = None
        if assignment is not None:
            self.assign(assignment)

    def get_domain(self):
        """
        Get the domain of the variable.

        :return: A list of PipeType objects representing the domain.
        """
        return list(self.domain)

    def get_assignment(self) -> PipeType | None:
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

    def assign(self, assignment: PipeType) -> bool:
        """
        Assign a PipeType to the variable.
        In the context of a csp, you probably don't want to call this function. Instead, call the csp's assign function with the Variable object as a parameter. That way, the csp can track which variables have been assigned.

        :param assignment: A PipeType to be assigned to the variable.
        :returns: True if assignment is successful, False if not
        """

        if assignment not in self.domain:
            print("Attempted to assign variable to value not in domain")
            return False
        self.assignment = assignment
        return True

    def unassign(self) -> bool:
        """
        Unassign the variable by setting the assignment to None.
        In the context of a csp, you probably don't want to call this function. Instead, call the csp's unassign function with the Variable object as a parameter. That way, the csp can track which variables have been assigned.

        :returns: True if variable had an assignment that was removed, false if not
        """
        if self.assignment is not None:
            self.assignment = None
            return True
        return False

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
        check_tuple: Callable[[list[PipeType]], bool],

        scope: list[Variable],
    ):
        """
        Initialize a Constraint with a name, satisfaction function, and scope.

        :param name: A string representing the name of the constraint.
        :param sat: A callable function that takes a list of Variables and returns a boolean indicating if the constraint is satisfied.
        :param scope: A list of Variable objects representing the scope of the constraint.
        """
        self.name = name
        self.check_tuple = check_tuple
        self.scope = scope
        self.check_tuple = check_tuple

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

    def check_fully_assigned(self):
        '''
        check if all the variables in the constraint have been assigned

        :returns: True if all variables in the constraint's scope have been assigned, false if there are some variables that are unassigned
        '''
        for var in self.scope:
            if var.get_assignment() is None:
                return False
        return True


class csp:
    """
    csp
    """

    def __init__(self, name: str, vars: list[Variable], cons: list[Constraint]):
        self.name = name
        self.vars: list[Variable] = []
        self.cons: list[Constraint] = cons
        self.vars_to_cons: dict[Variable, list[Constraint]] = {}
        self.unassigned_vars: list[Variable] = []

        for var in vars:
            self.add_var(var)

        for con in cons:
            self.add_con(con)

    def add_var(self, var: Variable):
        if type(var) is not Variable:
            raise Exception(
                "Tried to add a non-variable object as a variable in", self.name
            )
        if var not in self.vars:
            self.vars.append(var)
            self.vars_to_cons[var] = []
            if var.assignment is not None:
                self.unassigned_vars.append(var)

    def add_con(self, con: Constraint):
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

    def get_cons_with_var(self, var: Variable):
        return self.vars_to_cons[var].copy()
    
    def assign_var(self, var: Variable, assignment: PipeType) -> bool:
        '''
        Assign a value to a specified variable

        :param var: Variable that gets assigned to a value
        :param assignment: PipeType value to assign to the variable
        :returns: True if assignment successful, False if not
        '''
        if var.assign(assignment):
            self.unassigned_vars.remove(var)
            return True
        return False
    
    def unassign_var(self, var: Variable) -> bool:
        '''
        Unassign a value from a variable

        :param var: Variable to remove the assignment from
        :returns: True if value was removed, False if there was no value to remove
        '''
        if var.unassign():
            self.unassigned_vars.append(var)
            return True
        return False


    def backtracking_search(self) -> bool:
        '''
        Solves the csp using recursive backtracking search. Solution will be stored in the variable objects related to this csp.

        :returns: True if a solution was found, false if not.
        '''
        # if there are no unassigned variables in the csp, then this is a solution
        if not self.unassigned_vars:
            return True
        # get an unassigned variable to assign next
        to_assign = self.unassigned_vars[0]
        # try every assignment for the variable
        for to_assign_assignment in to_assign.active_domain:
            to_assign.assign(to_assign_assignment)
            # check if the assignment violates any constraint
            violation = False
            for con in self.cons:
                # if a constraint is not fully assigned, move to the next constraint
                if not con.check_fully_assigned(): continue

                pipes: list[PipeType] = []
                # check if all the variables in the constraint have been assigned
                for var in con.get_scope():
                    var_assignment = var.get_assignment()
                    assert var_assignment is not None
                    pipes.append(var_assignment)
                # if a constraint is violated, the assignment doesn't work
                if not con.check_tuple(pipes):
                    violation = True
                    break
            # this assignment will give a full solution once everything else is assigned
            # the variables will stay assigned after returning
            if not violation and self.backtracking_search(): return True
                
                
        # if the code gets here, then none of the assignable values for the variable work.
        # unassign the variable and return false to indicate that the csp is unsolvable
        self.unassign_var(to_assign)
        return False
