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
        validator: Callable[[list[PipeType]], bool],
        pruner: Callable[[list[Optional[PipeType]]], list[list[PipeType]]],
        scope: list[Variable],
    ):
        """
        Initialize a Constraint with a name, satisfaction function, and scope.

        :param name: A string representing the name of the constraint.
        :param validator: A callable function that takes a list of PipeTypes and returns a boolean indicating if the constraint is satisfied.
        :param pruner: A callable function that takes a fully or partially assignment of PipeTypes in its scope and outputs the pruned domains of the scoped variables
        :param scope: A list of Variable objects representing the scope of the constraint.
        """
        self.name = name
        self._validator = validator
        self._pruner = pruner
        self.scope = scope

    def get_scope(self):
        """
        Get the scope of the constraint.

        :return: A list of Variable objects representing the scope.
        """
        return list(self.scope)

    def var_has_active_domains(self):
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
        """
        check if all the variables in the constraint have been assigned

        :returns: True if all variables in the constraint's scope have been assigned, false if there are some variables that are unassigned
        """
        for var in self.scope:
            if var.get_assignment() is None:
                return False
        return True

    def violated(self):
        """
        Check if the constraint is violated.

        :returns: True if the constraint is violated, False if not.
        """
        if not self.check_fully_assigned():
            raise Exception(
                "Tried to check if a constraint was violated with unassigned variables"
            )
        pipes: list[PipeType] = []
        for var in self.scope:
            var_assignment = var.get_assignment()
            assert var_assignment is not None
            pipes.append(var_assignment)
        return not self._validator(pipes)

    def prune(self):
        """
        Prune the active domains of the variables in the constraint's scope.
        """
        pipes: list[Optional[PipeType]] = [var.get_assignment() for var in self.scope]
        new_active_domains = self._pruner(pipes)
        for i in range(len(self.scope)):
            self.scope[i].active_domain = new_active_domains[i]


class CSP:
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
        """
        Assign a value to a specified variable

        :param var: Variable that gets assigned to a value
        :param assignment: PipeType value to assign to the variable
        :returns: True if assignment successful, False if not
        """
        if var.assign(assignment):
            self.unassigned_vars.remove(var)
            return True
        return False

    def unassign_var(self, var: Variable) -> bool:
        """
        Unassign a value from a variable

        :param var: Variable to remove the assignment from
        :returns: True if value was removed, False if there was no value to remove
        """
        if var.unassign():
            self.unassigned_vars.append(var)
            return True
        return False

    def no_active_domains(self) -> bool:
        """
        Check if all unassigned variables have an empty active domain

        :returns: True if all unassigned variables have an empty active domain, False if there is at least one unassigned variable with a non-empty active domain
        """
        for var in self.unassigned_vars:
            if len(var.active_domain):
                return False
        return True

    def backtracking_search(self) -> bool:
        """
        Solves the csp using recursive backtracking search. Solution will be stored in the variable objects related to this csp.

        :returns: True if a solution was found, false if not.
        """
        # if there are no unassigned variables in the csp, then this is a solution
        if not self.unassigned_vars:
            return True
        # get an unassigned variable to assign next
        curr_var = self.unassigned_vars[0]
        # try every assignment for the variable
        for assignment in curr_var.active_domain:
            self.assign_var(curr_var, assignment)

            # check if the assignment violates any constraint
            violated = False
            for con in self.get_cons_with_var(curr_var):
                if not con.check_fully_assigned():
                    continue
                if con.violated():
                    violated = True
                    break
            # this assignment will give a full solution once everything else is assigned
            # the variables will stay assigned after returning
            if not violated and self.backtracking_search():
                return True

        # if the code gets here, then none of the assignable values for the variable work.
        # unassign the variable and return false to indicate that the csp is unsolvable
        self.unassign_var(curr_var)
        return False

    def forward_checking(self) -> bool:
        """
        Solves the csp using forward checking. Solution will be stored in the variable objects related to this csp.

        :returns: True if a solution was found, false if not.
        """
        # if there are no unassigned variables in the csp, then this is a solution
        if not self.unassigned_vars:
            return True
        # get an unassigned variable to assign next
        curr_var = self.unassigned_vars[0]
        # try every active assignment for the variable
        for assignment in curr_var.active_domain:
            self.assign_var(curr_var, assignment)

            # check if the assignment leads to a dead end (i.e. no variable has active domains)
            no_active_domains = False
            for con in self.get_cons_with_var(curr_var):
                con.prune()
                if self.no_active_domains():
                    no_active_domains = True
                    break
            # this assignment will give a full solution once everything else is assigned
            # the variables will stay assigned after returning
            if not no_active_domains and self.forward_checking():
                return True

        # if the code gets here, then none of the assignable values for the variable work.
        # unassign the variable and return false to indicate that the csp is unsolvable
        self.unassign_var(curr_var)
        return False
