from pipe_typings import *
from typing import Optional, Callable, Literal
from math import sqrt
from pipes_utils import *

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
        elif bottom:
            domain = [pipe for pipe in domain if not pipe[2]]
        if right:
            domain = [pipe for pipe in domain if not pipe[1]]
        elif left:
            domain = [pipe for pipe in domain if not pipe[3]]
        return domain


class Variable:
    """
    A class representing a variable for a pipe at each location in the CSP.
    """

    def __init__(
        self,
        location: int,
        domain: list[PipeType] = [],
        assignment: Optional[PipeType] = None,
    ):
        """
        Initialize a Variable with a name, domain, and an optional assignment.

        :param name: A tuple representing the name of the variable.
        :param domain: A list of PipeType objects representing the domain of the variable.
        :param assignment: An optional PipeType object representing the current assignment.
        """
        self.location = location
        self.domain = domain
        self.active_domain = domain.copy()
        self.assignment: Optional[PipeType] = None
        if assignment is not None:
            self.assign(assignment)

    def get_active_domain(self):
        """
        Get the domain of the variable.

        :return: A list of PipeType objects representing the active domain.
        """
        return list(self.active_domain)

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
        return f"Variable {self.location}: {ass} in {self.active_domain}"


class Constraint:
    """
    A class representing a constraint for a CSP.
    """

    def __init__(
        self,
        name: str,
        validator: Callable[[list[PipeType]], bool],
        pruner: Callable[[list[Variable]], dict[Variable, list[PipeType]]],
        scope: list[Variable],
    ):
        """
        Initialize a Constraint with a name, satisfaction function, and scope.

        :param name: A string representing the name of the constraint.
        :param validator: A callable function that takes a list of PipeTypes and returns a list of active domains for each variable in scope
        :param pruner: A callable function that takes a list of assigned or unassigned variables and prunes their active domains, returns {var -> pruned_domains}
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

    def prune(self) -> dict[Variable, list[PipeType]]:
        """
        Prune the active domains of the variables in the constraint's scope.

        :returns: Active domains of the variables in the constraint's scope before pruning
        """
        return self._pruner(self.scope)

    def __repr__(self):
        return self.name


class CSP:
    """
    csp
    """

    def __init__(self, name: str, vars: list[Variable], cons: list[Constraint]):
        self.name = name
        self.vars: list[Variable] = []
        self.cons: list[Constraint] = []
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
            if var.assignment is None:
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

    def get_assignment(self) -> Assignment:
        """
        Get the current assignment of the variables in the csp.

        :returns: A list of PipeType objects representing the assignment of the variables.
        """
        assignment: Assignment = []
        for var in self.vars:
            value = var.get_assignment()
            if value is None:
                raise Exception(
                    "Tried to get assignment when some variables are unassigned"
                )
            assignment.append(value)

        return assignment
    
    def fc_one(self) -> bool:
        """
        Solves the csp using forward checking. Solution will be stored in the variable objects related to this csp.

        :returns: True if a solution was found, false if not.
        """
        # if there are no unassigned variables in the csp and it's not already in solutions, then this is a new solution
        if not self.unassigned_vars:
            curr_assignment = self.get_assignment()
            print1DGrid(curr_assignment) # type: ignore
            return True

        # get an unassigned variable to assign next
        curr_var = self.unassigned_vars[0]
        # try every active assignment for the variable
        for assignment in curr_var.active_domain:
            self.unassign_var(curr_var)
            self.assign_var(curr_var, assignment)

            # prune values and accumulate pruned values
            pruned_domains: dict[Variable, list[PipeType]] = {}
            no_active_domains = False
            for con in self.get_cons_with_var(curr_var):
                pruned = con.prune()
                for var in pruned:
                    if var in pruned_domains:
                        pruned_domains[var] += pruned[var]
                    else:
                        pruned_domains[var] = pruned[var]

                if not con.var_has_active_domains():
                    no_active_domains = True
                    break

            # no dead-ends, keep going, adding to solutions if everything is assigned
            if not no_active_domains and self.fc_one():
                return True

            # restore the active domains and try another variable
            for var in pruned_domains:
                var.active_domain += pruned_domains[var]
        self.unassign_var(curr_var)
        return False

    def fc_all(self, solutions: list[Assignment]) -> None:
        """
        Solves the csp using forward checking. Solution will be stored in the variable objects related to this csp.

        :returns: True if a solution was found, false if not.
        """
        # if there are no unassigned variables in the csp and it's not already in solutions, then this is a new solution
        if not self.unassigned_vars:
            curr_assignment = self.get_assignment()
            if curr_assignment not in solutions:
                # print1DGrid(curr_assignment) # type: ignore
                for con in self.cons:
                    violated = con.violated()
                    if violated:
                        print(f"constraint {con.name} violated: {con.violated()}")
                        raise Exception("chyme")
                solutions.append(curr_assignment)
                # print(len(solutions))
                # print()
            return

        # get an unassigned variable to assign next
        curr_var = self.unassigned_vars[0]
        # try every active assignment for the variable
        for assignment in curr_var.active_domain:
            self.unassign_var(curr_var)
            self.assign_var(curr_var, assignment)

            # prune values and accumulate pruned values
            pruned_domains: dict[Variable, list[PipeType]] = {}
            no_active_domains = False
            for con in self.get_cons_with_var(curr_var):
                pruned = con.prune()
                for var in pruned:
                    if var in pruned_domains:
                        pruned_domains[var] += pruned[var]
                    else:
                        pruned_domains[var] = pruned[var]

                if not con.var_has_active_domains():
                    no_active_domains = True
                    break

            # no dead-ends, keep going, adding to solutions if everything is assigned
            if not no_active_domains:
                self.fc_all(solutions)

            # restore the active domains and try another variable
            for var in pruned_domains:
                var.active_domain += pruned_domains[var]
        self.unassign_var(curr_var)

    def gac_one(self) -> bool:
        """
        Solves the csp using generalized arc consistency. Solution will be stored in the variable objects related to this csp.

        :returns: True if a solution was found, false if not.
        """
        # all variables in the csp have been assigned
        if not self.unassigned_vars:
            curr_assignment = self.get_assignment()
            print1DGrid(curr_assignment) # type: ignore
            return True
        # get an unassigned variable to assign next
        curr_var = self.unassigned_vars[0]
        # try every active assignment for the variable
        for assignment in curr_var.active_domain:
            # print(f"assigning value {assignment} to variable {curr_var}")
            self.unassign_var(curr_var)
            self.assign_var(curr_var, assignment)
            pruned_domains: dict[Variable, list[PipeType]] = {}

            # check if the assignment leads to a dead end (i.e. any variable having no active domains)
            no_active_domains = False
            pruned_domains = self.ac3(self.get_cons_with_var(curr_var))
            for var in pruned_domains:
                if not var.get_active_domain():
                    no_active_domains = True
                    break
            # this assignment will give a full solution once everything else is assigned
            # the variables will stay assigned after returning
            if not no_active_domains and self.gac_one():
                return True

            # dead-end (no active domains for some variable) reached, restore the active domains
            for var in pruned_domains:
                var.active_domain += pruned_domains[var]

        # if the code gets here, then none of the assignable values for the variable work.
        # unassign the variable and return false to indicate that the csp is unsolvable
        self.unassign_var(curr_var)
        return False

    def ac3(self, q: list[Constraint]) -> dict[Variable, list[PipeType]]:
        pruned_domains: dict[Variable, list[PipeType]] = {}
        while len(q):
            # get the variables pruned when checking for satisfying tuples with the first constraint
            cur_con: Constraint = q.pop(0)
            pruned: dict[Variable, list[PipeType]] = cur_con.prune()
            for var in pruned:
                if var in pruned_domains:
                    pruned_domains[var] += pruned[var]
                else:
                    pruned_domains[var] = pruned[var]
                if not len(var.get_active_domain()):
                    # the active domain of a variable is empty, no need to bother computing any more for this assignment
                    return pruned_domains
                cons_to_add = self.get_cons_with_var(var)
                for c in cons_to_add:
                    if c not in q:
                        q.append(c)
        return pruned_domains
    
    def gac_all(self, solutions: list[Assignment]) -> None:
        """
        Finds all solutions to the csp using generalized arc consistency. Solutions will be stored in the solutions list that is passed in as a parameter.

        :params solutions: a list where the solutions will be stored
        """
        # all variables in the csp have been assigned
        if not self.unassigned_vars:
            curr_assignment = self.get_assignment()
            if curr_assignment not in solutions:
                # print1DGrid(curr_assignment) # type: ignore
                for con in self.cons:
                    violated = con.violated()
                    if violated:
                        print(f"constraint {con.name} violated: {con.violated()}")
                        raise Exception("chyme")
                solutions.append(curr_assignment)
                # print(len(solutions))
                # print()
            return
        # get an unassigned variable to assign next
        curr_var = self.unassigned_vars[0]
        # try every active assignment for the variable
        for assignment in curr_var.active_domain:
            # print(f"assigning value {assignment} to variable {curr_var}")
            self.unassign_var(curr_var)
            self.assign_var(curr_var, assignment)
            pruned_domains: dict[Variable, list[PipeType]] = {}

            # check if the assignment leads to a dead end (i.e. any variable having no active domains)
            no_active_domains = False
            pruned_domains = self.ac3(self.get_cons_with_var(curr_var))
            for var in pruned_domains:
                if not var.get_active_domain():
                    no_active_domains = True
                    break
            # this assignment will give a full solution once everything else is assigned
            # the variables will stay assigned after returning
            if not no_active_domains:
                self.gac_all(solutions)

            # dead-end (no active domains for some variable) reached, restore the active domains
            for var in pruned_domains:
                var.active_domain += pruned_domains[var]

        # if the code gets here, then none of the assignable values for the variable work.
        # unassign the variable
        self.unassign_var(curr_var)

PIPE_CHAR: dict[PipeType, str] = {
    (True, False, False, False): "╵",  # Open at the top
    (False, True, False, False): "╶",  # Open at the right
    (False, False, True, False): "╷",  # Open at the bottom
    (False, False, False, True): "╴",  # Open at the left
    (True, True, False, False): "└",  # Elbow (bottom-left)
    (True, False, True, False): "│",  # Vertical pipe
    (True, False, False, True): "┘",  # Elbow (bottom-right)
    (False, True, True, False): "┌",  # Elbow (top-left)
    (False, True, False, True): "─",  # Horizontal pipe
    (False, False, True, True): "┐",  # Elbow (top-right)
    (True, True, True, False): "├",  # T-junction (left, down, up)
    (True, True, False, True): "┴",  # T-junction (left, right, down)
    (True, False, True, True): "┤",  # T-junction (right, down, up)
    (False, True, True, True): "┬",  # T-junction (left, right, up)
}
PipeName = Literal[
    "Up",
    "Right",
    "Down",
    "Left",
    "UpRight",
    "UpDown",
    "UpLeft",
    "RightDown",
    "RightLeft",
    "DownLeft",
    "UpRightDown",
    "UpRightLeft",
    "UpDownLeft",
    "RightDownLeft",
]
PIPE: dict[PipeName, PipeType] = {
    "Up": (True, False, False, False),
    "Right": (False, True, False, False),
    "Down": (False, False, True, False),
    "Left": (False, False, False, True),
    "UpRight": (True, True, False, False),
    "UpDown": (True, False, True, False),
    "UpLeft": (True, False, False, True),
    "RightDown": (False, True, True, False),
    "RightLeft": (False, True, False, True),
    "DownLeft": (False, False, True, True),
    "UpRightDown": (True, True, True, False),
    "UpRightLeft": (True, True, False, True),
    "UpDownLeft": (True, False, True, True),
    "RightDownLeft": (False, True, True, True),
}


def print1DGrid(pipes: list[Optional[PipeType]]) -> None:
    n = int(sqrt(len(pipes)))
    for i in range(len(pipes)):
        pipe = pipes[i]
        if pipe is None:
            print("•", end="")
        else:
            print(PIPE_CHAR[pipe], end="")
        if i % n == n - 1:
            print()