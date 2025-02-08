class Variable:
    '''
    Variable for a CSP.
    '''
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain.copy()

    def get_domain(self):
        return self.domain.copy()
    
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