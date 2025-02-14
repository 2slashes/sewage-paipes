import unittest
from csp import PipeType, Variable, Constraint


class TestPipeType(unittest.TestCase):
    def test_valid_pipe_type(self):
        pt = PipeType([True, False, True, False])
        self.assertEqual(pt.arr, [True, False, True, False])

    def test_invalid_pipe_type(self):
        with self.assertRaises(Exception):
            PipeType([True, False])


class TestVariable(unittest.TestCase):
    def setUp(self):
        self.pt1 = PipeType([True, False, True, False])
        self.pt2 = PipeType([False, True, False, True])
        self.var = Variable(("row", "col"), [self.pt1, self.pt2])

    def test_get_domain(self):
        domain = self.var.get_domain()
        self.assertEqual(len(domain), 2)

    def test_assign_and_get_assignment(self):
        self.var.assign(self.pt1)
        assignment = self.var.get_assignment()
        self.assertEqual(assignment.arr, self.pt1.arr)

    def test_prune(self):
        self.var.prune([self.pt1])
        self.assertEqual(len(self.var.active_domain), 1)
        self.assertNotIn(self.pt1, self.var.active_domain)


class TestConstraint(unittest.TestCase):
    def setUp(self):
        self.pt1 = PipeType([True, False, True, False])
        self.pt2 = PipeType([False, True, False, True])
        self.var1 = Variable(("r1", "c1"), [self.pt1, self.pt2])
        self.var2 = Variable(("r1", "c2"), [self.pt1])
        self.constraint = Constraint("test", lambda vars: True, [self.var1])

    def test_get_scope(self):
        scope = self.constraint.get_scope()
        self.assertIn(self.var1, scope)

    def test_add_and_remove_from_scope(self):
        self.constraint.add_to_scope(self.var2)
        scope = self.constraint.get_scope()
        self.assertIn(self.var2, scope)
        self.constraint.remove_from_scope(self.var1)
        scope = self.constraint.get_scope()
        self.assertNotIn(self.var1, scope)

    def test_check_domains(self):
        with self.assertRaises(AttributeError):
            self.constraint.var_has_active_domains()


if __name__ == "__main__":
    unittest.main()
