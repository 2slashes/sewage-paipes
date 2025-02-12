import unittest
from tree import Node

class TestTreeCycle(unittest.TestCase):
    def test_no_cycle(self):
        # ...existing code for test setup...
        root = Node((0,))
        child1 = Node((1,))
        child2 = Node((2,))
        root.children.append(child1)
        root.children.append(child2)
        self.assertFalse(root.has_cycle())

    def test_with_cycle(self):
        # ...existing code for test setup...
        root = Node((0,))
        child1 = Node((1,))
        child2 = Node((2,))
        root.children.append(child1)
        child1.children.append(child2)
        # Introducing a cycle: child2 -> root
        child2.children.append(root)
        self.assertTrue(root.has_cycle())

if __name__ == "__main__":
    unittest.main()
