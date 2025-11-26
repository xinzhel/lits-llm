import unittest
import sys
sys.path.append('../..')
from langagent.search.mcts import MCTS, MCTSNode  # Adjust the import based on your project structure

class TestUCTSelect(unittest.TestCase):
    
    def test_uct_select_returns_child_with_higher_score_when_child_not_visited(self):
        """
        In this test, the parent node has 10 visits.
        Child1 has zero visits, while Child2 has 5 visits.
        With identical children_priority, child1 should get a higher score because its visit count is lower.
        """
        mcts = MCTS(w_exp=1.0)
        parent = MCTSNode(state=None, action=None)
        parent.cum_rewards = [0] * 10  # Simulate 10 visits at parent
        
        # Create two children with different visit counts.
        child1 = MCTSNode(state=None, action=None, parent=parent)
        child2 = MCTSNode(state=None, action=None, parent=parent)
        child1.cum_rewards = []            # 0 visits
        child2.cum_rewards = [1, 1, 1, 1, 1]  # 5 visits
        
        # Set children and their corresponding priority (both equal here).
        parent.children = [child1, child2]
        parent.children_priority = [1.0, 1.0]
        
        # Compute scores internally:
        # For child1: score = 1.0 * sqrt(log(10) / max(1, 0)) = sqrt(log(10))
        # For child2: score = 1.0 * sqrt(log(10) / 5)
        # Since sqrt(log(10)) > sqrt(log(10)/5), child1 should be selected.
        selected = mcts._uct_select(parent)
        self.assertEqual(selected, child1)

    def test_uct_select_returns_child_with_higher_priority(self):
        """
        In this test, even though child1 has zero visits (which tends to boost its score),
        we set a much higher priority for child2. With a larger number of parent's visits,
        the high priority should outweigh the effect of more child visits, and child2 should be selected.
        """
        mcts = MCTS(w_exp=1.0)
        parent = MCTSNode(state=None, action=None)
        parent.cum_rewards = [0] * 100  # Simulate 100 visits at parent
        
        # Create two children.
        child1 = MCTSNode(state=None, action=None, parent=parent)
        child2 = MCTSNode(state=None, action=None, parent=parent)
        child1.cum_rewards = []  # 0 visits
        child2.cum_rewards = [1, 1, 1, 1, 1]  # 5 visits
        
        # Set parent's children and adjust children_priority so child2 is more promising.
        parent.children = [child1, child2]
        parent.children_priority = [1.0, 5.0]
        
        # Expected: Despite child2 having more visits, its high priority should yield a higher UCT score.
        selected = mcts._uct_select(parent)
        self.assertEqual(selected, child2)
    
    def test_uct_select_returns_none_when_no_children(self):
        """
        If the node has no children, _uct_select should return None.
        """
        mcts = MCTS()
        parent = MCTSNode(state=None, action=None)
        parent.cum_rewards = [0] * 10
        parent.children = []
        parent.children_priority = []
        selected = mcts._uct_select(parent)
        self.assertIsNone(selected)

if __name__ == '__main__':
    unittest.main()
