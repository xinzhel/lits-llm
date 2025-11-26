from typing import List, Tuple, Optional, Callable, Generic, TypeVar
import itertools
import numpy as np
from typing import Dict, Any
from ...structures import StateT, ActionT
from ...structures.trace import _serialize_obj, _deserialize_obj
from ...type_registry import TYPE_REGISTRY

class SearchNode(Generic[StateT, ActionT]):
    
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self, 
        state: Optional[StateT], 
        action: Optional[ActionT], 
        parent: Optional['SearchNode'] = None, 
        fast_reward: float = -1, 
        children: Optional[List['SearchNode']] = None, 
        is_terminal: bool = False
    ):
        """
        A node in the search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        """
        self.id = next(SearchNode.id_iter)
        self.state = state
        self.action = action
        self.parent = parent
        self.children: List['SearchNode'] = children if children is not None else []
        self.is_continuous = False
        self.is_terminal = is_terminal
        self.is_terminal_for_repeat = False
        self.bn_score = -1
        self.state_conf = -1
        self.fast_reward = fast_reward
        # probability distribution over children for puct
        # self.children_priority = children_priority if children_priority is not None else []

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    def add_child(self, child: 'SearchNode'):
        self.children.append(child)
    
    def get_trace(self) -> List[Tuple[ActionT, StateT, float]]:
        """ Returns the sequence of actions and states from the root to the current node """
        node, path = self, []
        while node is not None:
            path.append((node.action, node.state, node.reward))
            node = node.parent
        path = path[::-1] # Reverse the path to get actions and states in order
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert self to a JSON-safe dict, serializing state/action recursively."""

        return {
            "id": self.id,
            "state": _serialize_obj(self.state),
            "action": _serialize_obj(self.action),
            "is_continuous": self.is_continuous,
            "is_terminal": self.is_terminal,
            "bn_score": self.bn_score,
            "state_conf": self.state_conf,
            "fast_reward": self.fast_reward,
        }

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> 'SearchNode':
        """Reconstruct node without parent/children links yet."""

        node = cls(state=None, action=None)
        node.id = dct["id"]
        node.state = _deserialize_obj(dct["state"])
        node.action = _deserialize_obj(dct["action"])
        node.is_continuous = dct.get("is_continuous", False)
        node.is_terminal = dct.get("is_terminal", False)
        node.bn_score = dct.get("bn_score", -1)
        node.state_conf = dct.get("state_conf", -1)
        node.fast_reward = dct.get("fast_reward", -1)
        return node

class MCTSNode(SearchNode[StateT, ActionT]):
    def __init__(self, state: Optional[StateT], action: Optional[ActionT], parent: Optional['MCTSNode'] = None,
                 fast_reward: float = -1, fast_reward_details=None,
                 is_terminal: bool = False, calc_q: Callable[[List[float]], float] = None):
        """
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        super().__init__(state, action, parent, children=None, is_terminal=is_terminal)
        
        self.fast_reward = fast_reward # reward for action (no state)
        self.reward = fast_reward
        self.fast_reward_details = fast_reward_details if fast_reward_details is not None else {}
        self.cum_rewards = []
        self.calc_q = calc_q if calc_q is not None else MCTSNode.DEFAULT_CALC_Q
        self.from_simulate= False  # whether this node is created in `_expand` called by `_simulate` 
        self.is_simulated = False  # whether this node has chosen for simulation 
        self.from_expand = False  # whether this node is created during the expansion phase
        self.from_continuation = False  # whether this node is created during the continuation phase but can be reused for expansion
    
    @classmethod
    def set_default_calc_q(cls, calc_q: Callable[[List[float]], float]):
        """
        Set the default Q-value calculation method for all new nodes.
        """
        cls.DEFAULT_CALC_Q = calc_q

        
    def to_dict(self) -> Dict[str, Any]:
        dct = super().to_dict()
        dct.update({
            "fast_reward": float(self.fast_reward),
            "from_simulate": self.from_simulate,
            "is_simulated": self.is_simulated,
            "from_expand": self.from_expand,
            "cum_rewards": [float(r) for r in self.cum_rewards],
            "from_continuation": self.from_continuation,

            
        })
        return dct
    
    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> 'MCTSNode':
        """Reconstruct MCTSNode without parent/children links yet."""
        node = super().from_dict(dct)
        node.fast_reward = dct.get("fast_reward", 0.0)
        node.from_simulate = dct.get("from_simulate", False)
        node.is_simulated = dct.get("is_simulated", False)
        node.from_expand =  dct.get("from_expand", False) if "from_expand" in dct else dct.get("is_expanded", False) # new version: from_expand; for old version: is_expanded 
        node.cum_rewards = dct.get("cum_rewards", [])
        node.from_continuation = dct.get("from_continuation", False)
       
        return node
    
    
    @property
    def is_all_children_visited(self) -> bool:
        return all(x.state is not None for x in self.children)
        
    @property
    def Q(self) -> float:
        if self.state is None :
            return self.fast_reward
        elif len(self.cum_rewards) == 0: # the state will not be materialized during simulation if it is "continuous"
            # assert self.is_continuous
            return self.fast_reward
        else:
            # Ideally, Q should only be used when node.is_all_children_visited is True
            return self.calc_q(self.cum_rewards)
MCTSNode.set_default_calc_q(np.mean)
  

class BeamSearchNode(SearchNode[StateT, ActionT]):
    def __init__(self, state: StateT, action: ActionT, reward: float, parent: Optional['BeamSearchNode'] = None, children: Optional[List['BeamSearchNode']] = None):
        super().__init__(state, action, parent, children)
        self.reward = reward
