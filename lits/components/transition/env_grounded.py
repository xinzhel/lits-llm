"""Base class for environment-grounded transitions.

EnvGroundedTransition is the base class for env_grounded task types (BlocksWorld, 
Crosswords, robotics planning). It defines the required interface for domain-specific
logic that domain experts must implement.

Domain experts can add new planning domains by implementing a single Transition class
with well-defined interfaces—no tree search algorithm knowledge required.

Required static methods:
    - goal_check(): Check if goals are met
    - generate_actions(): Generate valid actions from state

Required instance method:
    - _step(): Execute action and return new state

Example:
    ```python
    from lits.components.transition.env_grounded import EnvGroundedTransition
    from lits.registry import register_transition
    
    @register_transition("robot_arm", task_type="env_grounded")
    class RobotArmTransition(EnvGroundedTransition):
        @staticmethod
        def goal_check(target: np.ndarray, current: np.ndarray) -> Tuple[bool, float]:
            # Check if robot arm reached target position
            distance = np.linalg.norm(target - current)
            reached = distance < 0.01
            progress = max(0.0, 1.0 - distance / 10.0)
            return reached, progress
        
        @staticmethod
        def generate_actions(state: np.ndarray) -> List[str]:
            # Return valid robot arm movements
            return ["move_up", "move_down", "move_left", "move_right", "grip", "release"]
        
        def _step(self, state, action, query_or_goals, **kwargs):
            # Domain-specific state update logic
            ...
    ```
"""

from abc import abstractmethod
from typing import Tuple, List, Any, Union, Optional
from lits.components.base import LlmTransition
from lits.structures import StateT, ActionT


class EnvGroundedTransition(LlmTransition):
    """Base class for env_grounded Transition with required static methods.
    
    This class provides the interface for environment-grounded planning tasks.
    Domain experts implement this class to add new planning domains without
    understanding tree search internals.
    
    Subclasses must implement:
        - goal_check(): Static method to check if goals are met
        - generate_actions(): Static method to generate valid actions from state
        - _step(): Instance method to execute action and return new state
    
    Class Attributes:
        TASK_TYPE: Set to 'env_grounded' for automatic task type inference
    
    Static Method Signatures:
        The signatures for goal_check() and generate_actions() are flexible—different
        domains may have different state representations. Common patterns:
        
        goal_check:
            - String-based: goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]
            - Structured: goal_check(goals: List[Dict], state: np.ndarray) -> Tuple[bool, float]
        
        generate_actions:
            - String-based: generate_actions(env_state: str) -> List[str]
            - Structured: generate_actions(state: Dict, constraints: List = None) -> List[str]
    
    Example:
        ```python
        @register_transition("blocksworld", task_type="env_grounded")
        class BlocksWorldTransition(EnvGroundedTransition):
            @staticmethod
            def goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]:
                goals = parse_goals(query_or_goals)
                met = sum(1 for g in goals if g in env_state)
                return met == len(goals), met / len(goals)
            
            @staticmethod
            def generate_actions(env_state: str) -> List[str]:
                return ["pick red", "put red on blue", ...]
            
            def _step(self, state, action, query_or_goals, **kwargs):
                # Execute action and return new state
                ...
        ```
    """
    
    # Interface category for env_grounded tasks
    TASK_TYPE: str = "env_grounded"
    
    @staticmethod
    @abstractmethod
    def goal_check(*args, **kwargs) -> Tuple[bool, float]:
        """Check if goals are met.
        
        This static method checks whether the current state satisfies the goal
        conditions. It returns both a boolean indicating goal completion and
        a progress score for partial credit.
        
        Args:
            *args: Domain-specific arguments. Common patterns:
                   - (query_or_goals: str, env_state: str) for string-based domains
                   - (goals: List[Dict], state: np.ndarray) for structured domains
            **kwargs: Additional domain-specific keyword arguments
        
        Returns:
            Tuple[bool, float]: A tuple of:
                - goal_reached (bool): True if all goals are satisfied
                - progress (float): Progress score from 0.0 to 1.0
                  (e.g., fraction of goals met, distance to target)
        
        Note:
            The signature is flexible—different domains may have different
            state representations. Implement according to your domain's needs.
        
        Example (string-based):
            ```python
            @staticmethod
            def goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]:
                goals = parse_goals(query_or_goals)
                met = sum(1 for g in goals if g in env_state)
                return met == len(goals), met / len(goals)
            ```
        
        Example (structured):
            ```python
            @staticmethod
            def goal_check(target: np.ndarray, current: np.ndarray) -> Tuple[bool, float]:
                distance = np.linalg.norm(target - current)
                reached = distance < 0.01
                progress = max(0.0, 1.0 - distance / 10.0)
                return reached, progress
            ```
        """
        raise NotImplementedError("Subclasses must implement goal_check()")
    
    @staticmethod
    @abstractmethod
    def generate_actions(*args, **kwargs) -> List[str]:
        """Generate all valid actions from current state.
        
        This static method generates the list of valid actions that can be
        taken from the current state. The actions are typically strings
        representing domain-specific operations.
        
        Args:
            *args: Domain-specific arguments. Common patterns:
                   - (env_state: str,) for string-based domains
                   - (state: Dict, constraints: List = None) for structured domains
            **kwargs: Additional domain-specific keyword arguments
        
        Returns:
            List[str]: List of valid action strings for the current state
        
        Note:
            The signature is flexible—different domains may have different
            state representations. Implement according to your domain's needs.
        
        Example (string-based):
            ```python
            @staticmethod
            def generate_actions(env_state: str) -> List[str]:
                # Parse state and return valid block movements
                blocks = parse_blocks(env_state)
                actions = []
                for block in blocks:
                    if is_clear(block, env_state):
                        actions.append(f"pick {block}")
                return actions
            ```
        
        Example (structured):
            ```python
            @staticmethod
            def generate_actions(state: np.ndarray) -> List[str]:
                # Return valid robot arm movements
                return ["move_up", "move_down", "move_left", "move_right"]
            ```
        """
        raise NotImplementedError("Subclasses must implement generate_actions()")
