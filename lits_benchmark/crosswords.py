"""Crosswords benchmark module for mini crossword puzzles.

This module provides:
1. CrosswordsTransition - Domain-specific transition for mini crosswords
2. load_crosswords - Dataset loader for crossword puzzles
3. Crosswords-specific prompts for EnvGroundedPolicy and EnvGroundedPRM

This demonstrates how a user can add a new env_grounded domain to LiTS by:
1. Implementing a Transition class with goal_check and generate_actions static methods
2. Registering it with @register_transition decorator
3. Implementing a dataset loader with @register_dataset decorator
4. Registering domain-specific prompts with @register_user_prompt and @register_system_prompt

The crosswords domain is a 5x5 grid puzzle with 10 clues (5 horizontal h1-h5, 5 vertical v1-v5).
Actions are word placements in format "h1. apple" or "v1. world" (lowercase).

Reference: Adapted from llm-reasoners/examples/ToT/crosswords/utils.py MiniCrosswordsEnv

Example:
    ```python
    # Import to register components
    import lits_benchmark.crosswords
    
    # Use with main_search.py
    # python main_search.py --benchmark crosswords --import lits_benchmark.crosswords
    ```
"""

import re
import json
import copy
from typing import List, Dict, Tuple, Optional, Any

from lits.components.transition.env_grounded import EnvGroundedTransition
from lits.components.registry import register_transition
from lits.benchmarks.registry import register_dataset
from lits.structures.env_grounded import EnvState, EnvAction
from lits.prompts.prompt import PromptTemplate
from lits.prompts.registry import register_system_prompt, register_user_prompt


# =============================================================================
# Crosswords Prompts - EnvGroundedPolicy
# =============================================================================

@register_user_prompt('policy', 'env_grounded', 'crosswords')
def crosswords_policy_usr_prompt():
    """User prompt for EnvGroundedPolicy on crosswords domain.
    
    Placeholders:
        - {init_state}: Current board state with clues (includes Unfilled/Filled/Changed sections)
        - {goals}: Ground truth answers (not shown to LLM, used internally)
        - {actions}: Not used for crosswords (infinite action space)
    """
    return PromptTemplate(
        template="""I am solving a 5x5 mini crossword puzzle. I need to fill in words based on the clues.

The puzzle has 10 clues:
- h1-h5: Horizontal words (rows 1-5, left to right)
- v1-v5: Vertical words (columns 1-5, top to bottom)

Each word is exactly 5 letters. I can fill in one word at a time using the format "h1. apple" or "v1. world".

{init_state}

Choose ONE unfilled position and provide a 5-letter word that fits the clue.
Output ONLY the action in format "position. word" (e.g., "h1. apple"):""",
        input_variables=["init_state", "goals", "actions"]
    )


# =============================================================================
# Crosswords Prompts - EnvGroundedPRM (Process Reward Model)
# =============================================================================

@register_system_prompt('reward', 'env_grounded', 'crosswords')
def crosswords_reward_task_prompt():
    """System prompt for EnvGroundedPRM on crosswords domain.
    
    This prompt instructs the LLM to evaluate whether a proposed word placement
    is good, bad, or unknown based on the clue and current board state.
    """
    return """You are evaluating word placements in a 5x5 mini crossword puzzle.

## PUZZLE STRUCTURE
- 5x5 grid with 10 clues (h1-h5 horizontal, v1-v5 vertical)
- Each word is exactly 5 letters
- Words intersect at shared letter positions

## EVALUATION CRITERIA

### **1. CLUE MATCH**
Does the proposed word match the clue's definition?
- **good**: Word clearly matches the clue meaning
- **bad**: Word does not match the clue meaning
- **unknown**: Cannot determine if word matches clue

### **2. LETTER CONSISTENCY**
Does the word conflict with already-filled letters at intersection points?
- **good**: No conflicts with existing letters
- **bad**: Conflicts with existing letters at intersections
- **unknown**: Cannot determine consistency

### **3. WORD VALIDITY**
Is the proposed word a valid English word?
- **good**: Valid English word
- **bad**: Not a valid English word
- **unknown**: Uncertain about word validity

## OUTPUT FORMAT
Respond with ONLY one word: good, bad, or unknown"""


@register_user_prompt('reward', 'env_grounded', 'crosswords')
def crosswords_reward_usr_prompt():
    """User prompt for EnvGroundedPRM on crosswords domain.
    
    Placeholders:
        - <init_state>: Current board state with clues
        - <goals>: Ground truth answers (not used in evaluation)
        - <action>: Proposed word placement (e.g., "h1. apple")
    
    Note: Uses <placeholder> syntax for backward compatibility with EnvGroundedPRM.
    """
    return """[PUZZLE STATE]
<init_state>

[PROPOSED ACTION]
<action>

[EVALUATION]
Is this word placement good, bad, or unknown?"""


# =============================================================================
# CrosswordsTransition - Domain-specific Transition
# =============================================================================

@register_transition("crosswords", task_type="env_grounded")
class CrosswordsTransition(EnvGroundedTransition):
    """Crosswords Transition for mini crossword puzzles (5x5 grid, 10 clues).
    
    Adapted from llm-reasoners/examples/ToT/crosswords/utils.py MiniCrosswordsEnv.
    
    State representation:
        - board: List[str] of 25 characters (5x5 grid, row-major order)
        - ans: List[str] of 10 words (5 horizontal + 5 vertical)
        - status: List[int] of 10 status codes (0=unfilled, 1=filled, 2=changed)
        - clues: List[str] of 10 clues
        - ans_gt: List[str] of 10 ground truth answers
    
    Action format: "h1. apple" for horizontal row 1, "v1. world" for vertical column 1
        - Position: h1-h5 (horizontal rows) or v1-v5 (vertical columns)
        - Word: 5 lowercase letters
    """
    
    TASK_TYPE: str = None  # Benchmark-specific, no fallback
    
    def __init__(self, base_model, goal_check=None, task_name: str = "crosswords", **kwargs):
        """Initialize CrosswordsTransition.
        
        Args:
            base_model: LLM for generating word proposals (optional, may not be used)
            goal_check: Goal check function (passed by component_factory, but we use static method)
            task_name: Task name for prompt lookup
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(base_model=base_model, **kwargs)
        self.task_name = task_name
    
    @staticmethod
    def goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]:
        """Check if crossword is solved correctly.
        
        Args:
            query_or_goals: Ground truth answers (newline-separated, 10 words uppercase)
                Format: h1, h2, h3, h4, h5, v1, v2, v3, v4, v5
            env_state: Current grid state string (from render())
        
        Returns:
            (is_solved, r_word) tuple where r_word = correct_words / 10
        """
        # Parse ground truth answers (10 words: 5 horizontal + 5 vertical)
        answers_gt = [a.strip().upper() for a in query_or_goals.strip().split('\n') if a.strip()]
        
        if len(answers_gt) != 10:
            return False, 0.0
        
        # Extract board from "Current Board:" section
        board_match = re.search(r'Current Board:\n((?:[A-Z_]{5}\n){5})', env_state)
        if not board_match:
            return False, 0.0
        
        board_lines = board_match.group(1).strip().split('\n')
        board = list(''.join(board_lines))  # 25 characters
        
        if len(board) != 25:
            return False, 0.0
        
        # Extract 10 answers from board (5 horizontal + 5 vertical)
        current_ans = []
        # Horizontal (h1-h5): rows
        for i in range(5):
            current_ans.append(''.join(board[i*5:(i+1)*5]))
        # Vertical (v1-v5): columns
        for i in range(5):
            current_ans.append(''.join(board[i::5]))
        
        # Compare answers
        correct = sum(
            curr.upper() == gt.upper() 
            for curr, gt in zip(current_ans, answers_gt)
        )
        r_word = correct / 10
        return r_word == 1.0, r_word
    
    @staticmethod
    def validate_action(env_state: str, action: str) -> bool:
        """Validate if an LLM-generated action is valid for the current state.
        
        For crosswords, validates:
        1. Action format: "position. word" (e.g., "h1. APPLE")
        2. Position is valid (h1-h5 or v1-v5)
        3. Position is currently unfilled
        4. Word has exactly 5 letters (alphabetic characters)
        
        Args:
            env_state: Current grid state string
            action: LLM-generated action string (e.g., "h4. SALON")
        
        Returns:
            True if action is valid, False otherwise
        """
        # Parse action format
        parts = action.split('. ')
        if len(parts) != 2:
            return False
        
        pos, word = parts
        word = word.strip()
        
        # Validate position format
        if not re.match(r'^[hv][1-5]$', pos):
            return False
        
        # Validate word: exactly 5 alphabetic characters
        if len(word) != 5 or not word.isalpha():
            return False
        
        # Check if position is unfilled
        unfilled_match = re.search(r'Unfilled:\n(.*?)(?:\n\n|Filled:|Changed:|$)', env_state, re.DOTALL)
        if unfilled_match:
            unfilled_section = unfilled_match.group(1)
            unfilled_positions = re.findall(r'([hv]\d)\.', unfilled_section)
            if pos not in unfilled_positions:
                return False
        
        return True
    
    def init_state(self, init_state_str: str = None, query_or_goals: str = None, 
                   clues: List[str] = None, board_gt: List[str] = None, **kwargs) -> EnvState:
        """Initialize crossword state from puzzle data.
        
        The crossword state is represented as a string (init_state_str) that contains:
        - Current Board: 5x5 grid showing filled letters and underscores for empty cells
        - Unfilled: List of clues with positions (h1-h5, v1-v5) that haven't been filled
        - Filled: List of clues with positions that have been filled with words
        - Changed: List of clues whose words were overwritten by subsequent actions
        
        Example init_state_str format::
        
            Current Board:
            _____
            _____
            _____
            _____
            _____
            
            Unfilled:
            h1. An agendum; something to be done: _____
            h2. An engine: _____
            ...
            v1. To heap: _____
            ...
            
            Filled:
            
            Changed:
        
        The query_or_goals contains the ground truth answers (10 words, newline-separated):
        - First 5 words are horizontal answers (h1-h5, rows top to bottom)
        - Last 5 words are vertical answers (v1-v5, columns left to right)
        
        Example query_or_goals::
        
            AGEND
            MOTOR
            ARTSY
            SALLE
            SLEER
            AMASS
            GORAL
            ETTLE
            NOSLE
            DRYER
        
        Args:
            init_state_str: Initial state string from MiniCrosswordsEnv.render().
                Contains the board visualization and clue listings by status.
                If None, generates from empty board with provided clues.
            query_or_goals: Ground truth answers (10 words, newline-separated).
                Used by goal_check() to evaluate puzzle completion.
            clues: List of 10 clue strings [h1, h2, h3, h4, h5, v1, v2, v3, v4, v5].
                Only needed if init_state_str is None.
            board_gt: List of 25 ground truth characters (row-major order).
                Only needed for computing letter-level accuracy.
            **kwargs: Additional arguments (ignored).
        
        Returns:
            EnvState with init_state set to the puzzle's initial state string.
            The state tracks the trajectory of actions via EnvStep objects.
        
        Note:
            EnvState uses init_state (str) to store the initial environment snapshot.
            The current state is accessed via env_state property, which returns
            the last step's next_state if available, otherwise init_state.
        """
        # If init_state_str is provided, use it directly
        if init_state_str:
            return EnvState(init_state=init_state_str)
        
        # Otherwise, generate initial state from clues
        # Initialize empty board: 25 underscores for 5x5 grid
        board = ['_'] * 25
        # Initialize answers: 10 words (5 horizontal + 5 vertical), all unfilled
        ans = ['_____'] * 10
        # Initialize status: 0=unfilled, 1=filled, 2=changed
        status = [0] * 10
        
        # Build state dict for rendering
        state_data = {
            'board': board,
            'ans': ans,
            'status': status,
            'clues': clues or [''] * 10,
        }
        
        return EnvState(init_state=self._render_state(state_data))
    
    def _render_state(self, metadata: Dict) -> str:
        """Render state as string.
        
        Args:
            metadata: State metadata dict with board, ans, status, clues
        
        Returns:
            State string
        """
        board = metadata['board']
        ans = metadata['ans']
        status = metadata['status']
        clues = metadata.get('clues', [''] * 10)
        
        # Render board
        s = "Current Board:\n"
        for i in range(5):
            s += ''.join(board[i*5:(i+1)*5]) + '\n'
        
        # Render answers by status
        def render_ans_section(status_filter: int, section_name: str) -> str:
            section = f"\n{section_name}:\n"
            for i in range(5):
                if status[i] == status_filter:
                    clue = clues[i] if i < len(clues) else ''
                    section += f'h{i+1}. {clue}: {ans[i]}\n'
            for i in range(5, 10):
                if status[i] == status_filter:
                    clue = clues[i] if i < len(clues) else ''
                    section += f'v{i-4}. {clue}: {ans[i]}\n'
            return section
        
        s += render_ans_section(0, "Unfilled")
        s += render_ans_section(1, "Filled")
        s += render_ans_section(2, "Changed")
        
        return s
    
    def _get_ans(self, board: List[str]) -> List[str]:
        """Extract answers from board.
        
        Args:
            board: List of 25 characters
        
        Returns:
            List of 10 words (5 horizontal + 5 vertical)
        """
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans
    
    def _apply_action(self, metadata: Dict, action: str) -> Tuple[Dict, str]:
        """Apply action to state and return new metadata and message.
        
        Args:
            metadata: Current state metadata
            action: Action string (e.g., "h1. apple")
        
        Returns:
            (new_metadata, message) tuple
        """
        metadata = copy.deepcopy(metadata)
        board = metadata['board']
        ans = metadata['ans']
        status = metadata['status']
        
        # Parse action
        action = action.split('\n')[-1]  # Take last line if multi-line
        parts = action.split('. ')
        
        if len(parts) != 2:
            return metadata, 'Invalid! Format should be like "h1. apple"'
        
        pos, word = parts
        word = word.strip()
        
        if len(word) != 5:
            return metadata, 'Invalid! Word should have 5 letters.'
        
        # Reject placeholder actions (words with only underscores or no letters)
        if not any(c.isalpha() for c in word):
            return metadata, 'Invalid! Word must contain actual letters, not placeholders.'
        
        if pos.startswith('h'):
            try:
                idx = int(pos[1:]) - 1
                if idx < 0 or idx > 4:
                    return metadata, 'Invalid! Position should be h1-h5 or v1-v5'
                board[idx*5:(idx+1)*5] = list(word.upper())
            except ValueError:
                return metadata, 'Invalid! Position should be h1-h5 or v1-v5'
        elif pos.startswith('v'):
            try:
                idx = int(pos[1:]) - 1
                if idx < 0 or idx > 4:
                    return metadata, 'Invalid! Position should be h1-h5 or v1-v5'
                board[idx::5] = list(word.upper())
                idx += 5  # For status update
            except ValueError:
                return metadata, 'Invalid! Position should be h1-h5 or v1-v5'
        else:
            return metadata, 'Invalid! Position should be h1-h5 or v1-v5'
        
        # Update answers and status
        new_ans = self._get_ans(board)
        new_status = []
        for i, (old_status, old_a, new_a) in enumerate(zip(status, ans, new_ans)):
            # Check if any letter changed (excluding unfilled)
            changed = any(
                old_letter != new_letter and old_letter != '_'
                for old_letter, new_letter in zip(old_a, new_a)
            )
            if changed:
                new_status.append(2)  # Changed
            else:
                new_status.append(old_status)
        
        # Mark the filled position
        if pos.startswith('h'):
            new_status[int(pos[1:]) - 1] = 1
        else:
            new_status[int(pos[1:]) - 1 + 5] = 1
        
        metadata['board'] = board
        metadata['ans'] = new_ans
        metadata['status'] = new_status
        metadata['steps'] = metadata.get('steps', 0) + 1
        
        return metadata, ''
    
    def _step(self, state: EnvState, step_or_action, query_or_goals: str, **kwargs) -> Tuple[EnvState, Dict]:
        """Execute action and return new state.
        
        This method parses the current state from env_state string, applies the action,
        and returns a new EnvState with the updated state string.
        
        Args:
            state: Current EnvState (contains env_state string property)
            step_or_action: Either EnvStep (from tree search) or EnvAction (from chain agent).
                - EnvStep has .action attribute which is an EnvAction
                - EnvAction (StringAction) has .action_str attribute
            query_or_goals: Ground truth answers (newline-separated, for goal checking)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            (new_state, info) tuple where:
            - new_state: EnvState with updated env_state string
            - info: Dict with 'goal_reached' (tuple), 'message' (error if any)
        """
        # Extract action string from step_or_action
        # - EnvStep (from tree search): has .action attribute -> EnvAction
        # - EnvAction/StringAction (from chain agent): has .action_str attribute
        if hasattr(step_or_action, 'get_action'):
            # It's an EnvStep - extract the EnvAction
            action = step_or_action.get_action()
        else:
            action = step_or_action
        
        # EnvAction is aliased to StringAction which has action_str attribute
        action_str = action.action_str if hasattr(action, 'action_str') else str(action)
        current_env_state = state.env_state
        
        # Parse current state from env_state string
        state_data = self._parse_state(current_env_state)
        
        # Apply action
        new_state_data, message = self._apply_action(state_data, action_str)
        
        # Render new state
        new_env_state = self._render_state(new_state_data)
        
        # Calculate word accuracy using goal_check
        is_solved, r_word = self.goal_check(query_or_goals, new_env_state)
        
        info = {
            'goal_reached': (is_solved, r_word),  # Tuple format required by EnvGroundedPRM.reward()
            'message': message,
        }
        
        # Create new EnvState by appending step to trajectory
        from lits.structures.env_grounded import EnvStep
        new_step = EnvStep(action=EnvAction(action_str), next_state=new_env_state)
        new_state = copy.deepcopy(state)
        new_state.append(new_step)
        
        return new_state, info
    
    def _parse_state(self, env_state: str) -> Dict:
        """Parse state data from env_state string.
        
        Extracts board, answers, status, and clues from the rendered state string.
        
        Args:
            env_state: State string from render()
        
        Returns:
            Dict with 'board', 'ans', 'status', 'clues' keys
        """
        # Initialize defaults
        board = ['_'] * 25
        ans = ['_____'] * 10
        status = [0] * 10
        clues = [''] * 10
        
        # Parse board from "Current Board:" section
        board_match = re.search(r'Current Board:\n((?:[A-Z_]{5}\n){5})', env_state)
        if board_match:
            board_lines = board_match.group(1).strip().split('\n')
            board = list(''.join(board_lines))
            ans = self._get_ans(board)
        
        # Parse clues and status from sections
        # Unfilled section (status=0)
        unfilled_match = re.search(r'Unfilled:\n(.*?)(?:\n\n|Filled:|$)', env_state, re.DOTALL)
        if unfilled_match:
            for match in re.finditer(r'([hv])(\d)\. ([^:]+): ([A-Z_]{5})', unfilled_match.group(1)):
                pos_type, pos_num, clue, word = match.groups()
                idx = int(pos_num) - 1 + (5 if pos_type == 'v' else 0)
                clues[idx] = clue.strip()
                status[idx] = 0
        
        # Filled section (status=1)
        filled_match = re.search(r'Filled:\n(.*?)(?:\n\n|Changed:|$)', env_state, re.DOTALL)
        if filled_match:
            for match in re.finditer(r'([hv])(\d)\. ([^:]+): ([A-Z_]{5})', filled_match.group(1)):
                pos_type, pos_num, clue, word = match.groups()
                idx = int(pos_num) - 1 + (5 if pos_type == 'v' else 0)
                clues[idx] = clue.strip()
                status[idx] = 1
        
        # Changed section (status=2)
        changed_match = re.search(r'Changed:\n(.*?)$', env_state, re.DOTALL)
        if changed_match:
            for match in re.finditer(r'([hv])(\d)\. ([^:]+): ([A-Z_]{5})', changed_match.group(1)):
                pos_type, pos_num, clue, word = match.groups()
                idx = int(pos_num) - 1 + (5 if pos_type == 'v' else 0)
                clues[idx] = clue.strip()
                status[idx] = 2
        
        return {
            'board': board,
            'ans': ans,
            'status': status,
            'clues': clues,
        }
    
    def _is_terminal(self, state: EnvState, query_or_goals: str, **kwargs) -> bool:
        """Check if state is terminal (all positions filled or max steps reached).
        
        Note: We do NOT check correctness here to avoid leaking ground truth.
        Terminal = all cells filled OR max steps reached.
        
        Args:
            state: Current EnvState
            query_or_goals: Ground truth answers (unused, kept for interface compatibility)
        
        Returns:
            True if terminal state
        """
        # Check max steps (20 is the limit from llm-reasoners)
        if len(state) >= 20:
            return True
        
        # Check if all positions are filled (no underscores in board)
        state_data = self._parse_state(state.env_state)
        board = state_data.get('board', ['_'] * 25)
        all_filled = '_' not in board
        return all_filled


# =============================================================================
# Dataset Loader Helper Functions
# =============================================================================

def _get_ans_from_board(board: List[str]) -> List[str]:
    """Extract 10 answers (5 horizontal + 5 vertical) from 25-char board."""
    ans = [''] * 10
    for i in range(5):
        ans[i] = ''.join(board[i*5:(i+1)*5])
    for i in range(5):
        ans[i+5] = ''.join(board[i::5])
    return ans


def _render_initial_state(clues: List[str], board: List[str] = None) -> str:
    """Render initial state string for a crossword puzzle.
    
    Args:
        clues: List of 10 clues
        board: Optional initial board (defaults to empty)
    
    Returns:
        State string in expected format
    """
    board = board or ['_'] * 25
    ans = _get_ans_from_board(board)
    
    # Render board
    s = "Current Board:\n"
    for i in range(5):
        s += ''.join(board[i*5:(i+1)*5]) + '\n'
    
    # All positions start as unfilled
    s += "\nUnfilled:\n"
    for i in range(5):
        s += f'h{i+1}. {clues[i]}: {ans[i]}\n'
    for i in range(5, 10):
        s += f'v{i-4}. {clues[i]}: {ans[i]}\n'
    
    s += "\nFilled:\n"
    s += "\nChanged:\n"
    
    return s


# =============================================================================
# Dataset Loader
# =============================================================================

@register_dataset("crosswords", task_type="env_grounded")
def load_crosswords(data_file: str = None, **kwargs) -> List[Dict]:
    """Load mini crossword puzzles from JSON data file.
    
    Data file format (from llm-reasoners/examples/ToT/crosswords/data/):
        [[clues_list, board_gt_list], ...]
        - clues_list: 10 clue strings [h1, h2, h3, h4, h5, v1, v2, v3, v4, v5]
        - board_gt_list: 25 ground truth characters (row-major order)
    
    The loader converts this to the format expected by CrosswordsTransition:
        - init_state_str: Rendered state string with board and clues
        - query_or_goals: 10 ground truth answers (newline-separated)
    
    Note:
        The raw 'clues' and 'board_gt' are embedded into init_state_str and 
        query_or_goals respectively. They are also returned for debugging/reference
        but are NOT used by CrosswordsTransition after initialization.
    
    Args:
        data_file: Path to mini0505.json. If None, returns a placeholder example.
        **kwargs: Additional arguments (ignored).
    
    Returns:
        List of dicts with keys:
        - 'init_state_str': Initial state string for CrosswordsTransition.init_state()
        - 'query_or_goals': Ground truth answers for goal_check()
        - 'clues': Raw clues list (for debugging only)
        - 'board_gt': Raw board characters (for debugging only)
    """
    if data_file is None:
        # Return placeholder example for testing
        return [{
            'init_state_str': (
                "Current Board:\n_____\n_____\n_____\n_____\n_____\n\n"
                "Unfilled:\n"
                "h1. An agendum; something to be done: _____\n"
                "h2. An engine: _____\n"
                "h3. Pretentious; flowery: _____\n"
                "h4. A salon; a hall: _____\n"
                "h5. To mock; to sneer: _____\n"
                "v1. To heap: _____\n"
                "v2. An Indian antelope: _____\n"
                "v3. To intend; to plan; to devise; a nettle; to guess: _____\n"
                "v4. A nozzle: _____\n"
                "v5. Desiccator; more dry: _____\n\n"
                "Filled:\n\nChanged:\n"
            ),
            'query_or_goals': "AGEND\nMOTOR\nARTSY\nSALLE\nSLEER\nAMASS\nGORAL\nETTLE\nNOSLE\nDRYER"
        }]
    
    # Load data from JSON file
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    examples = []
    for clues, board_gt in data:
        # Convert board_gt (25 chars) to 10 answers (5 horizontal + 5 vertical)
        ans_gt = _get_ans_from_board(board_gt)
        # Render initial state with empty board and clues
        init_state_str = _render_initial_state(clues)
        
        examples.append({
            'init_state_str': init_state_str,
            'query_or_goals': '\n'.join(ans_gt),
            # Raw data for debugging/reference only (not used by Transition)
            'clues': clues,
            'board_gt': board_gt
        })
    
    return examples
