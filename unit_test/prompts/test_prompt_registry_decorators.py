"""Unit tests for Prompt Registry Decorators.

Tests Property 5: Prompt Decorator Registration Round-Trip
- @register_system_prompt registers prompts retrievable via PromptRegistry.get()
- @register_user_prompt registers prompts retrievable via PromptRegistry.get_usr()

Validates: Requirements 7.1, 7.2, 7.3

run:
```
python -m unittest unit_test.prompts.test_prompt_registry_decorators -v
```
"""

import sys
sys.path.append('.')

import unittest

from lits.prompts.registry import (
    PromptRegistry,
    register_system_prompt,
    register_user_prompt,
)


class TestPromptDecoratorRoundTrip(unittest.TestCase):
    """Unit tests for prompt decorator registration round-trip.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    """
    
    def setUp(self):
        PromptRegistry.clear()
    
    def tearDown(self):
        PromptRegistry.clear()
    
    def test_system_prompt_registration_round_trip(self):
        """@register_system_prompt registers prompt retrievable via get()."""
        @register_system_prompt("policy", "rap", "my_task")
        def my_system_prompt():
            return "You are solving math problems."
        
        retrieved = PromptRegistry.get("policy", "rap", task_type="my_task")
        self.assertEqual(retrieved, "You are solving math problems.")
    
    def test_user_prompt_registration_round_trip(self):
        """@register_user_prompt registers prompt retrievable via get_usr()."""
        @register_user_prompt("policy", "rap", "my_task")
        def my_user_prompt():
            return {"question_format": "Problem: {question}"}
        
        retrieved = PromptRegistry.get_usr("policy", "rap", task_type="my_task")
        self.assertEqual(retrieved, {"question_format": "Problem: {question}"})
    
    def test_both_prompt_types_independent(self):
        """System and user prompts for same tuple are stored independently."""
        @register_system_prompt("policy", "concat", "gsm8k")
        def system():
            return "System prompt"
        
        @register_user_prompt("policy", "concat", "gsm8k")
        def user():
            return {"user": "template"}
        
        self.assertEqual(PromptRegistry.get("policy", "concat", task_type="gsm8k"), "System prompt")
        self.assertEqual(PromptRegistry.get_usr("policy", "concat", task_type="gsm8k"), {"user": "template"})
    
    def test_default_task_type_registration(self):
        """task_type=None registers as 'default' fallback."""
        @register_system_prompt("reward", "generative", None)
        def default_prompt():
            return "Default reward prompt"
        
        retrieved = PromptRegistry.get("reward", "generative")
        self.assertEqual(retrieved, "Default reward prompt")
    
    def test_decorated_function_remains_callable(self):
        """Decorated function is still callable and returns same value."""
        @register_system_prompt("transition", "blocksworld", "blocksworld")
        def my_prompt():
            return "Transition prompt"
        
        self.assertTrue(callable(my_prompt))
        self.assertEqual(my_prompt(), "Transition prompt")
    
    def test_dict_return_type(self):
        """Decorator accepts dict return type."""
        @register_system_prompt("policy", "test", "test_task")
        def dict_prompt():
            return {"template": "Hello {name}", "format": "json"}
        
        retrieved = PromptRegistry.get("policy", "test", task_type="test_task")
        self.assertEqual(retrieved, {"template": "Hello {name}", "format": "json"})


if __name__ == "__main__":
    unittest.main()
