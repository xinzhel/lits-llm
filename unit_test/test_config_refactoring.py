"""
Test suite for config refactoring to verify:
1. Common attributes are properly inherited from BaseConfig
2. max_steps is used consistently across all configs
3. to_dict() method works correctly from parent class
4. save_config() method works correctly from parent class
"""

import tempfile
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lits.agents.base import BaseConfig
from lits.agents.tree.base import BaseSearchConfig
from lits.agents.chain.env_chain import EnvChainConfig
from lits.agents.chain.react import ReactChatConfig


class TestBaseConfig:
    """Test BaseConfig common attributes and methods."""
    
    def test_base_config_has_common_attributes(self):
        """Verify BaseConfig has all common attributes."""
        config = BaseConfig(reasoning_method="test")
        
        assert hasattr(config, 'reasoning_method')
        assert hasattr(config, 'package_version')
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'gpu_device')
        assert hasattr(config, 'max_length')
        assert hasattr(config, 'max_steps')
    
    def test_base_config_to_dict(self):
        """Verify to_dict() method works correctly."""
        config = BaseConfig(
            reasoning_method="test",
            model_name="test-model",
            gpu_device="cuda:0",
            max_steps=15
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['reasoning_method'] == "test"
        assert config_dict['model_name'] == "test-model"
        assert config_dict['gpu_device'] == "cuda:0"
        assert config_dict['max_steps'] == 15
    
    def test_base_config_save_config(self):
        """Verify save_config() method works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BaseConfig(
                reasoning_method="test",
                model_name="test-model",
                max_steps=20
            )
            
            config.save_config(tmpdir)
            
            config_path = os.path.join(tmpdir, "test_config.json")
            assert os.path.exists(config_path)
            
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            
            assert saved_config['reasoning_method'] == "test"
            assert saved_config['model_name'] == "test-model"
            assert saved_config['max_steps'] == 20


class TestBaseSearchConfig:
    """Test BaseSearchConfig inherits common attributes."""
    
    def test_inherits_common_attributes(self):
        """Verify BaseSearchConfig inherits from BaseConfig."""
        config = BaseSearchConfig(reasoning_method="rest")
        
        # Check inherited attributes
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'gpu_device')
        assert hasattr(config, 'max_length')
        assert hasattr(config, 'max_steps')
        
        # Check specific attributes
        assert hasattr(config, 'n_actions')
        assert hasattr(config, 'eval_model_name')
        assert hasattr(config, 'force_terminating_on_depth_limit')
    
    def test_max_steps_replaces_depth_limit(self):
        """Verify max_steps is used instead of depth_limit."""
        config = BaseSearchConfig(reasoning_method="rest", max_steps=8)
        
        assert config.max_steps == 8
        assert not hasattr(config, 'depth_limit')
    
    def test_to_dict_includes_all_attributes(self):
        """Verify to_dict() includes both inherited and specific attributes."""
        config = BaseSearchConfig(
            reasoning_method="rest",
            model_name="test-model",
            max_steps=12,
            n_actions=5
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['model_name'] == "test-model"
        assert config_dict['max_steps'] == 12
        assert config_dict['n_actions'] == 5


class TestEnvChainConfig:
    """Test EnvChainConfig inherits common attributes."""
    
    def test_inherits_common_attributes(self):
        """Verify EnvChainConfig inherits from BaseConfig."""
        config = EnvChainConfig(reasoning_method="env_chain")
        
        # Check inherited attributes
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'gpu_device')
        assert hasattr(config, 'max_length')
        assert hasattr(config, 'max_steps')
        
        # Check specific attributes
        assert hasattr(config, 'temperature')
    
    def test_max_steps_default_override(self):
        """Verify EnvChainConfig overrides max_steps default to 30."""
        config = EnvChainConfig(reasoning_method="env_chain")
        
        assert config.max_steps == 30
    
    def test_no_duplicate_to_dict(self):
        """Verify EnvChainConfig uses inherited to_dict() method."""
        config = EnvChainConfig(
            reasoning_method="env_chain",
            model_name="test-model",
            max_steps=25,
            temperature=0.9
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['model_name'] == "test-model"
        assert config_dict['max_steps'] == 25
        assert config_dict['temperature'] == 0.9


class TestReactChatConfig:
    """Test ReactChatConfig inherits common attributes."""
    
    def test_inherits_common_attributes(self):
        """Verify ReactChatConfig inherits from BaseConfig."""
        config = ReactChatConfig(reasoning_method="react")
        
        # Check inherited attributes
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'gpu_device')
        assert hasattr(config, 'max_length')
        assert hasattr(config, 'max_steps')
        
        # Check specific attributes
        assert hasattr(config, 'enable_think')
        assert hasattr(config, 'timeout')
    
    def test_max_steps_attribute_exists(self):
        """Verify ReactChatConfig now has max_steps attribute."""
        config = ReactChatConfig(reasoning_method="react", max_steps=15)
        
        assert config.max_steps == 15
    
    def test_no_duplicate_to_dict(self):
        """Verify ReactChatConfig uses inherited to_dict() method."""
        config = ReactChatConfig(
            reasoning_method="react",
            model_name="test-model",
            max_steps=10,
            enable_think=False
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['model_name'] == "test-model"
        assert config_dict['max_steps'] == 10
        assert config_dict['enable_think'] is False


class TestConfigConsistency:
    """Test consistency across all config classes."""
    
    def test_all_configs_have_max_steps(self):
        """Verify all config classes have max_steps attribute."""
        base_config = BaseConfig(reasoning_method="base")
        search_config = BaseSearchConfig(reasoning_method="rest")
        env_config = EnvChainConfig(reasoning_method="env_chain")
        react_config = ReactChatConfig(reasoning_method="react")
        
        assert hasattr(base_config, 'max_steps')
        assert hasattr(search_config, 'max_steps')
        assert hasattr(env_config, 'max_steps')
        assert hasattr(react_config, 'max_steps')
    
    def test_all_configs_have_common_attributes(self):
        """Verify all config classes have common attributes."""
        configs = [
            BaseConfig(reasoning_method="base"),
            BaseSearchConfig(reasoning_method="rest"),
            EnvChainConfig(reasoning_method="env_chain"),
            ReactChatConfig(reasoning_method="react")
        ]
        
        common_attrs = ['model_name', 'gpu_device', 'max_length', 'max_steps']
        
        for config in configs:
            for attr in common_attrs:
                assert hasattr(config, attr), f"{config.__class__.__name__} missing {attr}"
    
    def test_all_configs_use_inherited_to_dict(self):
        """Verify all config classes use the inherited to_dict() method."""
        configs = [
            BaseConfig(reasoning_method="base", model_name="model1"),
            BaseSearchConfig(reasoning_method="rest", model_name="model2"),
            EnvChainConfig(reasoning_method="env_chain", model_name="model3"),
            ReactChatConfig(reasoning_method="react", model_name="model4")
        ]
        
        for config in configs:
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)
            assert 'model_name' in config_dict
            assert 'max_steps' in config_dict


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestBaseConfig,
        TestBaseSearchConfig,
        TestEnvChainConfig,
        TestReactChatConfig,
        TestConfigConsistency
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"✗ {method_name}: ERROR - {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print('='*60)
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
