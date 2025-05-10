import pytest
import yaml
from pathlib import Path

from src.config_loader import load_config, DEFAULT_CONFIG_FILE

# Test cases for config loading
def test_load_config_default_exists(mocker): # Use mocker if we want to patch the default path
    """Tests loading the default config file when it exists."""
    # This test assumes the actual default config file exists and is valid
    # If not, it should be mocked or skipped.
    if DEFAULT_CONFIG_FILE.exists():
        cfg = load_config()
        assert isinstance(cfg, dict)
        # Add more specific assertions based on expected keys in default config
        assert 'general' in cfg
        assert 'dataset' in cfg
        assert 'model' in cfg
        assert 'training' in cfg
    else:
        pytest.skip(f"Default config file {DEFAULT_CONFIG_FILE} not found, skipping test.")

def test_load_config_custom_path(tmp_path: Path):
    """Tests loading a config file from a specified path."""
    config_content = {
        'general': {'device': 'cpu', 'random_seed': 123},
        'training': {'batch_size': 32}
    }
    custom_config_file = tmp_path / "custom_config.yaml"
    with open(custom_config_file, 'w') as f:
        yaml.dump(config_content, f)

    cfg = load_config(custom_config_file)
    assert isinstance(cfg, dict)
    assert cfg['general']['device'] == 'cpu'
    assert cfg['training']['batch_size'] == 32

def test_load_config_file_not_found(tmp_path: Path):
    """Tests that FileNotFoundError is raised for a non-existent file."""
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(non_existent_file)

def test_load_config_invalid_yaml(tmp_path: Path):
    """Tests that YAMLError is raised for an invalid YAML file."""
    invalid_yaml_file = tmp_path / "invalid.yaml"
    # Example invalid YAML (e.g., incorrect indentation)
    invalid_yaml_file.write_text("key1: value1\n key2: value2") # Invalid indentation

    with pytest.raises(yaml.YAMLError):
        load_config(invalid_yaml_file)

def test_load_config_empty_yaml(tmp_path: Path):
    """Tests loading an empty YAML file."""
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("") # Empty file

    cfg = load_config(empty_yaml_file)
    assert isinstance(cfg, dict)
    assert len(cfg) == 0 