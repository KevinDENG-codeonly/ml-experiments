import os
import yaml
import json
from typing import Any

def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif ext.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {ext}")
    
    return config

def save_config(config: dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yml', '.yaml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif ext.lower() == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        raise ValueError(f"Unsupported config file format: {ext}") 