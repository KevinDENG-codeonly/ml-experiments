import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Define a base path for configuration files if this loader is in src/
# and configs are in ../configs/
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "config.yaml"

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
                     If None, defaults to `configs/config.yaml` relative to project root.

    Returns:
        A dictionary containing the configuration parameters.
    
    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty YAML file case
            return {}
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading config {config_path}: {e}")
        raise

# --- Pydantic Model (Optional - for validation and type hints) ---
# We can integrate Pydantic here for more robust config handling later.
# from pydantic import BaseModel, Field, FilePath, DirectoryPath
# from typing import Literal

# class DatasetConfig(BaseModel):
#     vocab_path: FilePath = Field(default=PROCESSED_DATA_DIR / "vocab.json")
#     vocab_freq_threshold: int = 5
#     max_caption_length: int = 30
#     image_size: int = 224

# class ModelEncoderConfig(BaseModel):
#     cnn_model: str = "resnet50"
#     pretrained: bool = True

# class ModelDecoderConfig(BaseModel):
#     hidden_size: int = 512
#     num_layers: int = 1
#     dropout_prob: float = 0.5

# class ModelConfig(BaseModel):
#     encoder: ModelEncoderConfig = ModelEncoderConfig()
#     decoder: ModelDecoderConfig = ModelDecoderConfig()
#     embed_size: int = 256

# class TrainingConfig(BaseModel):
#     batch_size: int = 64
#     num_epochs: int = 20
#     learning_rate: float = 0.001
#     lr_scheduler_step_size: int = 5
#     lr_scheduler_gamma: float = 0.1
#     optimizer_type: Literal["Adam", "SGD"] = "Adam"
#     gradient_clip_value: float = 5.0
#     log_interval: int = 100
#     save_every_epochs: int = 1
#     checkpoint_dir: Path = Field(default=Path("models/checkpoints"))

# class AppConfig(BaseModel):
#     experiment_name: str = "show_and_tell_coco"
#     device: str = "cuda"
#     random_seed: int = 42
#     dataset: DatasetConfig = DatasetConfig()
#     model: ModelConfig = ModelConfig()
#     training: TrainingConfig = TrainingConfig()

# def load_and_validate_config(config_path: Optional[Path] = None) -> AppConfig:
#     raw_config = load_config(config_path)
#     return AppConfig(**raw_config)