from abc import ABC, abstractmethod
from typing import Optional, Any
import torch
import torch.nn as nn
import sys
import os

# 在文件顶部添加项目根目录到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseHandler(ABC):
    """Base class for core components (trainer, evaluator, searcher)."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """Main method to run the component."""
        pass 