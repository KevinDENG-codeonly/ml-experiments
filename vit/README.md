# Vision Transformer (ViT) Project

This repository contains a modular implementation of Vision Transformer (ViT) with a focus on extensibility and good software engineering practices.

## Features

- **Modular Architecture**: Separation of concerns with well-defined components
- **Registry Pattern**: Easily register and use new models and datasets
- **Multiple Operation Modes**: Train, evaluate, search hyperparameters, or run inference
- **Extensible**: Add new models and datasets without changing existing code
- **Deployment Ready**: Docker support and FastAPI serving

## Project Structure

```
.
├── core/                  # Training and evaluation code
│   ├── trainer.py         # Model training
│   ├── evaluator.py       # Model evaluation
│   └── searcher.py        # Hyperparameter search
├── datasets/              # Data loading and processing
│   ├── cifar10.py         # CIFAR-10 dataset
│   └── pennfudanped.py    # PennFudan pedestrian dataset
├── deploy/                # Deployment utilities
│   ├── Dockerfile         # Docker configuration
│   ├── inference.py       # Inference utilities
│   └── serve.py           # API server (FastAPI)
├── models/                # Model definitions
│   ├── base.py            # Base model class
│   ├── registry.py        # Model registry
│   └── vit.py             # Vision Transformer implementation
├── utils/                 # Utility functions
│   ├── config.py          # Configuration utilities
│   ├── logger.py          # Logging utilities
│   └── utils.py           # General utilities
├── config.yaml            # Default configuration
├── main.py                # Entry point
└── requirements.txt       # Python dependencies
```

## Getting Started

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd vit
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

#### Training

Train a model with default configuration:
```
python main.py train --config config.yaml
```

Resume training from a checkpoint:
```
python main.py train --config config.yaml --resume path/to/checkpoint.pth
```

#### Hyperparameter Search

Run hyperparameter search:
```
python main.py search --config config.yaml
```

#### Evaluation

Evaluate a trained model:
```
python main.py eval --config config.yaml --model-path path/to/model.pth
```

#### Inference

Run inference on a single image:
```
python main.py infer --config config.yaml --model-path path/to/model.pth --input path/to/image.jpg
```

Run inference on a directory of images:
```
python main.py infer --config config.yaml --model-path path/to/model.pth --input path/to/image_dir
```

Get class probabilities instead of just the prediction:
```
python main.py infer --config config.yaml --model-path path/to/model.pth --input path/to/image.jpg --return-probs
```

### Deployment

#### Docker

Build the Docker image:
```
docker build -t vit-app .
```

Run the Docker container:
```
docker run -p 5000:5000 -v /path/to/models:/app/models vit-app
```

#### API Usage

FastAPI provides interactive documentation at http://localhost:5000/docs

```python
import requests
import base64
from PIL import Image
import io

# Load image
img = Image.open("path/to/image.jpg")
buffer = io.BytesIO()
img.save(buffer, format="JPEG")
img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

# Make prediction request
response = requests.post(
    "http://localhost:5000/predict",
    json={"image": img_str, "return_probs": True}
)

# Print results
print(response.json())
```

## Extending the Project

### Adding a New Model

1. Create a new model file in the `models/` directory
2. Inherit from `BaseModel` and implement required methods
3. Register the model with the `@register_model` decorator
4. Import the model in `models/__init__.py`

Example:
```python
from .base import BaseModel
from .registry import register_model

@register_model("my_model")
class MyModel(BaseModel):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        # Model definition
        
    def forward(self, x):
        # Forward pass
        return output
```

### Adding a New Dataset

1. Create a new dataset file in the `datasets/` directory
2. Inherit from `BaseDataModule` and implement required methods
3. Register the dataset with the `@register_dataset` decorator
4. Import the dataset in `datasets/__init__.py`

Example:
```python
from .base import BaseDataModule
from .registry import register_dataset

@register_dataset("my_dataset")
class MyDataset(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        # Dataset initialization
        
    def prepare_data(self):
        # Download or prepare data
        
    def setup(self, stage=None):
        # Setup train/val/test datasets
        
    def get_num_classes(self):
        return num_classes
        
    def get_input_shape(self):
        return input_shape
```

## License

[MIT License](LICENSE)