from setuptools import setup, find_packages
import os
import shutil

# Rename directory if needed
if os.path.exists('ml-experiments') and not os.path.exists('ml_experiments'):
    shutil.move('ml-experiments', 'ml_experiments')

setup(
    name="ml_experiments",
    version="0.1.0",
    packages=find_packages(),
    description="Vision Transformer Experiments",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        "torch==2.1.0",
        "torchvision==0.16.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "optuna>=3.0.0",
        "Pillow>=9.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
    ],
    python_requires=">=3.10",
) 