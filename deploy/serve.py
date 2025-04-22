import os
import sys
import json
import argparse
from typing import Union, Optional
import torch
from PIL import Image
import numpy as np
import base64
import io
import uvicorn
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deploy.inference import Predictor
from utils.config import load_config
from utils.logger import setup_logging, get_logger

# Initialize logger
logger = get_logger(__name__)

# Global predictor instance for app factory
_predictor = None

# Define request and response models
class InferenceRequest(BaseModel):
    image: Union[str, list[str]]
    batch: bool = False
    return_probs: bool = False

class InferenceResponse(BaseModel):
    predictions: Union[int, list[float], list[int], list[list[float]]]

def setup_predictor(model_path: str, config_path: str = None) -> Predictor:
    """
    Set up the predictor.
    
    Args:
        model_path: Path to the model checkpoint
        config_path: Path to the configuration file
        
    Returns:
        Predictor instance
    """
    # Load config if provided
    config = None
    if config_path:
        config = load_config(config_path)
        
    # Initialize predictor
    return Predictor(model_path, config=config)

def decode_image(image_data: str) -> Image.Image:
    """
    Decode base64 encoded image.
    
    Args:
        image_data: Base64 encoded image data
        
    Returns:
        PIL Image
    """
    # Remove data URL prefix if present
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]
        
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    return image

def process_request(predictor: Predictor, request_data: InferenceRequest) -> dict[str, Any]:
    """
    Process inference request.
    
    Args:
        predictor: Predictor instance
        request_data: Request data
        
    Returns:
        Response data
    """
    # Get request data
    image_data = request_data.image
    batch = request_data.batch
    return_probs = request_data.return_probs
    
    # Process request
    if batch:
        # Batch inference
        images = []
        for img_data in image_data:
            images.append(decode_image(img_data))
            
        predictions = predictor.batch_predict(images, return_probs=return_probs)
    else:
        # Single image inference
        image = decode_image(image_data)
        predictions = predictor.predict(image, return_probs=return_probs)
        
    # Format response
    return {
        "predictions": predictions
    }

def create_app(predictor: Predictor = None):
    """
    Create FastAPI app.
    
    Args:
        predictor: Predictor instance or None to use global predictor
        
    Returns:
        FastAPI app
    """
    from fastapi import FastAPI, HTTPException, Depends
    
    # Use provided predictor or global predictor
    global _predictor
    if predictor:
        _predictor = predictor
    
    app = FastAPI(title="Vision Transformer API", 
                  description="API for Vision Transformer inference",
                  version="1.0.0")
    
    # Dependency to get predictor
    async def get_predictor():
        if _predictor is None:
            raise HTTPException(status_code=500, detail="Predictor not initialized")
        return _predictor
    
    @app.post("/predict", response_model=InferenceResponse)
    async def predict(request_data: InferenceRequest, predictor: Predictor = Depends(get_predictor)):
        try:
            return process_request(predictor, request_data)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    return app

# App factory for uvicorn
def get_application():
    """App factory for uvicorn."""
    # For Docker CMD, this will be used directly by uvicorn
    # with the command: "uvicorn deploy.serve:get_application()"
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pth")
    config_path = os.environ.get("CONFIG_PATH")
    
    # Setup predictor
    predictor = setup_predictor(model_path, config_path)
    
    # Create app
    return create_app(predictor)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Model serving script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config-path", type=str, default=None, help="Path to config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to store logs")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    # Setup predictor
    predictor = setup_predictor(args.model_path, args.config_path)
    
    # Create app
    app = create_app(predictor)
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
        
if __name__ == "__main__":
    main() 