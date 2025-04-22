FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and serving dependencies
RUN pip install --no-cache-dir fastapi uvicorn

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/best_model.pth

# Create models directory
RUN mkdir -p /app/models

# Expose port for API
EXPOSE 5000

# Run the server
CMD ["uvicorn", "deploy.serve:get_application", "--host", "0.0.0.0", "--port", "5000"] 