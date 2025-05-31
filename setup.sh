#!/bin/bash

echo "Setting up DSPy-based Lean 4 Code Generator..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first."
    echo "Visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    uv venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Copy embeddings if they exist in parent directory
if [ -f "../lean4_embeddings_chunks.pkl" ]; then
    echo "Copying embeddings from parent directory..."
    cp ../lean4_embeddings_chunks.pkl .
fi

if [ -f "../lean4_embeddings.npy" ]; then
    cp ../lean4_embeddings.npy .
fi

# Create .env file from example if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env and add your API keys"
fi

# Generate embeddings
echo "Generating embeddings from Lean 4 documentation..."
python generate_embeddings.py

# Build dataset
echo "Building DSPy dataset..."
python src/build_dataset.py --analyze || echo "Dataset building skipped"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENROUTER_API_KEY"
echo "2. Test a single task: python run_dspy_tests.py --task task_id_0"
echo "3. Test all tasks: python run_dspy_tests.py --all"
echo "4. Optimize workflow: python src/main.py --optimize"
echo ""
echo "To activate the environment later: source venv/bin/activate"