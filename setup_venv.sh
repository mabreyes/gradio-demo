#!/bin/bash
# Setup script for virtual environment

set -e

echo "Setting up virtual environment..."

# Check if venv already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Activating it..."
    source venv/bin/activate
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To run the application, run: python main.py"

