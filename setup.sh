#!/bin/bash

# Stop script on error
set -e

# Define environment name and Python version
ENV_NAME="dsmc_env"
PYTHON_VERSION="3.12"

# Download and install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    curl -o miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    ./miniconda.exe /S /D=$HOME/miniconda
    export PATH="$HOME/miniconda/condabin:$PATH"
    echo "Miniconda installed."
else
    echo "Conda already installed."
fi

# Initialize Conda
eval "$(conda shell.bash hook)"

# Create a new environment
echo "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate the environment
echo "Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Install required packages
echo "Installing packages from requirements.txt"
pip install -r requirements.txt

echo "Setup complete. Activate your environment using 'conda activate $ENV_NAME'."