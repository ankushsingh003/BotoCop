#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting build process..."

# Install uv using pip (most reliable method on Render)
pip install uv

# Sync dependencies using uv
# This creates a .venv with all requirements from pyproject.toml
uv sync

echo "Build complete."
