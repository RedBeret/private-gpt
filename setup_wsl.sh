#!/bin/bash
set -e

# Install Python dependencies and UI extras
poetry install --no-interaction --extras ui

# Download models for offline use
poetry run python scripts/setup

echo "Setup completed"
