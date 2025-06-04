@echo off
poetry install --no-interaction --extras ui
poetry run python scripts/setup
ECHO Setup completed
