#!/bin/bash
set -e

# Default profile is local unless specified
export PGPT_PROFILES="${PGPT_PROFILES:-local}"

poetry run python -m private_gpt
