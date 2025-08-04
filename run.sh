#!/bin/bash
# Wrapper script to run Sentio CLI commands without installing the project

# Add current directory to PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Run CLI with passed arguments
python -m src.cli.main "$@"
