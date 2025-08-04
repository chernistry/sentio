"""Main CLI entry point for Sentio vNext.

This module serves as the entry point for the 'sentio' command-line tool.
It imports and exposes the CLI app defined in the src.cli package.
"""

from src.cli import app

if __name__ == "__main__":
    app()
