from __future__ import annotations

"""Sentio CLI entrypoint.

Simplified to expose sub-commands directly (no nested *ingest ingest*).
"""

import typer

from src.cli.api import api_app
from src.cli.ingest import ingest_app
from src.cli.run import run_app
from src.cli.studio import studio_app
from src.cli.ui import ui_app

app = typer.Typer(help="Sentio RAG CLI", add_completion=False)

# Register command apps
app.add_typer(ingest_app, name="ingest")
app.add_typer(ui_app, name="ui")
app.add_typer(api_app, name="api")
app.add_typer(run_app, name="run")
app.add_typer(studio_app, name="studio")

__all__ = ["app"]
