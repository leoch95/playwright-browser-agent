"""Main entrypoint for the Playwright Browser Agent CLI application.

This script imports the Typer application instance from the cli module
and runs it when the script is executed directly.
"""

# Entrypoint for the Playwright Browser Agent CLI

import typer

from playwright_browser_agent.cli import app

if __name__ == "__main__":
    app()
