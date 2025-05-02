# Entrypoint for the Playwright Browser Agent CLI

import typer

from playwright_browser_agent.cli import app

if __name__ == "__main__":
    app()
