"""Command-line interface for the Playwright Browser Agent."""

import typer

app = typer.Typer(
    name="pb-agent",
    help="A Playwright-powered browser agent controlled by an LLM.",
    add_completion=False, # Optional: disable shell completion
)

@app.callback()
def main_callback():
    """Main entry point callback for basic setup or version check (optional)."""
    # Placeholder for potential global options like --version
    pass

@app.command()
def chat():
    print("Starting chat mode...")

@app.command()
def batch(file: str):
    print(f"Starting batch mode with file: {file}")

if __name__ == "__main__":
    # This allows running the CLI directly via `python -m src.playwright_browser_agent.cli` for testing
    app()