"""Command-line interface for the Playwright Browser Agent."""

import asyncio  # Need asyncio to run the agent
import sys
from typing import Optional

import typer

from .agent import run_agent_chat_session  # Updated import
# Import config loading and agent logic (agent.py will be created later)
from .config import load_config

# from .agent import run_chat_loop # Placeholder for agent import

app = typer.Typer(
    name="pb-agent",
    help="A Playwright-powered browser agent controlled by an LLM.",
    add_completion=False, # Optional: disable shell completion
)

@app.callback()
def main_callback():
    """Main entry point callback, invoked before any command.

    Can be used for global setup, version checks, etc. Currently a placeholder.
    """
    # Placeholder for potential global options like --version
    pass

@app.command()
def chat(
    llm_provider: Optional[str] = typer.Option(
        None,
        "--llm-provider",
        "-p",
        help="LLM provider to use (e.g., 'openai', 'anthropic', 'google'). Reads from LLM_PROVIDER env var if not set.",
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        "--llm-model",
        "-m",
        help="Specific LLM model to use (e.g., 'gpt-4o', 'claude-3-opus-20240229'). Reads from LLM_MODEL env var if not set.",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Run the browser in headless mode (no GUI).",
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help="Record browser interactions by saving screenshots to the artifacts directory.",
    ),
):
    """Starts an interactive chat session to control the browser agent."""
    print("Starting interactive chat mode...")

    # Load configuration, applying CLI overrides
    # Note: Pydantic settings load env vars automatically
    # CLI args passed here will override env vars
    try:
        config = load_config(
            llm_provider=llm_provider,
            llm_model=llm_model,
            headless=headless,
            record=record,
        )
    except SystemExit:
        # Config validation failed (e.g., missing API key), error already printed
        raise typer.Exit(code=1)

    print(f"Using LLM: {config.llm_provider}/{config.llm_model}")
    print(f"Headless mode: {config.headless}")
    print(f"Recording screenshots: {config.record}")

    # Placeholder: Call the agent's chat loop (to be implemented in agent.py)
    # print("\n>>> Placeholder: Agent chat loop would start here <<<") # Remove this line
    # try:
    #     run_chat_loop(config)
    # except Exception as e:
    #     print(f"\nError during chat loop: {e}", file=sys.stderr)
    #     raise typer.Exit(code=1)

    # Call the agent's chat session runner
    try:
        asyncio.run(run_agent_chat_session(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    except Exception as e:
        print(f"\nAn error occurred during the chat session: {e}", file=sys.stderr)
        # Consider more specific error handling or logging traceback
        raise typer.Exit(code=1)

    print("Chat session ended.")

@app.command()
def batch(
    file: str = typer.Argument(
        ..., # Ellipsis marks it as required
        help="Path to the file containing batch instructions, one per line.",
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--llm-provider",
        "-p",
        help="LLM provider override for batch mode.",
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        "--llm-model",
        "-m",
        help="LLM model override for batch mode.",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Run the browser in headless mode for batch processing.",
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help="Record screenshots during batch processing.",
    ),
):
    """Processes browser instructions from a file in batch mode."""
    print(f"Starting batch mode with file: {file}")

    # Load configuration, applying CLI overrides
    try:
        config = load_config(
            llm_provider=llm_provider,
            llm_model=llm_model,
            headless=headless,
            record=record,
        )
    except SystemExit:
        raise typer.Exit(code=1)

    print(f"Using LLM: {config.llm_provider}/{config.llm_model}")
    print(f"Headless mode: {config.headless}")
    print(f"Recording screenshots: {config.record}")

    # Placeholder: Read file and call agent's batch processing function
    try:
        with open(file, 'r') as f:
            instructions = [line.strip() for line in f if line.strip()]
        print(f"Read {len(instructions)} instructions from {file}.")
        # Placeholder for batch processing logic
        print("\n>>> Placeholder: Agent batch processing would run here <<<")
        # run_batch_processing(config, instructions)
    except FileNotFoundError:
        print(f"Error: Batch file not found at {file}", file=sys.stderr)
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"\nError during batch processing: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    print("Batch processing finished.")

if __name__ == "__main__":
    # This allows running the CLI directly via `python -m src.playwright_browser_agent.cli` for testing
    app()