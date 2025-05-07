"""Command-line interface for the Playwright Browser Agent."""

import asyncio  # Need asyncio to run the agent
import sys
from enum import Enum  # Add Enum import
from typing import Optional

import typer

from .agent import run_agent_batch_session  # Updated import
from .agent import run_agent_chat_session
# Import config loading and agent logic (agent.py will be created later)
from .config import load_config
from .utils import wait_for_keypress  # Import wait function

# from .agent import run_chat_loop # Placeholder for agent import

# Define the modes using Enum
class MCPMode(str, Enum):
    snapshot = "snapshot"
    vision = "vision"

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
    record: bool = typer.Option(
        False,
        "--record",
        help="Record browser interactions by saving screenshots to the artifacts directory.",
    ),
    playwright_mcp_config_path: Optional[str] = typer.Option(
        None,
        "--playwright-mcp-config-path",
        help="Path to a custom Playwright MCP JSON configuration file.",
    ),
    artifacts_dir: str = typer.Option(
        "artifacts", # Default base directory for artifacts, timestamp will be appended
        "--screenshot-dir",
        help="Base directory to save screenshots. A timestamped subfolder will be created here.",
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
            record=record,
            playwright_mcp_config_path=playwright_mcp_config_path,
            artifacts_dir=artifacts_dir,
        )
    except SystemExit:
        # Config validation failed (e.g., missing API key), error already printed
        raise typer.Exit(code=1)

    print(f"Using LLM: {config.llm_provider}/{config.llm_model}")
    print(f"Recording screenshots: {config.record}")
    print(f"Screenshots will be saved in subdirectories of: {config.artifacts_dir}")

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
    record: bool = typer.Option(
        False,
        "--record",
        help="Record screenshots during batch processing.",
    ),
    playwright_mcp_config_path: Optional[str] = typer.Option(
        None,
        "--playwright-mcp-config-path",
        help="Path to a custom Playwright MCP JSON configuration file for batch mode.",
    ),
    artifacts_dir: str = typer.Option(
        "artifacts", # Default base directory for artifacts, timestamp will be appended
        "--screenshot-dir",
        help="Base directory to save screenshots during batch. A timestamped subfolder will be created here.",
    ),
):
    """Processes browser instructions from a file in batch mode."""
    print(f"Starting batch mode with file: {file}")

    # Load configuration, applying CLI overrides
    try:
        config = load_config(
            llm_provider=llm_provider,
            llm_model=llm_model,
            record=record,
            playwright_mcp_config_path=playwright_mcp_config_path,
            artifacts_dir=artifacts_dir,
        )
    except SystemExit:
        raise typer.Exit(code=1)

    print(f"Using LLM: {config.llm_provider}/{config.llm_model}")
    print(f"Recording screenshots: {config.record}")
    print(f"Screenshots will be saved in subdirectories of: {config.artifacts_dir}")

    # Placeholder: Read file and call agent's batch processing function
    try:
        with open(file, 'r') as f:
            # Read non-empty lines, skipping those starting with '#'
            instructions = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        print(f"Read {len(instructions)} instructions from {file}.")
        # Placeholder for batch processing logic
        # print("\n>>> Placeholder: Agent batch processing would run here <<<") # Remove this line
        # run_batch_processing(config, instructions) # Remove this line

        # Run the agent's batch processing
        try:
            asyncio.run(run_agent_batch_session(config, instructions))
        except KeyboardInterrupt:
            print("\nBatch processing interrupted by user. Exiting.")
        except Exception as e:
            print(f"\nAn error occurred during batch processing: {e}", file=sys.stderr)
            raise typer.Exit(code=1)

        # Wait for user keypress after successful batch completion
        print("\nBatch processing finished.")
        wait_for_keypress()

    except FileNotFoundError:
        print(f"Error: Batch file not found at {file}", file=sys.stderr)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    # This allows running the CLI directly via `python -m src.playwright_browser_agent.cli` for testing
    app()