"""Tests for the CLI interface (cli.py)."""

import pytest
from typer.testing import CliRunner

# TODO: Import the Typer app from cli.py
from playwright_browser_agent.cli import app

runner = CliRunner()

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

async def test_chat_command_help():
    """Test that the chat command shows help text."""
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "Usage: pb-agent chat" in result.stdout
    # pass # Remove the pass statement

@pytest.mark.skip(reason="CLI app import and tests not yet implemented")
async def test_batch_command_help():
    """Test that the batch command shows help text."""
    # result = runner.invoke(app, ["batch", "--help"])
    # assert result.exit_code == 0
    # assert "Usage: pb-agent batch" in result.stdout
    # assert "<file>" in result.stdout # Check for file argument
    pass

# TODO: Add tests for chat functionality
@pytest.mark.skip(reason="Chat interaction testing needs mocks")
async def test_chat_command_invocation():
    """Test invoking the chat command without interaction (requires mocking)."""
    # Mock agent.run_chat_loop here
    # result = runner.invoke(app, ["chat"])
    # assert result.exit_code == 0 # Or check for specific startup message
    pass

# TODO: Add tests for batch functionality
# TODO: Add tests for flags (--headless, --record, --provider, --model)