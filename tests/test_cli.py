"""Tests for the CLI interface (cli.py)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from playwright_browser_agent.cli import app
# Import the actual Settings class to create a mock instance
from playwright_browser_agent.config import Settings

runner = CliRunner()

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# @pytest.mark.skip(reason="CLI app import and tests not yet implemented")
async def test_chat_command_help():
    """Test that the chat command shows help text."""
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "Usage: pb-agent chat" in result.stdout
    # pass # Remove the pass statement

# @pytest.mark.skip(reason="CLI app import and tests not yet implemented")
async def test_batch_command_help():
    """Test that the batch command shows help text."""
    result = runner.invoke(app, ["batch", "--help"])
    assert result.exit_code == 0
    assert "Usage: pb-agent batch" in result.stdout
    assert " FILE " in result.stdout # Check for file argument (Typer uses uppercase)
    # pass # REMOVE THIS LINE


# TODO: Add tests for chat functionality
@patch("playwright_browser_agent.cli.load_config")
@patch("playwright_browser_agent.cli.run_agent_chat_session", new_callable=AsyncMock)
@patch("playwright_browser_agent.cli.asyncio.run") # Patch asyncio.run as used in cli.py
async def test_chat_command_invocation(mock_asyncio_run, mock_run_agent_chat_session, mock_load_config):
    """Test invoking the chat command, mocking asyncio.run used by cli.py."""
    # Configure the mock_load_config
    mock_config = MagicMock(spec=Settings)
    mock_config.llm_provider = "mock_provider"
    mock_config.llm_model = "mock_model"
    mock_config.mode = "snapshot"
    mock_config.headless = False
    mock_config.record = False
    mock_load_config.return_value = mock_config

    # Configure asyncio.run mock to schedule the coroutine on the existing loop
    def schedule_coro_on_running_loop(coro):
        return asyncio.create_task(coro)
    mock_asyncio_run.side_effect = schedule_coro_on_running_loop

    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}. Output:\n{result.stdout}\nException: {result.exception}"
    mock_load_config.assert_called_once()
    # Assert that the agent session coroutine was created
    mock_run_agent_chat_session.assert_called_once_with(mock_config)
    # Assert that our patched asyncio.run was called (it would receive the coroutine from above)
    mock_asyncio_run.assert_called_once() # Simpler assertion

# TODO: Add tests for batch functionality
@patch("playwright_browser_agent.cli.load_config")
@patch("playwright_browser_agent.cli.run_agent_batch_session", new_callable=AsyncMock)
@patch("playwright_browser_agent.cli.asyncio.run") # Patch asyncio.run as used in cli.py
@patch("playwright_browser_agent.cli.wait_for_keypress")
async def test_batch_command_invocation(mock_wait_for_keypress, mock_asyncio_run, mock_run_agent_batch_session, mock_load_config, tmp_path):
    """Test invoking the batch command, mocking asyncio.run used by cli.py."""
    # Configure the mock_load_config
    mock_config = MagicMock(spec=Settings)
    mock_config.llm_provider = "mock_provider"
    mock_config.llm_model = "mock_model"
    mock_config.mode = "snapshot"
    mock_config.headless = False
    mock_config.record = False
    mock_load_config.return_value = mock_config

    # Configure asyncio.run mock to schedule the coroutine on the existing loop
    def schedule_coro_on_running_loop(coro):
        return asyncio.create_task(coro)
    mock_asyncio_run.side_effect = schedule_coro_on_running_loop

    # Create a dummy batch file
    dummy_file = tmp_path / "batch_instructions.txt"
    instructions = ["Instruction 1", "Instruction 2"]
    dummy_file.write_text("\n".join(instructions))

    result = runner.invoke(app, ["batch", str(dummy_file)])
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}. Output:\n{result.stdout}\nException: {result.exception}"
    mock_load_config.assert_called_once()
    # Assert that the agent session coroutine was created
    mock_run_agent_batch_session.assert_called_once_with(mock_config, instructions)
    # Assert that our patched asyncio.run was called (it would receive the coroutine from above)
    mock_asyncio_run.assert_called_once() # Simpler assertion
    mock_wait_for_keypress.assert_called_once()

# TODO: Add tests for flags (--headless, --record, --provider, --model)