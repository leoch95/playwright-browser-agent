"""Core agent logic using LangChain and Playwright-MCP."""

import asyncio
import base64  # Import base64 for decoding
import inspect  # For inspect.isclass
import sys
import uuid  # Import uuid for generating thread IDs
from pathlib import Path  # Add Path import
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from langchain.agents import \
    AgentExecutor  # Assuming create_react_agent returns this type
from langchain_core.messages import \
    SystemMessage  # SystemMessage still needed for batch mode starting message
from langchain_core.messages import (  # Added ToolMessage for type check
    AIMessage, BaseMessage, HumanMessage, ToolMessage)
from langchain_core.prompts import \
    ChatPromptTemplate  # Import ChatPromptTemplate
# Import RunnableConfig
from langchain_core.runnables import RunnableConfig
from langchain_litellm import ChatLiteLLM
# Import the specific connection type
# Define connection types for casting
from langchain_mcp_adapters.client import StdioConnection  # Keep this
from langchain_mcp_adapters.client import (MultiServerMCPClient, SSEConnection,
                                           WebsocketConnection)
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import config type and prompt builder
from .config import Settings
from .prompts import build_system_prompt  # Correct import
from .utils import manage_screenshot_dir  # Import screenshot dir manager

# from .utils import wait_for_keypress # Remove wait function import from here

# from .prompts import build_system_prompt # TODO: Create this function

# Placeholder for configuration - this should come from config.py
# Example config structure (adapt based on actual config.py implementation)
# DEFAULT_MODEL = "gpt-4o" # Remove - Configured via Settings
# DEFAULT_MCP_CONFIG = { # Remove - Configured via Settings (partially hardcoded for now)
#     "playwright": {
#         # Default config to run playwright-mcp via npx with stdio
#         "command": "npx",
#         "args": ["@playwright/mcp@latest"],
#         "transport": "stdio", # Or "sse" if using HTTP
#     }
# }

# No need for MCPConnectionsType alias anymore

def _debug_print_structure(obj: Any, indent: int = 0, label: str = ""):
    """Recursively prints the type and structure of an object for debugging."""
    prefix = "  " * indent
    obj_type_str = str(type(obj))

    print(f"{prefix}{label}({obj_type_str}): ", end="")

    if obj is None or isinstance(obj, (str, int, float, bool)):
        print(repr(obj))
        return

    if isinstance(obj, dict):
        print("{")
        for k, v in obj.items():
            _debug_print_structure(v, indent + 1, label=f"'{k}'")
        print(f"{prefix}}}")
    elif isinstance(obj, (list, tuple, set)):
        print("[")
        for i, item in enumerate(obj):
            _debug_print_structure(item, indent + 1, label=f"[{i}]")
        print(f"{prefix}]")
    # Check for common LangChain/LangGraph message types that might have 'content' or 'artifact'
    elif hasattr(obj, 'artifact') and obj.artifact is not None : # Check artifact first as it's more specific for our case
        print(f" (has .artifact)")
        _debug_print_structure(obj.artifact, indent + 1, label=".artifact")
    elif hasattr(obj, 'content') and obj.content is not None:
         print(f" (has .content)")
         _debug_print_structure(obj.content, indent + 1, label=".content")
    elif hasattr(obj, '__dict__') and not inspect.isclass(obj): # Avoid trying to print class attributes
        print("{ (attributes via __dict__)")
        for attr, value in obj.__dict__.items():
            if not callable(value) and not attr.startswith("__"): # Avoid callables and private/magic methods
                 _debug_print_structure(value, indent + 1, label=f".{attr}")
        print(f"{prefix}}}")
    else:
        try:
            # Fallback for other types, try to represent them
            print(repr(obj))
        except Exception:
            print("<Unable to represent>")

def _handle_screenshot_tool_end(
    tool_output_data: Any,
    record_screenshots_flag: bool,
    screenshot_session_dir: Optional[Path],
    step_counter: int
) -> int:
    """Handles the output of the browser_take_screenshot tool, saves image, and logs."""
    if record_screenshots_flag and screenshot_session_dir:
        step_counter += 1
        screenshot_file_path = None
        try:
            # tool_output_data is expected to be a dictionary containing an 'artifact' list or a ToolMessage with 'content' string and 'artifact' list
            artifacts_list = None
            if isinstance(tool_output_data, dict):
                artifacts_list = tool_output_data.get("artifact")
            elif hasattr(tool_output_data, 'artifact'): # Handle ToolMessage-like objects
                artifacts_list = tool_output_data.artifact

            found_image_artifact = None # Renamed from image_artifact_dict

            if isinstance(artifacts_list, list):
                for item in artifacts_list:
                    # Correctly check for ImageContent object attributes
                    if (hasattr(item, 'type') and item.type == "image" and
                        hasattr(item, 'data') and item.data): # Check item.data is not None or empty
                        found_image_artifact = item # Store the object itself
                        break

            if found_image_artifact: # If an image artifact object was found
                image_data_b64 = found_image_artifact.data # Access .data attribute
                image_bytes = base64.b64decode(image_data_b64)
                screenshot_file_name = f"step_{step_counter}.png"
                screenshot_file_path = screenshot_session_dir / screenshot_file_name
                with open(screenshot_file_path, 'wb') as f:
                    f.write(image_bytes)
                print(f"📸 Screenshot taken: {screenshot_file_path.resolve()}", flush=True)
            else:
                # This message will still show tool_output_data if debugging is needed
                print(f"📸 Screenshot tool ran, but no image artifact found or artifact format unexpected. Debugging output structure:", flush=True)
                _debug_print_structure(tool_output_data, label="tool_output_data")
        except Exception as e:
            print(f"📸 Error saving screenshot: {e}. Path: {screenshot_file_path}", flush=True)
            # This message will still show tool_output_data if debugging is needed
            print(f"Debugging structure of tool_output_data that caused save error:", flush=True)
            _debug_print_structure(tool_output_data, label="tool_output_data_on_error")

    else:
        # This case handles when recording is off or dir is not set.
        # Do not print tool_output_data here as it's not relevant to a format issue.
        print(f"📸 Screenshot tool finished (recording disabled or screenshot directory issue).", flush=True)
    return step_counter

def _setup_agent_resources(config: Settings) -> tuple[ChatLiteLLM, Dict[str, Any], str]:
    """Helper to configure LLM, MCP, and system prompt string."""

    # Ensure default MCP config file exists
    default_mcp_config_file = Path("playwright-mcp-config.json")
    if not default_mcp_config_file.exists():
        print(f"Default MCP config file not found. Creating '{default_mcp_config_file}' with empty JSON.")
        try:
            with open(default_mcp_config_file, 'w') as f:
                f.write("{}")
        except OSError as e:
            print(f"Warning: Could not create default MCP config file '{default_mcp_config_file}': {e}")

    # 1. Build MCP Configuration Args
    mcp_args = ["@playwright/mcp@latest"]

    # Add custom config file path if provided
    if config.playwright_mcp_config_path:
        config_path = Path(config.playwright_mcp_config_path)
        if config_path.is_file():
            mcp_args.extend(["--config", str(config_path.resolve())]) # Pass absolute path
            print(f"Using Playwright MCP config file: {config_path.resolve()}")
        else:
            print(f"Warning: Provided MCP config path '{config.playwright_mcp_config_path}' does not exist or is not a file. Ignoring.")

    # Configure MCP mode based on config.mode
    if config.headless:
        mcp_args.append("--headless")
        print("Configuring Playwright MCP for Headless mode.")
    else:
        print("Configuring Playwright MCP for Headful mode (visible browser).")

    # Configure Vision/Snapshot mode
    if config.mode == "vision":
        mcp_args.append("--vision")
        print("Configuring Playwright MCP for Vision mode.")
    elif config.mode == "snapshot":
        print("Configuring Playwright MCP for Snapshot mode (default).")
        # No extra args needed for snapshot mode
    # else block removed as validation is done in config.py

    mcp_config = {
        "playwright": {
            "command": "npx",
            "args": mcp_args,
            "transport": "stdio",
        }
    }

    # 2. Initialize ChatLiteLLM
    llm = ChatLiteLLM(
        model_name=f"{config.llm_provider}/{config.llm_model}",
        streaming=True,
    )

    # 3. Build System Prompt String
    # build_system_prompt now returns a string
    system_prompt_string = build_system_prompt(
        mode=config.mode,
        headless=config.headless,
        record_screenshots=config.record
    )

    return llm, mcp_config, system_prompt_string

async def run_agent_chat_session(config: Settings):
    """Initializes the agent and runs the interactive chat loop with memory."""
    llm, mcp_config, system_prompt_string = _setup_agent_resources(config)

    # Initialize MemorySaver for in-memory history
    checkpointer = MemorySaver()

    # Manage screenshot directory for this session
    screenshot_session_dir: Optional[Path] = None
    if config.record:
        try:
            screenshot_session_dir = manage_screenshot_dir(config.artifacts_dir)
        except Exception as e:
            print(f"Warning: Could not create screenshot directory. Recording will be disabled. Error: {e}", file=sys.stderr)
            # Optionally, set config.record = False here if you want to disable recording fully

    # Initialize PlaywrightMCPTool via MultiServerMCPClient
    async with MultiServerMCPClient(connections=mcp_config) as client: # type: ignore
        tools = client.get_tools()

        # Use langgraph.prebuilt.create_react_agent (WITHOUT system_message/state_modifier)
        agent_executor = create_react_agent(
            llm,
            tools,
            # No system_message or state_modifier here
            checkpointer=checkpointer
        )

        # Start the interactive chat loop
        # Pass a unique thread_id for the session
        thread_id = str(uuid.uuid4())
        print(f"\nStarting chat session (Thread ID: {thread_id}). Type 'exit' or 'bye' to end.")
        await run_chat_loop(
            agent_executor,
            thread_id=thread_id,
            initial_system_prompt=system_prompt_string,
            record_screenshots=config.record,
            screenshot_dir=screenshot_session_dir
        )

async def run_chat_loop(
    agent_executor: Any,
    thread_id: str,
    initial_system_prompt: str,
    record_screenshots: bool,
    screenshot_dir: Optional[Path]
):
    """Runs the interactive chat loop using the agent's stream_events method with memory."""
    # Manually add system prompt ONLY IF history is empty (first run for this thread_id)
    # Checkpointer handles history, so we don't manage a local messages list across turns.
    is_first_turn = True # Assume first turn initially
    # TODO: A more robust check would involve querying the checkpointer/store
    # for existing messages for thread_id, but this requires more setup.
    step_counter = 0 # Initialize step counter for screenshots for this thread

    while True:
        user_input = input("\n💁 User: ")
        if user_input.lower() in ["exit", "bye"]:
            print("\n🌐 System: Goodbye! 👋")
            break

        # Construct messages for this turn
        current_turn_messages: List[BaseMessage] = []
        if is_first_turn:
            current_turn_messages.append(SystemMessage(content=initial_system_prompt))
            print(f"\n🌐 System (Initial): {initial_system_prompt}") # Show initial prompt
            is_first_turn = False # Don't add system prompt again

        current_turn_messages.append(HumanMessage(content=user_input))

        # Create RunnableConfig with thread_id
        run_config = RunnableConfig(configurable={"thread_id": thread_id})

        # Use agent's stream_events method with config for memory
        print("\n🤖 Agent: ", end="", flush=True)
        async for chunk in agent_executor.astream_events({"messages": current_turn_messages}, config=run_config, version="v2"):
            # Print intermediate steps/thoughts/tool calls/outputs
            event = chunk.get("event")
            data = chunk.get("data", {})
            name = chunk.get("name") # Tool name or agent name

            if event == "on_chat_model_stream":
                chunk_content = data.get("chunk")
                if chunk_content and hasattr(chunk_content, 'content') and chunk_content.content:
                    print(chunk_content.content, end="", flush=True)
                    # No need to manually accumulate AI response here, LangGraph handles state
            elif event == "on_tool_start" and name:
                # tool_input = data.get("input", {}) # LangGraph typically puts tool input here - No longer needed for logging
                if name == "browser_take_screenshot":
                    print(f"📸 Attempting to take screenshot...", flush=True)
                else:
                    # print(f"🛠️ Calling tool [{name}] with input: {tool_input}...", end="", flush=True)
                    print(f"🛠️ Calling tool [{name}]...", end="", flush=True)
            elif event == "on_tool_end" and name:
                tool_output_data = data.get("output") # LangGraph typically puts tool output here

                if name == "browser_take_screenshot":
                    step_counter = _handle_screenshot_tool_end(
                        tool_output_data,
                        record_screenshots,
                        screenshot_dir,
                        step_counter
                    )
                else:
                    # Generic success message for other tools, can be expanded if needed
                    print(f" success.", flush=True)
                # Handle potential errors by inspecting 'data' if LangGraph provides it
                # Example (pseudo-code): if 'error' in data: print(f" failed: {data['error']}", flush=True)

        print() # Add a newline after the full response stream

async def run_agent_batch_session(config: Settings, instructions: List[str]):
    """Initializes the agent and runs instructions from a list in batch mode with memory per instruction."""
    llm, mcp_config, system_prompt_string = _setup_agent_resources(config)

    # Initialize MemorySaver - used *within* each instruction's execution if needed
    # but state isn't intended to persist *between* instructions in batch mode currently.
    # A single checkpointer can be reused.
    checkpointer = MemorySaver()

    # Manage screenshot directory for this session
    screenshot_session_dir: Optional[Path] = None
    if config.record:
        try:
            screenshot_session_dir = manage_screenshot_dir(config.artifacts_dir)
        except Exception as e:
            print(f"Warning: Could not create screenshot directory for batch. Recording will be disabled. Error: {e}", file=sys.stderr)

    # Initialize PlaywrightMCPTool via MultiServerMCPClient
    async with MultiServerMCPClient(connections=mcp_config) as client: # type: ignore
        tools = client.get_tools()

        # Use langgraph.prebuilt.create_react_agent (WITHOUT system_message/state_modifier)
        agent_executor: AgentExecutor = create_react_agent( # type: ignore
            llm,
            tools,
            # No system_message or state_modifier here
            checkpointer=checkpointer
        )

        # Process instructions one by one
        # We need a way to pass the initial system prompt conceptually, but LangGraph handles it.
        # The state_modifier injects it before each LLM call within the agent run.

        # Create a unique thread_id for the entire batch run. Memory will accumulate
        # across instructions within this single batch run.
        batch_thread_id = f"batch-{uuid.uuid4()}"
        print(f"\nStarting batch process (Thread ID: {batch_thread_id})")

        step_counter = 0 # Initialize step counter for screenshots for this batch thread

        for i, instruction in enumerate(instructions):
            print(f"\n--- Instruction {i+1}/{len(instructions)} --- ")
            print(f"💁 User: {instruction}")

            # Construct messages for this turn, including System prompt only on the *very first* instruction
            current_turn_messages: List[BaseMessage] = []
            if i == 0:
                current_turn_messages.append(SystemMessage(content=system_prompt_string))
                print(f"\n🌐 System (Initial): {system_prompt_string}") # Show initial prompt

            current_turn_messages.append(HumanMessage(content=instruction))

            # Create RunnableConfig with thread_id for the batch run
            run_config = RunnableConfig(configurable={"thread_id": batch_thread_id})

            try:
                # Stream the agent's response for the current instruction using astream_events
                print("\n🤖 Agent: ", end="", flush=True)
                async for chunk in agent_executor.astream_events({"messages": current_turn_messages}, config=run_config, version="v2"):
                    # Print intermediate steps/thoughts/tool calls/outputs
                    event = chunk.get("event")
                    data = chunk.get("data", {})
                    name = chunk.get("name") # Tool name or agent name

                    if event == "on_chat_model_stream":
                        chunk_content = data.get("chunk")
                        if chunk_content and hasattr(chunk_content, 'content') and chunk_content.content:
                            print(chunk_content.content, end="", flush=True)
                            # No manual accumulation
                    elif event == "on_tool_start" and name:
                        # tool_input = data.get("input", {}) # LangGraph typically puts tool input here - No longer needed for logging
                        if name == "browser_take_screenshot":
                            print(f"📸 Attempting to take screenshot...", flush=True)
                        else:
                            # print(f"🛠️ Calling tool [{name}] with input: {tool_input}...", end="", flush=True)
                            print(f"🛠️ Calling tool [{name}]...", end="", flush=True)
                    elif event == "on_tool_end" and name:
                        tool_output_data = data.get("output") # LangGraph typically puts tool output here
                        if name == "browser_take_screenshot":
                            step_counter = _handle_screenshot_tool_end(
                                tool_output_data,
                                config.record,
                                screenshot_session_dir,
                                step_counter
                            )
                        else:
                            # Generic success message for other tools, can be expanded if needed
                            print(f" success.", flush=True)
                        # Handle potential errors by inspecting 'data' if LangGraph provides it
                        # Example (pseudo-code): if 'error' in data: print(f" failed: {data['error']}", flush=True)

                print() # Add a newline after the full response for the step

            except Exception as e:
                print(f"\n❌ Error processing instruction {i+1}: {e}")
                print("Skipping to next instruction...")
                # Optionally add an error message to the LangGraph state?
                # Requires modifying the graph or handling outside.
                continue # Move to the next instruction

        print("\n--- Batch processing finished --- ")

        # Pass the system prompt string to the loop/handler
        # await run_chat_loop(agent_executor, thread_id=batch_thread_id, initial_system_prompt=system_prompt_string) # Removed this line

# Example usage (for testing, will be called from cli.py)
async def main():
    pass # Keep async main structure if needed for direct testing later

if __name__ == "__main__":
    # Note: Running asyncio directly might require adjustments based on environment
    try:
        pass # Keep async main structure if needed for direct testing later
        # Example: configure and run chat
        # temp_config = Settings(llm_provider="openai", llm_model="gpt-4o", ...) # Load properly
        # asyncio.run(run_agent_chat_session(temp_config))
    except KeyboardInterrupt:
        print("\nExiting...")