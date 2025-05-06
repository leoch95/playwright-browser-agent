"""Core agent logic using LangChain and Playwright-MCP."""

import asyncio
import uuid  # Import uuid for generating thread IDs
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from langchain.agents import \
    AgentExecutor  # Assuming create_react_agent returns this type
from langchain_core.messages import \
    SystemMessage  # SystemMessage still needed for batch mode starting message
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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


def _setup_agent_resources(config: Settings) -> tuple[ChatLiteLLM, Dict[str, Any], str]:
    """Helper to configure LLM, MCP, and system prompt string."""
    # 1. Build MCP Configuration
    mcp_args = ["@playwright/mcp@latest"]

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
        await run_chat_loop(agent_executor, thread_id=thread_id, initial_system_prompt=system_prompt_string)

async def run_chat_loop(agent_executor: Any, thread_id: str, initial_system_prompt: str):
    """Runs the interactive chat loop using the agent's stream_events method with memory."""
    # Manually add system prompt ONLY IF history is empty (first run for this thread_id)
    # Checkpointer handles history, so we don't manage a local messages list across turns.
    is_first_turn = True # Assume first turn initially
    # TODO: A more robust check would involve querying the checkpointer/store
    # for existing messages for thread_id, but this requires more setup.

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
                # Check if it's the final agent step before printing tool call
                # This avoids printing tool calls when the LLM is just thinking
                # Heuristic: Assume a non-empty data dict means it's a real call?
                # A better check might be needed depending on langgraph event details
                # For now, print all tool starts
                print(f"🛠️ Calling tool [{name}]...", end="", flush=True)
            elif event == "on_tool_end" and name:
                # Check for errors in tool execution if possible from `data`
                # E.g., if data['output'] contains error info or if a specific error key exists
                # Assuming simple success print for now
                print(" success", flush=True)
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
                        print(f"🛠️ Calling tool [{name}]...", end="", flush=True)
                    elif event == "on_tool_end" and name:
                        print(" success", flush=True)
                        # TODO: Error handling based on 'data'

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