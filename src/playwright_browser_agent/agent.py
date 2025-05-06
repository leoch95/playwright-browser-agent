"""Core agent logic using LangChain and Playwright-MCP."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from langchain.agents import \
    AgentExecutor  # Assuming create_react_agent returns this type
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_litellm import ChatLiteLLM
# Define connection types for casting
from langchain_mcp_adapters.client import (MultiServerMCPClient, SSEConnection,
                                           StdioConnection,
                                           WebsocketConnection)
from langgraph.prebuilt import create_react_agent

# Import config type and prompt builder
from .config import Settings
from .prompts import build_system_prompt  # Uncomment import

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


MCPConnectionsType = Dict[str, Union[StdioConnection, SSEConnection, WebsocketConnection]]

# Import the specific connection type
from langchain_mcp_adapters.client import StdioConnection


def _setup_agent_resources(config: Settings) -> tuple[ChatLiteLLM, Dict[str, Any], str]:
    """Helper to configure LLM, MCP, and system prompt."""
    # 1. Build MCP Configuration
    mcp_args = ["@playwright/mcp@latest"]
    if config.headless:
        mcp_args.append("--headless")
        print("Configuring Playwright MCP for Headless mode.")
    else:
        mcp_args.append("--vision")
        print("Configuring Playwright MCP for Vision mode (non-headless).")

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

    # 3. Build System Prompt
    system_prompt = build_system_prompt(record_screenshots=config.record)

    return llm, mcp_config, system_prompt

async def run_agent_chat_session(config: Settings):
    """Initializes the agent and runs the interactive chat loop."""
    llm, mcp_config, system_prompt = _setup_agent_resources(config)

    # Initialize PlaywrightMCPTool via MultiServerMCPClient
    async with MultiServerMCPClient(connections=mcp_config) as client: # type: ignore
        tools = client.get_tools()

        # Use langgraph.prebuilt.create_react_agent
        agent_executor = create_react_agent(llm, tools)

        # Start the interactive chat loop
        await run_chat_loop(agent_executor, initial_prompt=system_prompt)

async def run_chat_loop(agent_executor: Any, initial_prompt: Optional[str] = None):
    """Runs the interactive chat loop using the agent's stream method."""
    messages = []
    if initial_prompt:
        messages.append(SystemMessage(content=initial_prompt))
        print(f"System: {initial_prompt}")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "bye"]:
            print("Exiting chat.")
            break

        messages.append(HumanMessage(content=user_input))

        # 4. Use agent's stream method
        async for chunk in agent_executor.astream({"messages": messages}):
            # TODO: Implement proper handling of streaming output chunks
            # Chunks can contain agent actions, tool outputs, final answers etc.
            # Need to parse and display appropriately.
            print(chunk, flush=True) # Basic printing for now

            # TODO: Update messages list with the AI's final response for history
            # For ReAct, the final response is usually in the 'output' key of the last chunk
            if "output" in chunk:
                 messages.append(AIMessage(content=chunk["output"]))

async def run_agent_batch_session(config: Settings, instructions: List[str]):
    """Initializes the agent and runs instructions from a list in batch mode."""
    llm, mcp_config, system_prompt = _setup_agent_resources(config)
    print(f"System: {system_prompt}") # Print system prompt once

    # Initialize PlaywrightMCPTool via MultiServerMCPClient
    async with MultiServerMCPClient(connections=mcp_config) as client: # type: ignore
        tools = client.get_tools()

        # Use langgraph.prebuilt.create_react_agent
        agent_executor: AgentExecutor = create_react_agent(llm, tools) # type: ignore

        # Process instructions one by one
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for i, instruction in enumerate(instructions):
            print(f"\n--- Instruction {i+1}/{len(instructions)} ---")
            print(f"User: {instruction}")
            print("Agent:")

            messages.append(HumanMessage(content=instruction))

            current_step_output = ""
            try:
                # Stream the agent's response for the current instruction
                async for chunk in agent_executor.astream({"messages": messages}):
                    # Print intermediate steps/thoughts/tool calls/outputs
                    # Adjust printing based on the actual structure of the chunk
                    print(chunk, flush=True)
                    # Accumulate the final answer if needed (depends on ReAct structure)
                    if "output" in chunk:
                        current_step_output = chunk["output"]

                # Append the final AI message from this step to maintain context for the next step
                if current_step_output:
                    messages.append(AIMessage(content=current_step_output))
                else:
                     # If no explicit output, maybe add a placeholder or log a warning?
                    print("Warning: Agent did not produce a final 'output' for this step.")
                    # Decide if we need to add a placeholder AIMessage

            except Exception as e:
                print(f"\nError processing instruction {i+1}: {e}")
                print("Skipping to next instruction...")
                # Optionally add a system message about the error? Or just continue?
                # messages.append(SystemMessage(content=f"Error encountered: {e}"))
                continue # Move to the next instruction

        print("\n--- Batch processing finished ---")
        # wait_for_keypress call moved to cli.py

# Example usage (for testing, will be called from cli.py)
async def main():
    pass # Add pass back to keep the block valid
    # TODO: Get config dynamically from config.py
    # Need to handle API key loading securely
    # await run_agent_chat_session() # Remove placeholder call

if __name__ == "__main__":
    # Note: Running asyncio directly might require adjustments based on environment
    try:
        pass # Add pass back to keep the block valid
        # asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")