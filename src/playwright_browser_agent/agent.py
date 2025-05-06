"""Core agent logic using LangChain and Playwright-MCP."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
# Define connection types for casting
from langchain_mcp_adapters.client import (MultiServerMCPClient, SSEConnection,
                                           StdioConnection,
                                           WebsocketConnection)
from langgraph.prebuilt import create_react_agent

# Import config type and prompt builder
from .config import Settings
from .prompts import build_system_prompt  # Uncomment import

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


async def run_agent_chat_session(config: Settings):
    """Initializes the agent and runs the interactive chat loop."""

    # 1. Build MCP Configuration
    mcp_args = ["@playwright/mcp@latest"]
    if config.headless:
        mcp_args.append("--headless")
        print("Configuring Playwright MCP for Headless mode.") # Added logging
    else:
        mcp_args.append("--vision") # Default to vision mode if not headless
        print("Configuring Playwright MCP for Vision mode (non-headless).") # Added logging

    mcp_config = {
        "playwright": {
            "command": "npx",
            "args": mcp_args,
            "transport": "stdio",
        }
    }

    # 2. Initialize ChatLiteLLM using loaded config
    # LiteLLM reads API keys directly from environment variables based on the provider/model string
    llm = ChatLiteLLM(
        model_name=f"{config.llm_provider}/{config.llm_model}",
        streaming=True,
        # Temperature, max_tokens etc. can be added if needed
    )

    # 3. Initialize PlaywrightMCPTool via MultiServerMCPClient
    # The client manages the connection(s) to MCP server(s)
    # Cast the config to satisfy the linter, assuming internal handling is correct
    async with MultiServerMCPClient(connections=mcp_config) as client: # type: ignore
        tools = client.get_tools()

        # 4. Build System Prompt (conditionally add recording instructions)
        system_prompt = build_system_prompt(record_screenshots=config.record) # Use the function
        # system_prompt = "You are a helpful assistant controlling a web browser. Describe your plan then execute actions using the available tools." # Remove fallback
        # Optionally print the prompt for debugging
        # print(f"--- System Prompt ---\\n{system_prompt}\\n--------------------")

        # 5. Use langgraph.prebuilt.create_react_agent
        agent_executor = create_react_agent(llm, tools)

        # 6. Start the interactive chat loop
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

async def send(agent_executor: Any, instruction: str, initial_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Sends a single instruction using the agent's invoke method."""
    messages = []
    if initial_prompt:
        messages.append(SystemMessage(content=initial_prompt))
    messages.append(HumanMessage(content=instruction))

    # 5. Use agent's invoke method
    result = await agent_executor.ainvoke({"messages": messages})

    # TODO: Process and return the result appropriately
    print(f"Agent Result: {result}")
    return result

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