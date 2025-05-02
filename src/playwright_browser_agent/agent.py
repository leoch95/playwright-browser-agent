"""Core agent logic using LangChain and Playwright-MCP."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import (MultiServerMCPClient, SSEConnection,
                                           StdioConnection,
                                           WebsocketConnection)
from langgraph.prebuilt import create_react_agent

# Placeholder for configuration - this should come from config.py
# Example config structure (adapt based on actual config.py implementation)
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MCP_CONFIG = {
    "playwright": {
        # Default config to run playwright-mcp via npx with stdio
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
        "transport": "stdio", # Or "sse" if using HTTP
    }
}

# Define the expected connection type for type hinting
MCPConnections = Dict[str, Union[StdioConnection, SSEConnection, WebsocketConnection]]

async def run_agent(
    model_name: str = DEFAULT_MODEL,
    mcp_config: Optional[MCPConnections] = None,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None, # Pass API key if needed by LiteLLM
):
    """Initializes and runs the agent components."""

    # Use default config if none provided
    if mcp_config is None:
        mcp_config = DEFAULT_MCP_CONFIG # type: ignore

    # 1. Initialize ChatLiteLLM
    llm = ChatLiteLLM(model=model_name, api_key=api_key, streaming=True)

    # 2. Initialize PlaywrightMCPTool via MultiServerMCPClient
    # The client manages the connection(s) to MCP server(s)
    async with MultiServerMCPClient(mcp_config) as client:
        # Fetch tools provided by the configured MCP server(s)
        tools = client.get_tools()

        # 3. Use langgraph.prebuilt.create_react_agent
        agent_executor = create_react_agent(llm, tools)

        # TODO: Implement run_chat_loop and send based on agent_executor

        print("Agent executor created. Need to implement chat loop and send.")
        # Placeholder for further implementation
        # await run_chat_loop(agent_executor, system_message)

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
    # TODO: Get config dynamically from config.py
    # Need to handle API key loading securely
    await run_agent()

if __name__ == "__main__":
    # Note: Running asyncio directly might require adjustments based on environment
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")