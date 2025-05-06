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

    # 3. Build System Prompt
    # Note: Prompts.py was reverted by user, so mode/headless are not passed here currently.
    # If prompts.py is updated later, pass config.mode and config.headless here.
    system_prompt = build_system_prompt(
        mode=config.mode,
        headless=config.headless,
        record_screenshots=config.record
    )

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
    """Runs the interactive chat loop using the agent's stream_events method."""
    messages = []
    if initial_prompt:
        messages.append(SystemMessage(content=initial_prompt))
        print(f"\n🌐 System: {initial_prompt}")

    while True:
        user_input = input("\n💁 User: ")
        if user_input.lower() in ["exit", "bye"]:
            print("\n🌐 System: Goodbye! 👋")
            break

        messages.append(HumanMessage(content=user_input))

        # 4. Use agent's stream_events method
        print("\n🤖 Agent: ", end="", flush=True)
        current_ai_response = "" # Accumulate AI response here
        async for chunk in agent_executor.astream_events({"messages": messages}, version="v2"):
            # TODO: Implement proper handling of streaming output chunks
            # Chunks can contain agent actions, tool outputs, final answers etc.
            # Need to parse and display appropriately.
            # print(chunk, flush=True) # Basic printing for now - REMOVED

            # Enhanced printing based on chunk type (example structure)
            event = chunk.get("event")
            data = chunk.get("data", {})
            name = chunk.get("name") # Tool name or agent name

            if event == "on_chat_model_stream":
                # Check if it's AIMessage content and the chunk key exists
                chunk_content = data.get("chunk")
                if chunk_content and hasattr(chunk_content, 'content') and chunk_content.content:
                    print(chunk_content.content, end="", flush=True)
                    current_ai_response += chunk_content.content # Append to accumulator
            elif event == "on_tool_start" and name:
                print(f"Calling tool [{name}]...", end="", flush=True) # Removed leading \n
            elif event == "on_tool_end" and name:
                # Assuming success if event fires. Check data for actual result if available.
                # result_str = data.get("output", "result unknown") # Example if output is in data
                print(" success", flush=True) # Append success status
                # TODO: If tool failed, print " failed" and maybe the error


            # TODO: Update messages list with the AI's final response for history
            # For ReAct, the final response is usually in the 'output' key of the last chunk
            # if "output" in data: # Assuming final output is in data key for final event
            #      messages.append(AIMessage(content=data["output"])) # Incorrectly appending here
        print() # Add a newline after the full response

        # Append the full AI response to messages *after* the stream is finished
        if current_ai_response:
            messages.append(AIMessage(content=current_ai_response))

async def run_agent_batch_session(config: Settings, instructions: List[str]):
    """Initializes the agent and runs instructions from a list in batch mode."""
    llm, mcp_config, system_prompt = _setup_agent_resources(config)
    print(f"\n🌐 System: {system_prompt}")

    # Initialize PlaywrightMCPTool via MultiServerMCPClient
    async with MultiServerMCPClient(connections=mcp_config) as client: # type: ignore
        tools = client.get_tools()

        # Use langgraph.prebuilt.create_react_agent
        agent_executor: AgentExecutor = create_react_agent(llm, tools) # type: ignore

        # Process instructions one by one
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for i, instruction in enumerate(instructions):
            print(f"\n--- Instruction {i+1}/{len(instructions)} ---")
            print(f"\n💁 User: {instruction}")
            # print("Agent:") # Removed, adding before loop

            messages.append(HumanMessage(content=instruction))

            current_step_output = "" # Reset accumulator for each instruction
            try:
                # Stream the agent's response for the current instruction using astream_events
                print("\n🤖 Agent: ", end="", flush=True)
                async for chunk in agent_executor.astream_events({"messages": messages}, version="v2"):
                    # Print intermediate steps/thoughts/tool calls/outputs
                    # Adjust printing based on the actual structure of the chunk
                    # print(chunk, flush=True) # REMOVED

                    # Enhanced printing based on chunk type (example structure)
                    event = chunk.get("event")
                    data = chunk.get("data", {})
                    name = chunk.get("name") # Tool name or agent name

                    if event == "on_chat_model_stream":
                        # Check if it's AIMessage content and the chunk key exists
                        chunk_content = data.get("chunk")
                        if chunk_content and hasattr(chunk_content, 'content') and chunk_content.content:
                            print(chunk_content.content, end="", flush=True)
                            current_step_output += chunk_content.content # Append to accumulator
                    elif event == "on_tool_start" and name:
                        print(f"Calling tool [{name}]...", end="", flush=True) # Removed leading \n
                    elif event == "on_tool_end" and name:
                        # Assuming success if event fires. Check data for actual result if available.
                        # result_str = data.get("output", "result unknown") # Example if output is in data
                        print(" success", flush=True) # Append success status
                        # TODO: If tool failed, print " failed" and maybe the error


                    # Final answer accumulation is handled by the on_chat_model_stream block.
                    # The 'output' key in data often contains structured state (full messages, tool calls)
                    # not just the final response string suitable for AIMessage content.
                    # REMOVED: if "output" in data: ... block


                print() # Add a newline after the full response for the step

                # Append the final AI message from this step (accumulated from the stream)
                # to maintain context for the next step
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