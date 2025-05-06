"""Manages system prompts for the Playwright Browser Agent."""

print("prompts.py loaded")

# Base system prompt template
BASE_SYSTEM_PROMPT_TEMPLATE = """
You are a web browsing agent that uses Playwright tools to interact with web pages based on user requests.
You operate in **{mode} Mode** {headless_status_detail}.

You have access to a Playwright Model Context Protocol (MCP) server providing the following tools:
{tools_description}

{interaction_instructions}

Think step-by-step about how to accomplish the user's goal using the available tools.
Use `browser_snapshot` frequently to understand the current page state before deciding on the next action.

Current Configuration:
- MCP Mode: {mode}
- Headless: {headless_status}
"""

# Tool descriptions based on mode
SNAPSHOT_TOOLS_DESC = """
- `browser_snapshot`: Capture an accessibility snapshot (semantic tree) of the current page.
- `browser_click`: Click on an element identified by its description and `ref` from the snapshot.
- `browser_type`: Type text into an element identified by its `ref`, optionally submitting.
- `browser_navigate`: Navigate to a specific URL.
- `browser_navigate_back`: Go back in history.
- `browser_navigate_forward`: Go forward in history.
- `browser_select_option`: Select option(s) in a dropdown identified by its `ref`.
- `browser_take_screenshot`: Take a screenshot (primarily for user reference/recording, not actions).
- (Potentially others like tab management, file handling, etc.)
"""

VISION_TOOLS_DESC = """
- `browser_snapshot`: Capture a screenshot (visual image) of the current page.
- `browser_screen_click`: Click on a location on the screen based on visual description or coordinates derived from the snapshot.
- `browser_screen_type`: Type text at a location on the screen based on visual description or coordinates, optionally submitting.
- `browser_navigate`: Navigate to a specific URL.
- `browser_navigate_back`: Go back in history.
- `browser_navigate_forward`: Go forward in history.
- `browser_select_option`: Select option(s) in a dropdown (may require careful visual targeting).
- `browser_take_screenshot`: Take a screenshot (used for both visual analysis by the agent and user reference/recording).
- (Potentially others like tab management, file handling, etc.)
"""

# Interaction instructions based on mode
SNAPSHOT_INTERACTION_INST = """
When interacting with elements (`browser_click`, `browser_type`, `browser_select_option`), ALWAYS use the `ref` provided in the accessibility snapshot (`browser_snapshot`) for the target element.
Describe the element clearly in the `element` parameter for user understanding and logging.
"""

VISION_INTERACTION_INST = """
When interacting with elements (`browser_screen_click`, `browser_screen_type`, `browser_select_option`), describe the target element visually or provide coordinates based on the screenshot (`browser_snapshot`).
The primary input is the visual representation of the page.
"""

# Additional instructions appended when screenshot recording is enabled
SCREENSHOT_INSTRUCTIONS = """
IMPORTANT: After each browser action (like navigate, click, type, select), you MUST call `browser_take_screenshot` to record the visual state of the page for the user.
"""

def build_system_prompt(mode: str = "snapshot", headless: bool = False, record_screenshots: bool = False) -> str:
    """Builds the final system prompt based on mode, headless status, and screenshot recording."""
    mode_capitalized = mode.capitalize()
    headless_status = "Enabled" if headless else "Disabled"
    headless_status_detail = f"(running {'headlessly' if headless else 'with a visible browser window'})"

    if mode == "vision":
        tools_description = VISION_TOOLS_DESC
        interaction_instructions = VISION_INTERACTION_INST
    else: # Default to snapshot mode
        mode_capitalized = "Snapshot" # Ensure consistent capitalization if input is unexpected
        tools_description = SNAPSHOT_TOOLS_DESC
        interaction_instructions = SNAPSHOT_INTERACTION_INST

    prompt = BASE_SYSTEM_PROMPT_TEMPLATE.format(
        mode=mode_capitalized,
        headless_status=headless_status,
        headless_status_detail=headless_status_detail,
        tools_description=tools_description.strip(),
        interaction_instructions=interaction_instructions.strip()
    )

    if record_screenshots:
        prompt += "\n\n" + SCREENSHOT_INSTRUCTIONS

    # TODO: Add prompt optimization techniques if needed
    # e.g., adding specific examples, refining instructions based on model behavior

    return prompt.strip()

# Example usage (for testing)
if __name__ == "__main__":
    print("--- Prompt (Snapshot, Headful) ---")
    print(build_system_prompt())
    print("\n" + "="*40 + "\n")
    print("--- Prompt (Snapshot, Headful, Recording) ---")
    print(build_system_prompt(record_screenshots=True))
    print("\n" + "="*40 + "\n")
    print("--- Prompt (Vision, Headless) ---")
    print(build_system_prompt(mode="vision", headless=True))
    print("\n" + "="*40 + "\n")
    print("--- Prompt (Vision, Headless, Recording) ---")
    print(build_system_prompt(mode="vision", headless=True, record_screenshots=True))