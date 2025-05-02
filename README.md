# Playwright Browser Agent

A browser automation agent powered by Playwright and driven by Large Language Models (LLMs) via LangGraph and LiteLLM.

## Features (Planned)

* Interactive chat mode for step-by-step browser control.
* Batch mode for executing predefined instruction sequences.
* Support for multiple LLM providers (OpenAI, Anthropic, Google, etc.) via LiteLLM.
* Uses Playwright's accessibility features via MCP for robust interaction (avoids brittle selectors).
* Optional screenshot recording for visual traceability.
* Headless browser support.

## Setup

1. **Prerequisites:**
    * Python 3.13+
    * Node.js (for `npx` to run the Playwright MCP server)
    * `uv` Python package manager (`pip install uv`)

2. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd playwright-browser-agent
    ```

3. **Create virtual environment and install dependencies:**

    ```bash
    uv venv
    uv sync
    ```

4. **Configure Environment Variables:**
    Create a `.env` file in the project root and add the required API keys for your chosen LLM provider(s). For example:

    ```dotenv
    # .env
    LLM_PROVIDER=openai
    LLM_MODEL=gpt-4o
    OPENAI_API_KEY=sk-...

    # Or for Anthropic:
    # LLM_PROVIDER=anthropic
    # LLM_MODEL=claude-3-sonnet-20240229
    # ANTHROPIC_API_KEY=sk-ant-...
    ```

    The agent requires `LLM_PROVIDER` and `LLM_MODEL` to be set either in `.env` or via CLI flags. API keys corresponding to the chosen provider must also be present as environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Usage

The agent provides two main commands: `chat` and `batch`.

**Important:** The Playwright MCP server (`@playwright/mcp`) needs to be running for the agent to function. The agent currently expects it to be started via `stdio` transport using `npx @playwright/mcp@latest` (this might be automated or configurable in the future).

### Interactive Chat Mode (`chat`)

Start an interactive session where you can give instructions to the browser agent.

```bash
# Basic usage (uses LLM_PROVIDER and LLM_MODEL from .env)
uv run pb-agent chat

# Override LLM provider and model
uv run pb-agent chat --llm-provider anthropic --llm-model claude-3-opus-20240229

# Run in headless mode
uv run pb-agent chat --headless

# Record screenshots after each action
uv run pb-agent chat --record
```

Inside the chat, type your instructions (e.g., `navigate to google.com`, `search for playwright documentation`, `click the first link`). Type `exit` or `bye` to end the session.

### Batch Mode (`batch`)

Execute a sequence of instructions from a file.

1. Create an instruction file (e.g., `tasks.txt`):

    ```
    navigate to https://google.com
    type "Playwright Python" into the search bar and submit
    click the link containing "Playwright for Python | Playwright Python"
    take a screenshot
    ```

2. Run the batch command:

    ```bash
    # Basic usage

uv run pb-agent batch tasks.txt

    # With headless and recording
uv run pb-agent batch tasks.txt --headless --record

    # Override LLM for the batch run
uv run pb-agent batch tasks.txt --llm-provider openai --llm-model gpt-3.5-turbo
    ```

### Global Options

* `--llm-provider`/`-p`: Specify the LLM provider (overrides `.env`).
* `--llm-model`/`-m`: Specify the LLM model (overrides `.env`).
* `--headless`: Run the browser without a GUI.
* `--record`: Save screenshots to `artifacts/<timestamp>/` after each browser action.

## Development

* **Testing:** Install dev dependencies (`uv sync --all-extras`) and run tests with `uv run pytest`.
* **Dependencies:** Managed using `uv`. Use `uv add <package>` or `uv remove <package>`.
