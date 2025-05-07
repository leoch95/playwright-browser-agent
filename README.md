# Playwright Browser Agent

A browser automation agent powered by Playwright and driven by Large Language Models (LLMs) via LangGraph and LiteLLM.

## Features

* Interactive chat mode for step-by-step browser control.
* Batch mode for executing predefined instruction sequences.
* Support for multiple LLM providers (OpenAI, Anthropic, Google, etc.) via LiteLLM.
* Uses Playwright's accessibility features via MCP for robust interaction.
* Optional screenshot recording to `artifacts/<timestamp>/step_<N>.png` for visual traceability.
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
    The agent relies on environment variables for configuration, especially for selecting the LLM and providing API keys. These can be set directly in your shell or, more conveniently, placed in a `.env` file in the project root.

    **Required Variables:**
    * `LLM_PROVIDER`: Specifies the LLM provider to use (e.g., `openai`, `anthropic`, `google`, `ollama`, etc.). Must match a provider supported by LiteLLM.
    * `LLM_MODEL`: Specifies the exact model name for the chosen provider (e.g., `gpt-4o`, `claude-3-opus-20240229`, `gemini/gemini-1.5-pro-latest`, `llama3`).
    * **Provider API Key:** You must set the corresponding API key environment variable for your chosen `LLM_PROVIDER`. Common examples include:
        * `OPENAI_API_KEY` (for `openai` provider)
        * `ANTHROPIC_API_KEY` (for `anthropic` provider)
        * `GOOGLE_API_KEY` (for `google` provider)
        * Refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for the specific key names required by other providers.

    **Optional Variables:**
    * Variables corresponding to CLI flags (like `HEADLESS`, `RECORD`) can also be set in `.env`, but CLI flags will always take precedence.

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

    The agent will validate these environment variables at startup. If `LLM_PROVIDER` or `LLM_MODEL` are missing, or if the required API key for the selected provider is not found in the environment, the application will exit with an error message.

## Usage

The agent provides two main commands: `chat` and `batch`.

**Important:** The Playwright MCP server (`@playwright/mcp`) is managed automatically by the agent.

### Interactive Chat Mode (`chat`)

Start an interactive session where you can give instructions to the browser agent.

```bash
# Basic usage (uses LLM_PROVIDER and LLM_MODEL from .env)
uv run pb-agent chat

# Override LLM provider and model
uv run pb-agent chat --llm-provider anthropic --llm-model claude-3-opus-20240229

# Record screenshots after each action
uv run pb-agent chat --record --screenshot-dir my_chat_screenshots
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

    # With recording and custom screenshot directory
    uv run pb-agent batch tasks.txt --record --screenshot-dir my_batch_screenshots

    # Override LLM for the batch run
    uv run pb-agent batch tasks.txt --llm-provider openai --llm-model gpt-3.5-turbo
    ```

### Global Options

* `--llm-provider`/`-p`: Specify the LLM provider (overrides `.env`).
* `--llm-model`/`-m`: Specify the LLM model (overrides `.env`).
* `--record`: Enable saving screenshots to a timestamped subfolder within the directory specified by `--screenshot-dir`. The actual files will be named `step_<N>.png`.
* `--screenshot-dir`: Specify the base directory for saving screenshots (defaults to `artifacts`). Used in conjunction with `--record`.
* `--playwright-mcp-config-path`: Path to a custom Playwright MCP JSON configuration file.

## Development

* **Testing:** Install dev dependencies (`uv sync --all-extras`) and run tests with `uv run pytest`.
* **Dependencies:** Managed using `uv`. Use `uv add <package>` or `uv remove <package>`.
