"""Configuration loading and management."""

import os
import sys
from typing import Optional

from dotenv import load_dotenv
from litellm import exceptions as litellm_exceptions
from litellm import validate_environment
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file first
load_dotenv()


class Settings(BaseSettings):
    """Application configuration settings."""

    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM Provider and Model
    llm_provider: Optional[str] = None  # e.g., 'openai', 'anthropic', 'google'
    llm_model: Optional[str] = None  # e.g., 'gpt-4o', 'claude-3-opus-20240229'

    # API Keys are read directly from environment by LiteLLM/SDKs
    # openai_api_key: Optional[str] = None
    # anthropic_api_key: Optional[str] = None
    # google_api_key: Optional[str] = None

    # Browser settings
    headless: bool = False

    # Recording settings
    record: bool = False
    artifacts_dir: str = "artifacts"


def load_config(**cli_overrides) -> Settings:
    """Loads configuration from environment variables and merges CLI overrides."""
    settings = Settings()

    # Apply CLI overrides (placeholder for now)
    for key, value in cli_overrides.items():
        if value is not None:
             # TODO: Implement proper merging logic when cli.py is built
            setattr(settings, key, value)

    print(f"Loaded configuration (pre-validation): {settings.model_dump()}") # Basic logging

    # --- Validation --- #
    # Ensure provider and model are specified
    if not settings.llm_provider:
        print("Error: LLM provider must be specified via --llm-provider flag or LLM_PROVIDER env var.", file=sys.stderr)
        sys.exit(1)
    if not settings.llm_model:
        print("Error: LLM model must be specified via --llm-model flag or LLM_MODEL env var.", file=sys.stderr)
        sys.exit(1)

    # Use LiteLLM's validator
    model_identifier = f"{settings.llm_provider}/{settings.llm_model}"
    try:
        # This function prints errors and potentially raises exceptions if validation fails
        validate_environment(model_identifier)
        print(f"LiteLLM environment validation successful for model: {model_identifier}")
    except litellm_exceptions.MissingEnvironmentVariableError as e:
        # validate_environment might also print its own errors, but we catch to ensure exit
        print(f"Error: Missing environment variables for model '{model_identifier}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch other potential errors during validation
        print(f"Error during environment validation for model '{model_identifier}': {e}", file=sys.stderr)
        sys.exit(1)

    return settings


# Example usage (can be removed later)
if __name__ == "__main__":
    config = load_config()
    print("--- Configuration Loaded ---")
    print(f"LLM Provider: {config.llm_provider}")
    print(f"LLM Model: {config.llm_model}")
    print(f"Headless: {config.headless}")
    print(f"Record: {config.record}")
    print(f"Artifacts Dir: {config.artifacts_dir}")
    # API keys are read from env, not stored in config object
    # print(f"OpenAI Key Set: {bool(config.openai_api_key)}")
    # print(f"Anthropic Key Set: {bool(config.anthropic_api_key)}")
    # print(f"Google Key Set: {bool(config.google_api_key)}")

# Remove the old dotenv loading check
# if loaded:
#     print("Loaded environment variables from .env file.")
# else:
#     print(".env file not found or empty, skipping dotenv loading.")
# print("config.py execution finished (dotenv part)")