"""Configuration loading and management."""

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
loaded = load_dotenv()

if loaded:
    print("Loaded environment variables from .env file.")
else:
    print(".env file not found or empty, skipping dotenv loading.")

# Rest of the config logic will go here
print("config.py execution finished (dotenv part)")