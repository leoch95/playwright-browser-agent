"""Utility functions for the Playwright Browser Agent."""

import datetime
import os
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Flag to indicate if shutdown is in progress
_shutdown_initiated = False
_original_sigint_handler = signal.getsignal(signal.SIGINT)
_original_sigterm_handler = signal.getsignal(signal.SIGTERM)

def get_timestamp() -> str:
    """Generates a timestamp string suitable for directory/file naming."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def wait_for_keypress(prompt: str = "Press Enter to continue..."):
    """Waits for the user to press Enter."""
    # Simple implementation using input().
    # More complex cross-platform single keypress detection could be added if needed.
    input(prompt)

def _handle_signal(signum, frame, cleanup_func: Optional[Callable[[], None]] = None):
    """Internal signal handler."""
    global _shutdown_initiated
    if _shutdown_initiated:
        print("Shutdown already in progress, force quitting...")
        sys.exit(1) # Force exit if called again

    _shutdown_initiated = True
    signal_name = signal.Signals(signum).name
    print(f"\nReceived {signal_name}. Initiating graceful shutdown...")

    if cleanup_func:
        try:
            print("Running cleanup tasks...")
            cleanup_func()
            print("Cleanup finished.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    print("Exiting.")

    # Restore original handler and exit
    if signum == signal.SIGINT:
        signal.signal(signal.SIGINT, _original_sigint_handler)
    elif signum == signal.SIGTERM:
        signal.signal(signal.SIGTERM, _original_sigterm_handler)

    sys.exit(0)

def register_signal_handlers(cleanup_func: Optional[Callable[[], None]] = None):
    """Registers signal handlers for SIGINT and SIGTERM."""
    signal.signal(signal.SIGINT, lambda s, f: _handle_signal(s, f, cleanup_func))
    signal.signal(signal.SIGTERM, lambda s, f: _handle_signal(s, f, cleanup_func))
    print("Signal handlers registered for graceful shutdown (SIGINT, SIGTERM).")

def manage_screenshot_dir(base_artifact_path: str = "artifacts") -> Path:
    """Creates and returns the path for the current session's screenshots."""
    session_timestamp = get_timestamp()
    screenshot_dir = Path(base_artifact_path) / session_timestamp
    try:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Screenshots will be saved to: {screenshot_dir.resolve()}")
        return screenshot_dir
    except OSError as e:
        print(f"Error creating screenshot directory '{screenshot_dir}': {e}", file=sys.stderr)
        # Fallback or re-raise depending on desired behavior
        raise # Re-raise for now

# Example usage (for testing)
if __name__ == "__main__":
    print(f"Generated timestamp: {get_timestamp()}")

    screenshot_path = manage_screenshot_dir()
    print(f"Created screenshot dir: {screenshot_path}")

    def sample_cleanup():
        print("Simulating cleanup... closing resources...")
        time.sleep(1) # Simulate work
        print("Cleanup simulation complete.")

    register_signal_handlers(sample_cleanup)
    print("Registered signal handlers. Press Ctrl+C to test graceful shutdown.")

    # Keep running to allow signal testing
    try:
        while not _shutdown_initiated:
            time.sleep(0.5)
    except KeyboardInterrupt:
        # This part might not be reached if SIGINT is handled gracefully
        pass

    # Test keypress wait
    # wait_for_keypress()