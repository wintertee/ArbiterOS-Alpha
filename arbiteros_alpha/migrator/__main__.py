"""Entry point for running the migrator module as a script.

Usage:
    uv run -m arbiteros_alpha.migrator path/to/agent.py
"""

from .cli import main

if __name__ == "__main__":
    main()
