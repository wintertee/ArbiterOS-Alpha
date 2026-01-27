"""Entry point for running the migrator module as a script.

Usage:
    # Single file migration (legacy)
    uv run -m arbiteros_alpha.migrator migrate path/to/agent.py

    # Repo-level transformation (new)
    uv run -m arbiteros_alpha.migrator transform /path/to/repo

    # Show help
    uv run -m arbiteros_alpha.migrator --help
"""

from .cli import main

if __name__ == "__main__":
    main()
