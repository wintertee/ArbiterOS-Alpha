"""Entry point for running the transform module as a script.

Usage:
    uv run -m arbiteros_alpha.transform path/to/agent.py
"""

from .cli import main

if __name__ == "__main__":
    main()

