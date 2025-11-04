# Installation

## Requirements

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/wintertee/ArbiterOS-alpha.git
cd ArbiterOS-alpha

# Install dependencies
uv sync
```

## Using pip

```bash
# Clone the repository
git clone https://github.com/wintertee/ArbiterOS-alpha.git
cd ArbiterOS-alpha

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Verify Installation

```bash
# Run the example
uv run -m examples.main
```

## Development Setup

For development, install additional dependencies:

```bash
# Install with all development dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

## Next Steps

- Continue to [Quick Start](quickstart.md)
- Check the [API Reference](../api/core.md)
