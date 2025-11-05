# ArbiterOS Alpha

**Policy-driven governance layer for LangGraph**

ArbiterOS-alpha is a lightweight governance framework that wraps LangGraph, enabling policy-based validation and dynamic routing without modifying the underlying graph structure.

## Quick Start

```bash
# Run example
uv run -m examples.main
```

## Features

- 🔒 **Policy-Driven Execution**: Validate execution constraints before and after instruction execution
- 🔀 **Dynamic Routing**: Route execution flow based on policy conditions
- 📊 **Execution History**: Track all instruction executions with timestamps and I/O
- 🎯 **LangGraph-Native**: Minimal migration cost from existing LangGraph code
- 🧩 **Decorator-Based**: Use `@instruction` decorator for lightweight governance
- 🔓 **Zero Lock-In**: Remove ArbiterOS by removing decorators and policies


## Installation

```bash
# Clone repository
git clone https://github.com/wintertee/ArbiterOS-alpha.git
cd ArbiterOS-alpha

# Install dependencies
uv sync
```

## Testing

```bash
# Run all tests
uv run pytest
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Setup pre-commit hooks
uv run pre-commit install
```

### Build Documentation

```bash
# Build documentation
uv run mkdocs build

# Serve documentation locally
uv run mkdocs serve
```

See [AGENTS.md](AGENTS.md) for AI development guidelines.

## License

See [LICENSE](LICENSE) file for details.
