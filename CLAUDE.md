# Interrupt Workshop Codebase Overview

This repository contains materials for building and understanding intelligent agents using LangGraph. The codebase focuses on an email assistant that can triage and respond to emails, with extensions for human-in-the-loop feedback and memory.

## Overview

The codebase is organized into four main sections:

1. **Basic Agent** - Email assistant that can triage and respond to emails
2. **Evaluation** - Tools for evaluating agent performance
3. **Human-in-the-Loop** - Adding human feedback to the agent
4. **Memory** - Enabling the agent to learn from past interactions

## Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Ensure you have a recent version of pip (required for editable installs with pyproject.toml)
python3 -m pip install --upgrade pip

# Install the package in editable mode
pip install -e .
```

## Running Tests

Execute the test suite with:

```bash
# Run tests for the default implementation
python tests/run_all_tests.py

# Run tests for a specific implementation
python tests/run_all_tests.py --implementation email_assistant_hitl

# Run tests for all available implementations
python tests/run_all_tests.py --all

# Add a specific experiment name for LangSmith tracking
python tests/run_all_tests.py --experiment-name "Custom Test Run"
```

### Testing Notebooks

You can also run tests to verify all notebooks execute without errors:

```bash
# Run all notebook tests directly
python tests/test_notebooks.py

# Or run via pytest
pytest tests/test_notebooks.py -v
```

Tests require:
- OpenAI API key
- LangSmith API key (for evaluation tracking)

Set these in a `.env` file or as environment variables.