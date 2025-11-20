# Zeta's Quant Tool Kit

A Python toolkit for quantitative finance and portfolio optimization.

## Installation

```bash
pip install -e .
```

For development with testing dependencies:

```bash
pip install -e ".[dev]"
```

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=zetakit --cov-report=html

# Run specific test file
pytest tests/test_optimization.py

# Run in parallel (faster)
pytest -n auto
```

See `tests/README.md` for detailed testing documentation.