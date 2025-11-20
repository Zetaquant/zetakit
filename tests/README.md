# Testing Guide for zetakit

This directory contains tests for the zetakit package.

## Running Tests

### Install test dependencies

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest
```

### Run tests with coverage

```bash
pytest --cov=zetakit --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

### Run tests in parallel

```bash
pytest -n auto
```

### Run specific test file

```bash
pytest tests/test_optimization.py
```

### Run specific test class or function

```bash
pytest tests/test_optimization.py::TestMeanVarianceOptimization
pytest tests/test_optimization.py::TestMeanVarianceOptimization::test_basic_optimization
```

### Run tests matching a pattern

```bash
pytest -k "test_basic"
```

### Run tests with verbose output

```bash
pytest -v
```

### Run tests and stop on first failure

```bash
pytest -x
```

### Run only fast tests (exclude slow markers)

```bash
pytest -m "not slow"
```

## Test Structure

- `conftest.py`: Shared fixtures and pytest configuration
- `test_*.py`: Test files matching the module structure
  - `test_optimization.py`: Tests for optimization module
  - `test_performance_metrics.py`: Tests for performance metrics module
  - (Add more as needed)

## Writing Tests

### Basic test function

```python
def test_function_name():
    """Test description."""
    result = some_function(input)
    assert result == expected_value
```

### Using fixtures

```python
def test_with_fixture(sample_returns):
    """Test using a fixture."""
    metrics = calculate_performance_metrics(sample_returns)
    assert metrics['total_return'] > 0
```

### Parametrized tests

```python
@pytest.mark.parametrize("risk_penalty", [0.5, 1.0, 2.0])
def test_different_risk_penalties(risk_penalty, sample_expected_returns, sample_covariance):
    weights = mean_variance_optimization(
        sample_expected_returns,
        sample_covariance,
        risk_penalty=risk_penalty
    )
    assert len(weights) == len(sample_expected_returns)
```

### Testing exceptions

```python
def test_raises_error():
    with pytest.raises(ValueError):
        function_that_raises_error()
```

## Best Practices

1. **Test names**: Use descriptive names starting with `test_`
2. **One assertion per test**: Keep tests focused and easy to debug
3. **Use fixtures**: Share common test data via fixtures in `conftest.py`
4. **Test edge cases**: Empty inputs, boundary values, error conditions
5. **Keep tests fast**: Mark slow tests with `@pytest.mark.slow`
6. **Test documentation**: Include docstrings explaining what each test verifies

