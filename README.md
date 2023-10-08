# Immunowave

Dynamical models for multi-scale immune responses

[![build and test](https://github.com/WSDeWitt/immunowave/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/WSDeWitt/immunowave/actions/workflows/build-and-test.yml)

## Installation

### Standard user

To install from the cloned repository:
``` bash
pip install .
```

### Developer

For an editable installation with additional tools for package development:
``` bash
pip install -e ".[dev]"
```

## Usage

Import in your Python code:
```python
import immunowave
```

## Development

### Formatting

We use [Black](https://github.com/psf/black) for code formatting. To format your code, run the following command:

```
black .
```

### Linting

We use [Flake8](https://flake8.pycqa.org/en/latest/) for linting. To lint your code, run the following command:
```bash
flake8 .
```

### Testing

We use [pytest](https://docs.pytest.org/en/latest/) for testing. To run the tests, run the following command:
```bash
pytest
```

## License

This project is licensed under the [MIT License](LICENSE).