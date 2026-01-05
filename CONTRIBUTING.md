# Contributing to FACT AI Group 23 Reproducibility Project

Thank you for your interest in contributing to our reproducibility project!

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your contribution
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/giacomograzia/fact-ai-group-23.git
cd fact-ai-group-23

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

We follow PEP 8 guidelines for Python code. Please ensure your code:

- Uses meaningful variable and function names
- Includes docstrings for all functions and classes
- Is properly formatted (use `black` for formatting)
- Passes linting checks (use `flake8`)

### Formatting

```bash
# Format code with black
black .

# Check linting
flake8 .

# Type checking
mypy src/
```

## Testing

Please ensure all tests pass before submitting a pull request:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## Documentation

- Update documentation for any new features
- Keep README.md up to date
- Document any deviations from the original paper
- Update REPRODUCTION.md if reproduction steps change

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the documentation
3. Ensure all tests pass
4. Request review from team members
5. Once approved, the PR will be merged

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what is best for the project
- Show empathy towards other contributors

### Scope

This code of conduct applies to all project spaces and when representing the project publicly.

## Questions?

If you have questions, please open an issue or contact the team members.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
