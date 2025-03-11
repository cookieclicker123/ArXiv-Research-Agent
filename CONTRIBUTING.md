# Contributing to Research Assistant

Thank you for considering contributing to the Research Assistant project! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ArXiv-Research-Agent.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: 
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
5. Install dependencies: `pip install -e ".[dev]"`
6. Copy environment variables: `cp .env.example .env` and edit as needed

## Development Workflow

1. Create a branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Format code: `black src tests`
5. Sort imports: `isort src tests`
6. Run linter: `ruff src tests`
7. Commit your changes: `git commit -m "Add your feature"`
8. Push to your fork: `git push origin feature/your-feature-name`
9. Create a pull request

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all functions, classes, and modules
- Keep functions small and focused on a single responsibility
- Use meaningful variable and function names

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage

## Documentation

- Update documentation for any changes to the API
- Add examples for new features
- Keep the README up to date

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if necessary
3. Link any related issues
4. Request review from a maintainer
5. Address any feedback from reviewers

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.