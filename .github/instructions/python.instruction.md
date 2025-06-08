---
applyTo: '**/*.py'
---
# Project coding standards for Python

Apply the [general coding guidelines](./general.instruction.md) to all code.

## Python Guidelines
- Use Python 3.8+ for all new code
- Follow PEP 8 style guidelines for formatting and naming conventions
- Use type hints for function signatures and variable declarations
- Prefer list comprehensions and generator expressions for concise data processing
- Handle exceptions gracefully and avoid bare except clauses
- Write functions and classes with single, well-defined responsibilities
- Include docstrings for all public modules, functions, classes, and methods
- Use meaningful variable and function names
- Avoid using global variables unless absolutely necessary
- Prefer built-in modules and standard libraries when possible
- Optimize code for readability first, then performance if needed
- Comment complex or non-obvious code sections to explain intent
- Avoid hardcoding values; use constants or configuration files
- Use virtual environments for dependency management
- Keep third-party dependencies up to date and minimal
- Document any external dependencies or setup steps in README files

## Testing Guidelines
- Write unit tests for all critical logic and edge cases
- Use pytest or unittest as the testing framework
- Mock external dependencies in tests
- Ensure tests are repeatable and independent

