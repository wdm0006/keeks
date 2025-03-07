# Keeks Documentation

This directory contains the Sphinx documentation for the Keeks package.

## Building the Documentation

To build the documentation, follow these steps:

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Build the HTML documentation:

```bash
make html
```

3. The built documentation will be available in the `build/html` directory.

## Documentation Structure

- `source/`: Contains the source files for the documentation
  - `conf.py`: Sphinx configuration file
  - `index.rst`: Main entry point for the documentation
  - `*.rst`: Documentation for different parts of the package
- `build/`: Contains the built documentation (generated)
- `Makefile` and `make.bat`: Scripts for building the documentation

## Adding New Documentation

To add new documentation:

1. Create a new `.rst` file in the `source/` directory
2. Add the file to the table of contents in `index.rst`
3. Build the documentation to see the changes

## Updating Existing Documentation

Most of the API documentation is automatically generated from docstrings in the code.
To update the API documentation, update the docstrings in the code and rebuild the documentation.

## Documentation Standards

- Use proper heading hierarchy (=, -, ~, ^, ")
- Include examples where appropriate
- Document all public modules, classes, methods, and functions
- Follow NumPy docstring style consistently

## Online Documentation

The documentation is hosted online at [https://keeks.readthedocs.io/](https://keeks.readthedocs.io/).

## Repository

The source code for Keeks is available on GitHub at [https://github.com/wdm0006/keeks](https://github.com/wdm0006/keeks). 