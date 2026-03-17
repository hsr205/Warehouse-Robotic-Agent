# Warehouse Robotic Agent

<p align="center">
  <img src="https://img.shields.io/github/stars/hsr205/agentic_stock_predictor?style=for-the-badge" />
  <img src="https://img.shields.io/github/last-commit/hsr205/agentic_stock_predictor?style=for-the-badge" />
  <img src="https://img.shields.io/github/license/hsr205/agentic_stock_predictor?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Poetry-Dependency%20Management-blueviolet?style=for-the-badge" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Initialization](#initialization)
- [Environment Set-Up](#environment-set-up)
  - [Configuration Management](#configuration-management)
  - [Environment Variables](#environment-variables)
  - [Configuration Module Structure](#configuration-module-structure)
- [Application Execution](#application-execution)
- [Sources](#sources)
- [Conclusions](#conclusions)

---

## Overview

**ADD OVERVIEW SECTION MATERIAL**

---

## Initialization

This project uses **Poetry** for dependency management and virtual environment handling.

Poetry provides:
- Deterministic dependency resolution via `poetry.lock`
- Centralized configuration via `pyproject.toml`
- Automatic virtual environment management
- Separation of runtime and development dependencies

### 1. Installing Poetry (Mac)

Install Poetry using the official installer:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify installation:

```bash
poetry --version
```

---

### 2. Initializing Poetry in the Project

If this project has not yet been initialized with Poetry, run:

```bash
poetry init
```

This creates a `pyproject.toml` file containing project metadata and dependency definitions.

---

### 3. Installing Project Dependencies

After cloning the repository, install dependencies:

```bash
poetry install
```

This creates a virtual environment and installs all dependencies listed in `pyproject.toml` according to the locked versions in `poetry.lock`.

---

### 4. Adding Dependencies

Add a runtime dependency:

```bash
poetry add <library_name>
```

Example:

```bash
poetry add numpy
```

---

## Environment Set-Up

### Configuration Management

This project uses a centralized configuration module located at:

```
config/config.py
```

The configuration system is built using **Pydantic Settings** and is responsible for:

- Loading environment variables from a `.env` file
- Validating required configuration values
- Enforcing strong typing
- Failing fast if required variables are missing

---

### Environment Variables

A `.env` file must exist in the project root:

```
Final_Project_Materials/.env
```

Example:

```
APP_NAME=ROBOTIC_AGENT
```

The `Settings` class inside `config.py` defines all application configuration parameters.

The settings object is instantiated once:

```python
settings = Settings()
```

It can then be imported anywhere in the project:

```python
from config.config import settings
```

This ensures:

- Centralized configuration
- No duplicated environment reads
- Strong validation guarantees
- Clean separation of configuration from application logic

---

## Application Execution

The entry point of the application is:

```
main/main.py
```

To execute the application correctly, it must be run as a module from the project root.

---

### Required Project Structure

Ensure the following files exist:

```
main/__init__.py
config/__init__.py
```

These files mark the directories as Python packages and allow proper module imports.

---

### Run the Application (Recommended Method)

From the project root:

```bash
poetry run python -m main.main
```

This ensures:

- The correct Poetry-managed virtual environment is used
- Module imports resolve correctly
- The project root is added to the Python path

---

## Sources

**List all references used in the project.**

**Format:**

- Author. (Year). *Title*. Source. Link (if applicable)

---

## Conclusions

**Summarize findings and outcomes.**

Include:
- Key results
- Performance metrics
- Limitations
- Future improvements
- Lessons learned

Example:
- Model accuracy:
- Major insight:
- Bottlenecks:
- Next steps:

---
