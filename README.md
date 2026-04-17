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
- [Application Execution](#application-execution)
- [Sources](#sources)
- [Conclusions](#conclusions)

---

## Overview

In the modern day of Reinforcement Learning Agent adoption, there have emerged a wide variety of applications that agents could be used in the physical world to automate tasks previously completed by humans. A classical task of full warehouse automation using RL agents in a warehouse space, has displayed steady progress throughout the past few decades. Seeing this revelation unfolding, we as a group decided to train an RL agent in multiple 2-dimensional grid environments to first pick up and then deliver a package in ever increasingly difficult state space to navigate. Our project leverages two RL algorithms to highlight not only that this task is achievable but can be optimized through the Proximal Policy Optimization (PPO) and Asynchronous Advantage Actor-Critic (A3C) algorithms. It is through these two RL algorithms that we built the foundation of our project on. The ultimate goal being that our algorithms would help to optimize the actions of our warehouse package delivery agent.

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

- Dresser, S. (2025, July 1). Amazon launches a new AI Foundation model to power its robotic fleet and  deploys its 1 millionth robot. Amazon News. https://www.aboutamazon.com/news/operations/amazon-million-robots-ai-foundation-model 
- Gallagher, Ethan & Godwin Mahlangu, Israel. (2025). Reinforcement Learning Approaches to Dynamic Warehouse and Inventory Optimization. International Journal of Artificial Intelligence Tools. 13. 
- Walmart opens first of four next generation fulfillment centers in Joliet, IL. Walmart Corporate News and  Information. (2022, September 28). https://corporate.walmart.com/news/2022/09/28/walmart-opens-first-of-four-next-generation-fulfllment-centers-in-joliet-il
- Ziyan Wu, Wenhao Zhang, Rui Tang, Huilong Wang, Ivan Korolija, Reinforcement learning in building controls: A comparative study of algorithms considering model availability and policy representation, Journal of Building Engineering, Volume 90, 2024, 109497

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
