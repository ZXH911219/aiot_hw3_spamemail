# Project Context

## Purpose
This repository (HW3) is a small project workspace for building and experimenting with a spam classification pipeline and for practicing OpenSpec-driven design proposals.

## Tech Stack
- Python 3.8+ (recommended) with pandas and scikit-learn for ML work
- Node.js + npm (for optional tooling, proposal validators)
- OpenSpec CLI (`@fission-ai/openspec`) for proposal management
- Test runner: pytest (for Python) or Jest (for Node scripts)

## Project Conventions

### Code Style
- Python: follow PEP8; use black for formatting and flake8 for linting.
- JavaScript: use Prettier and ESLint if present.
- Commits: follow Conventional Commits (type(scope): summary).

### Architecture Patterns
- Keep the ML pipeline modular:
	- `data/` for raw and processed datasets
	- `src/` for reusable scripts and modules
	- `notebooks/` for exploratory work
	- `models/` for serialized trained models

### Testing Strategy
- Unit tests for data loaders and preprocessing functions.
- Integration test for the full training pipeline (smoke test).

### Git Workflow
- `main` branch holds the canonical code. Use feature branches `feat/...` and PRs for changes.

## Domain Context
- This project focuses on spam detection for SMS/email; performance metrics and fairness considerations are relevant.

## Important Constraints
- Development environment: Windows / PowerShell (examples will include PowerShell commands).

## External Dependencies
- Public dataset: Packt sample CSV (see proposal `openspec/proposals/0002-spam-classification.md`).
- Optional CI (GitHub Actions) for proposal validation and changelog generation.

## Useful Commands
- Install Python deps: `pip install -r requirements.txt` (when requirements are added)
- Run tests: `pytest`
- Initialize OpenSpec: `openspec init`

If this stack doesn't match your preferences (e.g., you prefer Node-only), tell me and I'll adapt these conventions.
