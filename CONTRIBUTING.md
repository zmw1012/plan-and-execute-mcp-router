# Contributing

This repository is a three-person course project. Please follow the workflow below.

## Branching

1. Always create a **feature branch from `main`**.
2. **Do not develop directly on `main`**.
3. Name branches by task, e.g. `feature/market-server`, `fix/eval-csv`, `docs/readme-install`.

```bash
git checkout main
git pull
git checkout -b feature/short-task-name
```

## Pull Requests

1. Merge only through a **Pull Request**.
2. Before merge, at least **one other teammate** should do a light review.
3. Keep each PR focused on **one clear task**.
4. Do not push secrets, `.env` files, real API keys, or large binary datasets.

## Local checks before opening a PR

```bash
python3 -m pip install -e ".[dev]"
pytest
```

## Secrets

- Copy `.env.example` to `.env` locally if needed.
- Never commit `.env` or real credentials.
- Prefer mock data for development.

## Communication

Coordinate ownership of domains (financial / market / news / evaluation) in the team chat before large refactors.
