# Contributing

## Setup

```bash
uv sync --all-extras          # runtime + ml/llm/serve extras + dev tools
```

CI installs without the `ml`/`llm` extras: tests use a fake embedder and a fake
LLM, so they never download models.

## Checks (same as CI)

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy                   # strict
uv run pytest
```

## Conventions

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
- Branches: `feature/<short-description>`, `fix/<short-description>`,
  `chore/<short-description>`.
- Public functions and classes have Google-style docstrings (enforced by ruff).
- `domain/` and `application/` never import from `infrastructure/`; see
  [`docs/architecture.md`](docs/architecture.md).
- No secrets or `.env` files in commits; configuration goes through `AIACT_*`
  environment variables (see [`.env.example`](.env.example)).
- Result numbers are never typed by hand: regenerate them with
  `ai-act-rag report`.
