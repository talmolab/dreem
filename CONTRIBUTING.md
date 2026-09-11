# Contributing to DREEM

Thanks for helping improve DREEM. This guide covers the mechanics: environment setup, the
checks CI runs, and what a reviewable pull request looks like.

## Development setup

DREEM targets **Python 3.12** (`requires-python = ">=3.12,<3.13"`) and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/talmolab/dreem
cd dreem

# dev dependencies + a CPU torch build (swap the extra for a GPU build if you have one)
uv sync --group dev --extra torch-cpu

# confirm the CLI is wired up
uv run dreem --help
```

Available torch extras are declared in `pyproject.toml`: `torch-cpu`, `torch-cuda118`,
`torch-cuda128`, `torch-cuda130`, plus `gpu` as an alias for the latest CUDA build. Pick the one
matching your machine — CI uses `torch-cpu`.

## Checks CI runs

Run these locally before pushing; they are exactly what the `CI` workflow enforces, so a green
local run means a green PR.

```bash
# 1. formatting — CI fails on any diff
uv run --frozen --extra torch-cpu ruff format --check dreem tests

# 2. tests with coverage
uv run --frozen --extra torch-cpu pytest --cov=dreem --cov-report=xml tests/

# 3. spelling (runs as its own workflow)
uvx codespell --skip="*.po,*.ts,./docs/_build,./docs/venv,./site,*.ipynb,.git,__pycache__,*.csv"
```

To apply formatting rather than just check it:

```bash
uv run ruff format dreem tests
```

### Lint rules

`ruff` is configured in `pyproject.toml` with `E`, `W`, `F`, `D` (google-style docstrings) and
`I` (import sorting) selected, `E501` ignored (the formatter owns line length), and a line
length of 88.

Note that `ruff check` is currently **commented out** in `.github/workflows/ci.yml` — only
`ruff format --check` is enforced. Running `uv run ruff check dreem` locally is still
worthwhile and will surface docstring and import issues before they accumulate.

## Tests

Tests live in `tests/` and run with pytest. Fixtures for small sample datasets are in
`tests/fixtures/`. When adding a feature:

- add a test that fails without your change,
- keep tests CPU-only and fast — CI has no GPU,
- avoid downloading data at test time; use or extend the existing fixtures.

## Documentation

Docs are built with mkdocs from `docs/`, and the navigation is declared in `mkdocs.yml`. To
preview locally:

```bash
uv run mkdocs serve
```

If you add a page under `docs/`, add it to the `nav:` block in `mkdocs.yml` — an unlisted page
will not appear on the site. Public API docstrings follow the google convention and are
rendered into the API reference automatically, so keep them accurate.

## Pull requests

- Branch off `main` and open the PR against `main`.
- Keep one logical change per PR; unrelated formatting churn makes review harder.
- Describe what changed and why. If behavior changes, say how a user would notice.
- Note any change to a config default or CLI flag explicitly — those silently alter other
  people's results, which matters more here than in most codebases since published numbers
  depend on them.
- Make sure the three checks above pass.

## Reporting issues

Open an issue at https://github.com/talmolab/dreem/issues. The most useful reports include:

- the exact command or script you ran,
- the full traceback,
- `dreem --version`, your OS, and your Python and torch versions,
- a description of your data: detection type (keypoints / boxes / masks), number of instances
  per frame, video format and resolution.

For tracking-quality problems rather than crashes, include the tracker settings you used
(`window_size`, `max_center_dist`, `overlap_thresh`) — they dominate results and are the first
thing anyone will ask about.
