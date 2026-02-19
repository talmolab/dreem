# Documentation Rules

## Documentation Stack

This project uses **MkDocs** with the Material theme for documentation, and **mike** for versioned deployments.

- Serve docs locally: `uv run mkdocs serve`
- Build docs: `uv run mkdocs build`
- Deploy a version: `uv run mike deploy <version> latest --update-aliases --push`
- Set default version: `uv run mike set-default latest --push`

Configuration is in `mkdocs.yml`. The site is hosted at https://dreem.sleap.ai/.

## Notebook-Markdown Synchronization

When any Jupyter notebook (`.ipynb`) in `docs/Examples/` is modified, the corresponding Markdown file(s) must also be updated:

1. Update the `.md` file in `docs/Examples/` that references the notebook
2. Update any related `.md` file in `docs/` (e.g., `quickstart.md`, `usage.md`)

Code snippets in Markdown files must use fenced code blocks with language identifiers:

```python
# Python code
import dreem
```

```bash
# Shell commands
dreem track ./data --checkpoint model.ckpt
```

This ensures documentation stays consistent across formats. This is particularly important if changes are being committed.

## CLI Documentation Sync

When CLI commands or flags change in `dreem/cli.py`:

- Update `docs/cli.md` with new/changed commands
- Update termynal demos in `docs/index.md` if the example commands change
- Update `docs/usage.md` with any workflow changes

## Navigation Updates

When adding new documentation pages:

- Add the page to the `nav:` section in `mkdocs.yml`
- Ensure proper hierarchy under the correct section

## Code Examples

All code examples in documentation must be:

- Tested and working with the current version
- Using the `dreem-track` package name (not `dreem-tracker` or other variants)
- Consistent with the CLI syntax in `dreem --help`

## API Reference

When public APIs change in `dreem/`:

- The mkdocstrings plugin auto-generates API docs, but docstrings must be updated in the source code
- Update any manual examples in `docs/` that reference changed APIs

## Assets

Store images and static files in `docs/assets/`. Reference them with relative paths.
