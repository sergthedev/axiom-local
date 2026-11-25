# Development log

Chronological, factual record. Failures and tradeoffs are included. Paths are sanitized.

## 2026-11-25 — Issue #1: bootstrap

### Starting state
- The remote repository had a single branch, `idk`, with one seed commit (`.gitignore`, `LICENSE`, `README.md`).
  There was no `main`. `main` was created from that seed commit and the foundation commit added on top
  (no force push; `idk` was left untouched).
- The seed `LICENSE` named a former GitHub username as copyright holder. The owner confirmed the current
  handle (`sergthedev`, same account) and asked for it to be updated; done in the foundation commit.

### Environment findings (failures/blockers encountered)
- The Mac initially had only system Python 3.9.6 (below the required 3.11/3.12), no `uv`, and no GitHub CLI.
  With the owner's explicit approval, `brew install python@3.12 uv gh` was run (Homebrew).
- Installed versions: Python 3.12.16, uv 0.13.10, GitHub CLI 2.105.0.

### Governance addendum and tradeoffs (approved 2026-11-25)
- Evidence-first files (REFERENCES, CLAIMS, ADRs, research register, experiment template) were created before any
  engine code. Tradeoff: more up-front documentation for a placeholder package; accepted to fix the habits early.
- Only two repository skills were added (`axiom-source`, `axiom-pr`); no third-party skills, agents or MCP servers.
  The skill format was checked against the official documentation (REF-001); part of that page was read through a
  summarized fetch rather than verbatim.
- Model/effort identifiers are not recorded in repository files by the implementing tool's policy; see
  `AI_CONTRIBUTIONS.md`.
- Runtime has zero third-party dependencies; dev tools pinned exactly in `pyproject.toml` and locked in `uv.lock`
  (tool and package versions chosen from PyPI metadata, REF-004).

### Local verification run (measured, single run)
- Hardware/OS: Apple Silicon (`arm64`), macOS 27.0.1.
- Python 3.12.16 (Homebrew), uv 0.13.10. Fresh virtual environment (previous `.venv` removed).

| Command | Exit status | Result |
|---------|-------------|--------|
| `uv sync --locked --python 3.12` | 0 | 7 packages installed (axiom-local, iniconfig 2.3.1, packaging 26.4, pluggy 1.7.0, pygments 2.21.0, pytest 9.2.0, ruff 0.17.6) |
| `uv run pytest` | 0 | 1 passed |
| `uv run ruff check .` | 0 | All checks passed |
| `uv run ruff format --check .` | 0 | Files already formatted |

Limitations: one machine, one run, not independently verified; Windows and CI results are reported on the issue
after the workflow actually runs, not asserted here. The single smoke test proves only that the package imports.
