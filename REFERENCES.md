# References

Only sources that were actually consulted are listed. Planned reading belongs in `docs/research/README.md`.
Access date for all entries below: 2026-11-25.

| ID | Author / publisher | Title | URL | Version / revision | Supported claim | Related |
|----|--------------------|-------|-----|--------------------|-----------------|---------|
| REF-001 | Anthropic | Extend Claude with skills (Claude Code docs) | https://code.claude.com/docs/en/skills | No revision shown on page | Project skills live at `.claude/skills/<skill-name>/SKILL.md` with optional YAML frontmatter (`name`, `description`). Part of the page was read via an automated summary rather than verbatim text. | Issue #1, `.claude/skills/` |
| REF-002 | Citation File Format contributors | Citation File Format schema | https://github.com/citation-file-format/citation-file-format/blob/1.2.0/schema.json | tag 1.2.0 | CFF 1.2.0 requires `authors`, `cff-version`, `message`, `title`; a person entry may use `alias`. | Issue #1, `CITATION.cff` |
| REF-003 | Astral | uv documentation: locking and syncing | https://docs.astral.sh/uv/concepts/projects/sync/ | docs as served on access date | `uv run --locked` / lockfile checking errors if the lockfile is out of date instead of updating it. | Issue #1, ADR 0001 |
| REF-004 | Python Package Index | PyPI JSON API metadata for pytest, ruff, hatchling | https://pypi.org/pypi/pytest/json (also `ruff`, `hatchling`) | pytest 9.2.0, ruff 0.17.6, hatchling 1.33.0 were the latest versions on the access date | Latest released versions, licenses (MIT) and `requires_python` of the dev/build tools at the time of choosing. | Issue #1, `THIRD_PARTY_NOTICES.md` |
| REF-005 | GitHub, Inc. | actions/setup-python README | https://github.com/actions/setup-python | README on `main`; usage examples show `@v7` | Current major version tag shown in usage examples. | Issue #1, `.github/workflows/ci.yml` |
| REF-006 | GitHub, Inc. | actions/checkout README | https://github.com/actions/checkout | README on `main`; usage examples show `@v7` | Current major version tag shown in usage examples. | Issue #1, `.github/workflows/ci.yml` |
