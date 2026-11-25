# Reproducibility

## Currently implemented

- Dependencies are locked in `uv.lock`; the runtime has no third-party dependencies.
- `uv sync --locked` fails if the lockfile and `pyproject.toml` disagree (REF-003).
- CI runs lint and tests on macOS and Windows with Python 3.12 (workflow state is reported on the issue, not assumed).

## Planned

- Seeds, input hashes and package versions recorded with every analysis result.
- Known-answer tests against independent references (see DR-001, DR-002).
- Experiment manifests per `docs/evidence/experiments/MANIFEST_TEMPLATE.md`.

## Caveats

- A lockfile pins versions; it does not prove numerical correctness.
- Bit-for-bit equality across platforms is **not** claimed; tolerances will be defined when the engine exists.
