# Security Policy

AXIOM is pre-release bootstrap software with no implemented analysis features. There are no supported releases yet.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Contact the repository owner through GitHub
(private security advisory if enabled on the repository, otherwise a direct message to the owner's GitHub handle).
No other contact channel is published yet.

## Design constraints

- The application runtime must not use telemetry, cloud inference or external APIs.
- Model-generated code must never be executed.
- No credentials, personal data or absolute local paths belong in the repository.
