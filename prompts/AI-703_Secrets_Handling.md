# Prompt: [AI-703] Secrets Handling

Persona: Builder (Implementation Engineer)

Objective
- Provide a simple secrets loader using environment variables or a local `.env` file; no secrets in code.

Context
- Supports any API keys if required (though NWS/Meteostat may not need keys).

Deliverables
- `src/nfl_pred/utils/secrets.py`:
  - Functions to read secrets by name from env or `.env`; clear errors if missing.

Constraints
- Do not hardcode secrets; document expected env names.

Steps
- Implement lookup with optional `python-dotenv` only if already in deps; otherwise manual `.env` parse.

Acceptance Criteria (DoD)
- Loader returns values when set; errors help users configure.

Verification Hints
- Set/unset env vars and verify behavior.

