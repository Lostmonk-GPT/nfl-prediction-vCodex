# Prompt: [AI-601] Feature Spec Generator

Persona: Scribe (Docs & ADR)

Objective
- Generate a feature specification table documenting name, definition, source columns, window, snapshot timing, null policy, and rule notes.

Context
- PRD Section 14 Documentation Artifact.

Deliverables
- `src/nfl_pred/docs/feature_spec.py`:
  - Introspect feature builders or maintain a structured spec; output a Markdown/CSV table.

Constraints
- Keep generation deterministic and easy to update.

Steps
- Define spec structure; write exporter to `docs/feature_spec.md`.

Acceptance Criteria (DoD)
- A current feature spec is produced and stored; entries cover all active features.

Verification Hints
- Manually verify a few entries match implementation.

