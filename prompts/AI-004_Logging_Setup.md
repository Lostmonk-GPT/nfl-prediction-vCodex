# Prompt: [AI-004] Logging Setup

Persona: Builder (Implementation Engineer)

Objective
- Add a simple, consistent console logging setup used by CLI and pipelines.

Context
- Depends on: [AI-001] scaffold.
- Logging should be adjustable via config/env.

Deliverables
- `src/nfl_pred/logging_setup.py` with `setup_logging(level: str|int = "INFO")`.
  - Formats with timestamp, level, module, message.
  - Respects env var `NFLPRED_LOG_LEVEL` if present.

Constraints
- Use stdlib `logging`; no external logging libs.
- Avoid double handlers on repeated setup.

Steps
- Implement setup function and default level handling.
- Example usage docstring.

Acceptance Criteria (DoD)
- Calling `setup_logging("DEBUG")` configures root logger and module loggers.
- No duplicate log lines on multiple calls.

Verification Hints
- Run a quick script importing the function and emitting logs; verify format/level.

