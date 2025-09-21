# Prompt: [AI-504] Tests: Rule Flags/Playoffs

Persona: Verifier (QA)

Objective
- Add tests verifying rule flag activation by season/date and playoff flagging.

Context
- Depends on: [AI-501], [AI-502].

Deliverables
- `tests/test_rule_flags.py`:
  - Cases around cutover seasons/dates; playoff weeks detection.

Constraints
- No network; deterministic inputs.

Steps
- Write tests covering boundary weeks and typical weeks.

Acceptance Criteria (DoD)
- Tests pass; failures indicate incorrect boundary logic.

Verification Hints
- Include cases exactly at cutover boundaries to confirm inclusivity/exclusivity.

