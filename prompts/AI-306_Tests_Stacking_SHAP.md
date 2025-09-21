# Prompt: [AI-306] Tests: Stacking/SHAP

Persona: Verifier (QA)

Objective
- Add tests for stacking OOF shapes/meta-learner training and SHAP sampling routine.

Context
- Depends on: [AI-302], [AI-304].

Deliverables
- `tests/test_stacking_shap.py`:
  - Verify OOF matrix dimensions; meta-learner trains and predicts.
  - Verify SHAP sampling selects the correct fraction and reproducibility via seed.

Constraints
- Keep tests light-weight; synthetic datasets only.

Steps
- Create toy data; run stacking and SHAP functions; assert expected properties.

Acceptance Criteria (DoD)
- Tests pass; failures indicate shape/seed issues clearly.

Verification Hints
- Compare sample size exact values for given fraction.

