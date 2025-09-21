# Prompt: [AI-304] SHAP Explainability

Persona: Forecaster (Modeling)

Objective
- Generate SHAP explanations for tree models on a sampled subset (10–20%) and produce weekly summary plots.

Context
- PRD Section 8 Explainability; performance-sensitive.

Deliverables
- `src/nfl_pred/explain/shap_utils.py`:
  - Functions to compute SHAP values with TreeExplainer; sample inputs; approximate/GPU fallback.
  - Save summary plots and link artifacts to MLflow runs.

Constraints
- Keep runtime manageable via sampling; document sampling seed.

Steps
- Implement computation and plotting utilities with configuration for sample fraction.

Acceptance Criteria (DoD)
- SHAP values computed on a sample; plots saved and registered as artifacts.

Verification Hints
- Dry-run on a tiny model and confirm non-zero importances for known strong features.

