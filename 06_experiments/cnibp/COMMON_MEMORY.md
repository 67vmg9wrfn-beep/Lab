# Common Engineering Memory

These are default execution rules for future projects.

## 1) Automation-first development and debugging
- Prefer automated pipelines over manual copy/paste debugging.
- Always write structured runtime artifacts to persistent storage:
  - `last_run.log`
  - `status.json`
  - `last_error.txt`
- Keep a stable output directory convention so failures can be diagnosed without UI access.
- Add clear stage markers in logs (data load, preprocess, train, eval, export).

## 2) Colab fault tolerance and resumability
- Assume Colab runtime can disconnect or be reclaimed.
- Design jobs to be restart-safe:
  - Reuse preprocessing outputs when inputs/config are unchanged.
  - Save fold/epoch checkpoints during training.
  - Skip completed folds on rerun.
  - Resume from latest checkpoint for interrupted folds.
- Keep all critical artifacts on Google Drive (not ephemeral `/content`).
- Validate required files/paths before long-running steps.

## 3) Practical default policy
- If a step can be reused, do not recompute it.
- If a step can fail due to environment instability, add fallback + retry + recover logic.
- If debugging requires UI-only logs, add file-based mirror logs automatically.
