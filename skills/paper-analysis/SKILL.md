---
name: paper-analysis
description: Systematic and structured analysis of research papers from PDF, URL, or full text input. Use when the user asks for paper review, screening/scoring, method dissection, reproducibility assessment, or standardized review outputs for one or many papers.
---

# Paper Analysis

Perform reproducible paper analysis with strict template compliance.

## Goal

For each paper, generate:

1. Structured review report (strictly follow `references/REPORT_TEMPLATE.md`)
2. Screening scores (1-5 with reasons)
3. PPT outline handoff file for slide generation
4. Action plan at 30 minutes / 1 day / 1 week

## Input Contract

Accept one of the following:

- Local PDF path (preferred)
- Paper URL
- Full text content

Prefer files under `/Users/codex/Lab/02_papers` for batch runs.

## Output Contract

For each paper, save to:

- `/Users/codex/Lab/03_summaries/paper_reviews/<paper_id>/review.md`
- `/Users/codex/Lab/03_summaries/paper_reviews/<paper_id>/ppt_outline.md`

Use paper id format:

- `YYYY-MM-DD_第一作者_简化标题`

For cross-paper work, additionally save:

- `/Users/codex/Lab/07_reports/paper-analysis-matrix.md`

## Mandatory Rules

- Follow section order in `references/REPORT_TEMPLATE.md` exactly.
- Do not fabricate data, metrics, or experimental outcomes.
- For missing details, write: `论文未说明`.
- Use concise bullet style; avoid filler text.
- Explicitly identify:
  - actual problem solved
  - technical core
  - key assumptions
  - weak points / failure modes

## Scoring Dimensions (1-5)

Score each paper and explain why:

- 相关性
- 创新性
- 技术严谨性
- 实验充分性
- 可复现性
- 实际应用价值

## Workflow

1. Build inventory (paper id, title, venue, year, source path/link).
2. Extract facts from source text only.
3. Fill `review.md` using `REPORT_TEMPLATE.md`.
4. Generate `ppt_outline.md` using `PPT_TEMPLATE.md`.
5. Produce matrix when processing multiple papers.
6. Run checklist before final output.

## Final Checklist

Before finishing, verify:

- problem definition is clear
- method core is explicit
- key assumptions are listed
- potential failure scenarios are identified
- reproducibility suggestions are actionable

## References

- `references/REPORT_TEMPLATE.md`
- `references/PPT_TEMPLATE.md`
- `references/analysis-framework.md`
