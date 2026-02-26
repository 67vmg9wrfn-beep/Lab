---
name: paper-to-ppt
description: Generate one presentation package per paper from paper PDFs or structured review files. Use when the user asks for paper presentation decks, slide outlines, speaker notes, or .pptx outputs for one or many papers.
---

# Paper To PPT

Create concise, evidence-grounded PPT materials for each paper.

## Input Contract

Accept as source:

- `/Users/codex/Lab/03_summaries/paper_reviews/<paper_id>/review.md` (preferred)
- Paper PDF
- Paper URL or extracted full text

## Output Contract

For each paper, save to:

- `/Users/codex/Lab/03_summaries/paper_reviews/<paper_id>/ppt_outline.md`
- `/Users/codex/Lab/07_reports/ppt/<paper_id>.slides.md`
- `/Users/codex/Lab/07_reports/ppt/<paper_id>.notes.md`

If a PPT toolchain is available, additionally output:

- `/Users/codex/Lab/07_reports/ppt/<paper_id>.pptx`

## Required Structure

- Must follow `references/PPT_TEMPLATE.md` for outline order.
- Keep narrative: problem -> method -> evidence -> risk -> decision.
- Every key claim must trace back to paper evidence.

## Slide Writing Rules

- One message per slide.
- Prefer quantitative bullets.
- Keep bullets short and scannable.
- Mark missing information as `论文未说明`.

## Workflow

1. Read `review.md` first when available.
2. Build `ppt_outline.md` using template.
3. Expand to slide script and notes.
4. Verify consistency with review scores and conclusions.
5. Generate `.pptx` only when toolchain exists.

## Batch Defaults

- Iterate all review folders under `/Users/codex/Lab/03_summaries/paper_reviews`.
- Produce one full presentation package per paper id.
- Preserve identical section order across all papers.

## References

- `references/PPT_TEMPLATE.md`
- `references/slide-template.md`
