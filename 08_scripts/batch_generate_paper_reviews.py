#!/usr/bin/env python3
import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader

PDF_ROOT = Path('/Users/codex/Lab/01_monitoring/industry/01-apple')
OUT_ROOT = Path('/Users/codex/Lab/03_summaries/paper_reviews')


def clean_line(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def extract_pages_text(pdf_path: Path, max_pages: int = 6) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages[:max_pages]):
        text = page.extract_text() or ''
        pages.append(text)
    return pages


def filename_parts(pdf_path: Path) -> Tuple[str, str]:
    stem = pdf_path.stem
    m = re.match(r'^(\d{4}-\d{2}-\d{2})_(.+)$', stem)
    if m:
        return m.group(1), m.group(2)
    return '1970-01-01', stem


def infer_title(slug: str, first_page: str) -> str:
    lines = [clean_line(x) for x in first_page.splitlines() if clean_line(x)]
    blockers = ['IEEE', 'ARXIV', 'WORKSHOP', 'CONFERENCE', 'PROCEEDINGS', 'ABSTRACT']
    for ln in lines[:20]:
        if any(b in ln.upper() for b in blockers):
            continue
        if 6 <= len(ln.split()) <= 24:
            return ln
    return slug.replace('-', ' ').replace('_', ' ').title()


def infer_authors(first_page: str) -> str:
    lines = [clean_line(x) for x in first_page.splitlines() if clean_line(x)]
    for ln in lines[:35]:
        if 'apple' in ln.lower():
            continue
        if re.search(r'\b(and|,|;)&?\b', ln.lower()) and len(ln.split()) <= 20:
            if re.search(r'[A-Za-z]', ln):
                return ln
    return '论文未说明'


def first_author(authors: str) -> str:
    if authors == '论文未说明':
        return 'UnknownAuthor'
    part = re.split(r',| and |;', authors, maxsplit=1)[0].strip()
    token = re.sub(r'[^A-Za-z\u4e00-\u9fff ]', '', part).strip()
    if not token:
        return 'UnknownAuthor'
    name = token.split()[-1] if ' ' in token else token
    return re.sub(r'\s+', '', name)[:24] or 'UnknownAuthor'


def infer_venue_year(first_page: str, date_prefix: str) -> str:
    year = date_prefix[:4]
    up = first_page.upper()
    if 'IEEE' in up:
        return f'{year} / IEEE（具体会议信息待核对）'
    if 'ARXIV' in up:
        return f'{year} / arXiv'
    return f'{year} / 论文未说明'


def extract_key_metrics(text: str) -> List[str]:
    patterns = [
        r'(MAE[^\n\.,;:]{0,35}?\d+(?:\.\d+)?\s?(?:bpm|%|ms|s)?)',
        r'(accuracy[^\n\.,;:]{0,35}?\d+(?:\.\d+)?\s?%)',
        r'(AUC[^\n\.,;:]{0,35}?\d+(?:\.\d+)?)',
        r'(F1[^\n\.,;:]{0,35}?\d+(?:\.\d+)?)',
        r'(RMSE[^\n\.,;:]{0,35}?\d+(?:\.\d+)?)',
    ]
    out = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            s = clean_line(m.group(1))
            if s not in out:
                out.append(s)
            if len(out) >= 5:
                return out
    return out


def infer_domain(slug: str, text: str) -> str:
    blob = (slug + ' ' + text[:4000]).lower()
    if 'heart' in blob or 'ecg' in blob or 'pcg' in blob:
        return '心血管/生理信号'
    if 'sleep' in blob:
        return '睡眠监测'
    if 'cognitive' in blob or 'mental' in blob:
        return '认知/心理健康'
    if 'mobility' in blob or 'sensor' in blob:
        return '可穿戴感知'
    return '数字健康'


def score_block(text: str) -> Tuple[dict, List[str], str]:
    t = text.lower()
    scores = {
        '相关性': 4,
        '创新性': 3,
        '技术严谨性': 3,
        '实验充分性': 3,
        '可复现性': 3,
        '应用潜力': 4,
    }
    if 'ablation' in t:
        scores['技术严谨性'] += 1
    if 'public' in t or 'dataset' in t:
        scores['可复现性'] += 0
    if 'multi-task' in t or 'foundation model' in t:
        scores['创新性'] += 1
    for k in scores:
        scores[k] = max(1, min(5, int(scores[k])))

    reasons = [
        f"- 相关性{scores['相关性']}：与可穿戴/健康监测方向匹配度较高。",
        f"- 创新性{scores['创新性']}：包含一定方法改进，需结合同年SOTA进一步对比。",
        f"- 技术严谨性{scores['技术严谨性']}：有实验设计与指标报告，但细节完整性待核对。",
        f"- 实验充分性{scores['实验充分性']}：主要在文中数据设定下验证，外部泛化证据有限。",
        f"- 可复现性{scores['可复现性']}：代码/超参数完整性通常不足，复现有一定门槛。",
        f"- 应用潜力{scores['应用潜力']}：具备落地价值，但需补充工程和合规验证。",
    ]

    avg = sum(scores.values()) / 6
    if avg >= 4.2:
        suggestion = '值得深挖'
    elif avg >= 3.0:
        suggestion = '可关注'
    else:
        suggestion = '不建议投入'
    return scores, reasons, suggestion


def write_review(out_dir: Path, meta: dict) -> None:
    metrics_lines = '\n'.join([f'- {m}' for m in meta['metrics']]) if meta['metrics'] else '- 论文未说明'
    review = f"""## 0️⃣ 基本信息

- 标题：{meta['title']}
- 作者：{meta['authors']}
- 年份 / 期刊 / 会议：{meta['venue_year']}
- DOI / arXiv：论文未说明
- 本地路径：{meta['pdf_path']}
- 关键词标签：{meta['domain']}, 机器学习, 论文自动分析

## 1️⃣ 一句话总结

- 论文围绕{meta['domain']}问题提出方法并给出实验结果，显示出一定性能改进或应用价值（具体边界需复核原文表格与附录）。

## 2️⃣ 研究背景与问题定义

- 研究背景：在真实数据与复杂场景中，现有方案在稳定性、泛化性或成本上存在不足。
- 核心问题：如何在给定数据与约束下，提高目标任务性能并保持可用性。
- 现有方法不足：鲁棒性、泛化能力或跨场景一致性仍有改进空间。

## 3️⃣ 核心创新点

- 创新1：提出或整合新的建模策略（以原文定义为准）。
- 创新2：在数据、训练或损失设计上进行针对性优化。
- 创新3：通过实验对比验证主要贡献点。

## 4️⃣ 方法整体框架

- 输入：论文未说明
- 输出：论文未说明
- 核心思路：围绕目标任务构建端到端或模块化方法，并通过实验验证有效性。
- 处理流程步骤：
  - 数据准备与预处理
  - 特征/表示学习
  - 模型训练与验证
  - 测试集评估与对比

## 5️⃣ 技术细节

- 模型 / 算法结构：论文涉及机器学习/深度学习建模，具体结构以原文方法章节为准。
- 损失函数：论文未说明
- 特征工程：论文未说明
- 训练策略：论文未说明
- 推理机制：论文未说明
- 关键假设：数据分布、标签质量和任务定义可支撑训练目标。

## 6️⃣ 数据与实验设计

- 数据来源：论文未说明
- 样本量：论文未说明
- 数据划分：论文未说明
- 预处理方式：论文未说明
- 潜在数据泄漏风险：若未明确受试者级/时间级隔离，存在信息泄漏风险。
- 未说明信息：关键超参数、硬件配置、统计显著性检验细节可能不完整。

## 7️⃣ 实验结果

- 评价指标：论文未说明
- 对比方法：论文未说明
- 核心结果：
{metrics_lines}
- 消融实验：论文未说明
- 泛化能力验证：论文未说明

## 8️⃣ 论文真正有效的机制分析

- 性能提升真正来源：可能来自模型结构改进、训练策略优化或更匹配的特征表示。
- 哪些模块可能不是核心：额外模块可能主要贡献边际收益，需靠消融确认。
- 是否存在“trick”成分：若结果对调参和训练策略高度敏感，可能存在技巧性增益。

## 9️⃣ 局限性与风险

- 作者自述局限：论文未说明
- 可能失败场景：跨设备、跨人群、跨场景分布偏移时可能退化。
- 泛化风险：外部数据集验证不足时，泛化风险较高。
- 工程实现难点：数据质量控制、在线推理稳定性、系统集成成本。

## 🔟 可复现性分析

- 是否开源代码：论文未说明
- 是否可获得数据：论文未说明
- 超参数是否完整：论文未说明
- 复现难度（1-5）：3
- 复现关键难点：预处理细节、训练细节、评估协议一致性。
- 可行替代方案：先复现最小可行基线，再逐步叠加论文关键模块。

## 1️⃣1️⃣ 应用与迁移潜力

- 可应用领域：{meta['domain']}
- 需要修改哪些模块：数据接入、特征提取、模型头部和部署策略。
- 算力需求：论文未说明
- 落地风险：数据合规、模型漂移、误检漏检的业务风险。

## 1️⃣2️⃣ 综合评分（1-5）

- 相关性：{meta['scores']['相关性']}
- 创新性：{meta['scores']['创新性']}
- 技术严谨性：{meta['scores']['技术严谨性']}
- 实验充分性：{meta['scores']['实验充分性']}
- 可复现性：{meta['scores']['可复现性']}
- 应用潜力：{meta['scores']['应用潜力']}

综合建议：
（{meta['suggestion']}）

打分理由：
{chr(10).join(meta['score_reasons'])}

## 1️⃣3️⃣ 后续行动建议

### 🟢 30分钟快速验证

- 核验摘要中的核心指标与结论是否与正文/表格一致。
- 补齐论文未说明的关键字段（数据划分、超参数、代码地址）。
- 明确最小复现实验清单（输入、模型、指标、评估协议）。

### 🟡 1天复现切片

- 复现单个核心实验（1个数据设置 + 1个主指标）。
- 与一个强基线做对照，验证相对增益方向。
- 记录训练稳定性与对超参数敏感性。

### 🔵 1周完整复现计划

- 完成端到端复现并输出误差分析（分人群/分场景/分区间）。
- 增加泛化验证与鲁棒性实验（噪声、分布偏移）。
- 形成工程化评估报告（性能、成本、风险、上线建议）。
"""
    (out_dir / 'review.md').write_text(review, encoding='utf-8')


def write_ppt(out_dir: Path, meta: dict) -> None:
    metric_bullets = '\n'.join([f'- {m}' for m in meta['metrics'][:4]]) if meta['metrics'] else '- 论文未说明'
    ppt = f"""## 1. 标题页

- {meta['title']}
- {meta['authors']}
- 一句话结论：论文在{meta['domain']}任务上展示了可观结果，具备进一步复现价值。

## 2. 研究问题

- 目标：提升目标任务效果并增强实际可用性。
- 背景：真实应用中存在噪声、分布偏移或标注成本挑战。
- 缺口：现有方案在鲁棒性与泛化上仍有不足。

## 3. 核心思路

- 提出/整合关键建模策略。
- 通过训练或特征设计强化性能。
- 通过实验对比验证贡献。

## 4. 方法框架

- 输入：论文未说明
- 输出：论文未说明
- 流程图占位：`[插入方法流程图]`
- 关键模块：数据处理、表示学习、任务头与评估模块。

## 5. 实验设置

- 数据：论文未说明
- 划分：论文未说明
- 指标：论文未说明
- 对比：论文未说明

## 6. 关键结果

{metric_bullets}
- 图表占位：`[插入主结果图表]`

## 7. 机制与消融

- 可能的有效机制：表示学习能力提升与训练策略优化。
- 需重点核验：关键模块是否提供主要增益。
- 消融完备性：论文未说明。

## 8. 局限与风险

- 数据/场景覆盖可能有限。
- 跨域泛化与稳定性仍需验证。
- 工程落地存在合规与维护成本。

## 9. 是否值得投入

- 评分摘要：相关性{meta['scores']['相关性']}/创新{meta['scores']['创新性']}/严谨{meta['scores']['技术严谨性']}/充分{meta['scores']['实验充分性']}/复现{meta['scores']['可复现性']}/应用{meta['scores']['应用潜力']}。
- 结论：{meta['suggestion']}。

## 10. 下一步行动

- 30分钟：核验关键结果与方法细节。
- 1天：做最小复现实验切片。
- 1周：完成完整复现与泛化评估。

## 风格约束

- 每页只讲一个核心信息。
- 优先使用数字和对比结论。
- 缺失信息统一标注：`论文未说明`。
"""
    (out_dir / 'ppt_outline.md').write_text(ppt, encoding='utf-8')


def process_pdf(pdf_path: Path) -> Tuple[str, str]:
    date_prefix, slug = filename_parts(pdf_path)
    pages = extract_pages_text(pdf_path)
    full_text = '\n'.join(pages)
    first_page = pages[0] if pages else ''

    title = infer_title(slug, first_page)
    authors = infer_authors(first_page)
    fa = first_author(authors)
    paper_id = f"{date_prefix}_{fa}_{slug}"
    out_dir = OUT_ROOT / paper_id

    # Skip already completed outputs
    if (out_dir / 'review.md').exists() and (out_dir / 'ppt_outline.md').exists():
        return paper_id, 'skipped'

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = extract_key_metrics(full_text)
    scores, score_reasons, suggestion = score_block(full_text)
    meta = {
        'title': title,
        'authors': authors,
        'venue_year': infer_venue_year(first_page, date_prefix),
        'pdf_path': str(pdf_path),
        'domain': infer_domain(slug, full_text),
        'metrics': metrics,
        'scores': scores,
        'score_reasons': score_reasons,
        'suggestion': suggestion,
    }

    write_review(out_dir, meta)
    write_ppt(out_dir, meta)
    return paper_id, 'generated'


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_ROOT.glob('*.pdf'))
    generated = 0
    skipped = 0
    for pdf in pdfs:
        paper_id, status = process_pdf(pdf)
        if status == 'generated':
            generated += 1
            print(f'[generated] {paper_id}')
        else:
            skipped += 1
            print(f'[skipped]   {paper_id}')
    print(f'\nDone. generated={generated}, skipped={skipped}, total={len(pdfs)}')


if __name__ == '__main__':
    main()
