# Large-Scale Training Deck

## Slide 1 - Title

- ICLR 2024 Apple biosignal foundation model framework
- Scale: 141K participants, 20M PPG, 3.75M ECG

## Slide 2 - Why it matters

- Label scarcity and free-living noise block traditional supervised pipelines
- Foundation encoders improve reuse and speed

## Slide 3 - Dataset

- AHMS large-scale longitudinal wearable dataset
- Participant-level split and modality-specific preprocessing

## Slide 4 - Pipeline

- 5-stage SSL pipeline with participant-level positive pairs
- Regularized InfoNCE + momentum branch

## Slide 5 - Objective

- Regularized InfoNCE objective and practical hyperparameters
- Modality-specific augmentation policy

## Slide 6 - Results

- Participant-level positive pairs outperform segment-level across PPG/ECG
- Large gains in age/BMI/sex probes

## Slide 7 - Modality distinction

- ECG appears easier to contrastively pretrain, but PPG was more predictive for many targets
- Use modality-specific policies

## Slide 8 - Ablation framework

- Ours outperforms SimCLR/BYOL variations on SER
- KoLeo regularization is important

## Slide 9 - Architecture tradeoff

- EfficientNet-style 1D encoder gives strong efficiency
- ViT competitive but heavier

## Slide 10 - Verification framework

- 4-layer QA framework from representation to deployment
- Directly adaptable to monitoring products

## Slide 11 - Roadmap

- 90-day execution plan for adapting the framework
- Covers data, training, validation, deployment

## Slide 12 - Limits

- Important limitations for productization and evidence claims
- Need external validation beyond this paper

## Slide 13 - Takeaways

- Representation-first, modality-aware, validation-layered development strategy

## Slide 14 - Appendix

- Consolidated key constants and metrics from the paper
