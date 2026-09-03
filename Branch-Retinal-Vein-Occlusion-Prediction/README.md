# BRVO Prediction from Pre-onset Fundus Hemisection Images

Reimplementation of [*Predicting branch retinal vein occlusion development using multimodal deep learning and pre-onset fundus hemisection images*](https://doi.org/10.1038/s41598-025-85777-7) (Scientific Reports, 2025).

Eun Young Choi\*, Dongyoung Kim\*, Jinyeong Kim, Eunjin Kim, Hyunseo Lee, Jinyoung Yeo, Tae Keun Yoo†, Min Kim†

> ⚠ This repository ports the paper's **pipeline** into runnable code. It is not the
> authors' official implementation and does not reproduce the reported numbers — see
> [Scope](#scope).

## Overview

Branch retinal vein occlusion (BRVO) is the most common form of retinal vein occlusion
and a leading cause of visual impairment in the working-age population. Predicting
*where* and *whether* it will occur from a fundus photograph taken before onset is hard:
BRVO is usually confined to the arteriovenous crossing of either the superior or the
inferior arcade of a single eye, and only about 10% of patients go on to develop it in
the fellow eye.

The paper's idea is that this asymmetry reflects **intra-individual differences in
retinal vascular structure**. To isolate structural signal from confounders like age
and systemic disease, it works on **metadata-matched hemisections**: each fundus image
is split into an upper and a lower half through the optic disc centre, and each half is
labelled by whether BRVO later appeared in it. The affected half is thus compared
against the patient's own unaffected halves.

A U-Net segments the optic disc and the retinal blood vessels; two EfficientNetB0
classifiers are trained, one on fundus halves and one on vessel halves; and a
**BV-enhanced multimodal model** concatenates the two.

## Method

### Segmentation — `src/segmentation.py`

A U-Net extracts the optic disc and the retinal blood vessels. The paper trains vessel
segmentation on DRIVE and FIVES, and optic-disc segmentation on REFUGE, then applies
both to the study images. Quality is reported with IoU and the Dice coefficient.

### Hemisection — `src/hemisection.py`

Each image is split along a horizontal line through the optic disc centre, taken from
the y-coordinate of the disc segmentation. Both halves are resized to 224 × 224 and
become one training datum each. The fundus half and its matched vessel half are split
at the same y, which is what makes them a pair for the multimodal model.

### Classification — `src/models.py`

| Model | Input | Backbone |
|---|---|---|
| Unimodal (fundus) | fundus hemisection | EfficientNetB0, ImageNet pretrained |
| Unimodal (vessel) | blood-vessel hemisection | EfficientNetB0, ImageNet pretrained |
| **BV-enhanced multimodal** | both | the two backbones concatenated |

The multimodal head replaces the last layers of the two trained unimodal models with
fully connected layers of **512, 128 and 52 nodes with dropout**, followed by a softmax
over the two classes (future BRVO occurrence or not).

Training follows the paper: SGD with momentum, lr 1e-4, mini-batch 20, 100 epochs.
Augmentation covers horizontal flipping, random rotation, blurring, brightening and
darkening, random noise, horizontal-shift cropping, and ±10% scaling.

### Evaluation — `src/evaluate.py`, `src/train.py`

The cohort is small, so evaluation is **leave-one-out cross-validation**: build N
models, each excluding one sample, score the held-out sample, and pool the softmax
outputs into a single ROC curve. Reported metrics are AUC, accuracy, sensitivity and
specificity, with the operating threshold chosen by Youden's index and 95% CIs from a
percentile bootstrap.

### Grad-CAM — `src/gradcam.py`

Attention maps from the last convolutional layer, used in the paper to confirm the
model attends to the arteriovenous crossing regions of the vascular arcade.

## Repository Structure

```
.
├── main.py                # CLI — toy / segment / hemisect / train / loocv / gradcam
├── requirements.txt
└── src/
    ├── config.py          # hyperparameters from the Methods section
    ├── data.py            # dataset, augmentation, synthetic cohort generator
    ├── segmentation.py    # U-Net + IoU / Dice
    ├── hemisection.py     # optic-disc-centred upper/lower split
    ├── models.py          # EfficientNetB0 unimodal + concatenation multimodal
    ├── train.py           # leave-one-out cross-validation
    ├── evaluate.py        # AUC, accuracy, sensitivity, specificity, Youden, bootstrap CI
    └── gradcam.py         # attention maps
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Synthetic cohort — 108 hemisections, 27 BRVO / 81 non-BRVO
python main.py toy

# 2. Segmentation and the hemisection split
python main.py segment --limit 8
python main.py hemisect

# 3. Train the two unimodal models, then assemble the multimodal one
python main.py train --kind fundus --tiny --epochs 3
python main.py train --kind vessel --tiny --epochs 3
python main.py train --kind multimodal --from-unimodal --tiny

# 4. Leave-one-out cross-validation
python main.py loocv --kind multimodal --tiny --epochs 2 --folds 12

# 5. Grad-CAM
python main.py gradcam --kind multimodal --tiny
```

`--tiny` swaps EfficientNetB0 for a small randomly initialized CNN so the pipeline runs
without downloading ImageNet weights; its numbers mean nothing. `--folds` truncates the
cross-validation for smoke tests — omit it for the real protocol, which is one fold per
sample. Drop both for the paper's setup:

```bash
python main.py train --kind fundus --device cuda
python main.py loocv --kind multimodal --device cuda
```

### Data

Patient images are not redistributable, so `main.py toy` synthesizes a cohort with the
paper's shape: 27 affected hemisections against 81 unaffected ones (27 counter halves
of the same eye and 54 contralateral halves). Each synthetic fundus carries a few
vessel arcs and an optic disc, and the positive class additionally gets an artery
crossing a vein at a shallow angle — a stand-in for the arteriovenous crossing, and the
only signal separating the classes.

To use real data, write a `data/<name>/hemisections.pt` holding `fundus` (N,3,224,224),
`vessel` (N,3,224,224), `label` (N,), and a `meta` list with `sample_id`, `eye_id`,
`patient_id`, `side` and `origin`. Then pass `--data-dir data/<name>`.

### Key options

| Option | Default | Corresponds to |
|---|---|---|
| `--epochs` | 100 | training epochs |
| `--batch-size` | 20 | mini-batch size |
| `--lr` | 1e-4 | SGD learning rate |
| `--kind` | — | `fundus` / `vessel` / `multimodal` |
| `--from-unimodal` | — | build the multimodal model from the trained unimodal backbones |
| `--no-pretrained` | — | EfficientNetB0 without ImageNet weights |
| `--no-augment` | — | disable augmentation |
| `--threads` | 4 | CPU thread cap; leaving it unset is far slower on many-core machines |

## Scope

**Implemented** — U-Net segmentation with IoU/Dice, the optic-disc-centred hemisection
split, both unimodal EfficientNetB0 classifiers, the BV-enhanced multimodal
concatenation model, the paper's augmentation list and optimizer settings, leave-one-out
cross-validation, AUC / accuracy / sensitivity / specificity with Youden's index and
bootstrap CIs, and Grad-CAM.

**Not implemented** — the following appear in the paper but not here:

- **The clinical cohort.** The 2,673 screened patients and the 38 with usable pre-onset
  images are not redistributable. Only the synthetic set ships here, so none of the
  reported numbers (AUC 0.76, accuracy 68.5%) can be reproduced.
- **DRIVE / FIVES / REFUGE.** The segmentation models are not pretrained on them; the
  U-Net trains on whatever masks it is given.
- **Clinical measurements.** Lesion area and location, central macular and subfoveal
  choroidal thickness, and arteriovenous crossing angles were measured by hand in
  Heidelberg Eye Explorer and ImageJ. None of that is here.
- **Metadata matching.** Real matching on age and systemic risk factors is replaced by
  the synthetic cohort's construction.
- **Statistical comparisons** between models and the supplementary analyses.

So this repository is the **pipeline**, not the study.

## Citation

```bibtex
@article{choi2025brvo,
    title   = {Predicting branch retinal vein occlusion development using multimodal
               deep learning and pre-onset fundus hemisection images},
    author  = {Choi, Eun Young and Kim, Dongyoung and Kim, Jinyeong and Kim, Eunjin and
               Lee, Hyunseo and Yeo, Jinyoung and Yoo, Tae Keun and Kim, Min},
    journal = {Scientific Reports},
    volume  = {15},
    pages   = {2729},
    year    = {2025},
    doi     = {10.1038/s41598-025-85777-7},
}
```
