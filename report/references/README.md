# References

Organised by topic. PDFs saved locally for offline access.

```text
references/
  dataset/          — UW-Sinus-Surgery-CL dataset paper
  kinematics/       — Kinematic-guided segmentation (core future direction)
  sam_surgical/     — SAM/SAM2 applied to surgical instruments
  README.md         — this file
```

---

## DATASET

### [D1] UW-Sinus-Surgery-CL Dataset Paper

**Title:** Towards Better Surgical Instrument Segmentation in Endoscopic Vision:
Multi-Angle Feature Aggregation and Contour Supervision

**Authors:** Fangbo Qin, Shan Lin, Yangming Li, Randall A. Bly, Kris S. Moe, Blake Hannaford

**Venue:** IEEE Robotics and Automation Letters (RA-L), 2020

**arXiv:** [arxiv.org/abs/2002.10675](https://arxiv.org/abs/2002.10675)

**Dataset page:** [digital.lib.washington.edu/researchworks](https://digital.lib.washington.edu/researchworks/items/80f00953-c8e0-46e1-9095-3c7cf0de6bed)

**PDF:** `dataset/qin2020_uw_sinus_surgery_cl.pdf`

---

## SOTA BASELINE

### [S1] HFRF-Net (Dice = 0.9374 on UW-Sinus-Surgery-Live)

**Title:** Dual-task hierarchical feature refinement and fusion network for precise
segmentation of surgical tools and polyps in endoscopy

**Venue:** Expert Systems with Applications, Elsevier, 2025

**URL:** [sciencedirect.com/...S0957417425022377](https://www.sciencedirect.com/science/article/abs/pii/S0957417425022377)

**PDF:** Not freely available (ScienceDirect paywall — access via institutional login)

---

## FOUNDATION MODEL

### [F1] SAM2 — Segment Anything Model 2

**Title:** SAM 2: Segment Anything in Images and Videos

**Authors:** Nikhila Ravi et al. (Meta AI)

**arXiv:** [arxiv.org/abs/2408.00714

**Note:** SAM3 (facebook/sam3) builds on SAM2. No separate SAM3 paper — model released on HuggingFace.

---

## KINEMATICS-GUIDED SEGMENTATION

### [K1] Self-Supervised Surgical Tool Segmentation using Kinematic Information

**Authors:** Cristian da Costa Rocha, Nicolas Padoy, Benoit Rosa

**Venue:** ICRA 2019, pp. 8720–8726

**arXiv:** [arxiv.org/abs/1902.04810

**PDF:** `kinematics/rocha2019_kinematic_self_supervised_segmentation.pdf`

**Key contribution:** First paper to use the robot's kinematic model to auto-generate
pixel-level training labels — zero manual annotation. Addresses unknown hand-eye
calibration via optimisation. No SAM, plain FCN backbone.

**Gap our work fills:** They use kinematics to generate training labels for a CNN.
We use kinematics to generate spatial prompts for SAM3+LoRA at inference —
stronger backbone, no retraining needed per deployment.

---

### [K2] SAF-IS: Spatial Annotation Free Framework for Instance Segmentation

**Authors:** Luca Sestini et al.

**Venue:** Medical Image Analysis, 2025

**arXiv:** [arxiv.org/abs/2309.01723

**PDF:** `kinematics/safis2025_spatial_annotation_free.pdf`

**Key contribution:** Instance segmentation requiring only binary tool masks and
tool-presence labels (freely available from robot logs). No spatial annotations.
Validated on EndoVis 2017/2018.

**Gap our work fills:** SAF-IS still needs binary masks as input. Our kinematic-prompt
pipeline would not need masks at all at inference.

---

## SAM APPLIED TO SURGICAL INSTRUMENTS

### [SA1] SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation

**Authors:** Wenxi Yue et al.

**Venue:** AAAI 2024

**arXiv:** [arxiv.org/abs/2308.08746

**Code:** https://github.com/wenxi-yue/SurgicalSAM

**PDF:** `sam_surgical/surgicalsam2024_aaai.pdf`

**Key contribution:** Lightweight prototype-based class prompt encoder for SAM.
Eliminates explicit prompts via contrastive prototype learning. SOTA on EndoVis 2017/2018.

---

### [SA2] Adapting SAM for Surgical Instrument Tracking and Segmentation

**Authors:** Yang et al.

**Venue:** arXiv 2024

**arXiv:** [arxiv.org/abs/2404.10640

**PDF:** `sam_surgical/yang2024_adapting_sam_endoscopic.pdf`

**Key contribution:** SAM fine-tuned for instrument tracking + segmentation in
endoscopic submucosal dissection (ESD) videos. Uses LoRA-style adaptation.

---

### [SA3] Augmenting Real-time Surgical Instrument Segmentation with Point Tracking and SAM

**Authors:** Zijian Wu, Adam Schmidt, Peter Kazanzides, Septimiu E. Salcudean

**Venue:** arXiv 2024 / PMC

**arXiv:** [arxiv.org/abs/2403.08003

**Code:** https://github.com/zijianwu1231/SIS-PT-SAM

**PDF:** `sam_surgical/wu2024_point_tracking_sam_surgical.pdf`

**Key contribution:** Online point tracker generates SAM prompts automatically across
video frames — no per-frame manual prompting. Achieves 25–90 FPS real-time on EndoVis 2015.
Dice = 91.0, IoU = 84.8.

**Directly relevant:** Point tracking replaces GT-derived point prompts.
Our kinematic approach is conceptually equivalent but uses robot state instead of
optical tracking — more robust under occlusion.

---

### [SA4] Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning

**Authors:** Haofeng Liu, Erli Zhang, Junde Wu et al.

**Venue:** arXiv 2024

**arXiv:** [arxiv.org/abs/2408.07931

**PDF:** `sam_surgical/surgsam2_2024_frame_pruning.pdf`

**Key contribution:** Adds Efficient Frame Pruning (EFP) to SAM2 to handle
long-range temporal redundancy in surgical video. Enables real-time inference.

---

## PAYWALLED / NOT DOWNLOADED

- **HFRF-Net [S1]** — ScienceDirect paywall
- **Real-time vision-based surgical tool segmentation with robot kinematics prior** (IEEE ICRA 2018, doi:10.1109/ICRA.2018.8333305) — IEEE paywall. Fuses CNN + kinematic pose at inference.

---

## Summary gap table

| Paper | Annotation-free | SAM backbone | Kinematic prompt |
|---|---|---|---|
| Rocha 2019 [K1] | YES | NO | YES (labels) |
| SAF-IS 2025 [K2] | YES | NO | NO |
| SurgicalSAM [SA1] | NO | YES | NO |
| Wu 2024 [SA3] | partial (point tracker) | YES | NO |
| **Our reframed work** | **YES** | **YES (SAM3+LoRA)** | **YES (prompt)** |

No published paper combines all three — annotation-free + SAM backbone + kinematic prompt.
