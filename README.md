# Occlusion-Aware Training and Post-Processing for Motorized Two-Wheeler Detection: Does Synthetic Occlusion Transfer to Real Occlusion?

**Does Synthetic Occlusion Transfer to Real Occlusion?**

Kukuh Agung Tri Pamuji, Semeia Rouli Sinaga, Ali Rahman Bayanaka, Wijdanadam Rafifilmi Nadarosula, Alifia Azzahra — Telkom University

We train YOLO11s on UVH-26 traffic-camera images with three training configurations (C0 standard, C1 targeted random erasing, C2 vehicle-crop pasting) and two post-processing options (hard NMS IoU 0.7, Soft-NMS σ = 0.3). We test them on paired synthetic occlusion (L0–L3, 0–80% box cover) and on 1,476 naturally occluded targets labeled clean / light / heavy.

## Main result (AP50, mean of 3 seeds, hard NMS 0.7)

| Set | C0 | C1 | C2 |
|---|---|---|---|
| Synthetic L1 (20–40%) | 53.1 | 53.7 | 73.5 |
| Synthetic L2 (40–60%) | 14.4 | 16.3 | 52.4 |
| Synthetic L3 (60–80%) | 1.7 | 2.0 | 28.6 |
| Natural light | 75.4 | 76.7 | 74.9 |
| Natural heavy | 58.2 | 59.0 | 58.4 |

C2 gains +20.3 / +38.0 / +26.9 points on synthetic cover (95% CI above zero), but its natural light / heavy changes (−0.45 / +0.26) have intervals that include zero. Synthetic gains do not establish transfer to natural occlusion. Full tables: `reports/Laporan_Run_Final.md`.

## Repository layout

```
notebooks/
  mbc-collab_kaggle.ipynb   final Kaggle run with outputs (long logs trimmed)
  mbc-colab_colab.ipynb     same pipeline adapted for Google Colab + Drive
src/
  occlusion_lib.py          occlusion generator, NMS post-processing, evaluator, bootstrap
  paket_tim_cell.py         notebook cell that packs checkpoints, raw predictions, panels, sha256 manifest
reports/
  Laporan_Run_Final.md      full results report (Indonesian)
  laporan_hasil.html        report with figures
  tables/                   all result tables as CSV, plus per-epoch training log of the 9 runs
  figures/                  figures exported from the notebook
labels/
  aturan_penandaan.json     labeling rules (clean / light / heavy / discard)
  label_pembanding_AI_claude.csv   150-target AI comparison labels (NOT a human annotator)
tools/
  penandaan-oklusi-motor.html      offline labeling tool for a second human annotator
```

## Reproduce

1. Open `notebooks/mbc-collab_kaggle.ipynb` on Kaggle (GPU T4, Internet on, secret `HF_TOKEN`).
2. Attach the natural-occlusion label CSV (`real_tags (1).csv`) as a Kaggle dataset and set `LABEL_KUKUH_DIR` in the config cell.
3. **Save Version → Save & Run All** (about 8–9 hours). Training is deterministic: six Kaggle runs gave identical per-epoch logs and tables.

Environment: Python 3.13.15, PyTorch 2.11.0+cu128, Ultralytics 8.4.175, Tesla T4. Colab (different CUDA build) gives slightly different numbers.

## Data

- UVH-26: https://huggingface.co/datasets/iisc-aim/UVH-26 (branch `v1.0`, majority-vote annotations, CC BY 4.0)
- Dataset card: https://huggingface.co/datasets/iisc-aim/UVH-26/blob/v1.0/README.md
- UVH-26 paper: https://arxiv.org/abs/2511.02563
- The class used is Two-wheeler (category 8: motorcycles and seated motor scooters with rider). Standing kick-scooters are not part of this class.

## Not in this repository

Model checkpoints (`best.pt`), raw predictions, matching files and comparison panels are large. They are produced by the final run in `PAKET_TIM_v6_e60_ulang.zip` (571 MB) and should be shared through GitHub Releases, Git LFS or a Kaggle dataset.

## Limitations

Natural-occlusion labels come from one human annotator. The 150-target comparison (κ = 0.283) was rated by an AI model and is not human inter-annotator agreement. Results cover one detector, one city, daytime images and targets of 32 px and larger.
