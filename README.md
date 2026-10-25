# Occlusion-Aware Training and Post-Processing for Motorized Two-Wheeler Detection: Does Synthetic Occlusion Transfer to Real Occlusion?

**Does Synthetic Occlusion Transfer to Real Occlusion?**

Kukuh Agung Tri Pamuji, Semeia Rouli Sinaga, Ali Rahman Bayanaka, Wijdanadam Rafifilmi Nadarosula (Mentor), Alifia Azzahra (Mentor) — Telkom University

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

# Laporan Run Final — mbc-collab (Kaggle, 25 Oktober 2026)

Disusun dari output yang tersimpan di `mbc-collab_3.ipynb`. Semua angka disalin dari output sel, tidak dihitung ulang.

## Ringkasan

- **Hasil identik dengan run paper.** Semua tabel (AP utama, ablasi bootstrap, RQ3, generator silang, evaluasi semua motor, cahaya, kategori 12c, kappa) sama persis dengan run paper (`mbc-final-keterima`) dan run ulang 24 Okt. Ini kali keenam hasil yang sama keluar di Kaggle.
- **Training lengkap.** 9 run utama dilatih dari awal (`resume=False`), masing-masing 60 epoch, sekitar 33 menit per run.
- **Paket tim lengkap.** `PAKET_TIM_v6_e60_ulang.zip` (571 MB, 107 file): 9 best.pt, 9 raw_preds, 7 panel, matching, prediksi generator silang, README, manifest sha256.
- **Catatan:** 3 run augmentasi cahaya (bagian 10b) tidak dilatih ulang di sesi ini (`lewati (sudah selesai)`), karena bobotnya masih tersimpan dari sesi Kaggle sebelumnya di folder yang sama. Hasil cahayanya tetap identik dengan run paper.

## 1. Identitas run

- Kaggle, Tesla T4, Python 3.13.15, torch 2.11.0+cu128, Ultralytics 8.4.175
- Notebook `mbc-collab`, EXP_TAG `_v6_e60_ulang`, mode PENUH: YOLO11s, imgsz 640, 60 epoch, patience 15, seed 0/1/2, deterministic=True
- UVH-26 v1.0 (anotasi MV); kelas 8 Two-wheeler (motor + skuter matic + pengendara) → 1 kelas `two_wheeler`; bicycle tidak dipakai
- Pool penutup: train 4801 / test 2152 crop; `OCCLUDER_CATS` memuat kategori 14 (Others, 6 crop di pool uji)
- Label nyata: `real_tags (1).csv` dari Kukuh, 1623 motor (cocok 100%) → 1476 target final (587 / 589 / 300), 147 dibuang
- NMS dipilih di validasi: N1 = hard IoU 0.7 (sama dengan default), N2 = Soft-NMS σ 0.3

## 2. Training

| run | menit | epoch | resume | status |
|:--|--:|--:|:--|:--|
| C0_s0 | 33.9 | 60 | tidak | selesai |
| C0_s1 | 33.2 | 60 | tidak | selesai |
| C0_s2 | 33.2 | 60 | tidak | selesai |
| C2_s0 | 33.3 | 60 | tidak | selesai |
| C2_s1 | 33.3 | 60 | tidak | selesai |
| C2_s2 | 33.0 | 60 | tidak | selesai |
| C1_s0 | 32.9 | 60 | tidak | selesai |
| C1_s1 | 32.8 | 60 | tidak | selesai |
| C1_s2 | 32.8 | 60 | tidak | selesai |

## 3. Data

| split | n_img | n_motor |
|:--|--:|--:|
| train | 2550 | 15323 |
| val | 450 | 2703 |
| test | 1150 | 6729 |

| level | target | citra | persen | penutup: - | kendaraan | lain | objek_diam |
|:--|--:|--:|--:|--:|--:|--:|--:|
| bersih | 587 | 365 | 39.8 | 587 | 0 | 0 | 0 |
| ringan | 589 | 362 | 39.9 | 0 | 535 | 20 | 34 |
| berat | 300 | 191 | 20.3 | 0 | 269 | 18 | 13 |

## 4. Hasil utama AP50 (×100, rata-rata ± sd 3 seed)

| cfg | NMS | L0 | L1 | L2 | L3 | nyata_bersih | nyata_ringan | nyata_berat |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| C0 | std 0.7 (N0=N1) | 85.2 ± 1.0 | 53.1 ± 1.8 | 14.4 ± 1.6 | 1.7 ± 0.6 | 84.7 ± 0.7 | 75.4 ± 0.6 | 58.2 ± 0.7 |
| C0 | Soft-NMS 0.3 (N2) | 85.6 ± 0.9 | 54.1 ± 1.8 | 14.9 ± 1.6 | 1.7 ± 0.6 | 83.8 ± 0.8 | 72.5 ± 0.6 | 56.8 ± 0.4 |
| C1 | std 0.7 (N0=N1) | 85.2 ± 0.3 | 53.7 ± 0.4 | 16.3 ± 1.6 | 2.0 ± 0.3 | 85.2 ± 0.6 | 76.7 ± 1.2 | 59.0 ± 2.3 |
| C1 | Soft-NMS 0.3 (N2) | 85.6 ± 0.3 | 54.5 ± 0.3 | 16.8 ± 1.6 | 2.1 ± 0.3 | 84.0 ± 0.8 | 73.5 ± 1.3 | 57.5 ± 2.3 |
| C2 | std 0.7 (N0=N1) | 85.3 ± 0.4 | 73.5 ± 0.7 | 52.4 ± 1.4 | 28.6 ± 2.2 | 85.6 ± 0.2 | 74.9 ± 1.4 | 58.4 ± 1.0 |
| C2 | Soft-NMS 0.3 (N2) | 85.7 ± 0.4 | 74.3 ± 0.6 | 53.8 ± 1.4 | 29.8 ± 2.4 | 84.8 ± 0.3 | 72.1 ± 1.0 | 56.6 ± 0.9 |

## 5. Ablasi bootstrap CI 95% (selisih terhadap C0 + NMS 0.7, skala 0–1; * = bermakna)

| perbandingan | L0 | L1 | L2 | L3 | nyata_bersih | nyata_ringan | nyata_berat |
|:--|:--|:--|:--|:--|:--|:--|:--|
| NMS saja (IoU terbaik) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| NMS saja (Soft-NMS) | +0.004 [+0.003, +0.005] * | +0.010 [+0.007, +0.012] * | +0.005 [+0.004, +0.007] * | +0.000 [+0.000, +0.001] * | -0.009 [-0.014, -0.005] * | -0.028 [-0.036, -0.020] * | -0.013 [-0.026, -0.002] * |
| Augmentasi saja (C1) | +0.000 [-0.006, +0.005] | +0.005 [-0.006, +0.017] | +0.019 [+0.009, +0.030] * | +0.004 [+0.001, +0.007] * | +0.005 [-0.004, +0.014] | +0.013 [+0.003, +0.026] * | +0.009 [-0.015, +0.029] |
| Augmentasi saja (C2) | +0.002 [-0.004, +0.008] | +0.203 [+0.185, +0.222] * | +0.380 [+0.355, +0.409] * | +0.269 [+0.248, +0.296] * | +0.009 [-0.001, +0.017] | -0.004 [-0.018, +0.010] | +0.003 [-0.025, +0.024] |
| Gabungan (C1 + IoU terbaik) | +0.000 [-0.006, +0.005] | +0.005 [-0.006, +0.017] | +0.019 [+0.009, +0.030] * | +0.004 [+0.001, +0.007] * | +0.005 [-0.004, +0.014] | +0.013 [+0.003, +0.026] * | +0.009 [-0.015, +0.029] |
| Gabungan (C1 + Soft-NMS) | +0.004 [-0.002, +0.009] | +0.014 [+0.003, +0.026] * | +0.024 [+0.014, +0.035] * | +0.004 [+0.001, +0.008] * | -0.007 [-0.018, +0.004] | -0.019 [-0.032, -0.004] * | -0.006 [-0.034, +0.020] |
| Gabungan (C2 + IoU terbaik) | +0.002 [-0.004, +0.008] | +0.203 [+0.185, +0.222] * | +0.380 [+0.355, +0.409] * | +0.269 [+0.248, +0.296] * | +0.009 [-0.001, +0.017] | -0.004 [-0.018, +0.010] | +0.003 [-0.025, +0.024] |
| Gabungan (C2 + Soft-NMS) | +0.006 [+0.001, +0.012] * | +0.211 [+0.193, +0.231] * | +0.394 [+0.368, +0.423] * | +0.281 [+0.260, +0.309] * | +0.001 [-0.009, +0.011] | -0.033 [-0.048, -0.018] * | -0.016 [-0.046, +0.008] |

## 6. RQ3: transfer sintetis ke nyata

| perbandingan | Δ sintetis (L1-L3) | Δ nyata (ringan+berat) | arah sama | rasio transfer |
|:--|--:|--:|:--|:--|
| NMS saja (IoU terbaik) | 0.0000 | 0.0000 | ya | - |
| NMS saja (Soft-NMS) | 0.0051 | -0.0209 | tidak | -4.11 |
| Augmentasi saja (C1) | 0.0092 | 0.0110 | ya | 1.19 |
| Augmentasi saja (C2) | 0.2841 | -0.0009 | tidak | -0.003 |
| Gabungan (C1 + Soft-NMS) | 0.0142 | -0.0127 | tidak | -0.89 |
| Gabungan (C2 + Soft-NMS) | 0.2954 | -0.0243 | tidak | -0.08 |

## 7. Uji generator silang (870 target sama, NMS 0.7)

| generator | C0 | C1 | C2 | Δ C1 | Δ C2 | n_target |
|:--|--:|--:|--:|--:|--:|--:|
| paste (= training C2) | 0.231 | 0.241 | 0.514 | 0.010 | 0.284 | 870 |
| erase (= training C1) | 0.218 | 0.549 | 0.496 | 0.331 | 0.279 | 870 |
| gray (netral) | 0.167 | 0.174 | 0.412 | 0.007 | 0.245 | 870 |

| generator | level | C0 | C1 | C2 |
|:--|:--|--:|--:|--:|
| erase | L1 | 0.489 | 0.721 | 0.716 |
| erase | L2 | 0.152 | 0.578 | 0.521 |
| erase | L3 | 0.012 | 0.347 | 0.252 |
| gray | L1 | 0.431 | 0.446 | 0.624 |
| gray | L2 | 0.069 | 0.075 | 0.418 |
| gray | L3 | 0.001 | 0.001 | 0.195 |
| paste | L1 | 0.533 | 0.539 | 0.734 |
| paste | L2 | 0.143 | 0.163 | 0.523 |
| paste | L3 | 0.016 | 0.020 | 0.286 |

## 8. Evaluasi semua motor dalam gambar

| level | motor_lain | tersentuh >0% | diabaikan >10% | tertutup >50% |
|:--|--:|--:|--:|--:|
| L1 | 4816 | 71 | 26 | 0 |
| L2 | 4816 | 78 | 29 | 1 |
| L3 | 4816 | 141 | 57 | 2 |

| set | nms | C0 | C1 | C2 |
|:--|:--|--:|--:|--:|
| L0 | soft:0.3 | 0.903 | 0.906 | 0.904 |
| L0 | std:0.7 | 0.906 | 0.909 | 0.907 |
| L1 | soft:0.3 | 0.870 | 0.875 | 0.897 |
| L1 | std:0.7 | 0.874 | 0.878 | 0.899 |
| L2 | soft:0.3 | 0.812 | 0.820 | 0.882 |
| L2 | std:0.7 | 0.818 | 0.824 | 0.883 |
| L3 | soft:0.3 | 0.767 | 0.772 | 0.861 |
| L3 | std:0.7 | 0.772 | 0.777 | 0.862 |
| nyata_semua_citra | soft:0.3 | 0.901 | 0.904 | 0.902 |
| nyata_semua_citra | std:0.7 | 0.904 | 0.908 | 0.905 |
| test_penuh | soft:0.3 | 0.896 | 0.900 | 0.899 |
| test_penuh | std:0.7 | 0.900 | 0.903 | 0.902 |

## 9. Cek penempatan penutup

| set | generator | level | rentang | n | gagal_target | cov_min | cov_mean | cov_max | di_luar_rentang |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| synthetic_test | paste | L1 | 0.20-0.40 | 910 | 14 | 0.2000 | 0.2827 | 0.3995 | 0 |
| synthetic_test | paste | L2 | 0.40-0.60 | 910 | 14 | 0.4001 | 0.4841 | 0.5992 | 0 |
| synthetic_test | paste | L3 | 0.60-0.80 | 910 | 14 | 0.6000 | 0.6776 | 0.8000 | 0 |
| synthetic_val | paste | L1 | 0.20-0.40 | 355 | 5 | 0.2003 | 0.2800 | 0.3970 | 0 |
| synthetic_val | paste | L2 | 0.40-0.60 | 355 | 5 | 0.4021 | 0.4799 | 0.5999 | 0 |
| synthetic_val | paste | L3 | 0.60-0.80 | 355 | 5 | 0.6003 | 0.6749 | 0.8000 | 0 |
| synthetic_test | erase | L1 | 0.20-0.40 | 910 | 0 | 0.2000 | 0.2821 | 0.3999 | 0 |
| synthetic_test | erase | L2 | 0.40-0.60 | 906 | 4 | 0.4000 | 0.4749 | 0.5985 | 0 |
| synthetic_test | erase | L3 | 0.60-0.80 | 874 | 36 | 0.6000 | 0.6701 | 0.7976 | 0 |
| synthetic_test | gray | L1 | 0.20-0.40 | 910 | 0 | 0.2000 | 0.2821 | 0.3999 | 0 |
| synthetic_test | gray | L2 | 0.40-0.60 | 906 | 4 | 0.4000 | 0.4749 | 0.5985 | 0 |
| synthetic_test | gray | L3 | 0.60-0.80 | 874 | 36 | 0.6000 | 0.6701 | 0.7976 | 0 |

Augmentasi training: penutup gagal ditempatkan 0/4827 kotak.

## 10. Panel C0 | C1 | C2 (hard NMS 0.7, seed 0) — jumlah per kategori

Dari sel paket tim. Setiap set punya satu gambar panel berisi satu contoh acak (seed 0) per kategori, dengan gambar yang sama untuk C0, C1, C2.

| set | semua berhasil | hanya C2 berhasil | C1 & C2 berhasil, C0 gagal | C0 berhasil, C2 gagal | semua gagal |
|:--|--:|--:|--:|--:|--:|
| L0 | 736 | 9 | 20 | 30 | 84 |
| L1 | 439 | 150 | 72 | 21 | 178 |
| L2 | 132 | 348 | 59 | 9 | 313 |
| L3 | 18 | 308 | 22 | 5 | 550 |
| nyata_bersih | 269 | 4 | 7 | 5 | 21 |
| nyata_ringan | 205 | 2 | 12 | 20 | 37 |
| nyata_berat | 85 | 1 | 6 | 11 | 27 |

## 11. Pemeriksaan label dengan penilai AI

- Label pembanding 150 motor (50 per tingkat menurut Kukuh) **diisi AI (Claude)**, tanpa melihat label Kukuh, memakai aturan yang sama; ditinjau Ali tanpa perubahan.
- Kesepakatan 50,7%, Cohen κ = 0,283, κ berbobot kuadratik 0,493, 74 label berbeda.
- Ini **bukan** agreement antar-manusia. Label evaluasi tetap dari Kukuh.

| Kukuh \ AI | berat | bersih | buang | ringan | total |
|:--|--:|--:|--:|--:|--:|
| berat | 20 | 3 | 2 | 24 | 49 |
| bersih | 0 | 18 | 4 | 28 | 50 |
| ringan | 5 | 3 | 5 | 38 | 51 |
| total | 25 | 24 | 11 | 90 | 150 |

## 12. Ketahanan cahaya

| cfg (400 citra uji, 2348 motor) | AP50 | AP50-95 | P | R | F1 |
|:--|:--|:--|:--|:--|:--|
| C0 | 0.892 ± 0.005 | 0.681 ± 0.004 | 0.864 ± 0.006 | 0.807 ± 0.011 | 0.834 ± 0.003 |
| C1 | 0.895 ± 0.002 | 0.686 ± 0.002 | 0.858 ± 0.011 | 0.814 ± 0.008 | 0.835 ± 0.001 |
| C2 | 0.893 ± 0.004 | 0.682 ± 0.003 | 0.866 ± 0.011 | 0.807 ± 0.011 | 0.836 ± 0.006 |

| kondisi (AP50) | C0 | C0+cahaya | C1 | C2 | selisih C0+cahaya − C0 |
|:--|--:|--:|--:|--:|--:|
| malam (sim.) | 0.797 | 0.817 | 0.809 | 0.791 | 0.020 |
| sangat gelap | 0.844 | 0.858 | 0.849 | 0.842 | 0.014 |
| gelap | 0.878 | 0.885 | 0.882 | 0.878 | 0.007 |
| redup | 0.889 | 0.891 | 0.892 | 0.890 | 0.002 |
| normal | 0.892 | 0.894 | 0.895 | 0.893 | 0.002 |
| kontras rendah | 0.878 | 0.886 | 0.881 | 0.881 | 0.008 |
| terang | 0.876 | 0.881 | 0.882 | 0.878 | 0.005 |
| sangat terang | 0.751 | 0.807 | 0.769 | 0.760 | 0.056 |

| model | skor ketahanan |
|:--|--:|
| C0 | 0.947 |
| C0+cahaya | 0.962 |
| C1 | 0.952 |
| C2 | 0.947 |

## 13. Isi file hasil

- `SEMUA_HASIL_v6_e60_ulang.zip` (1023 MB): laporan, results, paket_akhir (12 best.pt + raw_preds + README), raw_preds, sampel penanda 2, log training
- `PAKET_TIM_v6_e60_ulang.zip` (571 MB, 107 file): `weights/` 9 best.pt, `raw_preds/` (termasuk test_penuh), `matching.pkl`, `matching_semua_motor.pkl`, `gensilang_raw_preds/` + `target_gensilang.json`, `hasil_csv/`, `label/`, `log_training/`, `panel_C0_C1_C2_hardNMS07/` (7 PNG + daftar), `occlusion_lib.py`, `README.txt`, `MANIFEST_sha256.csv`

## 14. Status catatan tim

- **9 best.pt:** ✅ ada di PAKET_TIM
- **Prediksi mentah + test penuh + matching:** ✅ ada
- **Prediksi generator silang + cara evaluasi:** ✅ ada (gensilang_raw_preds, README)
- **Bukti satu run:** ✅ MANIFEST_sha256.csv
- **Panel C0–C1–C2 hard NMS 0.7, berhasil + gagal:** ✅ 7 panel (L0–L3, nyata bersih/ringan/berat)
- **Log 60 epoch lengkap:** ✅ 9 run, tanpa resume
- **Label penanda manusia kedua:** ❌ tidak dilakukan; ditulis sebagai keterbatasan, pembanding AI dilaporkan terpisah

## 15. Link dataset

| Dataset | Link | Keterangan |
|:--|:--|:--|
| UVH-26 (Hugging Face) | https://huggingface.co/datasets/iisc-aim/UVH-26 | Dataset citra + anotasi; notebook memakai cabang `v1.0`, anotasi MV |
| Dataset card UVH-26 v1.0 | https://huggingface.co/datasets/iisc-aim/UVH-26/blob/v1.0/README.md | Definisi kategori (Two-wheeler = kategori 8), lisensi CC BY 4.0 |
| Paper UVH-26 (arXiv) | https://arxiv.org/abs/2511.02563 | Sharma et al., 2025, Indian Institute of Science |
| Label oklusi nyata (Kaggle) | https://www.kaggle.com/datasets/alirahmanbayanaka/manual-penanda2 | `real_tags (1).csv` (label Kukuh) + `label_penanda2 (1).csv` (pembanding AI). Hanya bisa dibuka orang lain bila dataset di-set Public atau dibagikan |

## Data

- UVH-26: https://huggingface.co/datasets/iisc-aim/UVH-26 (branch `v1.0`, majority-vote annotations, CC BY 4.0)
- Dataset card: https://huggingface.co/datasets/iisc-aim/UVH-26/blob/v1.0/README.md
- UVH-26 paper: https://arxiv.org/abs/2511.02563
- The class used is Two-wheeler (category 8: motorcycles and seated motor scooters with rider). Standing kick-scooters are not part of this class.

## Not in this repository

Model checkpoints (`best.pt`), raw predictions, matching files and comparison panels are large. They are produced by the final run in `PAKET_TIM_v6_e60_ulang.zip` (571 MB) and should be shared through GitHub Releases, Git LFS or a Kaggle dataset.

## Limitations

Natural-occlusion labels come from one human annotator. The 150-target comparison (κ = 0.283) was rated by an AI model and is not human inter-annotator agreement. Results cover one detector, one city, daytime images and targets of 32 px and larger.


