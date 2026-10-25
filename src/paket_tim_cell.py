# ===== PAKET UNTUK TIM: panel C0|C1|C2 (hard NMS 0.7) + paket evaluasi yang bisa ditelusuri =====
# Jalankan di SESI YANG SAMA setelah Run all selesai (butuh MATCH, SETS, RAW, _gambar dari sel-sel sebelumnya).
import random, pickle, hashlib, json, zipfile, time, platform
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import ultralytics

TIM = W / f"paket_tim{TAG}"; TIM.mkdir(exist_ok=True)
S0 = SEEDS[0]; V7 = ("std", 0.7)
RUN_UTAMA = [f"{c}_s{s}" for c in ("C0", "C1", "C2") for s in SEEDS]

# ---------- A. PANEL C0 | C1 | C2, semua hard NMS 0.7, seed 0, gambar sama ----------
PANEL = TIM / "panel_C0_C1_C2_hardNMS07"; PANEL.mkdir(exist_ok=True)
def _hit(run, v, k, key):
    sc, tp, n = MATCH[(run, v, k)][key]; thr = CONF.get((run, v), 0.25)
    return int(((sc >= thr) & (tp[:, 0] == 1)).sum()), n
def _kategori(k, key):
    r = {c: _hit(f"{c}_s{S0}", V7, k, key) for c in ("C0", "C1", "C2")}
    ok = {c: h == n and n > 0 for c, (h, n) in r.items()}; no = {c: h == 0 for c, (h, n) in r.items()}
    if all(ok.values()): return "semua_berhasil"
    if ok["C2"] and no["C0"] and no["C1"]: return "hanya_C2_berhasil"
    if ok["C2"] and no["C0"]: return "C1_C2_berhasil_C0_gagal"
    if ok["C0"] and no["C2"]: return "C0_berhasil_C2_gagal"
    if all(no.values()): return "semua_gagal"
    return "campuran"
URUT_KAT = ["semua_berhasil", "hanya_C2_berhasil", "C1_C2_berhasil_C0_gagal", "C0_berhasil_C2_gagal", "semua_gagal"]
daftar, jumlah = [], []
for k in [x for x in ["L0", "L1", "L2", "L3", "nyata_bersih", "nyata_ringan", "nyata_berat"] if x in SETS]:
    kat = {}
    for key in sorted(SETS[k][0].gts):
        kat.setdefault(_kategori(k, key), []).append(key)
    jumlah.append(dict(set=k, **{c: len(kat.get(c, [])) for c in URUT_KAT + ["campuran"]}))
    pilih = [(c, random.Random(0).choice(kat[c])) for c in URUT_KAT if kat.get(c)]     # 1 contoh acak per kategori
    fig, axs = plt.subplots(len(pilih), 3, figsize=(11, 3.7 * len(pilih)), squeeze=False)
    for i, (c, key) in enumerate(pilih):
        for j, cfg in enumerate(("C0", "C1", "C2")):
            h, n = _hit(f"{cfg}_s{S0}", V7, k, key)
            axs[i, j].imshow(_gambar(f"{cfg}_s{S0}", V7, k, key)); axs[i, j].axis("off")
            axs[i, j].set_title(f"{cfg} | hard NMS 0.7 | {h}/{n} terdeteksi\n[{c.replace('_', ' ')}] {key}", fontsize=8)
            daftar.append(dict(set=k, kategori=c, gambar=key, cfg=cfg, seed=S0, nms="std:0.7", terdeteksi=h, target=n))
    plt.suptitle(f"{k}: C0 | C1 | C2, seed {S0}, hard NMS 0.7 (gambar sama per baris)\n"
                 "hijau = target, abu-abu = diabaikan, merah = prediksi (conf >= ambang validasi)", fontsize=10)
    plt.tight_layout(); plt.savefig(PANEL / f"panel_{k}.png", dpi=130); plt.close()
    print("panel:", k, {c: len(kat.get(c, [])) for c in URUT_KAT})
pd.DataFrame(daftar).to_csv(PANEL / "panel_daftar_contoh.csv", index=False)
pd.DataFrame(jumlah).to_csv(PANEL / "panel_jumlah_per_kategori.csv", index=False)

# ---------- B. Prediksi mentah generator silang (erase / gray) agar angka 11 bisa ditelusuri ----------
SIMPAN_PRED_GENSILANG = True        # ~15 menit GPU; False = lewati
GSR = TIM / "gensilang_raw_preds"; GSR.mkdir(exist_ok=True)
if SIMPAN_PRED_GENSILANG and "tg_sama" in globals():
    _t = time.time()
    for run in RUN_UTAMA:
        f = GSR / f"{run}.pkl"
        if f.exists(): continue
        model = YOLO(RUNS / run / "weights" / "best.pt"); out = {}
        for gen in ["erase", "gray"]:
            for lv in ["L1", "L2", "L3"]:
                paths = [str(DATA / f"synthetic_test_{gen}" / lv / "images" / f"{s}_{lv}.jpg") for s, _ in tg_sama]
                out[(gen, lv)] = L.predict_raw(model, paths, imgsz=IMGSZ, device=0)
        pickle.dump(out, open(f, "wb")); print(f"  gensilang raw {run} ({time.time() - _t:.0f} detik)")
    json.dump(dict(target_sama=[[s, int(k)] for s, k in tg_sama], dikeluarkan=sorted(GAGAL_SEMUA)),
              open(GSR / "target_gensilang.json", "w"))
if "MATCH_SEMUA" in globals():
    pickle.dump(dict(MATCH_SEMUA=MATCH_SEMUA, VAR_SEMUA=VAR_SEMUA, BATAS_IGN=BATAS_IGN), open(TIM / "matching_semua_motor.pkl", "wb"))

# ---------- C. README ----------
README = f"""PAKET UNTUK TIM — run {EXP_TAG} (satu run yang sama untuk semua isi paket)
dibuat: {time.strftime('%Y-%m-%d %H:%M')} | torch {torch.__version__} | ultralytics {ultralytics.__version__} | python {platform.python_version()}
model : YOLO11s, {EPOCHS} epoch, seed {SEEDS}, deterministic=True, imgsz {IMGSZ}

ISI
  weights/<C0|C1|C2>_s<seed>/best.pt   checkpoint 9 run utama (tanpa training ulang)
  raw_preds/<run>.pkl                  prediksi mentah SEBELUM NMS kustom (conf>=0.001, NMS bawaan iou=0.99, max_det=1000)
                                       kunci: val, valsyn_L0-L3, L0-L3, nyata_bersih/ringan/berat, test_penuh
  matching.pkl                         MATCH/CONF/BEST: pencocokan per citra untuk tabel utama, ablasi, bootstrap
  matching_semua_motor.pkl             MATCH_SEMUA: evaluasi semua motor (12b), termasuk test_penuh
  gensilang_raw_preds/<run>.pkl        prediksi mentah uji generator silang, kunci (generator, level)
  gensilang_raw_preds/target_gensilang.json   870 target yang sama untuk paste/erase/gray
  hasil_csv/                           semua tabel hasil (utama, ablasi, RQ3, generator silang, semua motor, cahaya, dll.)
  label/                               label final oklusi nyata + manifest sha256
  log_training/<run>/results.csv       log per epoch (60 baris) + args.yaml
  panel_C0_C1_C2_hardNMS07/            panel C0|C1|C2, seed {S0}, hard NMS 0.7, contoh berhasil & gagal
  occlusion_lib.py                     pustaka evaluasi
  MANIFEST_sha256.csv                  sidik jari semua file (bukti satu run)

CARA EVALUASI SINGKAT (tanpa GPU)
  import pickle, occlusion_lib as L
  M = pickle.load(open('matching.pkl','rb'))['MATCH']
  L.summarize(M[('C2_s0', ('std', 0.7), 'L2')])['AP50']          # satu sel tabel utama (seed 0)
  # NMS lain dari prediksi mentah:
  raw  = pickle.load(open('raw_preds/C2_s0.pkl','rb'))['L2']
  pred = L.postprocess(raw, ('soft', 0.3))                       # lalu L.EvalSet(...).match(pred)
  # semua motor / test penuh:
  MS = pickle.load(open('matching_semua_motor.pkl','rb'))['MATCH_SEMUA']
  L.summarize(MS[('C0_s0', ('std', 0.7), 'test_penuh')])['AP50']
Tabel paper = rata-rata 3 seed dari angka per-run di atas (lihat hasil_csv/).
"""
(TIM / "README.txt").write_text(README)

# ---------- D. ZIP + manifest sha256 ----------
isi = []
for r in RUN_UTAMA:
    isi += [(RUNS / r / "weights" / "best.pt", f"weights/{r}/best.pt"),
            (RUNS / r / "results.csv", f"log_training/{r}/results.csv"), (RUNS / r / "args.yaml", f"log_training/{r}/args.yaml")]
    isi.append((RAW / f"{r}.pkl", f"raw_preds/{r}.pkl"))
isi += [(RES / "ekspor" / "matching.pkl", "matching.pkl"), (Path(LIB_PATH), "occlusion_lib.py"),
        (REAL_TAGS_CSV, "label/real_tags.csv"), (RES / "real_labels_manifest.json", "label/real_labels_manifest.json"),
        (RES / "real_labels_final.csv", "label/real_labels_final.csv")]
isi += [(p, f"hasil_csv/{p.name}") for p in sorted(RES.glob("*.csv")) + sorted(RES.glob("*.json"))]
isi += [(p, p.relative_to(TIM).as_posix()) for p in sorted(TIM.rglob("*")) if p.is_file()]
def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
ZT = W / f"PAKET_TIM{TAG}.zip"; man = []
with zipfile.ZipFile(ZT, "w", zipfile.ZIP_DEFLATED) as z:
    for src, dst in isi:
        if Path(src).exists():
            z.write(src, dst); man.append(dict(file=dst, bytes=Path(src).stat().st_size, sha256=_sha(src)))
    z.writestr("MANIFEST_sha256.csv", pd.DataFrame(man).to_csv(index=False))
print(f"\nSELESAI: {ZT} ({ZT.stat().st_size / 1e6:.0f} MB, {len(man)} file)")
print("cek: best.pt =", sum(m['file'].endswith('best.pt') for m in man), "| raw_preds =", sum(m['file'].startswith('raw_preds/') for m in man),
      "| panel =", sum(m['file'].endswith('.png') for m in man))

# ---------- E. Link unduh ----------
from IPython.display import FileLink, display
display(FileLink(str(ZT.relative_to(W))))
