"""
occlusion_lib.py
Library untuk proyek "Occlusion-Aware Training and Post-Processing for Motorcycle Detection:
Does Synthetic Occlusion Transfer to Real Occlusion?"

Isi:
  A. Data      : unduh subset UVH-26, konversi ke YOLO 1 kelas, dedup, split          (FR-01..FR-05)
  B. Generator : pool penutup, penempatan penutup dengan tutupan terkontrol            (FR-06..FR-10)
  C. Set nyata : kandidat oklusi nyata (proksi), template penandaan, kappa              (FR-12..FR-14)
  D. Training  : augmentasi oklusi on-the-fly untuk YOLO11 (C1 erase, C2 paste)         (FR-15..FR-18)
  E. NMS       : NMS standar dan Soft-NMS Gaussian atas keluaran mentah                 (FR-19, FR-20)
  F. Evaluasi  : AP50, AP50:95, P, R dengan GT "ignore", bootstrap CI, analisis transfer (FR-21..FR-24)
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

MOTOR_CAT = 8            # UVH-26: 8 = Two-wheeler (motor + pengendara)
BICYCLE_CAT = 12         # sepeda: bukan motor, bukan sumber penutup
OCCLUDER_CATS = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13}  # kendaraan non-motor -> sumber penutup kontekstual
LEVELS = {"L1": (0.20, 0.40), "L2": (0.40, 0.60), "L3": (0.60, 0.80)}
HF_REVISION = "v1.0"     # cabang dataset UVH-26 yang dibahas laporannya; kunci versi agar data tidak berubah diam-diam


# ======================================================================================
# A. DATA
# ======================================================================================
def load_coco(path):
    with open(path) as f:
        coco = json.load(f)
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    imgs = {im["id"]: im for im in coco["images"]}
    return imgs, anns, coco.get("categories", [])


def inventory(imgs, anns):
    """Ringkasan untuk tabel inventaris (FR-01)."""
    n_motor = sum(1 for v in anns.values() for a in v if a["category_id"] == MOTOR_CAT)
    n_img_motor = sum(1 for v in anns.values() if any(a["category_id"] == MOTOR_CAT for a in v))
    return {"n_images": len(imgs), "n_images_with_motor": n_img_motor, "n_motor_boxes": n_motor}


def count_motor(anns, i):
    return sum(a["category_id"] == MOTOR_CAT for a in anns.get(i, []))


def pick_images(imgs, anns, n, seed, min_motor=1, max_motor=None):
    """Pilih n citra acak (seed) yang punya >= min_motor motor. max_motor: citra yang lebih padat DIKELUARKAN
    dari pool sebelum dipilih (saran aslab: kurangi instance lewat gambar, bukan lewat bounding box)."""
    ids = [i for i in imgs if count_motor(anns, i) >= min_motor and (max_motor is None or count_motor(anns, i) <= max_motor)]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return sorted(ids[:n])


def repo_path_index(repo_id, split_dir):
    """Peta nama file dasar -> path di repo HF (folder data/000, 001, ...)."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=HF_REVISION)
    idx = {}
    for f in files:
        if f.startswith(split_dir + "/") and f.lower().endswith((".png", ".jpg", ".jpeg")):
            idx[os.path.basename(f)] = f
    return idx


def _convert_one(args):
    """Unduh 1 citra, resize sisi terpanjang ke max_side, simpan JPG + label YOLO + meta semua kotak."""
    (repo_id, repo_file, im, ann_list, out_img, out_lbl, out_meta, max_side) = args
    if os.path.exists(out_img) and os.path.exists(out_meta):
        return "skip"
    from huggingface_hub import hf_hub_download
    import time
    try:
        for attempt in range(4):           # retry untuk 429 / koneksi putus
            try:
                p = hf_hub_download(repo_id, repo_file, repo_type="dataset", revision=HF_REVISION)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return "bad"
        h0, w0 = img.shape[:2]
        s = min(1.0, max_side / max(h0, w0))
        if s < 1.0:
            img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        cv2.imwrite(out_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes, lines = [], []
        for a in ann_list:
            x, y, bw, bh = a["bbox"]
            x1, y1 = max(0.0, x * s), max(0.0, y * s)
            x2, y2 = min(w, (x + bw) * s), min(h, (y + bh) * s)
            if x2 - x1 < 2 or y2 - y1 < 2:      # FR-05: buang kotak nol / di luar citra
                continue
            boxes.append({"cat": int(a["category_id"]), "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
            if a["category_id"] == MOTOR_CAT:
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        with open(out_lbl, "w") as f:
            f.write("\n".join(lines))
        with open(out_meta, "w") as f:
            json.dump({"source": repo_file, "w": w, "h": h, "boxes": boxes}, f)
        try:
            os.remove(os.path.realpath(p))   # hemat disk: hapus PNG mentah di cache HF
        except OSError:
            pass
        return "ok"
    except Exception as e:  # noqa
        return f"err:{e}"


def download_convert(repo_id, path_index, imgs, anns, ids, out_root, max_side=1280, workers=8):
    out_root = Path(out_root)
    for d in ("images", "labels", "meta"):
        (out_root / d).mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in ids:
        im = imgs[i]
        base = os.path.basename(im["file_name"])
        if base not in path_index:
            continue
        stem = Path(base).stem
        jobs.append((repo_id, path_index[base], im, anns.get(i, []),
                     str(out_root / "images" / f"{stem}.jpg"), str(out_root / "labels" / f"{stem}.txt"),
                     str(out_root / "meta" / f"{stem}.json"), max_side))
    stats = defaultdict(int)
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_convert_one, jobs)):
            stats[r.split(":")[0]] += 1
            if k % 250 == 0:
                print(f"  {k}/{len(jobs)} {dict(stats)}", flush=True)
    return dict(stats)


def load_meta(meta_dir, stem):
    with open(Path(meta_dir) / f"{stem}.json") as f:
        return json.load(f)


# ---------- FR-02: filter citra rusak / teraugmentasi (setara filter_dataset_images.py) ----------
def image_quality_flags(path, dark_thr=12, gray_thr=2.0, noise_thr=None):
    img = cv2.imread(str(path))
    flags = []
    if img is None:
        return ["unreadable"]
    if img.mean() < dark_thr:
        flags.append("black")
    b, g, r = [c.astype(np.float32) for c in cv2.split(img)]
    if np.abs(b - g).mean() < gray_thr and np.abs(g - r).mean() < gray_thr:
        flags.append("grayscale")
    if noise_thr is not None:
        lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if lap > noise_thr:
            flags.append("noise")
    return flags


# ---------- FR-03: dedup near-duplicate dengan dHash ----------
def dhash(path, size=8):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (img[:, 1:] > img[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_dup_groups(paths, max_hamming=6):
    """Union-find atas pasangan dHash dengan jarak Hamming <= max_hamming. O(n^2) cukup untuk ~5k citra."""
    hs = np.array([dhash(p) for p in paths], dtype=np.uint64)
    n = len(hs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        x = np.bitwise_xor(hs[i + 1:], hs[i])
        d = np.array([bin(int(v)).count("1") for v in x]) if len(x) else np.array([])
        for j in np.nonzero(d <= max_hamming)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[a] = b
    return [find(i) for i in range(n)], hs


def hamming_to_set(h, hs_ref, max_hamming=6):
    x = np.bitwise_xor(hs_ref, np.uint64(h))
    return any(bin(int(v)).count("1") <= max_hamming for v in x)


def group_split(stems, groups, val_frac, seed):
    """FR-04: split pada tingkat grup near-duplicate, seed tetap."""
    by_g = defaultdict(list)
    for s, g in zip(stems, groups):
        by_g[g].append(s)
    keys = sorted(by_g)
    random.Random(seed).shuffle(keys)
    n_val = int(round(val_frac * len(stems)))
    val, train = [], []
    for k in keys:
        (val if len(val) < n_val else train).extend(by_g[k])
    return sorted(train), sorted(val)


# ======================================================================================
# B. GENERATOR OKLUSI SINTETIS
# ======================================================================================
def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a, b):
    i = inter_area(a, b)
    u = box_area(a) + box_area(b) - i
    return i / u if u > 0 else 0.0


def covered_fraction(target, others, grid=32):
    """Fraksi luas target yang tertutup UNION kotak lain (raster grid agar irisan ganda tidak dihitung dua kali)."""
    x1, y1, x2, y2 = target
    if x2 <= x1 or y2 <= y1 or not others:
        return 0.0
    xs = np.linspace(x1, x2, grid, endpoint=False) + (x2 - x1) / grid / 2
    ys = np.linspace(y1, y2, grid, endpoint=False) + (y2 - y1) / grid / 2
    X, Y = np.meshgrid(xs, ys)
    m = np.zeros_like(X, dtype=bool)
    for o in others:
        m |= (X >= o[0]) & (X < o[2]) & (Y >= o[1]) & (Y < o[3])
    return float(m.mean())


def build_occluder_pool(meta_dir, stems, out_dir, min_side=40, max_motor_overlap=0.05, max_per_image=3, seed=0):
    """FR-06: potong kendaraan non-motor sebagai penutup. Potongan yang memuat motor dibuang."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    img_dir = Path(meta_dir).parent / "images"
    rows = []
    for stem in stems:
        meta = load_meta(meta_dir, stem)
        motors = [b["xyxy"] for b in meta["boxes"] if b["cat"] == MOTOR_CAT]
        cands = []
        for k, b in enumerate(meta["boxes"]):
            if b["cat"] not in OCCLUDER_CATS:
                continue
            x1, y1, x2, y2 = b["xyxy"]
            if min(x2 - x1, y2 - y1) < min_side:
                continue
            others = [o["xyxy"] for j, o in enumerate(meta["boxes"]) if j != k]
            if covered_fraction(b["xyxy"], others) > 0.10:      # penutup sendiri tidak boleh tertutup
                continue
            if motors and covered_fraction(b["xyxy"], motors) > max_motor_overlap:
                continue
            cands.append((k, b))
        rng.shuffle(cands)
        if not cands:
            continue
        img = cv2.imread(str(img_dir / f"{stem}.jpg"))
        for k, b in cands[:max_per_image]:
            x1, y1, x2, y2 = map(int, b["xyxy"])
            crop = img[y1:y2, x1:x2]
            oid = f"{stem}_{k}"
            cv2.imwrite(str(out_dir / f"{oid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"occluder_id": oid, "source_image": stem, "cat": b["cat"], "w": x2 - x1, "h": y2 - y1})
    with open(out_dir / "pool.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["occluder_id", "source_image", "cat", "w", "h"])
        wr.writeheader()
        wr.writerows(rows)
    return rows


class OccluderPool:
    def __init__(self, pool_dir):
        self.dir = Path(pool_dir)
        self.ids = sorted(p.stem for p in self.dir.glob("*.jpg"))
        self._cache = {}

    def __len__(self):
        return len(self.ids)

    def get(self, oid):
        if oid not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[oid] = cv2.imread(str(self.dir / f"{oid}.jpg"))
        return self._cache[oid]


def place_occluder(box, cov_range, aspect, img_w, img_h, rng, max_tries=50):
    """FR-07/FR-08: cari persegi penutup (xyxy, int) dengan tutupan = luas irisan / luas kotak motor
    yang jatuh di cov_range. Mengembalikan (rect, coverage) atau (None, None) jika gagal."""
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    A = bw * bh
    if A <= 0:
        return None, None
    for _ in range(max_tries):
        c = rng.uniform(*cov_range)
        S = c * A * rng.uniform(1.0, 1.6)            # penutup sedikit lebih besar dari irisan yang diinginkan
        ar = aspect * math.exp(rng.uniform(-0.25, 0.25))
        w = min(max(4.0, math.sqrt(S * ar)), 1.6 * bw + 1, img_w)
        h = min(max(4.0, S / w), 1.6 * bh + 1, img_h)
        cx = rng.uniform(bx1 - 0.3 * w, bx2 + 0.3 * w)
        cy = rng.uniform(by1 - 0.3 * h, by2 + 0.3 * h)
        x1 = int(round(min(max(0, cx - w / 2), img_w - w)))
        y1 = int(round(min(max(0, cy - h / 2), img_h - h)))
        rect = (x1, y1, int(round(x1 + w)), int(round(y1 + h)))
        cov = inter_area(rect, box) / A
        if cov_range[0] <= cov <= cov_range[1]:
            return rect, cov
    return None, None


def paint_occluder(img, rect, mode, crop=None, rng=None):
    """mode 'erase' = isi derau acak (random erasing, Zhong dkk.), 'paste' = tempel potongan kendaraan."""
    x1, y1, x2, y2 = rect
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return img
    if mode == "erase":
        rs = np.random.default_rng(rng.randrange(2**31) if rng else None)
        img[y1:y2, x1:x2] = rs.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    elif mode == "paste":
        img[y1:y2, x1:x2] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    elif mode == "gray":
        img[y1:y2, x1:x2] = 114
    return img


def occlude_box(img, box, cov_range, mode, pool=None, rng=None, occluder_id=None):
    """Tempel satu penutup pada satu kotak motor. Label tidak diubah (kotak penuh)."""
    rng = rng or random.Random()
    H, W = img.shape[:2]
    crop, oid = None, None
    if mode == "paste":
        oid = occluder_id or rng.choice(pool.ids)
        crop = pool.get(oid)
        aspect = crop.shape[1] / max(1, crop.shape[0])
    else:
        aspect = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))
    rect, cov = place_occluder(box, cov_range, aspect, W, H, rng)
    if rect is None:
        return img, None, None, None
    img = paint_occluder(img, rect, mode, crop, rng)
    return img, cov, oid, rect


def select_isolated_targets(meta, min_side=40, max_prior=0.10):
    """FR-09 (adaptasi): target sintetis = motor cukup besar yang hampir tidak tertutup apa pun,
    agar tingkat tutupan hanya berasal dari penutup buatan."""
    boxes = meta["boxes"]
    out = []
    for k, b in enumerate(boxes):
        if b["cat"] != MOTOR_CAT:
            continue
        x1, y1, x2, y2 = b["xyxy"]
        if min(x2 - x1, y2 - y1) < min_side:
            continue
        others = [o["xyxy"] for j, o in enumerate(boxes) if j != k and inter_area(o["xyxy"], b["xyxy"]) > 0]
        if covered_fraction(b["xyxy"], others) <= max_prior:
            out.append(k)
    return out


def build_synthetic_test(meta_dir, stems, pool_test, out_root, seed=0, mode="paste", min_side=40):
    """Set uji sintetis BERPASANGAN: target motor yang sama muncul di L0 (bersih), L1, L2, L3.
    Penutup yang sama dipakai di semua level untuk satu target; hanya ukuran/posisinya yang berubah.
    Tulis citra + label YOLO per level dan synthetic_log.csv (FR-10)."""
    out_root = Path(out_root)
    img_dir = Path(meta_dir).parent / "images"
    rng = random.Random(seed)
    log = []
    for lv in ["L0", *LEVELS]:
        (out_root / lv / "images").mkdir(parents=True, exist_ok=True)
        (out_root / lv / "labels").mkdir(parents=True, exist_ok=True)
    targets = {}
    skipped = 0
    for stem in stems:
        meta = load_meta(meta_dir, stem)
        cand = select_isolated_targets(meta, min_side=min_side)
        if not cand:
            continue
        k = rng.choice(cand)
        box = meta["boxes"][k]["xyxy"]
        base = cv2.imread(str(img_dir / f"{stem}.jpg"))
        oid = rng.choice(pool_test.ids) if mode == "paste" else None
        results = {}
        for lv, rr in LEVELS.items():
            r = random.Random(f"{seed}-{stem}-{lv}")
            img, cov, oid2, rect = occlude_box(base.copy(), box, rr, mode, pool_test, r, occluder_id=oid)
            if cov is None:
                break
            results[lv] = (img, cov, oid2, rect)
        if len(results) != len(LEVELS):       # FR-08: gagal 50 percobaan -> lewati dan catat
            skipped += 1
            log.append({"file": "", "source_image": stem, "level": "skip", "coverage": "", "occluder_id": oid or "",
                        "seed": seed, "target_idx": k, "target_xyxy": json.dumps(box), "rect": ""})
            continue
        lbl_src = Path(meta_dir).parent / "labels" / f"{stem}.txt"
        for lv in ["L0", *LEVELS]:
            fn = f"{stem}_{lv}.jpg"
            if lv == "L0":
                img, cov, oid2, rect = base, 0.0, "", None
            else:
                img, cov, oid2, rect = results[lv]
            cv2.imwrite(str(out_root / lv / "images" / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            os.link(lbl_src, out_root / lv / "labels" / f"{stem}_{lv}.txt") if not (out_root / lv / "labels" / f"{stem}_{lv}.txt").exists() else None
            log.append({"file": fn, "source_image": stem, "level": lv, "coverage": round(cov, 4),
                        "occluder_id": oid2 or "", "seed": seed, "target_idx": k,
                        "target_xyxy": json.dumps(box), "rect": json.dumps(rect) if rect else ""})
        targets[stem] = k
    with open(out_root / "synthetic_log.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        wr.writeheader()
        wr.writerows(log)
    print(f"target sintetis: {len(targets)}, dilewati: {skipped}")
    return targets


# ======================================================================================
# C. SET OKLUSI NYATA
# ======================================================================================
def real_occlusion_proxy(meta, front_margin=4):
    """FR-12: proksi oklusi nyata untuk kamera CCTV yang tinggi.
    Objek yang tepi bawahnya lebih rendah di citra (y2 lebih besar) dianggap lebih dekat ke kamera,
    jadi bisa menutupi. Proksi = fraksi luas motor yang tertutup union kotak 'di depan'."""
    boxes = meta["boxes"]
    out = []
    for k, b in enumerate(boxes):
        if b["cat"] != MOTOR_CAT:
            continue
        front = [o["xyxy"] for j, o in enumerate(boxes)
                 if j != k and o["xyxy"][3] > b["xyxy"][3] + front_margin and inter_area(o["xyxy"], b["xyxy"]) > 0]
        out.append((k, covered_fraction(b["xyxy"], front)))
    return out


def export_real_candidates(meta_dir, stems, out_dir, min_side=32, min_proxy=0.15, clean_max=0.02, seed=0):
    """Ekspor kandidat oklusi nyata (berperingkat) + target bersih acuan + template real_tags.csv.
    Tiap kandidat juga disimpan sebagai potongan konteks dengan kotak target digambar, untuk ditandai manual."""
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    img_dir = Path(meta_dir).parent / "images"
    rows = []
    rng = random.Random(seed)
    for stem in stems:
        meta = load_meta(meta_dir, stem)
        for k, p in real_occlusion_proxy(meta):
            x1, y1, x2, y2 = meta["boxes"][k]["xyxy"]
            if min(x2 - x1, y2 - y1) < min_side:
                continue
            if p >= min_proxy:
                rows.append({"stem": stem, "target_idx": k, "proxy": round(p, 3), "kind": "kandidat"})
            elif p <= clean_max and rng.random() < 0.15:
                rows.append({"stem": stem, "target_idx": k, "proxy": round(p, 3), "kind": "bersih"})
    rows.sort(key=lambda r: (r["kind"] != "kandidat", -r["proxy"]))
    for r in rows:
        if r["kind"] != "kandidat":
            continue
        meta = load_meta(meta_dir, r["stem"])
        img = cv2.imread(str(img_dir / f"{r['stem']}.jpg"))
        x1, y1, x2, y2 = map(int, meta["boxes"][r["target_idx"]]["xyxy"])
        pad = max(x2 - x1, y2 - y1)
        X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
        X2, Y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        crop = vis[Y1:Y2, X1:X2]
        s = 320 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s) if s < 1 or s > 1.5 else crop
        cv2.imwrite(str(out_dir / "crops" / f"{r['stem']}_{r['target_idx']}.jpg"), crop)
    with open(out_dir / "real_candidates.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["stem", "target_idx", "proxy", "kind"])
        wr.writeheader()
        wr.writerows(rows)
    # template penandaan (FR-13): satu baris per motor per penanda
    with open(out_dir / "real_tags_TEMPLATE.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "target_idx", "annotator", "level", "n_motor", "penutup", "kotak_ok", "catatan"])
        for r in rows:
            if r["kind"] == "kandidat":
                wr.writerow([f"{r['stem']}_{r['target_idx']}.jpg", r["target_idx"], "", "", "", "", "", ""])
    return rows


def cohen_kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_real_levels(tags_csv, candidates_csv, use_proxy_if_missing=True, light_max=0.5):
    """Gabungkan tag manual. Aturan: 'berat' = lebih dari separuh motor tidak terlihat.
    Bila sebuah target punya baris dengan annotator 'konsensus', baris itu dipakai. Jika tidak, dipakai
    label penanda pertama di berkas (karena itu simpan satu baris konsensus per target di real_tags.csv
    dan label mentah tiap penanda di real_tags_raw.csv untuk kappa).
    Tanpa tag manual, jatuh ke proksi (HANYA untuk uji pipeline, tandai sebagai 'proxy')."""
    levels = {}
    src = "manual"
    if tags_csv and Path(tags_csv).exists():
        by = defaultdict(list)
        with open(tags_csv) as f:
            for r in csv.DictReader(f):
                if r["level"].strip():
                    stem_k = r["file"].rsplit(".", 1)[0]
                    by[stem_k].append(((r.get("annotator") or "").strip().lower(), r["level"].strip().lower()))
        for key, labs in by.items():
            stem, k = key.rsplit("_", 1)
            cons = [lv for an, lv in labs if an == "konsensus"]
            levels[(stem, int(k))] = cons[0] if cons else labs[0][1]
    if levels:
        # Label manusia ada: HANYA label manusia yang dipakai. Sampel 'bersih' otomatis tidak boleh menimpa label manual.
        return levels, src
    if use_proxy_if_missing:
        src = "proxy"
        with open(candidates_csv) as f:
            for r in csv.DictReader(f):
                if r["kind"] == "kandidat":
                    levels[(r["stem"], int(r["target_idx"]))] = "ringan" if float(r["proxy"]) <= light_max else "berat"
    with open(candidates_csv) as f:
        for r in csv.DictReader(f):
            if r["kind"] == "bersih":
                levels[(r["stem"], int(r["target_idx"]))] = "bersih"
    return levels, src


# ======================================================================================
# D. TRAINING: augmentasi oklusi on-the-fly untuk Ultralytics YOLO11
# ======================================================================================
class OcclusionAug:
    """Transform Ultralytics: dipasang tepat sebelum Format, setelah mosaic/perspective/flip.
    Untuk tiap citra, dengan peluang p, pilih 1 kotak motor dan tutup dengan tutupan ~ U(cov_range).
    Label TIDAK diubah (kotak penuh tetap menjadi target)."""

    def __init__(self, mode, p=0.5, cov_range=(0.10, 0.80), pool_dir=None, min_side=8, max_targets=1, seed=0):
        assert mode in ("erase", "paste")
        self.mode, self.p, self.cov_range = mode, p, cov_range
        self.pool_dir, self.min_side, self.max_targets = pool_dir, min_side, max_targets
        self.seed = seed
        self._pool = None
        self._rng, self._rng_key = None, None

    @property
    def rng(self):
        """RNG per worker dataloader. Di worker, torch.initial_seed() = seed dasar iterator + id worker,
        sehingga tiap worker berbeda, deterministik untuk seed training yang sama, dan berubah saat
        loader dibangun ulang (close_mosaic). Tanpa ini, semua worker hasil fork memakai urutan acak yang sama."""
        try:
            import torch
            wi = torch.utils.data.get_worker_info()
            key = torch.initial_seed() if wi is not None else ("main", self.seed)
        except Exception:
            key = ("main", self.seed)
        if self._rng is None or key != self._rng_key:
            self._rng, self._rng_key = random.Random(str(key)), key
        return self._rng

    @property
    def pool(self):
        if self._pool is None and self.pool_dir:
            self._pool = OccluderPool(self.pool_dir)    # dimuat malas per worker dataloader
        return self._pool

    def __call__(self, labels):
        if self.rng.random() >= self.p:
            return labels
        inst = labels.get("instances")
        if inst is None or len(inst) == 0:
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        inst.convert_bbox("xyxy")
        boxes = inst.bboxes.copy()
        if inst.normalized:
            boxes = boxes * np.array([w, h, w, h], dtype=boxes.dtype)
        ok = [i for i, b in enumerate(boxes) if min(b[2] - b[0], b[3] - b[1]) >= self.min_side]
        if not ok:
            return labels
        self.rng.shuffle(ok)
        img = np.ascontiguousarray(img)
        for i in ok[: self.max_targets]:
            img, *_ = occlude_box(img, tuple(float(v) for v in boxes[i]), self.cov_range, self.mode, self.pool, self.rng)
        labels["img"] = img
        return labels


def make_trainer(mode=None, p=0.5, cov_range=(0.10, 0.80), pool_dir=None, seed=0):
    """Kembalikan kelas trainer. mode=None -> baseline C0 (trainer standar)."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    if mode is None:
        return DetectionTrainer

    aug = OcclusionAug(mode, p=p, cov_range=cov_range, pool_dir=pool_dir, seed=seed)

    class OccTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                attach_aug(ds, aug)
            return ds

    return OccTrainer


def attach_aug(ds, aug):
    """Sisipkan aug sebelum Format, dan pastikan tetap terpasang setelah close_mosaic membangun ulang transform."""
    def _insert(t):
        if not any(isinstance(x, OcclusionAug) for x in t.transforms):
            t.transforms.insert(len(t.transforms) - 1, aug)
        return t

    orig = ds.build_transforms
    ds.build_transforms = lambda hyp=None: _insert(orig(hyp))
    _insert(ds.transforms)
    return ds


# ======================================================================================
# E. POST-PROCESSING NMS
# ======================================================================================
def _iou_one_to_many(b, bs):
    xx1 = np.maximum(b[0], bs[:, 0]); yy1 = np.maximum(b[1], bs[:, 1])
    xx2 = np.minimum(b[2], bs[:, 2]); yy2 = np.minimum(b[3], bs[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1]); A = (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])
    return inter / np.maximum(a + A - inter, 1e-9)


def hard_nms(boxes, scores, iou_thr=0.7, max_det=300):
    if len(boxes) == 0:
        return boxes, scores
    try:
        import torch, torchvision
        keep = torchvision.ops.nms(torch.as_tensor(boxes, dtype=torch.float32),
                                   torch.as_tensor(scores, dtype=torch.float32), iou_thr).numpy()
    except Exception:
        order = scores.argsort()[::-1]; keep = []
        while len(order):
            i = order[0]; keep.append(i)
            order = order[1:][_iou_one_to_many(boxes[i], boxes[order[1:]]) <= iou_thr]
        keep = np.array(keep)
    keep = keep[:max_det]
    return boxes[keep], scores[keep]


def soft_nms_gaussian(boxes, scores, sigma=0.5, score_thr=0.001, max_det=300, pre_topk=1000):
    """Soft-NMS Gaussian (Bodla dkk., 2017): s_i <- s_i * exp(-IoU^2 / sigma)."""
    if len(boxes) == 0:
        return boxes, scores
    o = scores.argsort()[::-1][:pre_topk]
    B, S = boxes[o].astype(np.float64).copy(), scores[o].astype(np.float64).copy()
    keep_b, keep_s = [], []
    while len(S) and len(keep_b) < max_det:
        i = int(S.argmax())
        if S[i] < score_thr:
            break
        keep_b.append(B[i]); keep_s.append(S[i])
        B = np.delete(B, i, 0); S = np.delete(S, i)
        if len(S):
            S *= np.exp(-(_iou_one_to_many(keep_b[-1], B) ** 2) / sigma)
    return np.array(keep_b).reshape(-1, 4), np.array(keep_s)


def postprocess(raw, variant):
    """variant: ('std', iou) | ('soft', sigma)."""
    out = {}
    for k, (b, s) in raw.items():
        if variant[0] == "std":
            out[k] = hard_nms(b, s, variant[1])
        else:
            out[k] = soft_nms_gaussian(b, s, sigma=variant[1])
    return out


def predict_raw(model, paths, imgsz=640, batch=16, conf=0.001, device=None):
    """Keluaran 'mentah': NMS bawaan dibuat hampir tidak aktif (iou=0.99) agar semua varian NMS
    diterapkan dari kandidat yang sama."""
    raw = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = model.predict(chunk, imgsz=imgsz, conf=conf, iou=0.99, max_det=1000, verbose=False,
                            device=device, agnostic_nms=True)
        for p, r in zip(chunk, res):
            raw[Path(p).stem] = (r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32))
    return raw


# ======================================================================================
# F. EVALUASI
# ======================================================================================
IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def match_image(pb, ps, gt, care):
    """Pencocokan gaya COCO per citra dengan GT 'ignore'.
    Prediksi yang cocok dengan GT ignore (IoU >= t) tidak dihitung TP maupun FP.
    Return: scores (n_pred,), tp (n_pred, n_thr) dengan nilai 1=TP, 0=FP, -1=diabaikan; n_care."""
    n_thr = len(IOU_THRS)
    o = np.argsort(-ps)
    pb, ps = pb[o], ps[o]
    tp = np.zeros((len(ps), n_thr), dtype=np.int8)
    care = np.asarray(care, dtype=bool)
    if len(gt) == 0:
        return ps, tp, int(care.sum())
    gt = np.asarray(gt, dtype=np.float64)
    ious = np.stack([_iou_one_to_many(b, gt) for b in pb]) if len(pb) else np.zeros((0, len(gt)))
    for t, thr in enumerate(IOU_THRS):
        used = np.zeros(len(gt), dtype=bool)
        for i in range(len(pb)):
            cand = np.where(care & ~used & (ious[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                used[j] = True
                tp[i, t] = 1
            elif np.any(~care & (ious[i] >= thr)):
                tp[i, t] = -1
    return ps, tp, int(care.sum())


def ap_from_matches(scores, tp, n_gt):
    """AP 101 titik (COCO) per ambang IoU. tp: (n, n_thr) {1,0,-1}."""
    if n_gt == 0:
        return np.full(tp.shape[1], np.nan)
    o = np.argsort(-scores, kind="mergesort")
    tp = tp[o]
    aps = []
    rs = np.linspace(0, 1, 101)
    for t in range(tp.shape[1]):
        v = tp[:, t]
        v = v[v >= 0]
        if len(v) == 0:
            aps.append(0.0); continue
        ctp = np.cumsum(v == 1); cfp = np.cumsum(v == 0)
        rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1)
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec, rs, side="left")
        aps.append(float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx])))
    return np.array(aps)


class EvalSet:
    """Kumpulan GT untuk satu set evaluasi. gts[stem] = (boxes xyxy, care mask)."""

    def __init__(self, name, gts):
        self.name, self.gts = name, gts

    def match(self, preds):
        per = {}
        for stem, (gt, care) in self.gts.items():
            pb, ps = preds.get(stem, (np.zeros((0, 4)), np.zeros(0)))
            per[stem] = match_image(pb, ps, gt, care)
        return per


def summarize(per, stems=None, conf_thr=0.25):
    stems = list(per) if stems is None else stems
    S = np.concatenate([per[s][0] for s in stems]) if stems else np.zeros(0)
    T = np.concatenate([per[s][1] for s in stems]) if stems else np.zeros((0, len(IOU_THRS)), np.int8)
    n = sum(per[s][2] for s in stems)
    aps = ap_from_matches(S, T, n)
    m = (S >= conf_thr) & (T[:, 0] >= 0)
    tpc = int((T[m, 0] == 1).sum()); fpc = int((T[m, 0] == 0).sum())
    return {"AP50": float(aps[0]), "AP50_95": float(np.nanmean(aps)),
            "P": tpc / max(tpc + fpc, 1), "R": tpc / max(n, 1), "n_gt": n}


def best_f1_conf(per):
    """Ambang confidence dengan F1 maksimum pada IoU 0.5 — dipilih di val (FR-17)."""
    S = np.concatenate([v[0] for v in per.values()]); T = np.concatenate([v[1] for v in per.values()])
    n = sum(v[2] for v in per.values())
    keep = T[:, 0] >= 0
    S, t = S[keep], T[keep, 0]
    o = np.argsort(-S); S, t = S[o], t[o]
    ctp = np.cumsum(t == 1); cfp = np.cumsum(t == 0)
    p = ctp / np.maximum(ctp + cfp, 1); r = ctp / max(n, 1)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    i = int(np.argmax(f1))
    return float(S[i]), float(f1[i])


def bootstrap_delta(per_a_list, per_b_list, stems, metric="AP50", n_boot=1000, seed=0, conf_a=0.25, conf_b=0.25):
    """Bootstrap 95% CI untuk selisih (B - A) dengan resampling citra berpasangan.
    per_*_list = daftar hasil match per seed; metrik = rata-rata antar seed."""
    rng = np.random.default_rng(seed)
    stems = list(stems)
    def val(per_list, ss, c):
        return float(np.mean([summarize(p, ss, c)[metric] for p in per_list]))
    base = val(per_b_list, stems, conf_b) - val(per_a_list, stems, conf_a)
    ds = []
    for _ in range(n_boot):
        ss = [stems[i] for i in rng.integers(0, len(stems), len(stems))]
        ds.append(val(per_b_list, ss, conf_b) - val(per_a_list, ss, conf_a))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return base, float(lo), float(hi)


def gts_from_yolo(label_dir, img_dir, stems):
    """GT biasa (semua motor = care) dari label YOLO."""
    out = {}
    for s in stems:
        im = cv2.imread(str(Path(img_dir) / f"{s}.jpg"))
        h, w = im.shape[:2]
        boxes = []
        p = Path(label_dir) / f"{s}.txt"
        if p.exists():
            for line in p.read_text().split("\n"):
                if line.strip():
                    _, cx, cy, bw, bh = map(float, line.split())
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        out[s] = (np.array(boxes).reshape(-1, 4), np.ones(len(boxes), bool))
    return out


def gts_target_only(meta_dir, targets, name_fmt="{stem}"):
    """GT dengan hanya target yang 'care'; motor lain = ignore. targets: {stem: target_idx} atau {(stem,k): ...}."""
    out = {}
    for stem, k in targets:
        meta = load_meta(meta_dir, stem)
        motors = [(j, b["xyxy"]) for j, b in enumerate(meta["boxes"]) if b["cat"] == MOTOR_CAT]
        key = name_fmt.format(stem=stem)
        if key in out:     # beberapa target di citra yang sama (set nyata)
            gt, care = out[key]
            care = care | np.array([j == k for j, _ in motors])
            out[key] = (gt, care)
        else:
            out[key] = (np.array([b for _, b in motors]).reshape(-1, 4), np.array([j == k for j, _ in motors]))
    return out
