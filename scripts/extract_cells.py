#!/usr/bin/env python3
"""
extract_cells.py — ekstraksi sel tunggal dari citra lapang pandang apusan darah.

Dibuat untuk menyamakan granularitas dataset Taleqani (banyak sel per citra)
dengan C-NMC dan ALL-IDB (satu sel per citra).

Pipeline:
  blur -> HSV -> Otsu pada kanal saturasi -> (opsional) buang piksel terlalu
  terang (eritrosit) -> morfologi -> isi lubang -> (opsional) watershed untuk
  memisahkan sel bersentuhan -> connected component -> filter luas & soliditas
  -> crop bujur sangkar berpusat centroid -> simpan + laporan CSV

PENTING — penamaan keluaran:
  <PREFIX>_<kelas>_<id-lapang-pandang>_cell<NNN>.png
  ID lapang pandang WAJIB ikut terbawa, karena pembagian data harus dilakukan
  di tingkat lapang pandang: semua sel dari satu lapang harus masuk ke sisi
  yang sama (adaptasi ATAU tes), tidak boleh terpecah.

Dua mode pemakaian:
  1) --scan   : tidak menulis crop apa pun; hanya melaporkan sebaran luas objek
                supaya Anda bisa memilih --min-area / --max-area yang tepat.
  2) (default): ekstraksi penuh + laporan CSV.

Contoh:
  python extract_cells.py --input data/Taleqani/Original --scan --sample 40
  python extract_cells.py --input data/Taleqani/Original \\
      --output data/Taleqani/cells --min-area 1200 --max-area 60000 \\
      --qc-dir data/Taleqani/qc --split-touching
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, segmentation
from skimage.feature import peak_local_max

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP'}


# ---------------------------------------------------------------------------
# Segmentasi
# ---------------------------------------------------------------------------
def segment_nuclei(bgr, blur=5, v_max_pct=None, open_r=3, close_r=5,
                   min_area=500, split_touching=False, min_distance=15):
    """Kembalikan citra label (int32) berisi kandidat sel berinti.

    Inti sel pada pewarnaan Giemsa berwarna ungu pekat: SATURASI tinggi.
    Eritrosit berwarna merah muda pucat: saturasi lebih rendah dan value lebih
    tinggi. Otsu pada kanal S memisahkan keduanya pada sebagian besar kasus;
    --v-max-pct menambah penyaringan untuk kasus yang sulit.
    """
    if blur and blur >= 3:
        bgr = cv2.GaussianBlur(bgr, (blur | 1, blur | 1), 0)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask > 0

    if v_max_pct is not None:
        mask &= v <= np.percentile(v, v_max_pct)

    # Morfologi lewat OpenCV: bebas dari perubahan API skimage antar versi
    m8 = mask.astype(np.uint8)
    if open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * open_r + 1, 2 * open_r + 1))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * close_r + 1, 2 * close_r + 1))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
    mask = ndi.binary_fill_holes(m8 > 0)

    # buang objek di bawah luas minimum
    lab, _ = ndi.label(mask)
    if lab.max() > 0:
        counts = np.bincount(lab.ravel())
        too_small = np.where(counts < max(1, int(min_area)))[0]
        mask[np.isin(lab, too_small[too_small > 0])] = False

    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    if split_touching:
        dist = ndi.distance_transform_edt(mask)
        coords = peak_local_max(dist, min_distance=min_distance, labels=mask)
        markers = np.zeros(dist.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        if markers.max() == 0:
            markers, _ = ndi.label(mask)
        else:
            markers, _ = ndi.label(markers > 0)
        labels = segmentation.watershed(-dist, markers, mask=mask)
    else:
        labels, _ = ndi.label(mask)

    return labels.astype(np.int32)


def region_candidates(labels, min_area, max_area, min_solidity,
                      drop_border=True, border_margin=2):
    """Filter regionprops berdasarkan luas, soliditas, dan sentuhan tepi."""
    h, w = labels.shape
    keep = []
    for rp in measure.regionprops(labels):
        if rp.area < min_area or rp.area > max_area:
            continue
        if min_solidity > 0 and rp.solidity < min_solidity:
            continue
        if drop_border:
            r0, c0, r1, c1 = rp.bbox
            if (r0 <= border_margin or c0 <= border_margin or
                    r1 >= h - border_margin or c1 >= w - border_margin):
                continue
        keep.append(rp)
    keep.sort(key=lambda r: (r.centroid[0], r.centroid[1]))
    return keep


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------
def crop_square(bgr, rp, labels, pad=0.35, out_size=128, masked=False,
                min_crop=24):
    """Crop bujur sangkar berpusat centroid, lalu ubah ukuran ke out_size.

    masked=False -> crop mentah, latar eritrosit ikut terbawa (mirip ALL-IDB
                    dan cocok sebagai domain target)
    masked=True  -> latar dihitamkan (mirip C-NMC)
    """
    h, w = bgr.shape[:2]
    r0, c0, r1, c1 = rp.bbox
    side = int(max(r1 - r0, c1 - c0) * (1.0 + 2.0 * pad))
    side = max(side, min_crop)

    cy, cx = rp.centroid
    top, left = int(round(cy - side / 2)), int(round(cx - side / 2))

    # padding refleksi bila kotak melewati tepi citra
    pad_t, pad_l = max(0, -top), max(0, -left)
    pad_b, pad_r = max(0, top + side - h), max(0, left + side - w)
    if any((pad_t, pad_l, pad_b, pad_r)):
        bgr = cv2.copyMakeBorder(bgr, pad_t, pad_b, pad_l, pad_r,
                                 cv2.BORDER_REFLECT_101)
        labels = np.pad(labels, ((pad_t, pad_b), (pad_l, pad_r)),
                        mode='constant')
        top += pad_t
        left += pad_l

    patch = bgr[top:top + side, left:left + side]
    if patch.size == 0:
        return None

    if masked:
        m = (labels[top:top + side, left:left + side] == rp.label)
        patch = patch * m[:, :, None].astype(patch.dtype)

    interp = cv2.INTER_AREA if side > out_size else cv2.INTER_CUBIC
    return cv2.resize(patch, (out_size, out_size), interpolation=interp)


# ---------------------------------------------------------------------------
# Pembantu
# ---------------------------------------------------------------------------
def list_images(root):
    """Kembalikan [(kelas, path)]; kelas diambil dari nama subfolder langsung."""
    root = Path(root)
    out = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in IMG_EXT:
            rel = p.relative_to(root)
            cls = rel.parts[0] if len(rel.parts) > 1 else '_root'
            out.append((cls, p))
    return out


def qc_overlay(bgr, kept, labels):
    vis = bgr.copy()
    for rp in kept:
        m = (labels == rp.label).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        r0, c0, r1, c1 = rp.bbox
        cv2.rectangle(vis, (c0, r0), (c1, r1), (255, 0, 0), 1)
    return vis


def imread_unicode(path):
    """cv2.imread gagal pada path non-ASCII di sebagian sistem."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Mode --scan
# ---------------------------------------------------------------------------
def run_scan(items, args):
    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(items):
        idx = rng.choice(len(items), size=args.sample, replace=False)
        items = [items[i] for i in sorted(idx)]

    areas, per_img, dims = [], [], []
    funnel = Counter()          # berapa objek tersisa setelah tiap filter
    lost_area_pct = []
    for cls, path in items:
        bgr = imread_unicode(path)
        if bgr is None:
            continue
        dims.append(bgr.shape[:2])
        labels = segment_nuclei(bgr, args.blur, args.v_max_pct, args.open_r,
                                args.close_r, min_area=20,
                                split_touching=args.split_touching,
                                min_distance=args.min_distance)
        props = measure.regionprops(labels)
        a = [rp.area for rp in props]
        areas.extend(a)
        per_img.append(len(a))

        # corong filter, memakai ambang yang SEDANG DIPAKAI pengguna
        h, w = labels.shape
        funnel['0_mentah'] += len(props)
        s1 = [r for r in props if r.area >= args.min_area]
        funnel['1_lolos_min_area'] += len(s1)
        s2 = [r for r in s1 if r.area <= args.max_area]
        funnel['2_lolos_max_area'] += len(s2)
        s3 = [r for r in s2 if args.min_solidity <= 0 or
              r.solidity >= args.min_solidity]
        funnel['3_lolos_soliditas'] += len(s3)
        if args.keep_border:
            s4 = s3
        else:
            m = args.border_margin
            s4 = [r for r in s3
                  if not (r.bbox[0] <= m or r.bbox[1] <= m or
                          r.bbox[2] >= h - m or r.bbox[3] >= w - m)]
        funnel['4_lolos_tepi'] += len(s4)

    if not areas:
        print("Tidak ada objek terdeteksi sama sekali. Segmentasi gagal: "
              "coba turunkan --open-r, atau setel --v-max-pct 85.")
        return

    areas = np.asarray(areas)
    hh = np.array([d[0] for d in dims])
    ww = np.array([d[1] for d in dims])
    print(f"\nDipindai {len(per_img)} citra.")
    print(f"Ukuran citra: {ww.min()}x{hh.min()} s/d {ww.max()}x{hh.max()} "
          f"(median {int(np.median(ww))}x{int(np.median(hh))})")
    print(f"Objek mentah per citra: rerata {np.mean(per_img):.1f}, "
          f"median {np.median(per_img):.0f}\n")

    print("Sebaran luas objek mentah (piksel):")
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{q:<3d} = {np.percentile(areas, q):10.0f}")

    # --- corong: filter mana yang membuang objek
    print(f"\nCORONG FILTER (ambang yang sedang dipakai: "
          f"min-area={args.min_area}, max-area={args.max_area}, "
          f"min-solidity={args.min_solidity}, "
          f"buang-tepi={'tidak' if args.keep_border else 'ya'}):")
    order = ['0_mentah', '1_lolos_min_area', '2_lolos_max_area',
             '3_lolos_soliditas', '4_lolos_tepi']
    label = {'0_mentah': 'objek mentah',
             '1_lolos_min_area': 'setelah --min-area',
             '2_lolos_max_area': 'setelah --max-area',
             '3_lolos_soliditas': 'setelah --min-solidity',
             '4_lolos_tepi': 'setelah filter tepi'}
    prev = funnel[order[0]]
    for k in order:
        cur = funnel[k]
        drop = prev - cur
        bar = '#' * int(40 * cur / max(1, funnel[order[0]]))
        note = f"  (-{drop})" if k != order[0] and drop else ''
        print(f"  {label[k]:<26s} {cur:>7d} {bar}{note}")
        prev = cur
    print(f"  -> {funnel['4_lolos_tepi'] / max(1, len(per_img)):.2f} "
          f"sel per citra dengan ambang saat ini")

    worst = None
    for a, b in zip(order, order[1:]):
        d = funnel[a] - funnel[b]
        if worst is None or d > worst[1]:
            worst = (b, d)
    if worst and worst[1] > 0:
        hint = {
            '1_lolos_min_area': "--min-area terlalu TINGGI untuk resolusi citra ini",
            '2_lolos_max_area': "--max-area terlalu RENDAH, atau sel menggumpal "
                                "(pakai --split-touching)",
            '3_lolos_soliditas': "--min-solidity terlalu ketat; coba 0.5 atau 0",
            '4_lolos_tepi': "sel banyak menyentuh tepi; coba --keep-border "
                            "atau turunkan --pad",
        }[worst[0]]
        print(f"\n  PENYEBAB UTAMA: {label[worst[0]]} membuang {worst[1]} objek.")
        print(f"  {hint}")

    lo, hi = np.percentile(areas, 25), np.percentile(areas, 99.5)
    print(f"\nSaran berdasarkan data ini: --min-area {int(lo)} "
          f"--max-area {int(hi)}")
    print("Selalu verifikasi dengan --qc-dir sebelum ekstraksi penuh.")


# ---------------------------------------------------------------------------
# Mode ekstraksi
# ---------------------------------------------------------------------------
def run_extract(items, args):
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    qc_root = Path(args.qc_dir) if args.qc_dir else None
    if qc_root:
        qc_root.mkdir(parents=True, exist_ok=True)

    rows = []
    per_class = Counter()
    fields_per_class = Counter()
    cells_per_field = defaultdict(int)
    empty_fields = []
    unreadable = []

    for n, (cls, path) in enumerate(items, start=1):
        bgr = imread_unicode(path)
        if bgr is None:
            unreadable.append(str(path))
            continue

        labels = segment_nuclei(bgr, args.blur, args.v_max_pct, args.open_r,
                                args.close_r, min_area=args.min_area,
                                split_touching=args.split_touching,
                                min_distance=args.min_distance)
        kept = region_candidates(labels, args.min_area, args.max_area,
                                 args.min_solidity, not args.keep_border,
                                 args.border_margin)

        field_id = path.stem
        fields_per_class[cls] += 1
        cls_dir = out_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for k, rp in enumerate(kept, start=1):
            patch = crop_square(bgr, rp, labels, args.pad, args.out_size,
                                args.masked)
            if patch is None:
                continue
            name = f"{args.prefix}_{cls}_{field_id}_cell{k:03d}.png"
            cv2.imwrite(str(cls_dir / name), patch)
            per_class[cls] += 1
            cells_per_field[field_id] += 1
            rows.append({
                'file': name, 'class': cls, 'field_id': field_id,
                'source_path': str(path), 'cell_index': k,
                'area_px': int(rp.area),
                'centroid_y': round(float(rp.centroid[0]), 1),
                'centroid_x': round(float(rp.centroid[1]), 1),
                'solidity': round(float(rp.solidity), 3),
                'eccentricity': round(float(rp.eccentricity), 3),
                'bbox_side_px': int(max(rp.bbox[2] - rp.bbox[0],
                                        rp.bbox[3] - rp.bbox[1])),
            })

        if not kept:
            empty_fields.append(str(path))
        if qc_root and (n % max(1, args.qc_every) == 0):
            d = qc_root / cls
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"qc_{field_id}.png"), qc_overlay(bgr, kept, labels))

        if args.verbose and n % 100 == 0:
            print(f"  {n}/{len(items)} citra diproses...", file=sys.stderr)

    # ---- laporan ----
    csv_path = out_root / 'extraction_report.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                            ['file', 'class', 'field_id'])
        wr.writeheader()
        wr.writerows(rows)

    sum_path = out_root / 'summary_per_class.csv'
    with open(sum_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['class', 'n_fields', 'n_cells', 'cells_per_field'])
        for cls in sorted(set(list(per_class) + list(fields_per_class))):
            nf, nc = fields_per_class[cls], per_class[cls]
            wr.writerow([cls, nf, nc, round(nc / nf, 2) if nf else 0])

    print("\n=== LAPORAN EKSTRAKSI ===")
    print(f"{'kelas':<28s} {'lapang':>8s} {'sel':>8s} {'sel/lapang':>12s}")
    print("-" * 60)
    for cls in sorted(set(list(per_class) + list(fields_per_class))):
        nf, nc = fields_per_class[cls], per_class[cls]
        print(f"{cls:<28s} {nf:>8d} {nc:>8d} {(nc/nf if nf else 0):>12.2f}")
    print("-" * 60)
    print(f"{'TOTAL':<28s} {sum(fields_per_class.values()):>8d} "
          f"{sum(per_class.values()):>8d}")

    if empty_fields:
        print(f"\nPERHATIAN: {len(empty_fields)} lapang pandang tidak "
              f"menghasilkan sel apa pun. Periksa beberapa di --qc-dir; "
              f"kemungkinan --min-area terlalu tinggi.")
        for p in empty_fields[:5]:
            print(f"  {p}")
    if unreadable:
        print(f"\nPERHATIAN: {len(unreadable)} berkas gagal dibaca.")

    print(f"\nCSV per sel   : {csv_path}")
    print(f"CSV per kelas : {sum_path}")
    print("\nLangkah berikutnya: gunakan kolom 'field_id' untuk membagi data.\n"
          "Semua sel dengan field_id yang sama WAJIB masuk ke sisi yang sama\n"
          "(adaptasi ATAU tes). Jangan mengacak di tingkat sel.")


# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Ekstraksi sel tunggal dari citra lapang pandang apusan darah.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input', required=True,
                   help='folder akar; subfolder tingkat pertama dianggap kelas')
    p.add_argument('--output', default='cells_out', help='folder keluaran')
    p.add_argument('--prefix', default='TALEQANI', help='awalan nama berkas')

    p.add_argument('--scan', action='store_true',
                   help='hanya laporkan sebaran luas objek, tidak menulis crop')
    p.add_argument('--sample', type=int, default=40,
                   help='jumlah citra yang dipindai pada mode --scan')

    g = p.add_argument_group('segmentasi')
    g.add_argument('--blur', type=int, default=5, help='ukuran kernel Gaussian')
    g.add_argument('--v-max-pct', type=float, default=None,
                   help='buang piksel dengan value di atas persentil ini '
                        '(menyaring eritrosit terang); contoh 85')
    g.add_argument('--open-r', type=int, default=3, help='radius opening')
    g.add_argument('--close-r', type=int, default=5, help='radius closing')
    g.add_argument('--split-touching', action='store_true',
                   help='pisahkan sel bersentuhan dengan watershed')
    g.add_argument('--min-distance', type=int, default=15,
                   help='jarak minimum antar puncak pada watershed')

    g = p.add_argument_group('filter')
    g.add_argument('--min-area', type=int, default=1000, help='luas minimum (px)')
    g.add_argument('--max-area', type=int, default=100000, help='luas maksimum (px)')
    g.add_argument('--min-solidity', type=float, default=0.80,
                   help='soliditas minimum; menyaring gumpalan dan debris')
    g.add_argument('--keep-border', action='store_true',
                   help='pertahankan sel yang menyentuh tepi citra '
                        '(bawaan: dibuang karena terpotong)')
    g.add_argument('--border-margin', type=int, default=2,
                   help='toleransi piksel untuk deteksi sentuhan tepi')

    g = p.add_argument_group('crop')
    g.add_argument('--pad', type=float, default=0.35,
                   help='margin di sekitar sel, sebagai rasio sisi bbox')
    g.add_argument('--out-size', type=int, default=128, help='ukuran keluaran')
    g.add_argument('--masked', action='store_true',
                   help='hitamkan latar (mirip C-NMC). Bawaan: latar '
                        'dipertahankan, mirip ALL-IDB dan sesuai domain target')

    g = p.add_argument_group('lain')
    g.add_argument('--qc-dir', default=None, help='folder overlay pemeriksaan')
    g.add_argument('--qc-every', type=int, default=25,
                   help='simpan overlay tiap N citra')
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--verbose', action='store_true')
    return p


def main():
    args = build_parser().parse_args()
    items = list_images(args.input)
    if not items:
        sys.exit(f"Tidak ada citra ditemukan di {args.input}")
    print(f"Ditemukan {len(items)} citra dalam "
          f"{len(set(c for c, _ in items))} kelas.")
    if args.scan:
        run_scan(items, args)
    else:
        run_extract(items, args)


if __name__ == '__main__':
    main()