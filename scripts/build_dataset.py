#!/usr/bin/env python3
"""
build_dataset.py — menyusun folder dataset untuk tahap GAN dan tahap classifier.

Menghasilkan DUA struktur sekaligus dari satu perintah:

  <out>/gan/                 <- format repo junyanz, TANPA label
      trainA/                   citra source (untuk melatih generator)
      trainA_mask/              mask source opsional, sejajar dengan trainA
      trainB/                   sel target subset ADAPTASI
      testA/                    sampel source untuk inspeksi visual
      testB/                    sampel target untuk inspeksi visual

  <out>/clf/                 <- untuk classifier, DENGAN label
      train/<kelas>/            source (nanti diganti hasil translasi)
      val/<kelas>/              source, dipisah per subjek — untuk pilih checkpoint
      test/<kelas>/             sel target subset TES — label hanya dipakai di sini

Aturan yang ditegakkan skrip ini:
  * Source dipecah per SUBJEK (C-NMC) atau per LAPANG PANDANG (ALL-IDB);
    tidak ada subjek yang muncul di train dan val sekaligus.
  * Target dipecah per LAPANG PANDANG; semua sel dari satu lapang masuk ke
    sisi yang sama. Ini satu-satunya penjaga kebocoran yang tersedia karena
    Taleqani tidak memuat identitas pasien.
  * Kelas source diseimbangkan lewat undersampling.
  * Kelas target TIDAK diseimbangkan — menyeimbangkannya berarti membaca
    label target, dan itu melanggar protokol unsupervised.

Contoh (Skenario A, C-NMC -> Taleqani):
  python build_dataset.py \\
      --source-root data/CNMC --source-id-mode cnmc \\
      --target-cells data/Taleqani/cells \\
      --target-report data/Taleqani/cells/extraction_report.csv \\
      --out datasets/cnmc2taleqani \\
      --source-per-class 1000 --gan-target-cap 2000
"""

import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# kelas yang dianggap positif (ALL / malignant); sisanya dianggap Normal
POSITIVE_HINTS = ('all', 'malignant', 'abnormal', 'cancer', 'blast',
                  'early', 'pre-b', 'preb', 'pro-b', 'prob', 'leukemi')


def norm_class(name):
    """Petakan nama folder/kelas apa pun ke 'ALL' atau 'Normal'."""
    low = name.lower().replace('_', '-')
    if any(h in low for h in POSITIVE_HINTS):
        return 'ALL'
    return 'Normal'


# ---------------------------------------------------------------------------
# Pengelompokan source
# ---------------------------------------------------------------------------
def source_group_id(path, mode):
    """Kembalikan ID kelompok yang tidak boleh terpecah antara train dan val."""
    stem = path.stem
    if mode == 'cnmc':
        # CNMC_UID_<P>_<N>_<C>_<kelas> -> subjek = <P>
        m = re.search(r'UID[_-](\d+)', stem)
        if m:
            return f"subj{m.group(1)}"
        parts = stem.split('_')
        return f"subj{parts[2]}" if len(parts) > 2 else stem
    if mode == 'allidb':
        # ALLIDB1_Im001_1_cell005 -> lapang pandang = Im001
        m = re.search(r'(Im\d+)', stem, flags=re.IGNORECASE)
        return m.group(1) if m else stem
    return stem       # mode 'filename': tiap citra berdiri sendiri


def collect_source(root, mode):
    """[(kelas_biner, group_id, path)] dari subfolder kelas."""
    root = Path(root)
    out = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            rel = p.relative_to(root)
            cls = norm_class(rel.parts[0] if len(rel.parts) > 1 else p.stem)
            out.append((cls, source_group_id(p, mode), p))
    return out


# ---------------------------------------------------------------------------
# Pengelompokan target
# ---------------------------------------------------------------------------
def collect_target(cells_dir, report_csv):
    """[(kelas_biner, field_id, path)] dibaca dari laporan ekstraksi."""
    cells_dir = Path(cells_dir)
    rows = []
    with open(report_csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            p = cells_dir / r['class'] / r['file']
            if not p.exists():
                alt = list(cells_dir.rglob(r['file']))
                if not alt:
                    continue
                p = alt[0]
            rows.append((norm_class(r['class']), r['field_id'], p))
    return rows


# ---------------------------------------------------------------------------
# Pembagian berbasis kelompok
# ---------------------------------------------------------------------------
def split_by_group(items, frac_first, rng, stratify=True):
    """Pecah [(cls, group, path)] menjadi dua, utuh per kelompok.

    Stratifikasi dilakukan pada tingkat kelompok: kelompok dikelompokkan
    menurut kelas mayoritasnya, lalu dibagi proporsional per kelas. Ini
    menjamin kedua sisi memuat kedua kelas.
    """
    by_group = defaultdict(list)
    for cls, g, p in items:
        by_group[g].append((cls, p))

    group_cls = {g: Counter(c for c, _ in v).most_common(1)[0][0]
                 for g, v in by_group.items()}

    first, second = [], []
    buckets = defaultdict(list)
    for g in by_group:
        buckets[group_cls[g] if stratify else '_'].append(g)

    for _, groups in sorted(buckets.items()):
        groups = sorted(groups)
        rng.shuffle(groups)
        k = int(round(len(groups) * frac_first))
        k = min(max(k, 1), len(groups) - 1) if len(groups) > 1 else len(groups)
        for g in groups[:k]:
            first += [(c, g, p) for c, p in by_group[g]]
        for g in groups[k:]:
            second += [(c, g, p) for c, p in by_group[g]]
    return first, second


def balance_by_class(items, per_class, rng):
    """Undersampling agar tiap kelas punya jumlah sama.

    Pengambilan dilakukan berimbang antar kelompok supaya satu subjek tidak
    mendominasi. per_class=None berarti pakai ukuran kelas terkecil.
    """
    by_cls = defaultdict(list)
    for it in items:
        by_cls[it[0]].append(it)
    n = min(len(v) for v in by_cls.values())
    if per_class:
        n = min(n, per_class)

    out = []
    for cls, lst in sorted(by_cls.items()):
        by_group = defaultdict(list)
        for it in lst:
            by_group[it[1]].append(it)
        for v in by_group.values():
            rng.shuffle(v)
        groups = sorted(by_group)
        rng.shuffle(groups)
        picked, i = [], 0
        while len(picked) < n:                    # ronde bergilir antar kelompok
            progressed = False
            for g in groups:
                if i < len(by_group[g]):
                    picked.append(by_group[g][i])
                    progressed = True
                    if len(picked) == n:
                        break
            if not progressed:
                break
            i += 1
        out += picked
    return out


def cap_by_group(items, cap, rng):
    """Batasi jumlah citra, tetapi buang/ambil per kelompok secara utuh."""
    if not cap or len(items) <= cap:
        return items
    by_group = defaultdict(list)
    for it in items:
        by_group[it[1]].append(it)
    groups = sorted(by_group)
    rng.shuffle(groups)
    out = []
    for g in groups:
        if len(out) >= cap:
            break
        out += by_group[g]
    return out


# ---------------------------------------------------------------------------
def place(items, dest, by_class=False, link=False):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for cls, _g, src in items:
        d = dest / cls if by_class else dest
        d.mkdir(parents=True, exist_ok=True)
        tgt = d / src.name
        if tgt.exists():
            tgt = d / f"{src.stem}__{abs(hash(str(src))) % 9999:04d}{src.suffix}"
        if link:
            try:
                tgt.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, tgt)
        else:
            shutil.copy2(src, tgt)
        n += 1
    return n


def place_source_with_masks(items, mask_root, image_dest, mask_dest, link=False):
    """Tempatkan citra source dan crop_mask dengan basename yang tetap berpasangan."""
    mask_root = Path(mask_root)
    image_dest, mask_dest = Path(image_dest), Path(mask_dest)
    image_dest.mkdir(parents=True, exist_ok=True)
    mask_dest.mkdir(parents=True, exist_ok=True)
    used_names = set()
    for cls, _group, src in items:
        mask_src = mask_root / cls / src.name
        if not mask_src.is_file():
            raise FileNotFoundError(
                f"Mask source tidak ditemukan untuk {src}: {mask_src}. "
                "Pastikan --source-mask-root menunjuk ke crop_mask.")
        name = src.name
        if name in used_names:
            name = f"{src.stem}__{abs(hash(str(src))) % 9999:04d}{src.suffix}"
        used_names.add(name)
        _place_one(src, image_dest / name, link)
        _place_one(mask_src, mask_dest / name, link)
    return len(items)


def _place_one(src, dest, link):
    if link:
        try:
            dest.symlink_to(Path(src).resolve())
        except OSError:
            shutil.copy2(src, dest)
    else:
        shutil.copy2(src, dest)


def report(title, items):
    c = Counter(i[0] for i in items)
    g = len(set(i[1] for i in items))
    tot = sum(c.values())
    detail = '  '.join(f"{k}={v}" for k, v in sorted(c.items()))
    ratio = (f"{c['ALL']/c['Normal']:.2f}:1"
             if c.get('Normal') else '-')
    print(f"{title:<34s} {tot:>7d} citra | {g:>5d} kelompok | {detail:<26s} rasio {ratio}")
    return {'split': title, 'n_images': tot, 'n_groups': g,
            'n_ALL': c.get('ALL', 0), 'n_Normal': c.get('Normal', 0)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Susun folder dataset untuk tahap GAN dan classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--source-root', required=True,
                    help='folder source dengan subfolder per kelas')
    ap.add_argument('--source-mask-root', default=None,
                    help='opsional: folder crop_mask dengan subfolder kelas; '
                         'jika diisi, buat gan/trainA_mask berpasangan')
    ap.add_argument('--source-id-mode', default='cnmc',
                    choices=['cnmc', 'allidb', 'filename'],
                    help='cara membaca ID subjek/lapang dari nama berkas')
    ap.add_argument('--target-cells', required=True,
                    help='folder hasil extract_cells.py')
    ap.add_argument('--target-report', required=True,
                    help='extraction_report.csv dari extract_cells.py')
    ap.add_argument('--out', required=True)

    ap.add_argument('--source-per-class', type=int, default=None,
                    help='jumlah citra per kelas di source; kosong = pakai '
                         'kelas terkecil (C-NMC: 3389)')
    ap.add_argument('--source-val-frac', type=float, default=0.15,
                    help='porsi subjek source untuk validation classifier')
    ap.add_argument('--target-adapt-frac', type=float, default=0.60,
                    help='porsi lapang pandang target untuk adaptasi')
    ap.add_argument('--gan-target-cap', type=int, default=None,
                    help='batasi jumlah sel target untuk training GAN agar '
                         'epoch tidak terlalu panjang; pemotongan per lapang')
    ap.add_argument('--n-preview', type=int, default=50,
                    help='jumlah citra untuk testA/testB (inspeksi visual)')
    ap.add_argument('--symlink', action='store_true',
                    help='buat symlink, bukan salinan (hemat ruang)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)

    src = collect_source(args.source_root, args.source_id_mode)
    tgt = collect_target(args.target_cells, args.target_report)
    if not src:
        raise SystemExit(f"Tidak ada citra source di {args.source_root}")
    if not tgt:
        raise SystemExit("Tidak ada sel target terbaca dari laporan.")

    print("\n=== DATA MENTAH ===")
    rows = [report('source (mentah)', src), report('target (mentah)', tgt)]

    # --- source: seimbangkan, lalu pecah per subjek jadi train/val
    src_bal = balance_by_class(src, args.source_per_class, rng)
    src_train, src_val = split_by_group(src_bal, 1 - args.source_val_frac, rng)

    # --- target: pecah per lapang pandang jadi adaptasi/tes
    tgt_adapt, tgt_test = split_by_group(tgt, args.target_adapt_frac, rng)
    gan_B = cap_by_group(tgt_adapt, args.gan_target_cap, rng)

    print("\n=== SETELAH PEMBAGIAN ===")
    rows.append(report('source seimbang', src_bal))
    rows.append(report('  -> clf/train', src_train))
    rows.append(report('  -> clf/val', src_val))
    rows.append(report('target adaptasi', tgt_adapt))
    rows.append(report('  -> gan/trainB', gan_B))
    rows.append(report('target tes -> clf/test', tgt_test))

    # --- tulis struktur GAN (tanpa label)
    if args.source_mask_root:
        place_source_with_masks(src_bal, args.source_mask_root,
                                out / 'gan' / 'trainA',
                                out / 'gan' / 'trainA_mask', link=args.symlink)
    else:
        place(src_bal, out / 'gan' / 'trainA', link=args.symlink)
    place(gan_B, out / 'gan' / 'trainB', link=args.symlink)
    place(rng.sample(src_bal, min(args.n_preview, len(src_bal))),
          out / 'gan' / 'testA', link=args.symlink)
    place(rng.sample(gan_B, min(args.n_preview, len(gan_B))),
          out / 'gan' / 'testB', link=args.symlink)

    # --- tulis struktur classifier (dengan label)
    place(src_train, out / 'clf' / 'train', by_class=True, link=args.symlink)
    place(src_val, out / 'clf' / 'val', by_class=True, link=args.symlink)
    place(tgt_test, out / 'clf' / 'test', by_class=True, link=args.symlink)

    # --- manifest: catat sisi mana tiap kelompok berada
    with open(out / 'split_manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['split', 'domain', 'class', 'group_id', 'file'])
        for name, its, dom in [('clf_train', src_train, 'source'),
                               ('clf_val', src_val, 'source'),
                               ('gan_trainB', gan_B, 'target'),
                               ('target_adapt', tgt_adapt, 'target'),
                               ('clf_test', tgt_test, 'target')]:
            for cls, g, p in its:
                w.writerow([name, dom, cls, g, p.name])

    with open(out / 'split_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- pemeriksaan kebocoran
    print("\n=== PEMERIKSAAN KEBOCORAN ===")
    ok = True
    for nm, a, b in [('source train vs val (subjek)', src_train, src_val),
                     ('target adaptasi vs tes (lapang)', tgt_adapt, tgt_test)]:
        inter = set(i[1] for i in a) & set(i[1] for i in b)
        print(f"  {nm:<36s} {'BERSIH' if not inter else f'BOCOR: {len(inter)}'}")
        ok &= not inter
    print("  " + ("Semua pemeriksaan lolos." if ok else
                  "ADA KEBOCORAN — periksa parsing ID kelompok."))

    print(f"\nStruktur GAN        : {out / 'gan'}")
    print(f"Struktur classifier : {out / 'clf'}")
    print(f"Manifest            : {out / 'split_manifest.csv'}")
    print("\nCatatan: clf/train dan clf/val saat ini masih berisi citra source\n"
          "ASLI. Setelah generator selesai dilatih, translasikan kedua folder\n"
          "itu dan ganti isinya dengan hasil translasi. clf/test TIDAK pernah\n"
          "ditranslasikan — isinya sel target asli.")


if __name__ == '__main__':
    main()
