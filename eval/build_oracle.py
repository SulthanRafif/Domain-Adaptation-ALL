#!/usr/bin/env python3
"""
build_oracle.py — susun himpunan latih berlabel dari domain target (baseline oracle).

Oracle adalah BATAS ATAS: classifier yang dilatih langsung pada sel target
berlabel, lalu diuji pada subset tes target yang sama dengan eksperimen lain.
Tanpa angka ini, Anda tidak bisa menghitung gap closure — hanya punya akurasi
mentah yang tidak bisa ditafsirkan.

Sumber datanya adalah subset ADAPTASI (baris `target_adapt` pada
split_manifest.csv), bukan subset tes. Label subset adaptasi memang tersedia;
protokol UDA hanya melarang memakainya saat melatih model DA, bukan saat
membangun baseline oracle.

Pembagian train/val tetap per LAPANG PANDANG, sama seperti di mana pun.

Contoh:
  python build_oracle.py \\
      --manifest datasets/cnmc2taleqani/split_manifest.csv \\
      --cells    data/Taleqani/cells \\
      --out      datasets/cnmc2taleqani/clf_oracle \\
      --val-frac 0.15
"""

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Susun himpunan oracle dari subset adaptasi target.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--manifest', required=True,
                    help='split_manifest.csv keluaran build_dataset.py')
    ap.add_argument('--cells', required=True,
                    help='folder sel keluaran extract_cells.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--symlink', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cells = Path(args.cells)

    # indeks nama berkas -> path sebenarnya
    index = {p.name: p for p in cells.rglob('*') if p.is_file()}

    rows, missing = [], 0
    with open(args.manifest, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] != 'target_adapt':
                continue
            p = index.get(r['file'])
            if p is None:
                missing += 1
                continue
            rows.append((r['class'], r['group_id'], p))

    if not rows:
        raise SystemExit("Tidak ada baris 'target_adapt' pada manifest.")
    if missing:
        print(f"PERINGATAN: {missing} berkas tercatat di manifest tapi tidak "
              f"ditemukan di {cells}")

    # pembagian per lapang pandang, distratifikasi per kelas
    by_group = defaultdict(list)
    for cls, g, p in rows:
        by_group[g].append((cls, p))
    group_cls = {g: Counter(c for c, _ in v).most_common(1)[0][0]
                 for g, v in by_group.items()}

    buckets = defaultdict(list)
    for g in by_group:
        buckets[group_cls[g]].append(g)

    train, val = [], []
    for _, groups in sorted(buckets.items()):
        groups = sorted(groups)
        rng.shuffle(groups)
        k = max(1, int(round(len(groups) * args.val_frac)))
        k = min(k, len(groups) - 1) if len(groups) > 1 else 0
        for g in groups[:k]:
            val += [(c, g, p) for c, p in by_group[g]]
        for g in groups[k:]:
            train += [(c, g, p) for c, p in by_group[g]]

    out = Path(args.out)
    for name, items in [('train', train), ('val', val)]:
        for cls, _g, src in items:
            d = out / name / cls
            d.mkdir(parents=True, exist_ok=True)
            dst = d / src.name
            if args.symlink:
                try:
                    dst.symlink_to(src.resolve())
                    continue
                except OSError:
                    pass
            shutil.copy2(src, dst)

    inter = set(g for _, g, _ in train) & set(g for _, g, _ in val)
    print("\n=== HIMPUNAN ORACLE ===")
    for name, items in [('train', train), ('val', val)]:
        c = Counter(i[0] for i in items)
        print(f"  {name:<6s} {len(items):>6d} citra | "
              f"{len(set(i[1] for i in items)):>4d} lapang | " +
              '  '.join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"  kebocoran lapang train/val: "
          f"{'BERSIH' if not inter else f'BOCOR {len(inter)}'}")
    print(f"\nTersimpan di {out}")
    print("Uji baseline ini dengan --test-dir clf/test yang SAMA seperti "
          "eksperimen lain,\nagar ketiga angka benar-benar sebanding.")


if __name__ == '__main__':
    main()