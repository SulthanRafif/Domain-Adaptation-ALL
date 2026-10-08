#!/usr/bin/env python3
"""
preflight.py — periksa seluruh jalur yang bisa gagal DIAM-DIAM, tanpa melatih.

Crash akan ketahuan di menit pertama training. Yang berbahaya adalah kesalahan
yang tidak menghentikan skrip tapi merusak angkanya — dan itu baru terlihat
setelah training selesai. Skrip ini menguji bagian-bagian itu dalam hitungan
detik.

Yang diperiksa:
  1. Folder kelas ada, namanya konsisten antar split
  2. Pemetaan kelas ImageFolder identik di train, val, dan test
  3. Kelas positif (ALL) terdeteksi, dan polaritas metrik benar
  4. field_id terbaca dari nama berkas subset tes
  5. Satu field_id tidak tersebar ke beberapa kelas
  6. Subset tes timpang (kalau seimbang, protokol kemungkinan dilanggar)

Contoh:
  python preflight.py --train-dir datasets/cnmc2taleqani/clf_translated/train \\
      --val-dir datasets/cnmc2taleqani/clf_translated/val \\
      --test-dir datasets/cnmc2taleqani/clf/test
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  GAGAL  {msg}")


def warn(msg):
    print(f"  INFO   {msg}")


def good(msg):
    print(f"  OK     {msg}")


def field_id_of(path):
    """Harus sama persis dengan fungsi di run_classifier.py."""
    m = re.match(r'.+?_(.+)_cell\d+$', Path(path).stem)
    return m.group(1) if m else Path(path).stem


def scan(d):
    """{kelas: [path]} dari subfolder tingkat pertama."""
    root = Path(d)
    if not root.is_dir():
        return None
    out = defaultdict(list)
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(sub.rglob('*')):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                out[sub.name].append(f)
    return dict(out)


def main():
    ap = argparse.ArgumentParser(
        description="Pemeriksaan pra-training untuk run_classifier.py.")
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--test-dir', required=True)
    args = ap.parse_args()

    splits = {}
    print("\n[1] Struktur folder")
    for name, d in [('train', args.train_dir), ('val', args.val_dir),
                    ('test', args.test_dir)]:
        s = scan(d)
        if s is None:
            fail(f"{name}: folder tidak ada — {d}")
            continue
        if not s:
            fail(f"{name}: tidak ada subfolder kelas berisi citra — {d}")
            continue
        splits[name] = s
        tot = sum(len(v) for v in s.values())
        detail = '  '.join(f"{k}={len(v)}" for k, v in sorted(s.items()))
        good(f"{name:<5s} {tot:>6d} citra | {detail}")

    if len(splits) < 3:
        print("\nTIDAK LOLOS — perbaiki dulu sebelum menjalankan training.")
        sys.exit(1)

    print("\n[2] Konsistensi pemetaan kelas (urutan ImageFolder)")
    maps = {n: {c: i for i, c in enumerate(sorted(s))}
            for n, s in splits.items()}
    if len(set(map(str, maps.values()))) == 1:
        good(f"identik di ketiga split: {maps['train']}")
    else:
        for n, m in maps.items():
            print(f"         {n}: {m}")
        fail("pemetaan kelas BERBEDA antar split — metrik akan kacau. "
             "Samakan nama subfolder kelas.")

    print("\n[3] Polaritas kelas positif")
    classes = sorted(splits['train'])
    if 'ALL' in classes:
        pos = classes.index('ALL')
        good(f"kelas positif 'ALL' pada indeks {pos}; dipetakan ke 1 "
             f"saat metrik dihitung")
        if pos == 0:
            warn("ALL berada di indeks 0 — inilah yang dulu membalik AUC. "
                 "Sudah ditangani di versi sekarang.")
    else:
        fail(f"tidak ada kelas bernama 'ALL'; yang ada {classes}. "
             f"Ganti nama folder kelas positif menjadi persis 'ALL'.")

    print("\n[4] Pembacaan field_id pada subset tes")
    paths = [p for v in splits['test'].values() for p in v]
    sample = paths[:5]
    parsed = [(p.name, field_id_of(str(p))) for p in sample]
    n_fields = len(set(field_id_of(str(p)) for p in paths))
    for nm, fid in parsed:
        print(f"         {nm}  ->  {fid}")
    if n_fields == len(paths):
        fail(f"setiap citra menghasilkan field_id unik ({n_fields} dari "
             f"{len(paths)}). Pola nama berkas tidak cocok — agregasi tingkat "
             f"lapang pandang tidak akan bermakna.")
    elif n_fields < 3:
        fail(f"hanya {n_fields} field_id untuk {len(paths)} citra — "
             f"parsing kemungkinan salah.")
    else:
        good(f"{n_fields} lapang pandang dari {len(paths)} sel "
             f"(rerata {len(paths)/n_fields:.1f} sel/lapang)")

    print("\n[5] Satu lapang pandang tidak tersebar ke beberapa kelas")
    fc = defaultdict(set)
    for cls, v in splits['test'].items():
        for p in v:
            fc[field_id_of(str(p))].add(cls)
    mixed = [f for f, c in fc.items() if len(c) > 1]
    if mixed:
        fail(f"{len(mixed)} lapang pandang muncul di lebih dari satu kelas, "
             f"mis. {mixed[:3]}")
    else:
        good("setiap lapang pandang hanya milik satu kelas")

    print("\n[6] Keseimbangan kelas subset tes (pemeriksaan protokol)")
    c = Counter({k: len(v) for k, v in splits['test'].items()})
    if len(c) == 2:
        a, b = sorted(c.values(), reverse=True)
        ratio = a / max(1, b)
        if ratio < 1.3:
            fail(f"subset tes nyaris seimbang (rasio {ratio:.2f}:1). "
                 f"Distribusi alami Taleqani sekitar 5:1 — kalau diseimbangkan, "
                 f"berarti label target ikut dipakai dan protokol dilanggar.")
        else:
            good(f"rasio {ratio:.2f}:1 — distribusi alami terjaga. "
                 f"Laporkan balanced accuracy, bukan accuracy mentah.")

    print("\n" + ("LOLOS — aman dijalankan." if ok else
                  "TIDAK LOLOS — perbaiki dulu."))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()