#!/usr/bin/env python3
"""
eval_translation.py — evaluasi model domain adaptation ITU SENDIRI.

Terpisah dari evaluasi classifier. Skrip ini menjawab: seberapa dekat citra hasil
translasi dengan domain target, dan seberapa banyak isi citra asalnya berubah.

Menghitung dua hal yang saling menyeimbangkan:

  1. SSIM dan PSNR antara citra source ASLI dan hasil TRANSLASI.
     Mengukur pelestarian konten. Nilai terlalu tinggi = generator nyaris tidak
     mengubah apa pun. Terlalu rendah = morfologi sel ikut rusak.

  2. Sebaran fitur lewat t-SNE, untuk tiga himpunan sekaligus: source asli,
     source tertranslasi, dan target. Kalau adaptasi berhasil, titik-titik
     "source tertranslasi" bergeser mendekati awan "target".

FID TIDAK dihitung di sini — pakai implementasi rujukan `pytorch-fid` supaya
angkanya sebanding dengan yang dilaporkan di literatur. Perintahnya dicetak di
akhir eksekusi.

PENTING: seluruh pembandingan memakai subset target ADAPTASI (gan/trainB),
bukan subset tes. Memakai subset tes akan mencemarinya.

Contoh:
  python eval_translation.py \\
      --source-orig       datasets/cnmc2taleqani/clf/train \\
      --source-translated datasets/cnmc2taleqani/clf_translated/train \\
      --target            datasets/cnmc2taleqani/gan/trainB \\
      --out results/translation_eval --max-per-set 600
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Palet kategorikal, sudah lolos validator (lightness, chroma, CVD, kontras)
PALETTE = {
    'source_asli': '#3B5BDB',
    'source_translasi': '#E8590C',
    'target': '#9C36B5',
}


def list_images(root):
    return [p for p in sorted(Path(root).rglob('*'))
            if p.is_file() and p.suffix.lower() in IMG_EXT]


def load_rgb(path, size):
    return np.asarray(Image.open(path).convert('RGB').resize(
        (size, size), Image.BICUBIC))


# ---------------------------------------------------------------------------
# 1. Pelestarian konten: SSIM dan PSNR pada pasangan citra
# ---------------------------------------------------------------------------
def content_preservation(orig_dir, trans_dir, size, limit):
    from skimage.metrics import (peak_signal_noise_ratio as psnr,
                                 structural_similarity as ssim)

    orig_root, trans_root = Path(orig_dir), Path(trans_dir)
    pairs = []
    for p in list_images(orig_root):
        rel = p.relative_to(orig_root)
        # translate.py menyimpan sebagai .png
        cand = trans_root / rel.with_suffix('.png')
        if cand.exists():
            pairs.append((p, cand))
    if not pairs:
        raise SystemExit(
            "Tidak ada pasangan citra yang cocok. Pastikan --source-translated "
            "memang hasil translate.py atas --source-orig, dan struktur "
            "subfoldernya sama.")
    if limit and len(pairs) > limit:
        idx = np.random.default_rng(0).choice(len(pairs), limit, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]

    ss, ps, ss_cell = [], [], []
    for a, b in pairs:
        ia, ib = load_rgb(a, size), load_rgb(b, size)
        ss.append(ssim(ia, ib, channel_axis=2, data_range=255))
        ps.append(psnr(ia, ib, data_range=255))

        # SSIM DI DALAM SEL SAJA.
        # SSIM global membandingkan latar yang memang SENGAJA diubah: C-NMC
        # berlatar hitam, hasil translasi berlatar eritrosit. Latar menempati
        # sebagian besar bidang, sehingga nilainya anjlok meski selnya utuh.
        # Yang bermakna adalah kesetiaan di wilayah sel.
        m = _cell_mask(ia)
        if m is not None and m.sum() > 20:
            _, smap = ssim(ia, ib, channel_axis=2, data_range=255, full=True)
            ss_cell.append(float(smap.mean(axis=2)[m].mean()))

    out = {
        'n_pairs': len(pairs),
        'ssim_mean': round(float(np.mean(ss)), 4),
        'ssim_std': round(float(np.std(ss)), 4),
        'ssim_cell_mean': (round(float(np.mean(ss_cell)), 4)
                           if ss_cell else None),
        'ssim_cell_std': (round(float(np.std(ss_cell)), 4)
                          if ss_cell else None),
        'psnr_mean': round(float(np.mean(ps)), 4),
        'psnr_std': round(float(np.std(ps)), 4),
    }
    return out


# ---------------------------------------------------------------------------
# 1b. Pelestarian BENTUK: Dice, IoU, dan pergeseran deskriptor morfologi
# ---------------------------------------------------------------------------
def _cell_mask(rgb):
    """Segmentasi sel berinti; metode yang sama untuk citra asli maupun hasil
    translasi, supaya perbandingannya adil.

    Otsu pada kanal saturasi memisahkan inti (ungu pekat, saturasi tinggi) dari
    latar — baik latar hitam C-NMC (S = 0) maupun latar eritrosit hasil karangan
    generator (saturasi sedang). Hanya komponen terbesar yang diambil, karena
    citra ini berisi satu sel di tengah.
    """
    import cv2
    from scipy import ndimage as ndi

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    s = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)[:, :, 1]
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ndi.binary_fill_holes(m > 0)
    lab, n = ndi.label(m)
    if n == 0:
        return None
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def shape_preservation(orig_dir, trans_dir, size, limit):
    """Seberapa setia bentuk sel bertahan setelah translasi.

    SSIM mencampur bentuk dengan warna dan tekstur, sehingga bisa terlihat baik
    meski sel berubah bentuk. Metrik di sini memisahkan bentuk saja:

      dice, iou        - tumpang tindih wilayah sel sebelum dan sesudah
      d_eccentricity   - NEGATIF berarti sel menjadi lebih bulat
      d_solidity       - POSITIF berarti lekukan kontur hilang
      area_ratio       - > 1 berarti sel membesar

    Untuk klasifikasi ALL, ketiga deskriptor terakhir lebih penting daripada
    dice: lekukan inti dan ketakberaturan kontur adalah sinyal diagnostiknya.
    Generator yang membulatkan semua sel menghapus fitur yang justru harus
    dipelajari classifier.
    """
    from skimage import measure

    orig_root, trans_root = Path(orig_dir), Path(trans_dir)
    pairs = []
    for p in list_images(orig_root):
        cand = trans_root / p.relative_to(orig_root).with_suffix('.png')
        if cand.exists():
            pairs.append((p, cand))
    if limit and len(pairs) > limit:
        idx = np.random.default_rng(0).choice(len(pairs), limit, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]

    dice, iou, d_ecc, d_sol, ratio, skipped = [], [], [], [], [], 0
    for a, b in pairs:
        ma = _cell_mask(load_rgb(a, size))
        mb = _cell_mask(load_rgb(b, size))
        if ma is None or mb is None or ma.sum() == 0 or mb.sum() == 0:
            skipped += 1
            continue
        inter = np.logical_and(ma, mb).sum()
        dice.append(2 * inter / (ma.sum() + mb.sum()))
        iou.append(inter / np.logical_or(ma, mb).sum())
        ra = measure.regionprops(ma.astype(np.uint8))[0]
        rb = measure.regionprops(mb.astype(np.uint8))[0]
        d_ecc.append(rb.eccentricity - ra.eccentricity)
        d_sol.append(rb.solidity - ra.solidity)
        ratio.append(rb.area / max(1, ra.area))

    if not dice:
        return {'n_pairs': 0, 'catatan': 'segmentasi gagal pada semua pasangan'}

    def stat(v):
        return round(float(np.mean(v)), 4), round(float(np.std(v)), 4)

    dm, ds = stat(dice)
    im, isd = stat(iou)
    em, es = stat(d_ecc)
    sm, ss_ = stat(d_sol)
    rm, rs = stat(ratio)
    return {
        'n_pairs': len(dice), 'n_skipped': skipped,
        'dice_mean': dm, 'dice_std': ds,
        'iou_mean': im, 'iou_std': isd,
        'd_eccentricity_mean': em, 'd_eccentricity_std': es,
        'd_solidity_mean': sm, 'd_solidity_std': ss_,
        'area_ratio_mean': rm, 'area_ratio_std': rs,
    }


# ---------------------------------------------------------------------------
# 2. Sebaran fitur: t-SNE atas tiga himpunan
# ---------------------------------------------------------------------------
def extract_features(paths, size, batch, device):
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    net = models.resnet34(weights='IMAGENET1K_V1')
    net.fc = nn.Identity()
    net.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            xs = torch.stack([tf(Image.open(p).convert('RGB'))
                              for p in paths[i:i + batch]]).to(device)
            feats.append(net(xs).cpu().numpy())
    return np.concatenate(feats)


def tsne_plot(sets, out_png, size, batch, device, seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    names, all_feats, group = [], [], []
    for i, (name, paths) in enumerate(sets):
        if not paths:
            continue
        names.append(name)
        all_feats.append(extract_features(paths, size, batch, device))
        group += [len(names) - 1] * len(paths)
    X = np.concatenate(all_feats)
    group = np.asarray(group)

    perp = float(min(30, max(5, len(X) / 10)))
    emb = TSNE(n_components=2, init='pca', perplexity=perp,
               random_state=seed).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    for i, name in enumerate(names):
        m = group == i
        ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.75, linewidths=0.6,
                   edgecolors='#fcfcfb', c=PALETTE.get(name, '#666666'),
                   label=name.replace('_', ' '))
    ax.set_title('Sebaran fitur sebelum dan sesudah translasi (t-SNE)',
                 fontsize=12, color='#1a1a1a', pad=12)
    ax.tick_params(colors='#8a8a8a', labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d8d5')
    leg = ax.legend(frameon=False, fontsize=9, loc='best')
    for t in leg.get_texts():
        t.set_color('#3a3a3a')
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {'n_points': int(len(X)), 'perplexity': perp, 'sets': names}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluasi kualitas translasi model domain adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--source-orig', required=True)
    ap.add_argument('--source-translated', required=True)
    ap.add_argument('--target', required=True,
                    help='subset target ADAPTASI (gan/trainB), bukan subset tes')
    ap.add_argument('--out', default='results/translation_eval')
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--max-per-set', type=int, default=600,
                    help='batas jumlah citra per himpunan untuk t-SNE')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--skip-tsne', action='store_true')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("== 1. Pelestarian konten (source asli vs hasil translasi) ==")
    cp = content_preservation(args.source_orig, args.source_translated,
                              args.img_size, args.max_per_set)
    print(f"  pasangan        {cp['n_pairs']}")
    print(f"  SSIM global     {cp['ssim_mean']} ± {cp['ssim_std']}")
    print(f"  SSIM dalam sel  {cp['ssim_cell_mean']} ± {cp['ssim_cell_std']}"
          "   <- yang bermakna")
    print(f"  PSNR            {cp['psnr_mean']} ± {cp['psnr_std']} dB")
    if cp['ssim_mean'] < 0.35:
        print("  Catatan: SSIM global rendah WAJAR bila latar source hitam "
              "(C-NMC) dan\n  hasil translasi berlatar eritrosit — latar "
              "memang sengaja diubah.\n  Nilai yang dinilai adalah SSIM dalam "
              "sel.")
    sc = cp.get('ssim_cell_mean')
    if sc is not None:
        if sc > 0.95:
            print("  PERINGATAN: SSIM dalam sel sangat tinggi — generator "
                  "nyaris tidak mengubah sel.")
        elif sc < 0.40:
            print("  PERINGATAN: SSIM dalam sel rendah — tekstur dan struktur "
                  "sel ikut rusak,\n  bukan sekadar bergaya ulang.")

    result = {'content_preservation': cp}

    print("\n== 1b. Pelestarian bentuk sel ==")
    sp = shape_preservation(args.source_orig, args.source_translated,
                            args.img_size, args.max_per_set)
    result['shape_preservation'] = sp
    if sp.get('n_pairs'):
        print(f"  pasangan     {sp['n_pairs']}  (gagal tersegmentasi: "
              f"{sp['n_skipped']})")
        print(f"  Dice         {sp['dice_mean']} ± {sp['dice_std']}")
        print(f"  IoU          {sp['iou_mean']} ± {sp['iou_std']}")
        print(f"  Δ eksentrisitas {sp['d_eccentricity_mean']:+.4f} "
              f"± {sp['d_eccentricity_std']}")
        print(f"  Δ soliditas     {sp['d_solidity_mean']:+.4f} "
              f"± {sp['d_solidity_std']}")
        print(f"  rasio luas      {sp['area_ratio_mean']} "
              f"± {sp['area_ratio_std']}")
        if sp['dice_mean'] < 0.70:
            print("  PERINGATAN: Dice rendah — bentuk sel bergeser banyak. "
                  "Naikkan --lambda_pixel.")
        # Pemeriksaan DUA ARAH. Versi sebelumnya hanya menangkap pembulatan,
        # sehingga peregangan bentuk (eksentrisitas NAIK) lolos tanpa peringatan.
        de, ds_ = sp['d_eccentricity_mean'], sp['d_solidity_mean']
        if de < -0.10:
            print("  PERINGATAN: sel MEMBULAT (eksentrisitas turun). "
                  "Naikkan --lambda_pixel.")
        elif de > 0.10:
            print("  PERINGATAN: sel MEMANJANG (eksentrisitas naik). "
                  "Generator meregangkan\n  bentuk sel; morfologi tidak "
                  "terjaga. Naikkan --lambda_pixel.")
        if abs(ds_) > 0.05:
            arah = 'hilang' if ds_ > 0 else 'bertambah'
            print(f"  PERINGATAN: lekukan kontur {arah} (Δ soliditas "
                  f"{ds_:+.3f}). Morfologi inti berubah.")
    else:
        print(f"  {sp.get('catatan', 'tidak ada pasangan')}")

    if not args.skip_tsne:
        import torch
        dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n== 2. Sebaran fitur (t-SNE, device={dev}) ==")

        def sample(d):
            ps = list_images(d)
            if args.max_per_set and len(ps) > args.max_per_set:
                idx = rng.choice(len(ps), args.max_per_set, replace=False)
                ps = [ps[i] for i in sorted(idx)]
            return ps

        png = out / 'tsne.png'
        info = tsne_plot([('source_asli', sample(args.source_orig)),
                          ('source_translasi', sample(args.source_translated)),
                          ('target', sample(args.target))],
                         png, args.img_size, args.batch_size, dev)
        result['tsne'] = info
        print(f"  {info['n_points']} titik -> {png}")
        print("  Yang dinilai: awan 'source translasi' harus bergeser "
              "mendekati awan 'target'.")

    with open(out / 'translation_eval.json', 'w') as f:
        json.dump(result, f, indent=2)
    with open(out / 'content_preservation.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cp.keys()))
        w.writeheader()
        w.writerow(cp)

    print(f"\n== 3. FID — jalankan dengan implementasi rujukan ==")
    print("  pip install pytorch-fid")
    print(f"  # jarak AWAL (sebelum adaptasi):")
    print(f"  python -m pytorch_fid {args.source_orig} {args.target}")
    print(f"  # jarak AKHIR (sesudah adaptasi):")
    print(f"  python -m pytorch_fid {args.source_translated} {args.target}")
    print("  Bukti keberhasilan adaptasi adalah FID kedua LEBIH RENDAH "
          "dari yang pertama.")
    print(f"\nTersimpan di {out}")


if __name__ == '__main__':
    main()