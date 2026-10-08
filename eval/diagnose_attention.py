#!/usr/bin/env python3
"""
diagnose_attention.py — apakah mask atensi kolaps, atau hanya salah render?

Tampilan di checkpoints/<name>/web/ TIDAK bisa dipercaya untuk menilai mask:
`tensor2im` bawaan repo mengasumsikan tensor berada di rentang [-1, 1] dan
menerapkan (x+1)/2*255, sedangkan s_a adalah keluaran sigmoid di [0, 1].
Akibatnya s_a=0 dirender sebagai abu-abu 127 dan s_a=1 sebagai putih 255 —
mask sehat pun tampak pucat tanpa garis tepi yang tegas.

Skrip ini membaca nilai s_a secara langsung dan melaporkan statistiknya, lalu
menyimpan versi render yang kontrasnya benar.

Penafsiran:
  std < 0.02                  -> KOLAPS. Mask praktis konstan.
  mean < 0.05 atau > 0.95     -> KOLAPS ke salah satu ujung.
  std > 0.15 dan 0.1<mean<0.9 -> SEHAT. Mask membedakan wilayah.

Contoh:
  python diagnose_attention.py \\
      --repo-root ./pytorch-CycleGAN-and-pix2pix \\
      --checkpoints ./pytorch-CycleGAN-and-pix2pix/checkpoints/attn_cnmc2taleqani \\
      --input datasets/cnmc2taleqani/gan/trainA \\
      --out results/attn_diag --n 64 --compare-epochs 5,50,100,200
"""

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_pair(repo_root, ckpt_dir, epoch, bg_source, ngf, use_skip, device):
    sys.path.insert(0, str(Path(repo_root).resolve()))
    from models.attn_networks import AttnResnetGenerator, AttentionGate

    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                   track_running_stats=False)
    G = AttnResnetGenerator(3, 3, ngf, n_blocks=6, norm_layer=norm_layer,
                            use_skip=use_skip)
    A = AttentionGate(7, bg_source)
    for net, name in [(G, 'G_ST'), (A, 'A_S')]:
        p = Path(ckpt_dir) / f"{epoch}_net_{name}.pth"
        if not p.exists():
            return None, None
        sd = torch.load(p, map_location='cpu', weights_only=True)
        sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
              for k, v in sd.items()}
        net.load_state_dict(sd)
        net.eval().to(device)
    return G, A


@torch.no_grad()
def mask_stats(G, A, batch):
    raw = G(batch)
    s_a, _ = A(raw, batch)
    a = s_a.cpu().numpy()
    return {
        'min': float(a.min()), 'max': float(a.max()),
        'mean': float(a.mean()), 'std': float(a.std()),
        'p05': float(np.percentile(a, 5)), 'p50': float(np.percentile(a, 50)),
        'p95': float(np.percentile(a, 95)),
        'frac_below_0.1': float((a < 0.1).mean()),
        'frac_above_0.9': float((a > 0.9).mean()),
        # keragaman ANTAR citra: kalau ~0, semua sampel memberi mask sama
        'std_between_images': float(a.reshape(a.shape[0], -1).mean(1).std()),
    }, s_a


def verdict(st):
    if st['std'] < 0.02:
        return 'KOLAPS', 'mask praktis konstan di seluruh citra'
    if st['mean'] < 0.05:
        return 'KOLAPS', 'mask jatuh ke 0 — generator diabaikan, s\' ≈ citra asal'
    if st['mean'] > 0.95:
        return 'KOLAPS', 'mask jatuh ke 1 — latar ikut diterjemahkan seluruhnya'
    if st['std'] > 0.15 and 0.1 < st['mean'] < 0.9:
        return 'SEHAT', 'mask membedakan wilayah dengan jelas'
    return 'MERAGUKAN', 'mask bervariasi tapi lemah; periksa render kontras'


def main():
    ap = argparse.ArgumentParser(
        description="Diagnosis kolaps mask atensi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--repo-root', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--input', required=True, help='folder citra source')
    ap.add_argument('--out', default='results/attn_diag')
    ap.add_argument('--epoch', default='200')
    ap.add_argument('--compare-epochs', default=None,
                    help='daftar epoch dipisah koma, mis. 5,50,100,200')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--bg-source', default='input',
                    choices=['input', 'generated'])
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--ngf', type=int, default=64)
    ap.add_argument('--no-skip', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(Path(args.input).rglob('*'))
             if p.is_file() and p.suffix.lower() in IMG_EXT][:args.n]
    if not files:
        raise SystemExit(f"Tidak ada citra di {args.input}")

    tf = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size],
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    batch = torch.stack([tf(Image.open(p).convert('RGB'))
                         for p in files]).to(dev)
    print(f"{len(files)} citra dari {args.input}\n")

    epochs = (args.compare_epochs.split(',') if args.compare_epochs
              else [args.epoch])
    last = None

    print(f"{'epoch':>8s} {'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s} "
          f"{'<0.1':>7s} {'>0.9':>7s} {'antar-citra':>12s}  vonis")
    print('-' * 86)
    for ep in epochs:
        G, A = load_pair(args.repo_root, args.checkpoints, ep.strip(),
                         args.bg_source, args.ngf, not args.no_skip, dev)
        if G is None:
            print(f"{ep:>8s}  (checkpoint tidak ada)")
            continue
        st, s_a = mask_stats(G, A, batch)
        v, why = verdict(st)
        print(f"{ep:>8s} {st['mean']:7.4f} {st['std']:7.4f} {st['min']:7.4f} "
              f"{st['max']:7.4f} {st['frac_below_0.1']:7.3f} "
              f"{st['frac_above_0.9']:7.3f} {st['std_between_images']:12.5f}  {v}")
        last = (ep, st, s_a, v, why)

    if last is None:
        raise SystemExit("Tidak ada checkpoint yang bisa dimuat.")

    ep, st, s_a, v, why = last
    print(f"\nVONIS epoch {ep}: {v} — {why}")

    # --- render mask dengan kontras yang benar, plus versi ter-stretch
    grid = []
    for i in range(min(8, s_a.shape[0])):
        m = s_a[i, 0].cpu().numpy()
        raw = (m * 255).astype(np.uint8)                       # skala sebenarnya
        lo, hi = m.min(), m.max()
        stretched = (((m - lo) / (hi - lo + 1e-8)) * 255).astype(np.uint8)
        grid.append(np.concatenate([raw, stretched], axis=0))
    Image.fromarray(np.concatenate(grid, axis=1)).save(out / f'mask_epoch{ep}.png')
    print(f"\nRender mask: {out / f'mask_epoch{ep}.png'}")
    print("  Baris ATAS  = skala sebenarnya (0=hitam, 1=putih)")
    print("  Baris BAWAH = kontras diregangkan ke rentang penuh")
    print("  Kalau baris bawah menampilkan bentuk sel tapi baris atas rata,")
    print("  berarti mask BEKERJA namun amplitudonya terlalu kecil —")
    print("  berbeda dari kolaps total, dan penanganannya juga berbeda.")


if __name__ == '__main__':
    main()