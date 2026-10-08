#!/usr/bin/env python3
"""
translate.py — Tahap 4: terjemahkan citra source ke gaya domain target.

Memuat generator G_ST dan modul atensi A_S hasil training, lalu menjalankannya
atas sebuah folder. Nama berkas dipertahankan PERSIS supaya label kelas (dari
subfolder) dan ID subjek (dari nama berkas) tetap terbawa.

`test.py` bawaan repo tidak bisa dipakai: skrip itu mengasumsikan generator
bernama `G` dan tidak mengenal pasangan G_ST + A_S.

Contoh:
  python translate.py \\
      --repo-root ./pytorch-CycleGAN-and-pix2pix \\
      --checkpoints ./pytorch-CycleGAN-and-pix2pix/checkpoints/attn_cnmc2taleqani \\
      --epoch 200 --bg-source input \\
      --input  datasets/cnmc2taleqani/clf/train \\
      --output datasets/cnmc2taleqani/clf_translated/train

Jalankan terpisah untuk train dan val. JANGAN pernah dijalankan atas clf/test —
subset itu berisi citra target asli dan tidak boleh diterjemahkan.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_nets(args):
    """Bangun G_ST dan A_S dengan arsitektur yang sama seperti saat training,
    lalu muat bobotnya."""
    sys.path.insert(0, str(Path(args.repo_root).resolve()))
    from models.attn_networks import AttnResnetGenerator, AttentionGate
    import functools

    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                   track_running_stats=False)
    G = AttnResnetGenerator(3, 3, args.ngf, n_blocks=6, norm_layer=norm_layer,
                            use_skip=not args.no_skip)
    A = AttentionGate(7, args.bg_source)

    ck = Path(args.checkpoints)
    for net, name in [(G, 'G_ST'), (A, 'A_S')]:
        p = ck / f"{args.epoch}_net_{name}.pth"
        if not p.exists():
            raise SystemExit(f"Checkpoint tidak ditemukan: {p}")
        sd = torch.load(p, map_location='cpu', weights_only=True)
        # buang awalan 'module.' bila checkpoint berasal dari DataParallel/DDP
        sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
              for k, v in sd.items()}
        net.load_state_dict(sd)
        net.eval()
    return G, A


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(
        description="Terjemahkan citra source ke gaya domain target.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--repo-root', required=True,
                    help='folder kloningan pytorch-CycleGAN-and-pix2pix')
    ap.add_argument('--checkpoints', required=True,
                    help='folder checkpoints/<name> hasil training')
    ap.add_argument('--epoch', default='200',
                    help="epoch checkpoint; 'latest' juga bisa")
    ap.add_argument('--input', required=True,
                    help='folder masukan; subfolder kelas dipertahankan')
    ap.add_argument('--output', required=True)
    ap.add_argument('--bg-source', default='input',
                    choices=['input', 'generated'],
                    help='HARUS sama dengan yang dipakai saat training')
    ap.add_argument('--load-size', type=int, default=128)
    ap.add_argument('--ngf', type=int, default=64)
    ap.add_argument('--no-skip', action='store_true',
                    help='hanya bila training memakai --no_skip')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--save-mask', default=None,
                    help='folder opsional untuk menyimpan mask atensi s_a')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    G, A = load_nets(args)
    dev = torch.device(args.device)
    G.to(dev)
    A.to(dev)

    tf = transforms.Compose([
        transforms.Resize([args.load_size, args.load_size],
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    in_root, out_root = Path(args.input), Path(args.output)
    files = [p for p in sorted(in_root.rglob('*'))
             if p.is_file() and p.suffix.lower() in IMG_EXT]
    if not files:
        raise SystemExit(f"Tidak ada citra di {in_root}")
    print(f"{len(files)} citra akan diterjemahkan dari {in_root}")

    mask_root = Path(args.save_mask) if args.save_mask else None
    to_pil = transforms.ToPILImage()
    done = 0

    for i in range(0, len(files), args.batch_size):
        chunk = files[i:i + args.batch_size]
        batch = torch.stack([tf(Image.open(p).convert('RGB')) for p in chunk]).to(dev)

        raw = G(batch)                     # G(s)
        s_a, out = A(raw, batch)           # s_a, s'

        out = (out.clamp(-1, 1) + 1.0) / 2.0
        for j, p in enumerate(chunk):
            rel = p.relative_to(in_root)
            dst = out_root / rel.with_suffix('.png')
            dst.parent.mkdir(parents=True, exist_ok=True)
            to_pil(out[j].cpu()).save(dst)

            if mask_root is not None:
                md = mask_root / rel.with_suffix('.png')
                md.parent.mkdir(parents=True, exist_ok=True)
                to_pil(s_a[j].cpu().expand(3, -1, -1)).save(md)
        done += len(chunk)
        if done % 500 < args.batch_size:
            print(f"  {done}/{len(files)}", file=sys.stderr)

    print(f"Selesai. {done} citra tersimpan di {out_root}")
    print("Nama berkas dan struktur subfolder dipertahankan, sehingga label "
          "kelas dan ID subjek tetap terbawa.")


if __name__ == '__main__':
    main()