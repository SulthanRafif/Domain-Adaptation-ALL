#!/usr/bin/env python3
"""
run_classifier.py — Tahap 5: latih classifier dan hitung metrik evaluasi.

Satu skrip untuk SEMUA baseline. Yang membedakan hanya --train-dir:

  source-only   --train-dir clf/train                (source ASLI, tanpa adaptasi)
  DA (usulan)   --train-dir clf_translated/train     (source hasil translasi)
  oracle        --train-dir clf_oracle/train         (sel target berlabel)

--val-dir dipakai untuk MEMILIH checkpoint. --test-dir dievaluasi SEKALI di akhir.
Subset tes tidak pernah menyentuh proses pelatihan maupun pemilihan model.

Metrik yang dilaporkan: accuracy, balanced accuracy, F1, AUC-ROC, sensitivity,
specificity, confusion matrix — di tingkat sel DAN di tingkat lapang pandang.

Contoh:
  python run_classifier.py \\
      --train-dir datasets/cnmc2taleqani/clf_translated/train \\
      --val-dir   datasets/cnmc2taleqani/clf_translated/val \\
      --test-dir  datasets/cnmc2taleqani/clf/test \\
      --tag da_skenarioA --seed 0
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_loaders(args):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(), norm])

    ds = {}
    for split, d, tf in [('train', args.train_dir, train_tf),
                         ('val', args.val_dir, eval_tf),
                         ('test', args.test_dir, eval_tf)]:
        ds[split] = datasets.ImageFolder(d, transform=tf)

    # Kelas harus dipetakan identik di ketiga split
    if not (ds['train'].class_to_idx == ds['val'].class_to_idx ==
            ds['test'].class_to_idx):
        raise SystemExit(f"Pemetaan kelas berbeda antar split:\n"
                         f"  train {ds['train'].class_to_idx}\n"
                         f"  val   {ds['val'].class_to_idx}\n"
                         f"  test  {ds['test'].class_to_idx}\n"
                         "Pastikan nama subfolder kelas sama persis.")

    loaders = {
        'train': DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=False),
        'val': DataLoader(ds['val'], batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers),
        'test': DataLoader(ds['test'], batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers),
    }
    return ds, loaders


def make_model(arch, n_classes):
    fn = {'resnet34': models.resnet34, 'resnet50': models.resnet50}[arch]
    net = fn(weights='IMAGENET1K_V1')
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


@torch.no_grad()
def predict(net, loader, dev):
    net.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(net(x.to(dev)), dim=1)
        probs.append(p.cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def field_id_of(path):
    """Ambil ID lapang pandang dari nama berkas hasil extract_cells.py:
    <PREFIX>_<kelas>_<field_id>_cell<NNN>.png"""
    m = re.match(r'.+?_(.+)_cell\d+$', Path(path).stem)
    return m.group(1) if m else Path(path).stem


def metrics(y_bin, p_pos, label=''):
    """y_bin: 1 untuk ALL, 0 untuk Normal.  p_pos: peluang kelas ALL.

    PENTING: ImageFolder mengurutkan kelas secara alfabetis, sehingga 'ALL'
    mendapat indeks 0 dan 'Normal' indeks 1. Seluruh metrik sklearn
    mengasumsikan kelas POSITIF berada di indeks 1. Tanpa pemetaan ulang di
    sini, sensitivity dan specificity tertukar, F1 dihitung untuk kelas Normal,
    dan AUC menjadi kebalikannya (0,02 alih-alih 0,98).
    """
    y_pred = (p_pos >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    try:
        auc = roc_auc_score(y_bin, p_pos)
    except ValueError:
        auc = float('nan')
    return {
        'level': label,
        'n': int(len(y_bin)),
        'accuracy': round(float(accuracy_score(y_bin, y_pred)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y_bin, y_pred)), 4),
        'f1': round(float(f1_score(y_bin, y_pred, zero_division=0)), 4),
        'auc_roc': round(float(auc), 4),
        'sensitivity': round(float(sens), 4),
        'specificity': round(float(spec), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp),
                             'fn': int(fn), 'tp': int(tp)},
    }


def aggregate_fields(paths, y_bin, p_pos, how='mean'):
    """Agregasi prediksi semua sel dalam satu lapang pandang.

    Bekerja langsung pada (y_bin, p_pos) — ALL = 1 — sehingga tidak ada lagi
    pembalikan indeks kelas.
    """
    buckets = {}
    for p, yt, pr in zip(paths, y_bin, p_pos):
        f = field_id_of(p)
        buckets.setdefault(f, {'y': int(yt), 'p': []})['p'].append(float(pr))
    ys = np.array([v['y'] for v in buckets.values()])
    agg = max if how == 'max' else (lambda v: float(np.mean(v)))
    ps = np.array([agg(v['p']) for v in buckets.values()])
    return ys, ps


def main():
    ap = argparse.ArgumentParser(
        description="Latih classifier dan hitung metrik pada domain target.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--tag', required=True, help='nama eksperimen untuk keluaran')
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--arch', default='resnet34', choices=['resnet34', 'resnet50'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--field-agg', default='mean', choices=['mean', 'max'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    ds, loaders = build_loaders(args)
    classes = ds['train'].classes
    pos_idx = classes.index('ALL') if 'ALL' in classes else 1
    print(f"Kelas: {ds['train'].class_to_idx} | kelas positif = {classes[pos_idx]}")
    for s in ('train', 'val', 'test'):
        cnt = np.bincount([y for _, y in ds[s].samples], minlength=len(classes))
        print(f"  {s:5s}: {len(ds[s]):6d} citra  " +
              '  '.join(f"{c}={n}" for c, n in zip(classes, cnt)))

    net = make_model(args.arch, len(classes)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{args.tag}_seed{args.seed}_best.pth"
    best_val, history = -1.0, []

    for ep in range(1, args.epochs + 1):
        net.train()
        tot = 0.0
        for x, y in loaders['train']:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        sched.step()

        vp, vy = predict(net, loaders['val'], dev)
        vbal = balanced_accuracy_score(vy, vp.argmax(1))
        history.append({'epoch': ep,
                        'train_loss': round(tot / len(ds['train']), 4),
                        'val_balanced_acc': round(float(vbal), 4)})
        marker = ''
        if vbal > best_val:                       # pemilihan checkpoint: VAL saja
            best_val = vbal
            torch.save(net.state_dict(), best_path)
            marker = '  <- terbaik'
        print(f"  epoch {ep:3d}/{args.epochs}  loss {tot/len(ds['train']):.4f}  "
              f"val_bal_acc {vbal:.4f}{marker}")

    # ---------- evaluasi SEKALI pada subset tes ----------
    net.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
    probs, ty = predict(net, loaders['test'], dev)
    paths = [p for p, _ in ds['test'].samples]
    # petakan ke soal biner dengan ALL = 1, sekali di sini, lalu seluruh
    # metrik di bawah memakai konvensi yang sama
    y_bin = (ty == pos_idx).astype(int)
    p_pos = probs[:, pos_idx]

    res = {
        'tag': args.tag, 'seed': args.seed, 'arch': args.arch,
        'train_dir': args.train_dir, 'test_dir': args.test_dir,
        'classes': ds['train'].class_to_idx,
        'best_val_balanced_acc': round(float(best_val), 4),
        'positive_class': classes[pos_idx],
        'cell_level': metrics(y_bin, p_pos, 'sel'),
    }
    fy, fp_ = aggregate_fields(paths, y_bin, p_pos, args.field_agg)
    res['field_level'] = metrics(fy, fp_,
                                 f'lapang pandang ({args.field_agg})')
    res['history'] = history

    with open(out_dir / f"{args.tag}_seed{args.seed}.json", 'w') as f:
        json.dump(res, f, indent=2)

    print(f"\n=== HASIL: {args.tag} (seed {args.seed}) ===")
    for key in ('cell_level', 'field_level'):
        m = res[key]
        print(f"\nTingkat {m['level']}  (n={m['n']})")
        for k in ('accuracy', 'balanced_accuracy', 'f1', 'auc_roc',
                  'sensitivity', 'specificity'):
            print(f"  {k:<20s} {m[k]}")
        c = m['confusion_matrix']
        print(f"  confusion            TN={c['tn']} FP={c['fp']} "
              f"FN={c['fn']} TP={c['tp']}")
    print(f"\nTersimpan: {out_dir / f'{args.tag}_seed{args.seed}.json'}")
    print("\nJangan memakai angka ini untuk menyetel apa pun. Kalau Anda kembali "
          "mengubah\nhyperparameter setelah melihatnya, subset tes sudah "
          "tercemar dan harus diganti.")


if __name__ == '__main__':
    main()