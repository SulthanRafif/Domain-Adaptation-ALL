#!/usr/bin/env python3
"""Compare source, translated, and target image distributions quantitatively.

The style metrics compare *sets* of unpaired images. The shape metrics pair
source and translated images by relative filename. Output includes masked Lab
statistics, LBP texture histograms, Dice/IoU shape preservation, and optional
standard Inception FID (via pytorch-fid).

Example:
  python eval/compare_translation_metrics.py \
      --source datasets/.../gan/testA \
      --translated results/epoch30/testA \
      --target datasets/.../gan/trainB \
      --out results/metrics_epoch30

For checkpoint selection, use a target validation folder that was held out
from GAN training. Keep testB untouched for the final evaluation. If explicit
segmentation masks are available, pass the corresponding --*-mask-dir options.
Otherwise masks are estimated automatically (black-background threshold when
the image has a dark border, Otsu on saturation otherwise).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance
from skimage import color
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LAB_FEATURES = ("L", "a", "b")
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2  # uniform LBP: P + 2 bins


def list_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXT)


def load_rgb(path, size):
    return np.asarray(Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.BICUBIC))


def match_mask(mask_root, image_path, image_root):
    if mask_root is None:
        return None
    rel = image_path.relative_to(image_root)
    candidates = [Path(mask_root) / rel,
                  (Path(mask_root) / rel).with_suffix(".png")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_mask(path, size):
    mask = Image.open(path).convert("L").resize(
        (size, size), Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def estimate_mask(rgb, mode="auto"):
    """Estimate foreground mask; explicit crop masks are preferable."""
    rgb_f = rgb.astype(np.float32) / 255.0
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    dark_border = float(np.median(border)) < 24
    selected = "black" if mode == "auto" and dark_border else mode
    if mode == "auto" and not dark_border:
        selected = "otsu-saturation"

    if selected == "black":
        mask = np.max(rgb, axis=2) > 16
    elif selected == "otsu-saturation":
        saturation = color.rgb2hsv(rgb_f)[:, :, 1]
        try:
            threshold = threshold_otsu(saturation)
        except ValueError:
            return np.zeros(rgb.shape[:2], dtype=bool)
        mask = saturation > max(0.08, float(threshold))
    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    mask = binary_closing(mask, footprint=disk(2))
    mask = ndi.binary_fill_holes(mask)
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    largest = int(np.argmax(areas))
    result = labels == largest
    # Reject degenerate masks; in that case use the non-filled threshold mask.
    fraction = float(result.mean())
    if fraction < 0.005 or fraction > 0.98:
        return np.asarray(mask, dtype=bool)
    return result


def get_mask(rgb, explicit_path, mode):
    if explicit_path is not None:
        mask = read_mask(explicit_path, rgb.shape[0])
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Mask size mismatch: {explicit_path}")
        return mask, "provided"
    return estimate_mask(rgb, mode), mode


def image_features(rgb, mask, rng, pixel_sample=512):
    if int(mask.sum()) < 16:
        return None
    lab = color.rgb2lab(rgb.astype(np.float32) / 255.0)
    pixels = lab[mask]
    if len(pixels) > pixel_sample:
        pixels = pixels[rng.choice(len(pixels), pixel_sample, replace=False)]
    medians = np.median(lab[mask], axis=0)

    gray = color.rgb2gray(rgb.astype(np.float32) / 255.0)
    gray8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray8, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp[mask], bins=np.arange(LBP_BINS + 1), density=False)
    hist = hist.astype(np.float64)
    hist /= max(1.0, hist.sum())
    return {
        "lab_pixels": pixels,
        "lab_median": medians,
        "lbp_hist": hist,
        "foreground_pixels": int(mask.sum()),
    }


def collect_set(root, mask_root, size, mask_mode, max_images, seed, set_name):
    root = Path(root)
    paths = list_images(root)
    if max_images and len(paths) > max_images:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(paths), max_images, replace=False))
        paths = [paths[i] for i in chosen]
    rows, skipped = [], []
    for index, path in enumerate(paths):
        rgb = load_rgb(path, size)
        mpath = match_mask(mask_root, path, root)
        mask, used_mode = get_mask(rgb, mpath, mask_mode)
        rng = np.random.default_rng(seed + index)
        feat = image_features(rgb, mask, rng)
        if feat is None:
            skipped.append(str(path))
            continue
        rows.append({
            "name": path.relative_to(root).as_posix(),
            "mask_mode": used_mode,
            **feat,
        })
    if not rows:
        raise SystemExit(f"Tidak ada citra bermask valid pada {root}")
    return rows, skipped


def bootstrap_w1(a, b, repeats, seed):
    point = float(wasserstein_distance(a, b))
    if repeats <= 0 or len(a) < 2 or len(b) < 2:
        return {"distance": point, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    a, b = np.asarray(a), np.asarray(b)
    for i in range(repeats):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        estimates[i] = wasserstein_distance(aa, bb)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"distance": point, "ci95_low": float(low), "ci95_high": float(high)}


def compare_style(source_rows, translated_rows, target_rows, bootstrap, seed):
    result = {}
    sets = {
        "source_to_target": source_rows,
        "translated_to_target": translated_rows,
    }
    for comparison, rows in sets.items():
        result[comparison] = {}
        for channel_index, channel in enumerate(LAB_FEATURES):
            left = [r["lab_median"][channel_index] for r in rows]
            right = [r["lab_median"][channel_index] for r in target_rows]
            result[comparison][f"Lab_{channel}_cell_median_W1"] = bootstrap_w1(
                left, right, bootstrap, seed + channel_index)

        left_lbp = np.mean([r["lbp_hist"] for r in rows], axis=0)
        right_lbp = np.mean([r["lbp_hist"] for r in target_rows], axis=0)
        result[comparison]["LBP_histogram_mean_L1"] = float(
            np.abs(left_lbp - right_lbp).sum())
    return result


def paired_shape(source_rows, translated_rows, source_root, translated_root,
                 source_mask_root, translated_mask_root, size, mask_mode):
    translated_by_name = {r["name"]: r for r in translated_rows}
    # Re-read paired images so masks are generated consistently at the same size.
    source_root, translated_root = Path(source_root), Path(translated_root)
    dices, ious, area_ratios = [], [], []
    for row in source_rows:
        rel = Path(row["name"])
        src_path = source_root / rel
        fake_rel = rel.with_suffix(".png")
        fake_path = translated_root / fake_rel
        if not fake_path.exists():
            fake_rel = rel
            fake_path = translated_root / rel
        if (not src_path.exists() or not fake_path.exists()
                or fake_rel.as_posix() not in translated_by_name):
            continue
        src_rgb, fake_rgb = load_rgb(src_path, size), load_rgb(fake_path, size)
        src_mp = match_mask(source_mask_root, src_path, source_root)
        fake_mp = match_mask(translated_mask_root, fake_path, translated_root)
        src_mask, _ = get_mask(src_rgb, src_mp, mask_mode)
        fake_mask, _ = get_mask(fake_rgb, fake_mp, mask_mode)
        union = np.logical_or(src_mask, fake_mask).sum()
        if src_mask.sum() == 0 or fake_mask.sum() == 0 or union == 0:
            continue
        inter = np.logical_and(src_mask, fake_mask).sum()
        dices.append(float(2 * inter / (src_mask.sum() + fake_mask.sum())))
        ious.append(float(inter / union))
        area_ratios.append(float(fake_mask.sum() / src_mask.sum()))
    if not dices:
        return {"n_pairs": 0, "note": "No matched source/translated masks."}

    def summary(values):
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }
    return {
        "n_pairs": len(dices),
        "dice": summary(dices),
        "iou": summary(ious),
        "area_ratio_fake_over_source": summary(area_ratios),
    }


def maybe_fid(source_root, translated_root, target_root, batch_size, device):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError:
        return {"available": False,
                "note": "Install pytorch-fid and rerun with --fid to calculate."}
    return {
        "available": True,
        "feature_note": "Standard Inception FID on full RGB images; not color-specific.",
        "source_to_target": float(calculate_fid_given_paths(
            [str(source_root), str(target_root)], batch_size, device, 2048, 0)),
        "translated_to_target": float(calculate_fid_given_paths(
            [str(translated_root), str(target_root)], batch_size, device, 2048, 0)),
    }


def write_per_image_csv(path, named_sets):
    fields = ["set", "image", "mask_mode", "foreground_pixels",
              "Lab_L_median", "Lab_a_median", "Lab_b_median"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for set_name, rows in named_sets.items():
            for row in rows:
                writer.writerow({
                    "set": set_name,
                    "image": row["name"],
                    "mask_mode": row["mask_mode"],
                    "foreground_pixels": row["foreground_pixels"],
                    "Lab_L_median": row["lab_median"][0],
                    "Lab_a_median": row["lab_median"][1],
                    "Lab_b_median": row["lab_median"][2],
                })


def main():
    ap = argparse.ArgumentParser(
        description="Bandingkan distribusi warna/tekstur dan bentuk translasi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="citra source yang menjadi input translasi, mis. testA")
    ap.add_argument("--translated", required=True,
                    help="hasil translate.py dari --source")
    ap.add_argument("--target", required=True,
                    help="folder referensi target; gunakan validation split untuk pemilihan epoch")
    ap.add_argument("--out", default="results/translation_metrics")
    ap.add_argument("--source-mask-dir", default=None)
    ap.add_argument("--translated-mask-dir", default=None)
    ap.add_argument("--target-mask-dir", default=None)
    ap.add_argument("--mask-mode", choices=["auto", "black", "otsu-saturation"],
                    default="auto")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=0,
                    help="0 berarti semua citra; sampling deterministik")
    ap.add_argument("--bootstrap", type=int, default=500,
                    help="jumlah bootstrap untuk CI 95%% jarak Lab")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fid", action="store_true",
                    help="hitung juga Inception FID (perlu pytorch-fid)")
    ap.add_argument("--fid-batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu",
                    help="device untuk --fid; gunakan cuda jika tersedia")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Mengumpulkan fitur foreground per citra...")
    source_rows, skipped_source = collect_set(
        args.source, args.source_mask_dir, args.img_size, args.mask_mode,
        args.max_images, args.seed, "source")
    translated_rows, skipped_translated = collect_set(
        args.translated, args.translated_mask_dir, args.img_size, args.mask_mode,
        args.max_images, args.seed, "translated")
    target_rows, skipped_target = collect_set(
        args.target, args.target_mask_dir, args.img_size, args.mask_mode,
        args.max_images, args.seed, "target")

    style = compare_style(source_rows, translated_rows, target_rows,
                          args.bootstrap, args.seed)
    shape = paired_shape(source_rows, translated_rows, args.source,
                         args.translated, args.source_mask_dir,
                         args.translated_mask_dir, args.img_size, args.mask_mode)
    result = {
        "counts": {
            "source": len(source_rows), "translated": len(translated_rows),
            "target": len(target_rows),
        },
        "mask_failures": {
            "source": skipped_source, "translated": skipped_translated,
            "target": skipped_target,
        },
        "style_distribution": style,
        "shape_preservation": shape,
        "notes": [
            "Lab distances compare per-image cell median distributions; lower is closer.",
            "Lab W1 is measured in Lab channel units; LBP histogram L1 is in [0, 2].",
            "Dice/IoU compare source and translated masks paired by relative filename.",
            "Automatic masks are approximate; inspect segmentation or pass explicit masks.",
        ],
    }
    if args.fid:
        result["fid"] = maybe_fid(args.source, args.translated, args.target,
                                  args.fid_batch_size, args.device)

    json_path = out / "metrics.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_per_image_csv(out / "per_image_features.csv", {
        "source": source_rows,
        "translated": translated_rows,
        "target": target_rows,
    })

    print(f"Source/translasi/target valid: {len(source_rows)}/"
          f"{len(translated_rows)}/{len(target_rows)}")
    print("Jarak gaya (lebih kecil lebih dekat ke target):")
    for comparison, metrics in style.items():
        labs = []
        for channel in LAB_FEATURES:
            m = metrics[f"Lab_{channel}_cell_median_W1"]
            ci = (f" [{m['ci95_low']:.3f}, {m['ci95_high']:.3f}]"
                  if m["ci95_low"] is not None else "")
            labs.append(f"{channel}={m['distance']:.3f}{ci}")
        print(f"  {comparison}: " + ", ".join(labs)
              + f", LBP-L1={metrics['LBP_histogram_mean_L1']:.4f}")
    print(f"Pelestarian bentuk: {json.dumps(shape, ensure_ascii=False)}")
    if args.fid:
        print(f"FID: {json.dumps(result['fid'], ensure_ascii=False)}")
    print(f"Hasil tersimpan: {json_path} dan {out / 'per_image_features.csv'}")


if __name__ == "__main__":
    main()
