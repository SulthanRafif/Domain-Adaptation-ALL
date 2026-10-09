#!/usr/bin/env python3
"""Ekstraksi crop sel tunggal dari citra lapang pandang Taleqani.

CLAHE pada kanal saturasi + Otsu dipakai untuk membuat kandidat mask. Filter
hue membantu mengurangi eritrosit pucat yang ikut terdeteksi. Crop RGB berasal
dari citra asli. Hasil ditulis ke folder baru dan tidak menimpa data lama.

Input:
    input_root/<kelas>/<citra_lapang_pandang>

Output:
    out_root/{crop_original,crop_mask,crop_foreground}/<kelas>/<field>_cellNNN.png
    out_root/{field_mask,qc}/<kelas>/<field>.png
    out_root/extraction_report.csv  (kompatibel dengan scripts/build_dataset.py)
    out_root/summary.csv

Catatan: Otsu diterapkan pada kanal saturasi yang ditingkatkan CLAHE; rentang hue
dan saturasi membantu mengurangi eritrosit pucat. Mask dan bounding box tetap
harus diperiksa melalui QC; segmentasi otomatis tidak dijamin sempurna.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, segmentation
from skimage.feature import peak_local_max

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def make_mask(bgr, args):
    """Mask kandidat leukosit dari CLAHE pada saturasi lalu Otsu."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    clahe = cv2.createCLAHE(
        clipLimit=args.clip_limit,
        tileGridSize=(args.tile_grid, args.tile_grid),
    )
    enhanced_saturation = clahe.apply(saturation)
    threshold, otsu = cv2.threshold(
        enhanced_saturation, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    hue = hsv[:, :, 0]
    mask = (
        (otsu > 0)
        & (saturation >= args.min_saturation)
        & (hue >= args.min_hue)
        & (hue <= args.max_hue)
    )

    mask8 = mask.astype(np.uint8) * 255
    if args.open_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.open_radius + 1, 2 * args.open_radius + 1),
        )
        mask8 = cv2.morphologyEx(mask8, cv2.MORPH_OPEN, k)
    if args.close_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.close_radius + 1, 2 * args.close_radius + 1),
        )
        mask8 = cv2.morphologyEx(mask8, cv2.MORPH_CLOSE, k)

    mask = ndi.binary_fill_holes(mask8 > 0)
    return threshold, mask


def label_objects(mask, args):
    if args.split_touching:
        distance = ndi.distance_transform_edt(mask)
        coords = peak_local_max(
            distance,
            min_distance=args.min_distance,
            labels=mask.astype(np.uint8),
            exclude_border=False,
        )
        markers = np.zeros(mask.shape, dtype=np.int32)
        for idx, (row, col) in enumerate(coords, start=1):
            markers[row, col] = idx
        markers, _ = ndi.label(markers > 0)
        if markers.max() > 0:
            return segmentation.watershed(-distance, markers, mask=mask)
    labels, _ = ndi.label(mask)
    return labels.astype(np.int32)


def select_regions(labels, saturation, args):
    h, w = labels.shape
    selected = []
    for region in measure.regionprops(labels):
        if region.area < args.min_area or region.area > args.max_area:
            continue
        if region.solidity < args.min_solidity:
            continue
        r0, c0, r1, c1 = region.bbox
        if not args.keep_border and (
            r0 <= args.border_margin or c0 <= args.border_margin
            or r1 >= h - args.border_margin or c1 >= w - args.border_margin
        ):
            continue
        region_mask = labels == region.label
        mean_sat = float(saturation[region_mask].mean())
        if mean_sat < args.min_saturation:
            continue
        selected.append((region, mean_sat))
    selected.sort(key=lambda item: (item[0].centroid[0], item[0].centroid[1]))
    return selected


def crop_square(image, mask, bbox, pad_fraction, out_size):
    r0, c0, r1, c1 = bbox
    side = max(r1 - r0, c1 - c0)
    side = max(1, int(round(side * (1.0 + 2.0 * pad_fraction))))
    cy, cx = (r0 + r1) / 2.0, (c0 + c1) / 2.0
    top, left = int(round(cy - side / 2)), int(round(cx - side / 2))

    h, w = image.shape[:2]
    pad_top, pad_left = max(0, -top), max(0, -left)
    pad_bottom = max(0, top + side - h)
    pad_right = max(0, left + side - w)
    if pad_top or pad_left or pad_bottom or pad_right:
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101,
        )
        mask = cv2.copyMakeBorder(
            mask, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0,
        )
        top += pad_top
        left += pad_left

    image_crop = image[top:top + side, left:left + side]
    mask_crop = mask[top:top + side, left:left + side]
    interpolation = cv2.INTER_AREA if side > out_size else cv2.INTER_CUBIC
    image_crop = cv2.resize(image_crop, (out_size, out_size), interpolation=interpolation)
    mask_crop = cv2.resize(mask_crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return image_crop, mask_crop


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Gagal menyimpan citra: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ekstrak sel Taleqani dengan CLAHE, Otsu, dan crop dari citra asli."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid", type=int, default=8)
    parser.add_argument("--min-saturation", type=float, default=45.0)
    parser.add_argument("--min-hue", type=int, default=118,
                        help="batas bawah rona HSV untuk kandidat ungu/biru")
    parser.add_argument("--max-hue", type=int, default=160,
                        help="batas atas rona HSV untuk kandidat ungu/biru")
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--max-area", type=int, default=12000)
    parser.add_argument("--min-solidity", type=float, default=0.15)
    parser.add_argument("--open-radius", type=int, default=1)
    parser.add_argument("--close-radius", type=int, default=2)
    parser.add_argument("--split-touching", action="store_true")
    parser.add_argument("--min-distance", type=int, default=12)
    parser.add_argument("--keep-border", action="store_true")
    parser.add_argument("--border-margin", type=int, default=2)
    parser.add_argument("--pad", type=float, default=0.65)
    parser.add_argument("--out-size", type=int, default=128)
    parser.add_argument(
        "--limit-fields", type=int, default=0,
        help="batasi jumlah lapang untuk QC awal; 0 memproses semuanya",
    )
    args = parser.parse_args()

    input_root, out_root = Path(args.input_root), Path(args.out_root)
    files = sorted(
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )
    if args.limit_fields > 0:
        files = files[:args.limit_fields]
    if not files:
        raise SystemExit(f"Tidak ada citra pada {input_root}")

    report_fields = [
        "class", "file", "field_id", "source_path", "otsu_threshold",
        "area_px", "solidity", "mean_saturation", "bbox_x", "bbox_y",
        "bbox_w", "bbox_h",
    ]
    counts = Counter()
    rows = []

    for index, path in enumerate(files, start=1):
        relative = path.relative_to(input_root)
        if len(relative.parts) < 2:
            counts["skipped_no_class"] += 1
            continue
        class_name = relative.parts[0]
        field_id = path.stem
        bgr = read_image(path)
        if bgr is None:
            counts["skipped_unreadable"] += 1
            continue

        threshold, mask = make_mask(bgr, args)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        labels = label_objects(mask, args)
        selected = select_regions(labels, hsv[:, :, 1], args)

        field_mask = np.zeros(mask.shape, dtype=np.uint8)
        qc = bgr.copy()
        for cell_index, (region, mean_sat) in enumerate(selected, start=1):
            region_mask = (labels == region.label).astype(np.uint8) * 255
            field_mask[region_mask > 0] = 255
            r0, c0, r1, c1 = region.bbox
            crop, crop_mask = crop_square(
                bgr, region_mask, region.bbox, args.pad, args.out_size
            )
            foreground = crop * (crop_mask > 0).astype(np.uint8)[:, :, None]
            filename = f"{field_id}_cell{cell_index:03d}.png"

            for folder, output in (
                ("crop_original", crop),
                ("crop_mask", crop_mask),
                ("crop_foreground", foreground),
            ):
                save_image(out_root / folder / class_name / filename, output)

            cv2.rectangle(qc, (c0, r0), (c1 - 1, r1 - 1), (0, 255, 0), 1)
            cv2.putText(
                qc, str(cell_index), (c0, max(10, r0 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
                cv2.LINE_AA,
            )
            rows.append({
                "class": class_name,
                "file": filename,
                "field_id": field_id,
                "source_path": str(path),
                "otsu_threshold": round(float(threshold), 3),
                "area_px": int(region.area),
                "solidity": round(float(region.solidity), 4),
                "mean_saturation": round(mean_sat, 3),
                "bbox_x": c0,
                "bbox_y": r0,
                "bbox_w": c1 - c0,
                "bbox_h": r1 - r0,
            })
            counts[f"cells_{class_name}"] += 1

        save_image(out_root / "field_mask" / class_name / f"{field_id}.png", field_mask)
        save_image(out_root / "qc" / class_name / f"{field_id}.png", qc)
        counts["fields_processed"] += 1
        if not selected:
            counts["fields_no_cells"] += 1
        if index % 100 == 0 or index == len(files):
            print(f"Proses {index}/{len(files)} lapang; crop tersimpan {len(rows)}")

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "extraction_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(rows)

    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["fields_found", len(files)])
        writer.writerow(["fields_processed", counts["fields_processed"]])
        writer.writerow(["fields_without_cells", counts["fields_no_cells"]])
        writer.writerow(["crops_total", len(rows)])
        for key, value in sorted(counts.items()):
            if key.startswith("cells_") or key.startswith("skipped_"):
                writer.writerow([key, value])

    print(f"Selesai. Crop: {len(rows)}; ringkasan: {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
