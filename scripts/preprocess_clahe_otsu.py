#!/usr/bin/env python3
"""CLAHE -> Otsu -> mask -> bounding-box crop.

Input:
    input_root/<kelas>/<gambar>

Output:
    out_root/{enhanced,mask,masked_enhanced,
              crop_original,crop_mask,crop_foreground,
              crop_masked_enhanced,qc}/<kelas>/<gambar>.png
    out_root/manifest.csv
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def clahe_lab(bgr, clip_limit=2.0, tile_grid=8):
    """Tingkatkan kontras pada kanal luminance, pertahankan kanal warna."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid, tile_grid),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced, lab[:, :, 0]


def otsu_mask(l_channel, foreground):
    """Mask berisi 255 untuk foreground dan 0 untuk background."""
    mode = (
        cv2.THRESH_BINARY_INV if foreground == "dark"
        else cv2.THRESH_BINARY
    )
    threshold, mask = cv2.threshold(
        l_channel, 0, 255, mode | cv2.THRESH_OTSU
    )
    return threshold, mask


def crop_square(image, mask, bbox, pad_fraction, out_size):
    """Crop persegi di sekitar bbox; padding citra reflektif, mask hitam."""
    x, y, w, h = bbox
    side = max(w, h)
    side = max(1, side + int(round(side * 2 * pad_fraction)))

    cx = x + w / 2
    cy = y + h / 2
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))

    image_h, image_w = image.shape[:2]
    left = max(0, -x0)
    top = max(0, -y0)
    right = max(0, x0 + side - image_w)
    bottom = max(0, y0 + side - image_h)

    if left or top or right or bottom:
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )
        mask = cv2.copyMakeBorder(
            mask,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        x0 += left
        y0 += top

    image_crop = image[y0:y0 + side, x0:x0 + side]
    mask_crop = mask[y0:y0 + side, x0:x0 + side]

    image_crop = cv2.resize(
        image_crop,
        (out_size, out_size),
        interpolation=cv2.INTER_AREA,
    )
    mask_crop = cv2.resize(
        mask_crop,
        (out_size, out_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return image_crop, mask_crop


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Gagal menyimpan gambar: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--foreground",
        choices=("dark", "bright"),
        default="dark",
    )
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid", type=int, default=8)
    parser.add_argument("--pad", type=float, default=0.15)
    parser.add_argument("--out-size", type=int, default=128)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    out_root = Path(args.out_root)

    files = sorted(
        path for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXT
    )
    if not files:
        raise SystemExit(f"Tidak ada gambar di {input_root}")

    manifest_path = out_root / "manifest.csv"
    fields = [
        "class",
        "file",
        "field_id",
        "source_path",
        "otsu_threshold",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]

    processed = 0
    skipped = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for path in files:
            bgr = read_image(path)
            if bgr is None:
                print(f"LEWATI: tidak dapat membaca {path}")
                skipped += 1
                continue

            relative = path.relative_to(input_root)
            if len(relative.parts) < 2:
                print(f"LEWATI: gambar tidak berada di folder kelas: {path}")
                skipped += 1
                continue

            class_name = relative.parts[0]
            class_relative = Path(*relative.parts[1:]).with_suffix(".png")
            stem = path.stem

            # CLAHE dipakai untuk menghasilkan mask Otsu.
            enhanced, l_channel = clahe_lab(
                bgr,
                clip_limit=args.clip_limit,
                tile_grid=args.tile_grid,
            )
            threshold, mask = otsu_mask(
                l_channel,
                foreground=args.foreground,
            )

            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                print(f"LEWATI: mask kosong untuk {path}")
                skipped += 1
                continue

            x = int(xs.min())
            y = int(ys.min())
            w = int(xs.max() - x + 1)
            h = int(ys.max() - y + 1)
            bbox = (x, y, w, h)

            masked_enhanced = cv2.bitwise_and(
                enhanced,
                enhanced,
                mask=mask,
            )

            # Crop citra asli dan crop mask memakai bbox yang sama.
            original_crop, crop_mask = crop_square(
                bgr,
                mask,
                bbox,
                args.pad,
                args.out_size,
            )

            # Mask biner mengalikan crop citra asli.
            # Piksel foreground tetap dari citra asli; background menjadi 0.
            binary_mask = (crop_mask > 0).astype(original_crop.dtype)
            foreground_crop = original_crop * binary_mask[:, :, None]

            # Disimpan sebagai keluaran tambahan untuk inspeksi/eksperimen.
            enhanced_crop, _ = crop_square(
                enhanced,
                mask,
                bbox,
                args.pad,
                args.out_size,
            )
            masked_enhanced_crop = cv2.bitwise_and(
                enhanced_crop,
                enhanced_crop,
                mask=crop_mask,
            )

            outputs = {
                "enhanced": enhanced,
                "mask": mask,
                "masked_enhanced": masked_enhanced,
                "crop_original": original_crop,
                "crop_mask": crop_mask,
                "crop_foreground": foreground_crop,
                "crop_masked_enhanced": masked_enhanced_crop,
            }

            for folder, image in outputs.items():
                save_image(
                    out_root / folder / class_name / class_relative,
                    image,
                )

            # Overlay untuk inspeksi bounding box.
            qc = bgr.copy()
            cv2.rectangle(
                qc,
                (x, y),
                (x + w - 1, y + h - 1),
                (0, 255, 0),
                2,
            )
            save_image(
                out_root / "qc" / class_name / class_relative,
                qc,
            )

            # Pertahankan ID lapang pandang untuk pengelompokan dataset.
            field_id = stem
            if "_cell" in stem:
                field_id = stem.rsplit("_cell", 1)[0]

            writer.writerow({
                "class": class_name,
                "file": class_relative.as_posix(),
                "field_id": field_id,
                "source_path": str(path),
                "otsu_threshold": round(float(threshold), 3),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
            })
            processed += 1

    print(f"Selesai: {processed} diproses, {skipped} dilewati.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()