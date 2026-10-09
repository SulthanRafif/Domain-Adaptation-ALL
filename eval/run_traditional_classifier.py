#!/usr/bin/env python3
"""Train SVM and Random Forest baselines on foreground color/texture features.

Expected layout (same class subfolders and names in all three splits)::

    train/<class>/*.{png,jpg,...}
    val/<class>/*.{png,jpg,...}
    test/<class>/*.{png,jpg,...}

Images may already be foreground crops with a black background. By default,
non-black pixels define the foreground. An optional mask directory can be
provided for each split; it must mirror the image directory structure and
contain matching filenames. Hyperparameters are selected on validation only;
the target test split is scored once after selection.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def canonical_class(name):
    """Unify the source and Taleqani naming used by this repository."""
    key = name.strip().lower().replace("_", "-")
    if key in {"all", "malignant"}:
        return "ALL"
    if key in {"normal", "benign"}:
        return "Normal"
    return name


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def foreground_mask(rgb, mask_path=None, black_threshold=2):
    if mask_path is not None:
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask pasangan tidak ditemukan: {mask_path}")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Ukuran mask {mask.shape} tidak sama dengan citra "
                             f"{rgb.shape[:2]}: {mask_path}")
    else:
        # Sesuai hasil crop_foreground: background bernilai 0.
        mask = np.max(rgb, axis=2) > black_threshold
    if not np.any(mask):
        raise ValueError("Mask foreground kosong; periksa citra/mask.")
    return mask


def lbp_histogram(gray, mask, bins=256):
    """Histogram LBP 8-neighbor dasar; dihitung pada piksel foreground."""
    g = gray.astype(np.int16)
    h, w = g.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    center = g[1:-1, 1:-1]
    neighbors = [g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:],
                 g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1],
                 g[2:, :-2], g[1:-1, :-2]]
    code = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    lbp[1:-1, 1:-1] = code
    values = lbp[mask]
    hist = np.bincount(values, minlength=bins).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def glcm_features(gray, mask, levels=16):
    """Small GLCM feature vector for four directions, restricted to mask."""
    quantized = np.minimum(gray.astype(np.int32) * levels // 256, levels - 1)
    h, w = gray.shape
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    features = []
    i, j = np.indices((levels, levels), dtype=np.float64)
    contrast_weight = (i - j) ** 2
    homogeneity_weight = 1.0 / (1.0 + np.abs(i - j))

    for dy, dx in directions:
        y0a, y1a = max(0, -dy), min(h, h - dy)
        x0a, x1a = max(0, -dx), min(w, w - dx)
        y0b, y1b = y0a + dy, y1a + dy
        x0b, x1b = x0a + dx, x1a + dx
        ma = mask[y0a:y1a, x0a:x1a]
        mb = mask[y0b:y1b, x0b:x1b]
        valid = ma & mb
        if not np.any(valid):
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue
        a = quantized[y0a:y1a, x0a:x1a][valid]
        b = quantized[y0b:y1b, x0b:x1b][valid]
        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (a, b), 1.0)
        matrix += matrix.T
        matrix /= matrix.sum()

        contrast = float(np.sum(matrix * contrast_weight))
        homogeneity = float(np.sum(matrix * homogeneity_weight))
        energy = float(np.sum(matrix ** 2))
        px, py = matrix.sum(axis=1), matrix.sum(axis=0)
        mx = float(np.sum(np.arange(levels) * px))
        my = float(np.sum(np.arange(levels) * py))
        sx = float(np.sqrt(np.sum(((np.arange(levels) - mx) ** 2) * px)))
        sy = float(np.sqrt(np.sum(((np.arange(levels) - my) ** 2) * py)))
        if sx > 1e-12 and sy > 1e-12:
            corr = float(np.sum(matrix * (i - mx) * (j - my)) / (sx * sy))
        else:
            corr = 0.0
        features.extend([contrast, homogeneity, energy, corr])
    return np.asarray(features, dtype=np.float64)


def extract_features(rgb, mask, lbp_weight=1.0, glcm_weight=1.0):
    # float RGB conversion gives Lab's conventional L*, a*, b* scale.
    rgb_float = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    color_features = []
    for channel in range(3):
        values = lab[:, :, channel][mask].astype(np.float64)
        color_features.extend([
            float(np.mean(values)), float(np.std(values)),
            float(np.median(values)),
            float(np.quantile(values, 0.10)),
            float(np.quantile(values, 0.90)),
        ])

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    texture = np.concatenate([
        lbp_histogram(gray, mask) * lbp_weight,
        glcm_features(gray, mask) * glcm_weight,
    ])
    return np.concatenate([np.asarray(color_features), texture])


def load_split(image_root, mask_root=None, black_threshold=2):
    root = Path(image_root)
    files = image_files(root)
    if not files:
        raise ValueError(f"Tidak ada citra di {root}")
    classes = sorted({canonical_class(p.name)
                      for p in root.iterdir() if p.is_dir()})
    if len(classes) < 2:
        raise ValueError(f"Diharapkan minimal dua subfolder kelas di {root}")
    class_to_idx = {name: i for i, name in enumerate(classes)}
    x, y, paths = [], [], []
    for path in files:
        rel = path.relative_to(root)
        label = canonical_class(rel.parts[0]) if rel.parts else None
        if label not in class_to_idx:
            raise ValueError(f"Citra tidak berada dalam subfolder kelas: {path}")
        rgb = load_rgb(path)
        paired_mask = Path(mask_root) / rel if mask_root else None
        mask = foreground_mask(rgb, paired_mask, black_threshold)
        x.append(extract_features(rgb, mask))
        y.append(class_to_idx[label])
        paths.append(str(path))
    return np.vstack(x), np.asarray(y, dtype=np.int64), paths, classes


def evaluate(y_true, y_pred, classes, positive_idx, positive_scores=None):
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(len(classes))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=np.arange(len(classes)),
            target_names=classes, output_dict=True, zero_division=0),
    }
    if (positive_scores is not None and len(classes) == 2
            and len(np.unique(y_true)) == 2):
        result["positive_class"] = classes[positive_idx]
        result["auc_roc"] = float(roc_auc_score(
            (y_true == positive_idx).astype(int),
            positive_scores))
    return result


def choose_svm(x_train, y_train, x_val, y_val, c_values, gamma_values,
               class_weight):
    candidates = []
    for c in c_values:
        for gamma in gamma_values:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("classifier", SVC(C=c, gamma=gamma, kernel="rbf",
                                   class_weight=class_weight)),
            ])
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            score = balanced_accuracy_score(y_val, pred)
            candidates.append((float(score), float(c), gamma, model))
    candidates.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    score, c, gamma, model = candidates[0]
    return model, {"validation_balanced_accuracy": score,
                   "C": c, "gamma": gamma}


def choose_rf(x_train, y_train, x_val, y_val, n_estimators, max_features,
              min_samples_leaf, seed, class_weight):
    candidates = []
    for trees in n_estimators:
        for max_feature in max_features:
            for leaf in min_samples_leaf:
                model = RandomForestClassifier(
                    n_estimators=trees, max_features=max_feature,
                    min_samples_leaf=leaf, class_weight=class_weight,
                    random_state=seed, n_jobs=-1)
                model.fit(x_train, y_train)
                pred = model.predict(x_val)
                score = balanced_accuracy_score(y_val, pred)
                candidates.append((float(score), int(trees), str(max_feature),
                                   int(leaf), model))
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    score, trees, max_feature, leaf, model = candidates[0]
    return model, {"validation_balanced_accuracy": score,
                   "n_estimators": trees, "max_features": max_feature,
                   "min_samples_leaf": leaf}


def parse_csv_values(value, converter, name):
    try:
        return [converter(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Nilai {name} tidak valid: {value}") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Baseline SVM/RF dari fitur warna Lab dan tekstur LBP/GLCM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", required=True,
                        help="target test; hanya dievaluasi setelah pemilihan")
    parser.add_argument("--train-mask-dir", default=None)
    parser.add_argument("--val-mask-dir", default=None)
    parser.add_argument("--test-mask-dir", default=None)
    parser.add_argument("--black-threshold", type=int, default=2)
    parser.add_argument("--out-dir", default="results/traditional_classifier")
    parser.add_argument("--tag", default="traditional_baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--svm-c", default="0.1,1,10,100")
    parser.add_argument("--svm-gamma", default="scale,0.001,0.01,0.1")
    parser.add_argument("--rf-trees", default="300,600")
    parser.add_argument("--rf-max-features", default="sqrt,0.5")
    parser.add_argument("--rf-min-leaf", default="1,2,4")
    parser.add_argument("--class-weight", choices=["none", "balanced"],
                        default="balanced")
    args = parser.parse_args()

    splits = {}
    for split, root, mask_root in [
            ("train", args.train_dir, args.train_mask_dir),
            ("val", args.val_dir, args.val_mask_dir),
            ("test", args.test_dir, args.test_mask_dir)]:
        splits[split] = load_split(root, mask_root, args.black_threshold)
    x_train, y_train, train_paths, classes = splits["train"]
    x_val, y_val, _, val_classes = splits["val"]
    x_test, y_test, test_paths, test_classes = splits["test"]
    if classes != val_classes or classes != test_classes:
        raise SystemExit("Nama/pemetaan kelas harus identik pada train, val, test: "
                         f"train={classes}, val={val_classes}, test={test_classes}")
    for name, labels, x in [("train", y_train, x_train),
                            ("val", y_val, x_val), ("test", y_test, x_test)]:
        print(f"{name:5s}: {len(labels)} citra, fitur={x.shape[1]}, "
              f"kelas=" + ", ".join(
                  f"{cls}={int(np.sum(labels == i))}"
                  for i, cls in enumerate(classes)))

    class_weight = None if args.class_weight == "none" else "balanced"
    c_values = parse_csv_values(args.svm_c, float, "svm-c")
    gamma_values = [v.strip() if v.strip() == "scale" else float(v.strip())
                    for v in args.svm_gamma.split(",")]
    trees = parse_csv_values(args.rf_trees, int, "rf-trees")
    max_features = [v.strip() if v.strip() in {"sqrt", "log2"}
                    else float(v.strip())
                    for v in args.rf_max_features.split(",")]
    leaves = parse_csv_values(args.rf_min_leaf, int, "rf-min-leaf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_idx = classes.index("ALL") if "ALL" in classes else len(classes) - 1

    svm, svm_params = choose_svm(x_train, y_train, x_val, y_val,
                                 c_values, gamma_values, class_weight)
    rf, rf_params = choose_rf(x_train, y_train, x_val, y_val,
                             trees, max_features, leaves, args.seed,
                             class_weight)
    results = {
        "tag": args.tag,
        "classes": classes,
        "positive_class": classes[pos_idx],
        "features": "Lab statistics + masked LBP histogram + masked GLCM",
        "foreground": ("paired mask" if any([args.train_mask_dir,
                                              args.val_mask_dir,
                                              args.test_mask_dir])
                       else f"non-black pixels (threshold={args.black_threshold})"),
        "splits": {"train": args.train_dir, "val": args.val_dir,
                   "test": args.test_dir},
        "models": {},
    }
    for name, model, params in [("svm", svm, svm_params), ("random_forest", rf, rf_params)]:
        # Test is touched only after validation-based hyperparameter selection.
        prediction = model.predict(x_test)
        if name == "svm":
            decision = model.decision_function(x_test)
            positive_scores = decision if pos_idx == 1 else -decision
        else:
            positive_scores = model.predict_proba(x_test)[:, pos_idx]
        result = evaluate(y_test, prediction, classes, pos_idx, positive_scores)
        result["selected_parameters"] = params
        results["models"][name] = result
        joblib.dump(model, out_dir / f"{args.tag}_{name}.joblib")
        print(f"\n{name}: val balanced accuracy="
              f"{params['validation_balanced_accuracy']:.4f}; "
              f"test balanced accuracy={result['balanced_accuracy']:.4f}; "
              f"test macro-F1={result['macro_f1']:.4f}")
        print("Confusion matrix (baris=label aktual, kolom=prediksi; "
              f"urutan kelas {classes}):\n{np.asarray(result['confusion_matrix'])}")

        with (out_dir / f"{args.tag}_{name}_test_predictions.csv").open(
                "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "true_class", "predicted_class"])
            for path, yt, yp in zip(test_paths, y_test, prediction):
                writer.writerow([path, classes[int(yt)], classes[int(yp)]])

    report_path = out_dir / f"{args.tag}_metrics.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nModel dan laporan tersimpan di: {out_dir}")
    print("Gunakan hasil test hanya sebagai evaluasi akhir; jangan memilih ulang "
          "parameter berdasarkan skor test.")


if __name__ == "__main__":
    main()
