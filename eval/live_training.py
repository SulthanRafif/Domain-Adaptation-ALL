"""In-process validation metrics and early stopping for Attention-CycleGAN."""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_translation_metrics import (  # noqa: E402
    collect_set,
    list_images,
    compare_style,
    paired_shape,
)


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_transform(opt, image_size):
    """Use CycleGAN's configured geometry with a deterministic center crop."""
    w, h = image_size
    new_w, new_h = w, h
    if opt.preprocess in ("resize_and_crop", "scale_width_and_crop"):
        new_h = new_w = opt.load_size
    crop_x = max(0, (new_w - opt.crop_size) // 2)
    crop_y = max(0, (new_h - opt.crop_size) // 2)
    params = {"crop_pos": (crop_x, crop_y), "flip": False}
    eval_opt = SimpleNamespace(**vars(opt))
    eval_opt.no_flip = True
    from data.base_dataset import get_transform
    return get_transform(eval_opt, params=params)


def _mask_transform(opt, image_size):
    w, h = image_size
    new_w, new_h = w, h
    if opt.preprocess in ("resize_and_crop", "scale_width_and_crop"):
        new_h = new_w = opt.load_size
    params = {"crop_pos": (max(0, (new_w - opt.crop_size) // 2),
                            max(0, (new_h - opt.crop_size) // 2)),
              "flip": False}
    eval_opt = SimpleNamespace(**vars(opt))
    eval_opt.no_flip = True
    from data.base_dataset import get_transform
    return get_transform(eval_opt, params=params,
                         method=transforms.InterpolationMode.NEAREST,
                         convert=False)


def _resolve_mask(mask_root, image_path, image_root):
    if not mask_root:
        return None
    rel = image_path.relative_to(image_root)
    for candidate in (Path(mask_root) / rel,
                      (Path(mask_root) / rel).with_suffix(".png")):
        if candidate.is_file():
            return candidate
    return None


def _atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def _style_score(style):
    """Mean generated/source distance ratio; lower is closer to target."""
    src = style["source_to_target"]
    fake = style["translated_to_target"]
    ratios = []
    for channel in ("L", "a", "b"):
        key = f"Lab_{channel}_cell_median_W1"
        base = src[key]["distance"]
        adapted = fake[key]["distance"]
        ratios.append(adapted / max(base, 1e-6))
    base_lbp = src["LBP_histogram_mean_L1"]
    fake_lbp = fake["LBP_histogram_mean_L1"]
    ratios.append(fake_lbp / max(base_lbp, 1e-6))
    return float(np.mean(ratios))


class LiveTrainingEvaluator:
    """Evaluate a fixed validation subset and maintain early-stop state."""

    def __init__(self, opt):
        self.opt = opt
        self.source_root = Path(opt.eval_source_dir).resolve()
        self.target_root = Path(opt.eval_target_dir).resolve()
        self.source_mask_root = (Path(opt.eval_source_mask_dir).resolve()
                                 if opt.eval_source_mask_dir else None)
        self.dashboard_dir = (Path(opt.eval_dashboard_dir).resolve()
                              if opt.eval_dashboard_dir else
                              Path(opt.checkpoints_dir).resolve() / opt.name /
                              "evaluation_dashboard")
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.dashboard_dir / "history.csv"
        self.state_path = self.dashboard_dir / "metrics.json"
        self.preview_path = self.dashboard_dir / "preview.png"
        self.index_path = self.dashboard_dir / "index.html"
        template = Path(__file__).with_name("early_stop_dashboard.html")
        if template.exists():
            shutil.copyfile(template, self.index_path)

        previous = {}
        if self.state_path.exists():
            try:
                previous = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        es = previous.get("early_stopping", {})
        self.best_score = es.get("best_score")
        self.best_epoch = es.get("best_epoch")
        self.bad_checks = int(es.get("bad_checks", 0))
        self.checkpoint_dir = Path(opt.checkpoints_dir) / opt.name
        self.image_size = int(opt.crop_size)

        if not self.source_root.is_dir() or not self.target_root.is_dir():
            raise FileNotFoundError("eval_source_dir dan eval_target_dir harus menunjuk ke folder citra.")
        if self.source_mask_root is not None and not self.source_mask_root.is_dir():
            raise FileNotFoundError(f"Folder mask validasi tidak ditemukan: {self.source_mask_root}")
        if not list_images(self.source_root):
            raise FileNotFoundError(f"Tidak ada citra validasi di {self.source_root}")
        if not list_images(self.target_root):
            raise FileNotFoundError(f"Tidak ada citra target validasi di {self.target_root}")

    def _select_source_paths(self):
        paths = list_images(self.source_root)
        limit = int(self.opt.eval_max_images)
        if limit > 0 and len(paths) > limit:
            rng = np.random.default_rng(0)
            keep = np.sort(rng.choice(len(paths), limit, replace=False))
            paths = [paths[i] for i in keep]
        return paths

    @torch.no_grad()
    def _translate_and_preview(self, model, epoch, work_dir):
        source_paths = self._select_source_paths()
        translated_root = work_dir / "translated"
        translated_root.mkdir(parents=True, exist_ok=True)
        g = getattr(model, "netG_ST")
        a = getattr(model, "netA_S")
        g_module = g.module if hasattr(g, "module") else g
        a_module = a.module if hasattr(a, "module") else a
        old_g_mode, old_a_mode = g_module.training, a_module.training
        g_module.eval()
        a_module.eval()
        device = model.device
        preview_items = []

        try:
            for path in source_paths:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    transform = _image_transform(self.opt, image.size)
                    source_tensor = transform(image).unsqueeze(0).to(device)
                    raw = g_module(source_tensor)
                    attention, translated = a_module(raw, source_tensor)
                    out = ((translated[0].clamp(-1, 1) + 1.0) * 127.5)
                    out = out.byte().permute(1, 2, 0).cpu().numpy()
                    att = (attention[0, 0].clamp(0, 1) * 255).byte().cpu().numpy()
                    rel = path.relative_to(self.source_root).with_suffix(".png")
                    destination = translated_root / rel
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(out, mode="RGB").save(destination)
                    if len(preview_items) < 8:
                        src_preview = image.resize((self.image_size, self.image_size),
                                                   Image.Resampling.BICUBIC)
                        preview_items.append((src_preview, Image.fromarray(out),
                                              Image.fromarray(att, mode="L").convert("RGB")))
        finally:
            g_module.train(old_g_mode)
            a_module.train(old_a_mode)

        self._write_preview(preview_items, epoch)
        return translated_root

    def _write_preview(self, items, epoch):
        tile = self.image_size
        label_h = 24
        canvas = Image.new("RGB", (tile * 3, (tile + label_h) * max(1, len(items))),
                           "white")
        draw = ImageDraw.Draw(canvas)
        labels = ("source", "fake_T", "attention")
        for row, trio in enumerate(items):
            y = row * (tile + label_h)
            for col, (im, label) in enumerate(zip(trio, labels)):
                canvas.paste(im.resize((tile, tile)), (col * tile, y))
                draw.text((col * tile + 4, y + tile + 4), label, fill="black")
        temp = self.preview_path.with_suffix(".tmp.png")
        canvas.save(temp)
        temp.replace(self.preview_path)

    def _attention_iou(self, model):
        if self.source_mask_root is None:
            return None
        paths = self._select_source_paths()
        a = model.netA_S.module if hasattr(model.netA_S, "module") else model.netA_S
        g = model.netG_ST.module if hasattr(model.netG_ST, "module") else model.netG_ST
        old_g_mode, old_a_mode = g.training, a.training
        g.eval()
        a.eval()
        scores = []
        try:
            with torch.no_grad():
                for path in paths:
                    mask_path = _resolve_mask(self.source_mask_root, path, self.source_root)
                    if mask_path is None:
                        continue
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        source_size = image.size
                        transform = _image_transform(self.opt, image.size)
                        x = transform(image).unsqueeze(0).to(model.device)
                    with Image.open(mask_path) as mask_image:
                        mask_image = mask_image.convert("L")
                        if mask_image.size != source_size:
                            raise ValueError(
                                "Ukuran crop_mask harus sama dengan citra source "
                                f"sebelum transformasi: {path}={source_size}, "
                                f"{mask_path}={mask_image.size}."
                            )
                        mt = _mask_transform(self.opt, mask_image.size)(mask_image)
                        mask_values = np.asarray(mt)
                        threshold = 0 if mask_values.max(initial=0) <= 1 else 127
                        target = torch.from_numpy(mask_values > threshold).to(model.device)
                    pred = a(g(x), x)[0][0, 0] >= 0.5
                    if tuple(target.shape) != tuple(pred.shape):
                        raise ValueError(
                            "Ukuran attention dan crop_mask setelah transformasi "
                            f"berbeda: {tuple(pred.shape)} vs {tuple(target.shape)} "
                            f"(citra {path})."
                        )
                    target = target.bool()
                    intersection = torch.logical_and(pred, target).sum().item()
                    union = torch.logical_or(pred, target).sum().item()
                    if union:
                        scores.append(intersection / union)
        finally:
            g.train(old_g_mode)
            a.train(old_a_mode)
        return float(np.mean(scores)) if scores else None

    def evaluate(self, model, epoch):
        with tempfile.TemporaryDirectory(prefix=f"epoch{epoch:04d}_",
                                         dir=self.dashboard_dir) as temp_name:
            translated_root = self._translate_and_preview(model, epoch, Path(temp_name))
            source_rows, _ = collect_set(
                self.source_root, None, self.image_size, "auto",
                self.opt.eval_max_images, 0, "source")
            translated_rows, _ = collect_set(
                translated_root, None, self.image_size, "auto",
                self.opt.eval_max_images, 0, "translated")
            target_rows, _ = collect_set(
                self.target_root, None, self.image_size, "auto",
                self.opt.eval_max_images, 1, "target")
            style = compare_style(source_rows, translated_rows, target_rows,
                                  bootstrap=200, seed=0)
            shape = paired_shape(
                source_rows, translated_rows, self.source_root,
                translated_root, None, None, self.image_size, "auto")
            attention_iou = self._attention_iou(model)
            score = _style_score(style)

        dice = (shape.get("dice", {}).get("mean")
                if shape.get("n_pairs", 0) else None)
        shape_ok = dice is not None and dice >= self.opt.early_stop_min_dice
        attention_ok = (self.opt.early_stop_min_attention_iou <= 0
                        or (attention_iou is not None
                            and attention_iou >= self.opt.early_stop_min_attention_iou))
        eligible = bool(shape_ok and attention_ok)
        improved = (eligible and (self.best_score is None
                                  or score < self.best_score - self.opt.early_stop_min_delta))
        if improved:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.bad_checks = 0
            model.save_networks("best")
        elif epoch >= self.opt.early_stop_start_epoch:
            self.bad_checks += 1

        stop = (self.opt.early_stop_patience > 0
                and epoch >= self.opt.early_stop_start_epoch
                and self.best_score is not None
                and self.bad_checks >= self.opt.early_stop_patience)
        result = {
            "epoch": int(epoch),
            "timestamp": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
            "style_score": score,
            "style_distribution": style,
            "shape_preservation": shape,
            "attention_iou": attention_iou,
            "checkpoint_eligible": eligible,
            "improved": bool(improved),
            "early_stopping": {
                "enabled": self.opt.early_stop_patience > 0,
                "patience": int(self.opt.early_stop_patience),
                "bad_checks": int(self.bad_checks),
                "min_delta": float(self.opt.early_stop_min_delta),
                "start_epoch": int(self.opt.early_stop_start_epoch),
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "stop_now": bool(stop),
            },
            "validation_counts": {
                "source": len(source_rows), "translated": len(translated_rows),
                "target": len(target_rows),
            },
        }
        _atomic_json(self.state_path, result)
        self._append_history(result)
        self._write_dashboard_status(result)
        print(f"[validation epoch {epoch}] style_score={score:.4f}, "
              f"Dice={dice if dice is not None else 'n/a'}, "
              f"attention_IoU={attention_iou if attention_iou is not None else 'n/a'}, "
              f"best_epoch={self.best_epoch}, patience={self.bad_checks}/"
              f"{self.opt.early_stop_patience}")
        return stop

    def _append_history(self, result):
        file_exists = self.history_path.exists()
        fields = ["epoch", "style_score", "lab_l_ratio", "lab_a_ratio",
                  "lab_b_ratio", "lbp_ratio", "dice", "iou", "attention_iou",
                  "eligible", "improved", "bad_checks", "best_epoch"]
        st = result["style_distribution"]
        src, fake = st["source_to_target"], st["translated_to_target"]
        ratios = {}
        for channel in ("L", "a", "b"):
            k = f"Lab_{channel}_cell_median_W1"
            ratios[f"lab_{channel.lower()}_ratio"] = (
                fake[k]["distance"] / max(src[k]["distance"], 1e-6))
        ratios["lbp_ratio"] = (fake["LBP_histogram_mean_L1"] /
                               max(src["LBP_histogram_mean_L1"], 1e-6))
        shape = result["shape_preservation"]
        row = {
            "epoch": result["epoch"], "style_score": result["style_score"],
            **ratios,
            "dice": shape.get("dice", {}).get("mean"),
            "iou": shape.get("iou", {}).get("mean"),
            "attention_iou": result["attention_iou"],
            "eligible": result["checkpoint_eligible"],
            "improved": result["improved"],
            "bad_checks": result["early_stopping"]["bad_checks"],
            "best_epoch": result["early_stopping"]["best_epoch"],
        }
        with self.history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_dashboard_status(self, result):
        (self.dashboard_dir / "status.txt").write_text(
            f"Evaluated epoch {result['epoch']}\n",
            encoding="utf-8")
