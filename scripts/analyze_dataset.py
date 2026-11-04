"""Profile the TXL-PBC dataset and write stats for the reports and dashboard.

Outputs:
  reports/dataset_stats.json      full statistics
  dashboard/data/dataset.js       same stats, loadable by the dashboard over file://
  reports/figures/samples/*.jpg   annotated example images
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "TXL-PBC"
SPLITS = ["train", "val", "test"]
COLORS = {0: (42, 120, 214), 1: (235, 104, 52), 2: (27, 175, 122)}


def percentiles(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {k: round(float(v), 2) for k, v in zip(
        ["min", "p25", "median", "p75", "max"], np.percentile(a, [0, 25, 50, 75, 100]))} | {
        "mean": round(float(a.mean()), 2)}


def histogram(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [round(float(e), 2) for e in edges], "counts": counts.tolist()}


def main():
    names = yaml.safe_load((DATA / "data.yaml").read_text())["names"]
    stats = {"classes": names, "splits": {}, "issues": []}
    all_box_px = defaultdict(list)      # class -> sqrt(w*h) in pixels
    all_aspect = defaultdict(list)
    cells_per_image = []
    sizes = Counter()
    modes = Counter()

    for split in SPLITS:
        imgs = sorted((DATA / "images" / split).glob("*.png"))
        cls_counts = Counter()
        per_img = []
        for img_path in imgs:
            with Image.open(img_path) as im:
                w, h = im.size
                modes[im.mode] += 1
            sizes[f"{w}x{h}"] += 1
            lbl = DATA / "labels" / split / f"{img_path.stem}.txt"
            if not lbl.exists():
                stats["issues"].append(f"{split}/{img_path.name}: missing label file")
                continue
            rows = [r.split() for r in lbl.read_text().splitlines() if r.strip()]
            per_img.append(len(rows))
            if not rows:
                stats["issues"].append(f"{split}/{img_path.name}: no annotations")
            seen = set()
            for r in rows:
                c, x, y, bw, bh = int(r[0]), *map(float, r[1:5])
                if tuple(r) in seen:
                    stats["issues"].append(f"{split}/{img_path.name}: duplicate box")
                seen.add(tuple(r))
                if not all(0 <= v <= 1 for v in (x, y, bw, bh)):
                    stats["issues"].append(f"{split}/{img_path.name}: coordinates out of [0,1]")
                cls_counts[c] += 1
                all_box_px[c].append((bw * w * bh * h) ** 0.5)
                all_aspect[c].append(bw * w / max(bh * h, 1e-6))
        cells_per_image += per_img
        stats["splits"][split] = {
            "images": len(imgs),
            "cells": sum(cls_counts.values()),
            "per_class": {names[i]: cls_counts[i] for i in range(len(names))},
        }

    stats["image_sizes"] = dict(sizes.most_common())
    stats["color_modes"] = dict(modes)
    stats["cells_per_image"] = percentiles(cells_per_image) | {
        "hist": histogram(cells_per_image, bins=range(0, max(cells_per_image) + 2))}
    max_px = max(max(v) for v in all_box_px.values())
    edges = np.linspace(0, max_px, 31)
    stats["box_size_px"] = {names[c]: percentiles(v) | {"hist": histogram(v, edges)}
                            for c, v in sorted(all_box_px.items())}
    stats["box_aspect"] = {names[c]: percentiles(v) for c, v in sorted(all_aspect.items())}
    stats["totals"] = {
        "images": sum(s["images"] for s in stats["splits"].values()),
        "cells": sum(s["cells"] for s in stats["splits"].values()),
        "per_class": {n: sum(s["per_class"][n] for s in stats["splits"].values()) for n in names},
    }
    stats["issue_count"] = len(stats["issues"])
    stats["issues"] = stats["issues"][:50]
    stats["samples"] = draw_samples(names)

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (ROOT / "dashboard" / "data" / "dataset.js").write_text(
        "window.DATASET = " + json.dumps(stats) + ";\n")
    t = stats["totals"]
    print(f"{t['images']} images, {t['cells']} cells, per class {t['per_class']}, "
          f"{stats['issue_count']} issues")


def draw_samples(names, n=6, seed=0):
    out = ROOT / "reports" / "figures" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((DATA / "images" / "train").glob("*.png"))
    random.Random(seed).shuffle(imgs)
    paths = []
    for img_path in imgs[:n]:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        for r in (DATA / "labels" / "train" / f"{img_path.stem}.txt").read_text().split("\n"):
            if not r.strip():
                continue
            c, x, y, bw, bh = int(r.split()[0]), *map(float, r.split()[1:5])
            d.rectangle([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h],
                        outline=COLORS[c], width=max(2, w // 200))
        im.thumbnail((480, 480))
        p = out / f"{img_path.stem[:12]}.jpg"
        im.save(p, quality=85)
        paths.append(str(p.relative_to(ROOT)))
    return paths


if __name__ == "__main__":
    main()
