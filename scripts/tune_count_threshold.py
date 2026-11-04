"""Choose a per-class confidence threshold for cell counting on the validation split.

The default 0.25 threshold over-counts RBCs on test because border-cut and occluded
RBCs are often unlabeled in TXL-PBC. This runs best.pt once on val and test at a low
confidence, keeps every box score, then picks the RBC threshold that minimizes
per-image count MAE on val. Test is only used to report the result. WBC and platelet
counts are already near exact at 0.25 (val MAE <= 0.1), so they keep the default; the
sweep is still recorded for every class.

Inputs:   runs/kaggle_output/output/best.pt, data/TXL-PBC/{images,labels}/{val,test}
Outputs:  reports/count_thresholds.json (read by build_reports.py)

Needs torch + ultralytics (not in .venv):  <py3.11 venv>/bin/python scripts/tune_count_threshold.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "TXL-PBC"
OUT = ROOT / "reports" / "count_thresholds.json"
DEFAULT_CONF = 0.25
TUNED = {"RBC"}
GRID = [round(0.05 + 0.01 * i, 2) for i in range(86)]  # 0.05 .. 0.90


def predict_split(model, split, imgsz, device):
    """Per image: true count per class and (class, confidence) of every box at conf >= 0.01."""
    rows = []
    for img in sorted((DATA / "images" / split).glob("*.png")):
        gt = [0] * len(model.names)
        for line in (DATA / "labels" / split / f"{img.stem}.txt").read_text().splitlines():
            if line.strip():
                gt[int(line.split()[0])] += 1
        r = model.predict(str(img), imgsz=imgsz, conf=0.01, device=device, verbose=False)[0]
        boxes = [(int(c), round(float(s), 4)) for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        rows.append({"image": img.name, "gt": gt, "boxes": boxes})
    return rows


def counts(rows, cls, conf):
    return [(r["gt"][cls], sum(1 for c, s in r["boxes"] if c == cls and s >= conf)) for r in rows]


def score(pairs):
    errs = [abs(p - g) for g, p in pairs]
    return {"mae": sum(errs) / len(errs), "exact_match_rate": sum(e == 0 for e in errs) / len(errs),
            "gt_total": sum(g for g, _ in pairs), "pred_total": sum(p for _, p in pairs)}


def pick(rows, cls):
    """Lowest val MAE; ties go to the smallest total bias, then to the threshold nearest the default."""
    def key(t):
        s = score(counts(rows, cls, t))
        return (round(s["mae"], 6), abs(s["pred_total"] - s["gt_total"]), abs(t - DEFAULT_CONF))
    return min(GRID, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "kaggle_output" / "output" / "best.pt"))
    ap.add_argument("--device", default=None, help="e.g. mps, cpu, 0 (default: auto)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(args.weights)
    imgsz = model.ckpt.get("train_args", {}).get("imgsz", 640) if getattr(model, "ckpt", None) else 640
    names = model.names
    val, test = (predict_split(model, s, imgsz, device) for s in ("val", "test"))

    thresholds = {n: pick(val, i) if n in TUNED else DEFAULT_CONF for i, n in names.items()}
    result = {"criterion": "minimum per-image count MAE on the validation split", "default_conf": DEFAULT_CONF,
              "device": device, "imgsz": imgsz, "thresholds": thresholds,
              "sweep": {n: {"conf": GRID, "val_mae": [round(score(counts(val, i, t))["mae"], 4) for t in GRID]}
                        for i, n in names.items()}}
    for split, rows in (("val", val), ("test", test)):
        result[split] = {
            "default": {n: score(counts(rows, i, DEFAULT_CONF)) for i, n in names.items()},
            "tuned": {n: score(counts(rows, i, thresholds[n])) for i, n in names.items()},
        }
    result["test"]["per_image"] = [
        {"image": r["image"], "gt": r["gt"],
         "pred": [sum(1 for c, s in r["boxes"] if c == i and s >= thresholds[names[i]]) for i in names]}
        for r in test]
    OUT.write_text(json.dumps(result, indent=2))

    print(f"{'class':<10}{'conf':>6}  val MAE (0.25 -> tuned)   test MAE (0.25 -> tuned)   test total (true / 0.25 / tuned)")
    for n, t in thresholds.items():
        v, te = result["val"], result["test"]
        print(f"{n:<10}{t:>6.2f}  {v['default'][n]['mae']:>6.2f} -> {v['tuned'][n]['mae']:<6.2f}"
              f"           {te['default'][n]['mae']:>6.2f} -> {te['tuned'][n]['mae']:<6.2f}"
              f"           {te['default'][n]['gt_total']} / {te['default'][n]['pred_total']} / {te['tuned'][n]['pred_total']}")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
