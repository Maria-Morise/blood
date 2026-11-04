"""Turn the dataset stats and the downloaded Kaggle output into reports + dashboard data.

Inputs:   reports/dataset_stats.json, runs/kaggle_output/output/{results.csv,summary.json,...}
Optional: reports/count_thresholds.json (scripts/tune_count_threshold.py)
Outputs:  reports/{dataset,training,evaluation}_report.md, reports/figures/model/*,
          dashboard/data/results.js
"""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures" / "model"


def find_output():
    hits = list((ROOT / "runs" / "kaggle_output").rglob("summary.json"))
    return hits[0].parent if hits else None


def fmt(v, d=3):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def dataset_report(ds):
    names = ds["classes"]
    t = ds["totals"]
    rows = [[s, d["images"], d["cells"], *[d["per_class"][n] for n in names]] for s, d in ds["splits"].items()]
    rows.append(["**Total**", t["images"], t["cells"], *[t["per_class"][n] for n in names]])
    share = ", ".join(f"{n} {100 * t['per_class'][n] / t['cells']:.1f}%" for n in names)
    size_rows = [[n, *[fmt(ds["box_size_px"][n][k], 1) for k in ("min", "median", "max")]] for n in names]
    samples = "\n".join(f"![sample](../{p})" for p in ds["samples"])
    return f"""# Dataset report: TXL-PBC

Blood smear images annotated for detection of three cell types: {", ".join(names)}.

## Splits and class counts

{table(["Split", "Images", "Cells", *names], rows)}

The classes are heavily imbalanced: {share}. RBCs make up most of the cells, while
platelets are both rare and small, so platelet recall is the metric most likely to lag.

## Images

- Sizes: {", ".join(f"{k} ({v})" for k, v in ds["image_sizes"].items())}
- Color modes: {", ".join(f"{k} ({v})" for k, v in ds["color_modes"].items())}. RGBA images are read
  as 3-channel by the training loader.
- Cells per image: median {fmt(ds["cells_per_image"]["median"], 0)}, range
  {fmt(ds["cells_per_image"]["min"], 0)}–{fmt(ds["cells_per_image"]["max"], 0)}.

## Box sizes (square root of box area, in pixels)

{table(["Class", "Min", "Median", "Max"], size_rows)}

Boxes are close to square for all classes (median aspect ratio ~1.0).

## Label quality

{ds["issue_count"]} exact duplicate boxes were found (same class and coordinates
repeated within one label file). No missing label files, empty images or
out-of-range coordinates. Ultralytics removes duplicate labels when it caches the
dataset, so the source files were left unchanged.

## Annotated samples (training split)

Blue = WBC, orange = RBC, green = Platelets.

{samples}
"""


def training_report(summary, epochs):
    cfg = summary["config"]
    best = max(epochs, key=lambda e: e["metrics/mAP50-95(B)"])
    last = epochs[-1]
    rows = [[e["epoch"], fmt(e["train/box_loss"]), fmt(e["train/cls_loss"]), fmt(e["val/box_loss"]),
             fmt(e["val/cls_loss"]), fmt(e["metrics/mAP50(B)"]), fmt(e["metrics/mAP50-95(B)"])]
            for e in epochs if e["epoch"] % 10 == 0 or e["epoch"] in (1, len(epochs))]
    stopped = (f"Early stopping ended training after {len(epochs)} of {cfg['epochs']} epochs "
               f"(patience {cfg['patience']})." if len(epochs) < cfg["epochs"]
               else f"All {cfg['epochs']} epochs ran.")
    return f"""# Training report

| Setting | Value |
|---|---|
| Base model | `{cfg["model"]}` (COCO-pretrained YOLO26-S) |
| Hardware | Kaggle {summary["gpu"]} |
| Ultralytics | {summary["ultralytics"]} |
| Image size | {cfg["imgsz"]} |
| Batch | {cfg["batch"]} |
| Epochs (max) | {cfg["epochs"]}, patience {cfg["patience"]} |
| LR schedule | cosine |
| Augmentation | Ultralytics defaults + vertical flip {cfg["flipud"]}, rotation ±{cfg["degrees"]}° |
| Wall time | {summary["total_time_s"] / 60:.1f} min (incl. setup and evaluation) |

{stopped} The best validation mAP50-95 was **{fmt(best["metrics/mAP50-95(B)"])}** at epoch
{best["epoch"]} (mAP50 {fmt(best["metrics/mAP50(B)"])}); those weights (`best.pt`) are used for evaluation.
The final epoch reached mAP50-95 {fmt(last["metrics/mAP50-95(B)"])}.

## Progress

{table(["Epoch", "Train box", "Train cls", "Val box", "Val cls", "mAP50", "mAP50-95"], rows)}

![results](figures/model/train_results.png)
"""


def evaluation_report(summary, ct=None):
    names = list(summary["test"]["per_class"])

    def rows(split):
        s = summary[split]
        r = [[n, *[fmt(s["per_class"][n][k]) for k in ("precision", "recall", "mAP50", "mAP50-95")]] for n in names]
        r.append(["**All**", *[f"**{fmt(s['overall'][k])}**" for k in ("precision", "recall", "mAP50", "mAP50-95")]])
        return table(["Class", "Precision", "Recall", "mAP50", "mAP50-95"], r)

    cnt = summary["counting"]
    count_rows = [[n, v["gt_total"], v["pred_total"], fmt(v["mae"], 2), f"{100 * v['exact_match_rate']:.0f}%"]
                  for n, v in cnt["per_class"].items()]
    sp = summary["test"]["speed_ms"]
    worst = min(names, key=lambda n: summary["test"]["per_class"][n]["mAP50-95"])
    tuned = count_threshold_section(ct) if ct else ""
    examples = "\n".join(f"![example](figures/model/examples/{p.name})"
                         for p in sorted((FIG / "examples").glob("*.jpg")))
    return f"""# Evaluation report

Model: best checkpoint from the training run, evaluated at image size
{summary["config"]["imgsz"]}. The test split (126 images) was never used
during training or model selection.

## Test split

{rows("test")}

## Validation split

{rows("val")}

The weakest class on test is **{worst}**.

## Cell counting on the test split

Counts are the number of predicted boxes per image at confidence ≥ {cnt["conf"]}.
MAE is the mean absolute difference between predicted and true counts per image.

{table(["Class", "True total", "Predicted total", "MAE per image", "Exact count"], count_rows)}

Where the predicted total is well above the true total, check the ground truth before
blaming the detector. In TXL-PBC, RBCs cut off by the image border or partly hidden
under other cells are often left unlabeled. The model still finds them, usually at
lower confidence, and every box above the threshold is counted. Box-level precision
stays high because Ultralytics scores it at the confidence that maximizes F1, not at
{cnt["conf"]}. A count-specific threshold should be chosen on the validation split,
not on test.
{tuned}
## Speed

Per image on {summary["gpu"]}: preprocess {fmt(sp["preprocess"], 2)} ms, inference
{fmt(sp["inference"], 2)} ms, postprocess {fmt(sp["postprocess"], 2)} ms.

## Plots

![confusion matrix](figures/model/test_confusion_matrix_normalized.png)
![PR curve](figures/model/test_BoxPR_curve.png)

## Example predictions (test split)

{examples}
"""


def count_threshold_section(ct):
    """Per-class counting thresholds chosen on val, compared with the default on both splits."""
    t = ct["thresholds"]
    d = ct["default_conf"]

    def arrow(split, n, k, f):
        return f"{f(ct[split]['default'][n][k])} → {f(ct[split]['tuned'][n][k])}"

    pct = lambda v: f"{100 * v:.0f}%"
    rows = [[n, fmt(t[n], 2), arrow("val", n, "mae", lambda v: fmt(v, 2)), arrow("test", n, "mae", lambda v: fmt(v, 2)),
             f"{ct['test']['default'][n]['gt_total']} / {arrow('test', n, 'pred_total', str)}",
             arrow("test", n, "exact_match_rate", pct)] for n in t]
    changed = [n for n in t if t[n] != d]
    sweep = ct["sweep"]
    curve = "; ".join(f"{n}: " + ", ".join(f"{c:.2f} → {m:.2f}" for c, m in zip(sweep[n]["conf"], sweep[n]["val_mae"])
                                          if round(c * 100) % 10 == 0 or c == t[n])
                      for n in changed)
    return f"""
## Count-specific thresholds

`scripts/tune_count_threshold.py` picks the confidence threshold that minimizes per-image
count MAE on the **validation** split{" for " + ", ".join(changed) if changed else ""}; other classes keep {d}.
The test split is only used to report the result below. Arrows read default {d} → tuned.

{table(["Class", "Threshold", "Val MAE", "Test MAE", "Test true / predicted total", "Test exact count"], rows)}

Validation MAE by threshold ({curve}).

These thresholds are for counting only. Detection metrics above are unchanged.
"""


def main():
    ds = json.loads((REPORTS / "dataset_stats.json").read_text())
    (REPORTS / "dataset_report.md").write_text(dataset_report(ds))
    out = find_output()
    if out is None:
        print("No Kaggle output yet; wrote dataset report only.")
        return
    summary = json.loads((out / "summary.json").read_text())
    with open(out / "results.csv") as f:
        epochs = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for e in epochs:
        e["epoch"] = int(e["epoch"])

    if FIG.exists():
        shutil.rmtree(FIG)
    (FIG / "examples").mkdir(parents=True)
    for f in (out / "plots").glob("*.png"):
        shutil.copy(f, FIG / f.name)
    for f in (out / "examples").glob("*.jpg"):
        shutil.copy(f, FIG / "examples" / f.name)

    (REPORTS / "training_report.md").write_text(training_report(summary, epochs))
    ct_file = REPORTS / "count_thresholds.json"
    ct = json.loads(ct_file.read_text()) if ct_file.exists() else None
    (REPORTS / "evaluation_report.md").write_text(evaluation_report(summary, ct))
    lean = {k: v for k, v in summary.items() if k != "counting"}
    lean["counting"] = {k: v for k, v in summary["counting"].items() if k != "per_image"}
    lean["counting"]["per_image"] = summary["counting"]["per_image"]
    if ct:
        lean["count_thresholds"] = {"thresholds": ct["thresholds"], "default_conf": ct["default_conf"],
                                    "default": ct["test"]["default"], "per_class": ct["test"]["tuned"],
                                    "per_image": ct["test"]["per_image"]}
    lean["epochs"] = epochs
    lean["plots"] = sorted(f"../reports/figures/model/{p.name}" for p in FIG.glob("*.png"))
    lean["examples"] = sorted(f"../reports/figures/model/examples/{p.name}" for p in (FIG / "examples").glob("*.jpg"))
    (ROOT / "dashboard" / "data" / "results.js").write_text("window.RESULTS = " + json.dumps(lean) + ";\n")
    print(f"Reports written. Test mAP50-95 = {summary['test']['overall']['mAP50-95']:.4f}")


if __name__ == "__main__":
    main()
