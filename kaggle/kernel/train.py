"""Fine-tune YOLO26 on TXL-PBC blood smears. Runs as a Kaggle GPU script kernel.

The kernel has no internet access, so Ultralytics, its missing dependencies, the
pretrained weights and the plot font come from the private dataset built from
kaggle/deps/ and are installed offline.

Progress is printed as single-line `@@<TAG> {json}` records so the local
monitor (scripts/monitor.py) can follow the run through `kaggle kernels logs`.
Everything needed for the reports ends up in /kaggle/working/output/.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG = {
    "model": "yolo26s.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 25,
    "optimizer": "auto",
    "cos_lr": True,
    "seed": 0,
    # Smears have no canonical orientation, so vertical flips are valid augmentation.
    "flipud": 0.5,
    "fliplr": 0.5,
    "degrees": 10.0,
    "close_mosaic": 10,
}
WORK = Path("/kaggle/working")
OUT = WORK / "output"


def emit(tag, payload):
    print(f"@@{tag} {json.dumps(payload, default=float)}", flush=True)


def find_input(marker):
    hits = sorted(Path("/kaggle/input").rglob(marker))
    if not hits:
        sys.exit(f"{marker} not found under /kaggle/input")
    return hits[0].parent


def install_offline():
    deps = find_input("ultralytics-*.whl")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                    "--find-links", str(deps), "ultralytics"], check=True)
    os.environ["YOLO_OFFLINE"] = "1"
    # yolo26n.pt is what the AMP check loads; both resolve from the working dir.
    for f in ("yolo26s.pt", "yolo26n.pt"):
        shutil.copy(deps / f, WORK / f)
    from ultralytics.utils import USER_CONFIG_DIR
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(deps / "Arial.ttf", USER_CONFIG_DIR / "Arial.ttf")


def find_dataset():
    yamls = [p for p in Path("/kaggle/input").rglob("data.yaml") if (p.parent / "images").is_dir()]
    if not yamls:
        sys.exit("TXL-PBC data.yaml not found under /kaggle/input")
    root = yamls[0].parent
    data_yaml = WORK / "txl_pbc.yaml"
    data_yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 3\nnames: ['WBC', 'RBC', 'Platelets']\n")
    return root, data_yaml


def per_class(metrics, names):
    box = metrics.box
    rows = {}
    for i, c in enumerate(box.ap_class_index):
        rows[names[int(c)]] = {"precision": box.p[i], "recall": box.r[i],
                               "mAP50": box.ap50[i], "mAP50-95": box.ap[i]}
    return rows


def evaluate(model, data_yaml, split, names):
    m = model.val(data=str(data_yaml), split=split, imgsz=CONFIG["imgsz"], batch=16,
                  plots=True, project=str(WORK / "eval"), name=split, exist_ok=True)
    res = {
        "overall": {"precision": m.box.mp, "recall": m.box.mr,
                    "mAP50": m.box.map50, "mAP50-95": m.box.map},
        "per_class": per_class(m, names),
        "speed_ms": m.speed,
        # Ultralytics layout: rows = predicted class, cols = true class, last index = background.
        "confusion_matrix": m.confusion_matrix.matrix.tolist(),
        "confusion_labels": list(names.values()) + ["background"],
    }
    for f in Path(m.save_dir).glob("*.png"):
        shutil.copy(f, OUT / "plots" / f"{split}_{f.name}")
    return res


def count_accuracy(model, root, names, conf=0.25):
    """Per-image cell counting error on the test split, the number a lab would read off."""
    per_img = []
    for img in sorted((root / "images" / "test").glob("*.png")):
        gt = [0] * len(names)
        for line in (root / "labels" / "test" / f"{img.stem}.txt").read_text().splitlines():
            if line.strip():
                gt[int(line.split()[0])] += 1
        r = model.predict(str(img), imgsz=CONFIG["imgsz"], conf=conf, verbose=False)[0]
        pred = [0] * len(names)
        for c in r.boxes.cls.tolist():
            pred[int(c)] += 1
        per_img.append({"image": img.name, "gt": gt, "pred": pred})
    summary = {}
    for i, n in names.items():
        errs = [abs(p["pred"][i] - p["gt"][i]) for p in per_img]
        gt_total = sum(p["gt"][i] for p in per_img)
        summary[n] = {"mae": sum(errs) / len(errs),
                      "exact_match_rate": sum(e == 0 for e in errs) / len(errs),
                      "gt_total": gt_total, "pred_total": sum(p["pred"][i] for p in per_img)}
    return {"conf": conf, "per_class": summary, "per_image": per_img}


def save_examples(model, root, n=8):
    dst = OUT / "examples"
    dst.mkdir(parents=True, exist_ok=True)
    for img in sorted((root / "images" / "test").glob("*.png"))[:n]:
        r = model.predict(str(img), imgsz=CONFIG["imgsz"], conf=0.25, verbose=False)[0]
        r.save(filename=str(dst / f"{img.stem[:12]}.jpg"), line_width=2)


def main():
    t0 = time.time()
    os.chdir(WORK)
    install_offline()
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        emit("ERROR", {"message": "No GPU in this session. Kaggle grants GPUs only to phone-verified accounts."})
        sys.exit(1)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    root, data_yaml = find_dataset()
    emit("START", {"config": CONFIG, "ultralytics": ultralytics.__version__,
                   "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "dataset": str(root)})

    model = YOLO(CONFIG["model"])
    epoch_t = {"t": time.time()}

    def on_fit_epoch_end(trainer):
        if "val/box_loss" not in trainer.metrics:  # the final best.pt validation re-fires this hook
            return
        now = time.time()
        rec = {"epoch": trainer.epoch + 1, "epochs": trainer.epochs,
               "epoch_time_s": round(now - epoch_t["t"], 1), "elapsed_s": round(now - t0, 1)}
        epoch_t["t"] = now
        rec.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
        rec.update({k: v for k, v in trainer.metrics.items()})
        rec.update({f"lr/{k}": v for k, v in trainer.lr.items()})
        if torch.cuda.is_available():
            rec["gpu_mem_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        emit("EPOCH", rec)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    train_args = {k: v for k, v in CONFIG.items() if k != "model"}
    model.train(data=str(data_yaml), project=str(WORK / "runs"), name="train", exist_ok=True,
                device=0, workers=4, plots=True, **train_args)

    run_dir = WORK / "runs" / "train"
    best = YOLO(str(run_dir / "weights" / "best.pt"))
    names = best.names
    summary = {"config": CONFIG, "ultralytics": ultralytics.__version__,
               "gpu": torch.cuda.get_device_name(0), "train_time_s": round(time.time() - t0, 1)}
    emit("STAGE", {"stage": "evaluating"})
    summary["val"] = evaluate(best, data_yaml, "val", names)
    summary["test"] = evaluate(best, data_yaml, "test", names)
    summary["counting"] = count_accuracy(best, root, names)
    save_examples(best, root)

    shutil.copy(run_dir / "results.csv", OUT / "results.csv")
    shutil.copy(run_dir / "weights" / "best.pt", OUT / "best.pt")
    for f in run_dir.glob("*.png"):
        shutil.copy(f, OUT / "plots" / f"train_{f.name}")
    summary["total_time_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    emit("DONE", {"test_mAP50-95": summary["test"]["overall"]["mAP50-95"],
                  "test_mAP50": summary["test"]["overall"]["mAP50"],
                  "total_time_s": summary["total_time_s"]})


if __name__ == "__main__":
    main()
