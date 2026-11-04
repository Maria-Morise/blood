# YOLO26 blood cell detection

Fine-tunes a COCO-pretrained **YOLO26-S** model (Ultralytics) to detect and classify
white blood cells, red blood cells and platelets in blood smear images
([TXL-PBC](data/TXL-PBC/README.md), 1,260 images, 18,143 cells). Training runs on a
Kaggle GPU. A local HTML dashboard follows the run live and shows the final results.

## Layout

| Path | What it is |
|---|---|
| `kaggle/kernel/train.py` | Training + evaluation script that runs on Kaggle (T4 GPU) |
| `kaggle/kernel/kernel-metadata.json` | Kaggle kernel config (private, GPU, internet on) |
| `kaggle/dataset/dataset-metadata.json` | Private Kaggle dataset holding the images and labels |
| `scripts/analyze_dataset.py` | Dataset statistics and annotated samples |
| `scripts/monitor.py` | Polls the Kaggle run, updates the dashboard, downloads output when done |
| `scripts/tune_count_threshold.py` | Picks the RBC counting threshold on the validation split (runs `best.pt` locally) |
| `scripts/build_reports.py` | Builds the Markdown reports and the dashboard's results data |
| `dashboard/index.html` | Dashboard: training progress, model performance, dataset |
| `reports/` | `dataset_report.md`, `training_report.md`, `evaluation_report.md` and figures |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install kaggle six pillow numpy pyyaml
echo 'KAGGLE_API_TOKEN=<your token>' > .env      # never committed
```

## Run

```bash
set -a; . ./.env; set +a

# 1. Dataset stats and report
.venv/bin/python scripts/analyze_dataset.py

# 2. Upload the dataset (first time; use `datasets version` afterwards)
cp -R data/TXL-PBC kaggle/dataset/TXL-PBC
.venv/bin/kaggle datasets create -p kaggle/dataset --dir-mode zip

# 3. Start training on a Kaggle T4
.venv/bin/kaggle kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4

# 4. Follow the run; downloads the output and writes reports when it finishes
.venv/bin/python scripts/monitor.py --interval 60
```

Optional: tune the counting threshold locally, then rebuild the reports. This needs
torch and ultralytics, which don't support the Python in `.venv` yet, so use a separate
Python 3.11 environment:

```bash
python3.11 -m venv .venv-eval
.venv-eval/bin/pip install torch torchvision kaggle/deps/files/ultralytics-8.4.198-py3-none-any.whl
.venv-eval/bin/python scripts/tune_count_threshold.py   # writes reports/count_thresholds.json
.venv/bin/python scripts/build_reports.py
```

Open `dashboard/index.html` in a browser. It reloads its data every 20 seconds while
training is in progress and works directly from disk (no server needed).

## How monitoring works

`train.py` prints one `@@EPOCH {json}` line per epoch with losses, validation metrics,
learning rate and timing. `monitor.py` reads the kernel log through the Kaggle API,
parses those lines into `dashboard/data/live.js`, and when the kernel completes it
downloads `/kaggle/working/output` into `runs/kaggle_output/` (git-ignored; holds
`best.pt`) and runs `build_reports.py`.
