"""Konfigurasi terpusat: path proyek dan seluruh hyperparameter eksperimen."""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")              # hasil unduhan Kaggle (tidak di-commit)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoint")  # bobot model (tidak di-commit)
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")            # cache citra ter-resize (tidak di-commit)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")         # submission.csv (tidak di-commit)

CONFIG = {
    "model_name": "resnet18",                  # "resnet18" = fine-tuning pretrained, "docnet" = CNN dari nol
    "pretrained": True,
    "img_h": 384,
    "img_w": 288,
    "use_cache": True,
    "amp": True,
    "batch_size": 32,
    "n_splits": 5,                             # 1 fold dari 5 dipakai sebagai validasi (sekitar 20%)
    "random_state": 42,
    "num_workers": min(4, os.cpu_count() or 2),
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "lr": 3e-4,
    "weight_decay": 2e-2,
    "label_smoothing": 0.1,
    "grad_clip": 1.0,
    "epochs": 12,
    "patience": 5,
    "dropout": 0.5,
    "channels": [32, 64, 128, 256],            # hanya dipakai jika model_name = "docnet"
    "hidden": 256,
}
