"""
Utility Functions for Transactional Fraud Detection System
Provides logging, path resolution, serialization, and plotting helpers.
"""

import os
import sys
import logging
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents tkinter threading errors
import matplotlib.pyplot as plt
import seaborn as sns


# Project Root Resolution
def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_logger(name: str = "FraudDetection") -> logging.Logger:
    """
    Configure and return a standardized logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def save_artifact(obj, file_path: str | Path) -> None:
    """
    Safely serialize a Python object / ML model to disk using joblib.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger = get_logger()
    logger.info(f"Artifact saved successfully at: {path}")


def load_artifact(file_path: str | Path):
    """
    Safely load a serialized Python object / model from disk.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Requested artifact does not exist: {path}")
    return joblib.load(path)


def setup_visualization_style() -> None:
    """
    Apply a modern, publication-grade aesthetics theme for all Matplotlib & Seaborn plots.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "Segoe UI, Arial, DejaVu Sans, sans-serif"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#F1F5F9"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["figure.dpi"] = 150
