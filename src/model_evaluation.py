"""
Model Evaluation and Threshold Optimization Engine
Computes cost-sensitive metrics (PR-AUC, Recall, Precision, F1, ROC-AUC),
plots Precision-Recall and ROC curves, visualizes confusion matrices with business error costs,
and performs operational threshold optimization across decision boundaries.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.utils import get_logger, get_project_root, setup_visualization_style

logger = get_logger("ModelEvaluation")


class ModelEvaluator:
    """
    Exhaustive evaluation suite designed specifically for extreme class imbalance
    and financial risk calibration.
    """

    def __init__(self, figures_dir: Optional[Path | str] = None):
        self.figures_dir = Path(figures_dir) if figures_dir else get_project_root() / "reports" / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        setup_visualization_style()

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model against test partition using custom operating threshold.
        """
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-decision))
        else:
            y_proba = model.predict(X_test)

        y_pred = (y_proba >= threshold).astype(int)

        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        metrics = {
            "model_name": model_name,
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "y_proba": y_proba,
            "y_pred": y_pred,
        }
        return metrics

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        training_times: Optional[Dict[str, float]] = None,
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """
        Benchmark all models and compile into a comparative summary table.
        """
        rows = []
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name=name, threshold=threshold)
            row = {
                "Model": name,
                "Precision": metrics["precision"],
                "Recall (Detection Rate)": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC (Avg Precision)": metrics["pr_auc"],
                "False Positives (Alarms)": metrics["false_positives"],
                "False Negatives (Missed)": metrics["false_negatives"],
                "Training Time (s)": training_times.get(name, 0.0) if training_times else 0.0,
            }
            rows.append(row)

        comparison_df = pd.DataFrame(rows).sort_values("PR-AUC (Avg Precision)", ascending=False).reset_index(drop=True)
        return comparison_df

    def plot_pr_and_roc_curves(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Path:
        """
        Generate comparative Precision-Recall and ROC curves.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        palette = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]

        # Baseline no-skill PR line
        no_skill = y_test.sum() / len(y_test)
        axes[0].plot([0, 1], [no_skill, no_skill], linestyle="--", color="#64748B", label=f"No Skill ({no_skill*100:.2f}%)")

        for idx, (name, model) in enumerate(models.items()):
            color = palette[idx % len(palette)]
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = 1 / (1 + np.exp(-model.decision_function(X_test)))
            else:
                y_proba = model.predict(X_test)

            # Precision-Recall Curve
            p, r, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            axes[0].plot(r, p, color=color, linewidth=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            axes[1].plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (ROC-AUC = {roc_auc:.3f})")

        # ROC Chance line
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Random Chance (0.500)")

        axes[0].set_title("Precision-Recall Curves (Key Metric for Imbalance)", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_xlabel("Recall (Sensitivity)", fontsize=11)
        axes[0].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
        axes[0].legend(loc="lower left", frameon=True)
        axes[0].grid(True, linestyle="--", alpha=0.7)

        axes[1].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
        axes[1].set_xlabel("False Positive Rate (Fall-out)", fontsize=11)
        axes[1].set_ylabel("True Positive Rate (Recall)", fontsize=11)
        axes[1].legend(loc="lower right", frameon=True)
        axes[1].grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        save_path = self.figures_dir / "model_comparison_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved PR & ROC comparison curves to: {save_path}")
        return save_path

    def optimize_thresholds(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[pd.DataFrame, float, Path]:
        """
        Evaluate performance across operating thresholds [0.10 to 0.90] to balance
        fraud detection rate (recall) with false alarm rate (precision).
        """
        if thresholds is None:
            thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test)

        records = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            records.append({
                "Threshold": t,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "True Positives": int(tp),
                "False Positives": int(fp),
                "False Negatives": int(fn),
                "True Negatives": int(tn),
            })

        df_thresh = pd.DataFrame(records)
        best_row = df_thresh.loc[df_thresh["F1 Score"].idxmax()]
        recommended_threshold = float(best_row["Threshold"])

        # Plot Threshold Trade-Off
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_thresh["Threshold"], df_thresh["Precision"], marker="o", color="#2563EB", linewidth=2.2, label="Precision (Minimizes False Alarms)")
        ax.plot(df_thresh["Threshold"], df_thresh["Recall"], marker="s", color="#DC2626", linewidth=2.2, label="Recall (Maximizes Fraud Detection)")
        ax.plot(df_thresh["Threshold"], df_thresh["F1 Score"], marker="^", color="#10B981", linewidth=2.2, label="F1 Score (Balanced Metric)")

        ax.axvline(recommended_threshold, color="#7C3AED", linestyle="--", linewidth=1.8, label=f"Optimal F1 Threshold ({recommended_threshold:.2f})")
        ax.set_title("Threshold Optimization: Precision-Recall Operational Trade-off", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Decision Threshold", fontsize=11)
        ax.set_ylabel("Metric Score", fontsize=11)
        ax.set_xticks(thresholds)
        ax.legend(loc="center right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        save_path = self.figures_dir / "threshold_optimization.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved threshold optimization plot to: {save_path}. Recommended threshold: {recommended_threshold}")
        return df_thresh, recommended_threshold, save_path

    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str = "Best Model",
    ) -> Path:
        """
        Generate annotated Confusion Matrix with financial terminology.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        labels = [
            [f"True Negative (TN)\n{tn:,}\n(Legit Approved)", f"False Positive (FP)\n{fp:,}\n(False Alarm)"],
            [f"False Negative (FN)\n{fn:,}\n(Missed Fraud - LOSS)", f"True Positive (TP)\n{tp:,}\n(Fraud Blocked)"],
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            linewidths=1.5,
            linecolor="#1E293B",
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual Class", fontsize=11, fontweight="bold")
        ax.set_xticklabels(["0: Legitimate", "1: Fraudulent"])
        ax.set_yticklabels(["0: Legitimate", "1: Fraudulent"])

        plt.tight_layout()
        save_path = self.figures_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved confusion matrix figure to: {save_path}")
        return save_path

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 15,
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Calculate and plot ranked feature importance.
        """
        importance_values = None
        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance_values = np.abs(model.coef_[0])

        if importance_values is None:
            # Fallback uniform importance if model has neither
            importance_values = np.ones(len(feature_names))

        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_values
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        top_df = df_imp.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(top_df["Feature"][::-1], top_df["Importance"][::-1], color="#3B82F6", edgecolor="#1E3A8A", height=0.6)
        ax.set_title(f"Top {top_n} Most Predictive Fraud Features (Model Feature Importance)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Relative Importance Score", fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        for bar in bars:
            val = bar.get_width()
            ax.text(val + (val * 0.01), bar.get_y() + bar.get_height()/2.0, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        save_path = self.figures_dir / "feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved feature importance figure to: {save_path}")
        return df_imp, save_path
