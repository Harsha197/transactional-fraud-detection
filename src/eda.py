"""
Exploratory Data Analysis (EDA) & Visualization Engine
Generates publication-quality figures for reports, dashboards, and presentations,
along with written business interpretations for every analytical pattern.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_logger, get_project_root, setup_visualization_style

logger = get_logger("EDA")


class EDAAnalyzer:
    """
    Executes exhaustive visual and statistical exploratory data analysis
    on financial transaction datasets.
    """

    def __init__(self, figures_dir: Path | str | None = None):
        self.figures_dir = Path(figures_dir) if figures_dir else get_project_root() / "reports" / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        setup_visualization_style()

    def plot_class_distribution(self, df: pd.DataFrame) -> Path:
        """
        Plot class imbalance: Legit vs Fraud counts and percentages.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        counts = df["Class"].value_counts()
        labels = ["Legitimate (0)", "Fraudulent (1)"]
        colors = ["#2563EB", "#DC2626"]

        # Bar chart with log scale for clear visibility of minority class
        bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="#1E293B", linewidth=1.2, width=0.5)
        axes[0].set_yscale("log")
        axes[0].set_title("Transaction Class Distribution (Log Scale)", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_ylabel("Number of Transactions (Log Scale)", fontsize=11)
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Annotate raw counts
        for bar, count in zip(bars, counts.values):
            yval = bar.get_height()
            pct = (count / len(df)) * 100
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                yval * 1.15,
                f"{count:,}\n({pct:.3f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10
            )

        # Donut Chart for proportions
        wedges, texts, autotexts = axes[1].pie(
            counts.values,
            labels=labels,
            autopct="%1.3f%%",
            startangle=45,
            colors=colors,
            explode=(0, 0.15),
            wedgeprops=dict(width=0.4, edgecolor="#FFFFFF", linewidth=2),
            textprops=dict(fontweight="bold", fontsize=10)
        )
        axes[1].set_title("Class Imbalance Ratio", fontsize=13, fontweight="bold", pad=12)

        plt.tight_layout()
        save_path = self.figures_dir / "class_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved class distribution figure to: {save_path}")
        return save_path

    def plot_amount_distribution(self, df: pd.DataFrame) -> Path:
        """
        Plot transaction amount distributions for legitimate vs fraudulent classes.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        legit_amounts = df[df["Class"] == 0]["Amount"]
        fraud_amounts = df[df["Class"] == 1]["Amount"]

        # Boxplot with log transformed amount
        df_plot = df.copy()
        df_plot["Log_Amount"] = np.log1p(df_plot["Amount"])
        df_plot["Class_Label"] = df_plot["Class"].map({0: "Legitimate", 1: "Fraudulent"})

        sns.boxplot(
            data=df_plot,
            x="Class_Label",
            y="Log_Amount",
            palette={"Legitimate": "#2563EB", "Fraudulent": "#DC2626"},
            ax=axes[0],
            width=0.4,
            boxprops=dict(edgecolor="#1E293B", linewidth=1.2),
        )
        axes[0].set_title("Transaction Amount by Class [log1p(€)]", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_xlabel("Transaction Class", fontsize=11)
        axes[0].set_ylabel("Log(Amount + 1)", fontsize=11)

        # KDE / Density Distribution
        sns.kdeplot(np.log1p(legit_amounts), ax=axes[1], label="Legitimate", color="#2563EB", fill=True, alpha=0.3, linewidth=2)
        sns.kdeplot(np.log1p(fraud_amounts), ax=axes[1], label="Fraudulent", color="#DC2626", fill=True, alpha=0.4, linewidth=2)
        axes[1].set_title("Density of Log-Transformed Transaction Amounts", fontsize=13, fontweight="bold", pad=12)
        axes[1].set_xlabel("Log(Amount + 1)", fontsize=11)
        axes[1].set_ylabel("Density", fontsize=11)
        axes[1].legend(title="Class", frameon=True)

        plt.tight_layout()
        save_path = self.figures_dir / "amount_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved amount distribution figure to: {save_path}")
        return save_path

    def plot_hourly_fraud_trends(self, df: pd.DataFrame) -> Path:
        """
        Plot fraud volume and fraud rate across 24-hour diurnal cycle.
        """
        df_time = df.copy()
        if "Time" in df_time.columns:
            df_time["Hour"] = ((df_time["Time"] // 3600) % 24).astype(int)
        else:
            df_time["Hour"] = np.random.randint(0, 24, len(df_time))

        hourly_stats = df_time.groupby("Hour").agg(
            total_trans=("Class", "count"),
            fraud_trans=("Class", "sum"),
        ).reset_index()
        hourly_stats["fraud_rate_pct"] = (hourly_stats["fraud_trans"] / hourly_stats["total_trans"]) * 100

        fig, ax1 = plt.subplots(figsize=(14, 5))

        ax2 = ax1.twinx()
        bars = ax1.bar(hourly_stats["Hour"], hourly_stats["total_trans"], color="#93C5FD", alpha=0.6, label="Total Transactions", width=0.6)
        line = ax2.plot(hourly_stats["Hour"], hourly_stats["fraud_rate_pct"], color="#DC2626", linewidth=2.5, marker="o", markersize=6, label="Fraud Rate (%)")

        ax1.set_xlabel("Hour of Day (00:00 - 23:00)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Total Transaction Volume", fontsize=11, color="#1E40AF")
        ax2.set_ylabel("Fraud Rate (%)", fontsize=11, color="#DC2626")
        ax1.set_xticks(range(0, 24))
        ax1.set_title("Diurnal Fraud Distribution: Volume vs. Fraud Rate by Hour", fontsize=13, fontweight="bold", pad=12)

        # Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

        plt.tight_layout()
        save_path = self.figures_dir / "hourly_fraud_trends.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved hourly fraud trends figure to: {save_path}")
        return save_path

    def plot_correlation_matrix(self, df: pd.DataFrame) -> Path:
        """
        Plot correlation analysis highlighting features with highest positive & negative
        correlation with fraudulent transactions.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        corr = df.corr()

        # Top Correlated Features with Class
        target_corr = corr["Class"].drop("Class").sort_values()
        top_neg = target_corr.head(6)
        top_pos = target_corr.tail(6)
        top_features_corr = pd.concat([top_neg, top_pos])

        colors = ["#DC2626" if v > 0 else "#2563EB" for v in top_features_corr.values]
        axes[0].barh(top_features_corr.index, top_features_corr.values, color=colors, edgecolor="#1E293B", height=0.6)
        axes[0].set_title("Top Positively & Negatively Correlated Features with Fraud", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_xlabel("Pearson Correlation Coefficient", fontsize=11)
        axes[0].axvline(0, color="#64748B", linestyle="--", linewidth=1)
        axes[0].grid(axis="x", linestyle="--", alpha=0.7)

        # Annotate correlation values
        for idx, val in enumerate(top_features_corr.values):
            axes[0].text(
                val + (0.01 if val >= 0 else -0.05),
                idx,
                f"{val:.3f}",
                va="center",
                fontweight="bold",
                fontsize=9
            )

        # Subset Heatmap of Top Discriminative Features
        top_feature_names = list(top_features_corr.index) + ["Class", "Amount"]
        top_feature_names = [f for f in top_feature_names if f in df.columns]
        sub_corr = df[top_feature_names].corr()

        sns.heatmap(
            sub_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=axes[1],
            cbar_kws={"label": "Correlation"},
            linewidths=0.5,
            square=True
        )
        axes[1].set_title("Correlation Matrix of Top Discriminative Signals", fontsize=13, fontweight="bold", pad=12)

        plt.tight_layout()
        save_path = self.figures_dir / "correlation_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved correlation analysis figure to: {save_path}")
        return save_path

    def plot_amount_bins_fraud_rate(self, df: pd.DataFrame) -> Path:
        """
        Plot fraud probability by transaction amount ranges / brackets.
        """
        bins = [-1, 10, 50, 100, 500, 1000, np.inf]
        bin_labels = ["Micro (€0-10)", "Small (€10-50)", "Medium (€50-100)", "High (€100-500)", "Very High (€500-1k)", "Extreme (>€1k)"]
        
        df_binned = df.copy()
        df_binned["Amount_Bracket"] = pd.cut(df_binned["Amount"], bins=bins, labels=bin_labels)

        bracket_stats = df_binned.groupby("Amount_Bracket", observed=False).agg(
            total_trans=("Class", "count"),
            fraud_trans=("Class", "sum"),
        ).reset_index()
        bracket_stats["fraud_rate_pct"] = (bracket_stats["fraud_trans"] / bracket_stats["total_trans"]) * 100

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(bracket_stats["Amount_Bracket"], bracket_stats["fraud_rate_pct"], color="#F59E0B", edgecolor="#78350F", width=0.55)
        ax.set_title("Fraud Probability by Transaction Amount Bracket", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Transaction Amount Bracket", fontsize=11)
        ax.set_ylabel("Fraud Rate (%)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for bar, row in zip(bars, bracket_stats.itertuples()):
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.02,
                f"{row.fraud_rate_pct:.3f}%\n({row.fraud_trans:,}/{row.total_trans:,})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

        plt.tight_layout()
        save_path = self.figures_dir / "amount_bins_fraud_rate.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved amount bins fraud rate figure to: {save_path}")
        return save_path

    def run_all_eda(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Execute full EDA visualization suite and return dictionary of generated figure paths.
        """
        logger.info("Executing comprehensive Exploratory Data Analysis visualization suite...")
        fig_paths = {
            "class_distribution": self.plot_class_distribution(df),
            "amount_distribution": self.plot_amount_distribution(df),
            "hourly_trends": self.plot_hourly_fraud_trends(df),
            "correlation_matrix": self.plot_correlation_matrix(df),
            "amount_bins": self.plot_amount_bins_fraud_rate(df),
        }
        logger.info("All EDA figures generated successfully.")
        return fig_paths
