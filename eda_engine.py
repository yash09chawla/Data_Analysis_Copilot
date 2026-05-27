"""
EDA Engine — runs all analysis and generates charts automatically.
No LLM needed here; this always produces deterministic results.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import base64
import io
import json


# ── Styling ────────────────────────────────────────────────────────────────
BLUE   = "#3B5BDB"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
AMBER  = "#EF9F27"
GRAY   = "#888780"
BG     = "white"

plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "axes.titleweight":   "bold",
    "axes.titlesize":     12,
    "figure.facecolor":   BG,
    "axes.facecolor":     BG,
})


# ── Helper ──────────────────────────────────────────────────────────────────
def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor=BG)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


def _safe_axes_flat(axes, n: int):
    """Always return a flat list of axes regardless of subplot shape."""
    if n == 1:
        return [axes]
    arr = np.array(axes)
    return arr.flatten().tolist()


# ── Main class ───────────────────────────────────────────────────────────────
class EDAEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results: dict = {}

    # ── 1. Overview ──────────────────────────────────────────────────────────
    def get_overview(self) -> dict:
        df = self.df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

        overview = {
            "n_rows":           int(len(df)),
            "n_cols":           int(len(df.columns)),
            "numeric_cols":     num_cols,
            "categorical_cols": cat_cols,
            "bool_cols":        bool_cols,
            "dtypes":           df.dtypes.astype(str).to_dict(),
            "missing_values":   df.isnull().sum().to_dict(),
            "missing_pct":      (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_rows":   int(df.duplicated().sum()),
            "memory_usage_mb":  round(df.memory_usage(deep=True).sum() / 1024**2, 3),
            "describe":         df.describe(include="all").round(4).to_dict(),
        }
        self.results["overview"] = overview
        return overview

    # ── 2. Missing values bar chart ──────────────────────────────────────────
    def plot_missing_values(self) -> str | None:
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            self.results["missing_img"] = None
            return None

        pct = (missing / len(self.df) * 100).round(2)
        fig, ax = plt.subplots(figsize=(9, max(3, len(missing) * 0.55)))
        colors = [CORAL if p > 20 else AMBER if p > 5 else TEAL for p in pct.values]
        bars = ax.barh(pct.index[::-1], pct.values[::-1], color=colors[::-1], edgecolor="white", height=0.6)
        for bar, val in zip(bars, pct.values[::-1]):
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, color="#444")
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Values Per Column")
        ax.set_xlim(0, max(pct.values) * 1.15)
        plt.tight_layout()
        img = _fig_to_b64(fig)
        self.results["missing_img"] = img
        return img

    # ── 3. Distributions ────────────────────────────────────────────────────
    def plot_distributions(self) -> str | None:
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()[:9]
        if not num_cols:
            self.results["distributions_img"] = None
            return None

        n_cols = min(3, len(num_cols))
        n_rows = (len(num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axs = _safe_axes_flat(axes, len(num_cols))

        stats_summary = {}
        for i, col in enumerate(num_cols):
            ax = axs[i]
            data = self.df[col].dropna()
            ax.hist(data, bins=min(40, max(10, len(data.unique()))),
                    color=BLUE, edgecolor="white", alpha=0.85)
            ax.axvline(data.mean(),   color=CORAL, linestyle="--", lw=1.5, label=f"Mean {data.mean():.2f}")
            ax.axvline(data.median(), color=TEAL,  linestyle="--", lw=1.5, label=f"Median {data.median():.2f}")
            ax.set_title(col)
            ax.legend(fontsize=7)
            stats_summary[col] = {
                "mean":     round(float(data.mean()), 4),
                "median":   round(float(data.median()), 4),
                "std":      round(float(data.std()), 4),
                "skewness": round(float(data.skew()), 4),
                "kurtosis": round(float(data.kurtosis()), 4),
                "min":      round(float(data.min()), 4),
                "max":      round(float(data.max()), 4),
            }

        for j in range(len(num_cols), len(axs)):
            axs[j].set_visible(False)

        fig.suptitle("Distribution of Numeric Features", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        img = _fig_to_b64(fig)
        self.results["distributions_img"]    = img
        self.results["distribution_stats"]   = stats_summary
        return img

    # ── 4. Correlation heatmap ───────────────────────────────────────────────
    def plot_correlations(self) -> str | None:
        num_df = self.df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            self.results["correlations_img"] = None
            return None

        corr = num_df.corr()
        size = max(7, len(corr) * 0.9)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    ax=ax, square=True, linewidths=0.4,
                    annot_kws={"size": 9}, cbar_kws={"shrink": 0.75})
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        img = _fig_to_b64(fig)

        # top pairs
        cols, pairs = corr.columns.tolist(), []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], round(float(corr.iloc[i, j]), 3)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        self.results["correlations_img"]  = img
        self.results["top_correlations"]  = pairs[:8]
        return img

    # ── 5. Categorical bar charts ────────────────────────────────────────────
    def plot_categoricals(self) -> str | None:
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if 2 <= self.df[c].nunique() <= 30][:6]
        if not cat_cols:
            self.results["categorical_img"] = None
            return None

        n_cols = min(2, len(cat_cols))
        n_rows = (len(cat_cols) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
        axs = _safe_axes_flat(axes, len(cat_cols))

        cat_summary = {}
        for i, col in enumerate(cat_cols):
            ax = axs[i]
            vc = self.df[col].value_counts().head(10)
            colors_bar = [BLUE if k == 0 else "#8FA8EB" for k in range(len(vc))]
            bars = ax.bar(range(len(vc)), vc.values, color=colors_bar, edgecolor="white")
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels([str(v)[:18] for v in vc.index], rotation=35, ha="right", fontsize=9)
            ax.set_title(f"{col}  (unique: {self.df[col].nunique()})")
            ax.set_ylabel("Count")
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vc.values) * 0.01,
                        str(val), ha="center", va="bottom", fontsize=8)
            cat_summary[col] = {
                "n_unique":   int(self.df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.head(5).items()},
            }

        for j in range(len(cat_cols), len(axs)):
            axs[j].set_visible(False)

        fig.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        img = _fig_to_b64(fig)
        self.results["categorical_img"]     = img
        self.results["categorical_summary"] = cat_summary
        return img

    # ── 6. Outliers (standardised boxplots + IQR counts) ────────────────────
    def plot_outliers(self) -> str | None:
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()[:8]
        if not num_cols:
            self.results["outliers_img"] = None
            return None

        # standardise so all columns fit on one plot
        sub = self.df[num_cols].copy()
        for c in num_cols:
            std = sub[c].std()
            if std > 0:
                sub[c] = (sub[c] - sub[c].mean()) / std

        fig, ax = plt.subplots(figsize=(max(9, len(num_cols) * 1.4), 5))
        bp = sub.boxplot(
            ax=ax, patch_artist=True,
            boxprops=dict(facecolor=BLUE, alpha=0.45, color=BLUE),
            medianprops=dict(color=CORAL, linewidth=2.5),
            whiskerprops=dict(color=GRAY),
            capprops=dict(color=GRAY),
            flierprops=dict(marker="o", markerfacecolor=AMBER, markersize=4, alpha=0.6, linestyle="none"),
            return_type="dict",
        )
        ax.axhline(y=3,  color=CORAL, linestyle="--", alpha=0.4, lw=1.2, label="±3σ boundary")
        ax.axhline(y=-3, color=CORAL, linestyle="--", alpha=0.4, lw=1.2)
        ax.set_title("Outlier Detection — Standardised Boxplots")
        ax.set_ylabel("Standard deviations from mean")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9)
        plt.tight_layout()
        img = _fig_to_b64(fig)

        outlier_info = {}
        for col in num_cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            n_out = int(((self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)).sum())
            outlier_info[col] = {
                "count": n_out,
                "pct":   round(n_out / len(self.df) * 100, 2),
            }

        self.results["outliers_img"]  = img
        self.results["outlier_info"]  = outlier_info
        return img

    # ── 7. Target column analysis (optional) ────────────────────────────────
    def plot_target_analysis(self, target_col: str) -> str | None:
        """If a potential target column is detected, plot it specifically."""
        if target_col not in self.df.columns:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # left: distribution of target
        data = self.df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(data):
            axes[0].hist(data, bins=30, color=TEAL, edgecolor="white", alpha=0.85)
            axes[0].axvline(data.mean(), color=CORAL, linestyle="--", lw=1.5, label=f"Mean {data.mean():.2f}")
            axes[0].set_title(f"Target: {target_col} distribution")
            axes[0].legend()
        else:
            vc = data.value_counts().head(10)
            axes[0].bar(vc.index.astype(str), vc.values, color=TEAL, edgecolor="white")
            axes[0].set_title(f"Target: {target_col} classes")
            axes[0].tick_params(axis="x", rotation=30)

        # right: boxplot of top numeric vs target (if categorical target)
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != target_col]
        if num_cols and not pd.api.types.is_numeric_dtype(self.df[target_col]):
            top_col = num_cols[0]
            groups = [grp[top_col].dropna().values
                      for _, grp in self.df.groupby(target_col)]
            labels  = [str(k) for k in self.df[target_col].value_counts().index[:8]]
            groups  = groups[:8]
            axes[1].boxplot(groups, labels=labels, patch_artist=True,
                            boxprops=dict(facecolor=BLUE, alpha=0.5),
                            medianprops=dict(color=CORAL, lw=2))
            axes[1].set_title(f"{top_col} by {target_col}")
            axes[1].tick_params(axis="x", rotation=30)
        else:
            axes[1].set_visible(False)

        fig.suptitle("Target Column Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        img = _fig_to_b64(fig)
        self.results["target_img"] = img
        return img

    # ── Master runner ────────────────────────────────────────────────────────
    def run_all(self, target_col: str = None) -> dict:
        self.get_overview()
        self.plot_missing_values()
        self.plot_distributions()
        self.plot_correlations()
        self.plot_categoricals()
        self.plot_outliers()
        if target_col:
            self.plot_target_analysis(target_col)
        return self.results
