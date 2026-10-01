from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configurazione
# ============================================================

BASE_DIR = Path("data/profiling/final-20260913-analysis-01")

PCA_CSV = BASE_DIR / "final-clustering-evidence-final-53" / "dbscan-cosine-aligned-pca-coordinates.csv"
UMAP_CSV = BASE_DIR / "final-clustering-umap-final-53" / "dbscan-paper5-robust-cosine-ms4-q80-umap-coordinates.csv"

OUT_DIR = BASE_DIR / "final-threshold-sensitivity-53" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Se vuoi solo 2.5 e 15 lascia così.
# Se vuoi completezza come K-Means, puoi usare:
# THRESHOLDS = [2.5, 5, 10, 15, 20, 25]
THRESHOLDS = [2.5, 15.0]

# Colori usati per la ground truth
LABEL_ORDER = [
    "x86-preferred",
    "architecture-independent",
    "arm-preferred",
]

LABEL_COLORS = {
    "x86-preferred": "#1f77b4",               # blu
    "architecture-independent": "#7f7f7f",   # grigio
    "arm-preferred": "#d62728",               # rosso
}


# ============================================================
# Utility
# ============================================================

def threshold_suffix(t: float) -> str:
    """15 -> t15 ; 2.5 -> t2p5"""
    if float(t).is_integer():
        return f"t{int(t)}"
    return f"t{str(t).replace('.', 'p')}"


def ground_truth_label(delta_percent: float, threshold_percent: float) -> str:
    """
    Delta percent:
      positivo  -> x86 più lento? No: nei tuoi CSV è coerente con etichetta già usata.
      Dai file esistenti:
        delta > +threshold  -> x86-preferred
        delta < -threshold  -> arm-preferred
        altrimenti          -> architecture-independent
    """
    if delta_percent >= threshold_percent:
        return "x86-preferred"
    elif delta_percent <= -threshold_percent:
        return "arm-preferred"
    else:
        return "architecture-independent"


def add_ground_truth(df: pd.DataFrame, threshold_percent: float) -> pd.DataFrame:
    out = df.copy()
    out["ground_truth_label_threshold"] = out["architecture_delta_percent"].apply(
        lambda x: ground_truth_label(x, threshold_percent)
    )
    return out


def plot_ground_truth_scatter(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        output_png: Path,
        output_svg: Path,
        mark_noise: bool = True,
):
    plt.figure(figsize=(8, 6))

    # Prima i punti normali
    for label in LABEL_ORDER:
        subset = df[(df["ground_truth_label_threshold"] == label) & (~df["is_noise"])]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=52,
            alpha=0.9,
            label=label,
            c=LABEL_COLORS[label],
            edgecolors="black",
            linewidths=0.4,
        )

    # Eventuali noise con marker diverso
    if mark_noise and "is_noise" in df.columns:
        noise_df = df[df["is_noise"]]
        for label in LABEL_ORDER:
            subset = noise_df[noise_df["ground_truth_label_threshold"] == label]
            if len(subset) == 0:
                continue
            plt.scatter(
                subset[x_col],
                subset[y_col],
                s=70,
                alpha=0.95,
                marker="X",
                label=f"{label} (noise)",
                c=LABEL_COLORS[label],
                edgecolors="black",
                linewidths=0.5,
            )

    plt.title(title, fontsize=12)
    plt.xlabel(x_col.upper(), fontsize=10)
    plt.ylabel(y_col.upper(), fontsize=10)
    plt.legend(fontsize=8, frameon=True)
    plt.tight_layout()

    plt.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close()


def summarize_threshold(df: pd.DataFrame, threshold_percent: float, embedding_name: str):
    tmp = add_ground_truth(df, threshold_percent)
    counts = tmp["ground_truth_label_threshold"].value_counts().to_dict()

    return {
        "embedding": embedding_name,
        "threshold_percent": threshold_percent,
        "x86_preferred": counts.get("x86-preferred", 0),
        "architecture_independent": counts.get("architecture-independent", 0),
        "arm_preferred": counts.get("arm-preferred", 0),
        "noise_points": int(tmp["is_noise"].sum()) if "is_noise" in tmp.columns else 0,
        "total_points": len(tmp),
    }


# ============================================================
# Main
# ============================================================

def main():
    if not PCA_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {PCA_CSV}")
    if not UMAP_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {UMAP_CSV}")

    pca_df = pd.read_csv(PCA_CSV)
    umap_df = pd.read_csv(UMAP_CSV)

    # Controlli minimi
    required_pca = {"function_name", "pc1", "pc2", "is_noise", "architecture_delta_percent"}
    required_umap = {"function_name", "umap1", "umap2", "is_noise", "architecture_delta_percent"}

    if not required_pca.issubset(set(pca_df.columns)):
        missing = required_pca - set(pca_df.columns)
        raise ValueError(f"Nel PCA CSV mancano colonne: {missing}")

    if not required_umap.issubset(set(umap_df.columns)):
        missing = required_umap - set(umap_df.columns)
        raise ValueError(f"Nel UMAP CSV mancano colonne: {missing}")

    summary_rows = []

    for t in THRESHOLDS:
        suffix = threshold_suffix(t)

        # ---- PCA ----
        pca_thr = add_ground_truth(pca_df, t)
        pca_png = OUT_DIR / f"dbscan-paper5-robust-cosine-ms4-q80-cosine-pca-ground-truth-{suffix}.png"
        pca_svg = OUT_DIR / f"dbscan-paper5-robust-cosine-ms4-q80-cosine-pca-ground-truth-{suffix}.svg"

        plot_ground_truth_scatter(
            pca_thr,
            x_col="pc1",
            y_col="pc2",
            title=f"DBSCAN ground truth (PCA) – threshold = {t}%",
            output_png=pca_png,
            output_svg=pca_svg,
            mark_noise=True,
        )

        summary_rows.append(summarize_threshold(pca_df, t, "pca"))

        # ---- UMAP ----
        umap_thr = add_ground_truth(umap_df, t)
        umap_png = OUT_DIR / f"dbscan-paper5-robust-cosine-ms4-q80-umap-ground-truth-{suffix}.png"
        umap_svg = OUT_DIR / f"dbscan-paper5-robust-cosine-ms4-q80-umap-ground-truth-{suffix}.svg"

        plot_ground_truth_scatter(
            umap_thr,
            x_col="umap1",
            y_col="umap2",
            title=f"DBSCAN ground truth (UMAP) – threshold = {t}%",
            output_png=umap_png,
            output_svg=umap_svg,
            mark_noise=True,
        )

        summary_rows.append(summarize_threshold(umap_df, t, "umap"))

        print(f"[OK] Generati threshold {t}%:")
        print(f"     PCA : {pca_png}")
        print(f"     UMAP: {umap_png}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "dbscan-ground-truth-threshold-summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print()
    print("[DONE] Tutte le immagini richieste sono state generate.")
    print(f"[DONE] Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()