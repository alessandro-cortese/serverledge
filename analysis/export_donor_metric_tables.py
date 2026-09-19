from pathlib import Path
import pandas as pd

OUT = Path("data/profiling/analysis-transfer-donor-metrics-20260918")
EXPORT_DIR = OUT / "presentation_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Cerca automaticamente un CSV "summary" con le colonne che ci servono
candidates = sorted(OUT.rglob("*.csv"))

summary_path = None
for p in candidates:
    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    cols = {c.lower() for c in df.columns}
    needed = {"clusterer", "distance", "coverage", "arm_match"}
    if needed.issubset(cols):
        summary_path = p
        summary_df = df
        break

if summary_path is None:
    raise SystemExit("Nessun CSV summary compatibile trovato nell'output.")

print(f"Summary trovato: {summary_path}")

# Normalizzazione nomi colonna nel caso cambino leggermente
rename_map = {}
for c in summary_df.columns:
    lc = c.lower()
    if lc == "arm_match":
        rename_map[c] = "arm_match"
    elif lc == "median_gap_error":
        rename_map[c] = "median_gap_error"
    elif lc == "clusterer":
        rename_map[c] = "clusterer"
    elif lc == "distance":
        rename_map[c] = "distance"
    elif lc == "coverage":
        rename_map[c] = "coverage"

summary_df = summary_df.rename(columns=rename_map)

required = ["clusterer", "distance", "coverage", "arm_match"]
for c in required:
    if c not in summary_df.columns:
        raise SystemExit(f"Colonna mancante nel summary: {c}")

# Salva il summary completo
summary_df.to_csv(EXPORT_DIR / "donor_quality_full.csv", index=False)

# Solo KMeans
kmeans_df = summary_df[summary_df["clusterer"].str.lower() == "kmeans"].copy()
kmeans_df = kmeans_df.sort_values(by=["arm_match", "coverage"], ascending=[False, False])
kmeans_df.to_csv(EXPORT_DIR / "donor_quality_kmeans_only.csv", index=False)

# Confronto KMeans Manhattan vs Euclidean
kmeans_two = kmeans_df[
    kmeans_df["distance"].str.lower().isin(["manhattan", "euclidean", "native"])
].copy()
kmeans_two.to_csv(EXPORT_DIR / "donor_quality_kmeans_manhattan_vs_euclidean.csv", index=False)

# Anche DBSCAN per completezza
dbscan_df = summary_df[summary_df["clusterer"].str.lower() == "dbscan"].copy()
dbscan_df = dbscan_df.sort_values(by=["arm_match", "coverage"], ascending=[False, False])
dbscan_df.to_csv(EXPORT_DIR / "donor_quality_dbscan_only.csv", index=False)

print("\nFile esportati in:")
for p in sorted(EXPORT_DIR.iterdir()):
    print(" -", p)