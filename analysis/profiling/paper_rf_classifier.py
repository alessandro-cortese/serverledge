from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

"""Random Forest CPU per la classificazione ARM-speed del paper Chen et al. (UCC 2024).

Il training usa esclusivamente le feature del CSV x86 (schema 6). Le etichette
ARM-faster / ARM-slower / ARM-similar sono calcolate offline dal service time
warm misurato su entrambi i worker e non diventano mai input del modello.
Nessuna operazione GCP; nessuna modifica al collector o al runtime congelato.
"""

PIN = "9f9e9cfc5631b41f525f6d91fadad4415b93a8c1"
CLASSES = ("ARM-faster", "ARM-slower", "ARM-similar")
CLASS_TO_SERVERLEDGE = {
    "ARM-faster": "arm-preferred",
    "ARM-slower": "x86-preferred",
    "ARM-similar": "architecture-independent",
}
META_COLUMNS = (
    "paper_rf_csv_schema_version", "experiment_id", "paper_rf_feature_vector_schema_version",
    "request_id", "function_name", "machine_tag", "configured_cpus",
    "configured_memory_mb",
)
# Intestazione effettiva del selected.csv schema 6, verificata sulla campagna GCP.
# L'ordine e' parte della definizione delle feature: non accettiamo colonne arbitrarie.
EXPECTED_FEATURE_COLUMNS = (
    "totalMemory", "freeMemory", "frameworkRuntime", "userRuntime",
    "cpuUserDelta", "cpuKernelDelta", "cpuIdleDelta", "cpuIOWaitDelta",
    "cpuSoftIrqDelta", "cpuStealDelta", "pageFaultsDelta", "utilizedCPUs",
    "frameworkRuntimeDeltas", "runtime",
)
REFERENCE = "amd64"
TARGET_ROWS = 13400
EXPECTED_FUNCTIONS = 67
SAMPLES_PER_ARCH = 100
PAPER_THRESHOLD = 0.15
PAPER_TEST_FRACTION = 7 / 18


def sha256(path: Path) -> str:
    """Calcola SHA-256 streaming per verificare il dataset finale."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_final_dataset(path: Path) -> dict:
    """Accetta soltanto un export finale completo e verificato dal collector."""
    manifest_path = path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Manifest finale assente: la campagna non e' ancora conclusa")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pin") != PIN or manifest.get("rows") != TARGET_ROWS:
        raise ValueError("Manifest del dataset inatteso: commit o numerosita' errati")
    if manifest.get("dataset_sha256") != sha256(path):
        raise ValueError("SHA256 del dataset finale non valido")
    provenance = path.parent / "paper-rf-provenance.csv"
    if not provenance.is_file() or manifest.get("provenance_sha256") != sha256(provenance):
        raise ValueError("Provenienza dei blocchi assente o con SHA256 diverso")
    campaign = path.parent / "campaign.json"
    if not campaign.is_file() or manifest.get("campaign_manifest_sha256") != sha256(campaign):
        raise ValueError("Manifest della campagna assente o con SHA256 diverso")
    return manifest


def family(name: str) -> str:
    """Non distribuire una funzione e la variante twin fra train e test."""
    return name.removeprefix("twin-")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Individua le 14 feature mantenendo fuori gli otto metadati dello schema 6."""
    missing = sorted(set(META_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Metadati schema 6 mancanti: {missing}")
    if len(frame.columns) != 22 or len(set(frame.columns)) != 22:
        raise ValueError(f"CSV schema 6 inatteso: {len(frame.columns)} colonne anziche' 22")
    features = [col for col in frame.columns if col not in META_COLUMNS]
    if len(features) != 14:
        raise ValueError("Attese esattamente 14 feature numeriche")
    unsafe = [col for col in features if "arm_" in col.lower() or "preference" in col.lower() or "target" in col.lower()]
    if unsafe:
        raise ValueError(f"Possibile leakage negli input: {unsafe}")
    if features != list(EXPECTED_FEATURE_COLUMNS):
        raise ValueError(f"Feature schema 6 inattese: {features}; attese: {list(EXPECTED_FEATURE_COLUMNS)}")
    return features


def load_dataset(path: Path, *, check_counts: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Carica tutte le esecuzioni selezionate senza fare aggregazioni premature."""
    frame = pd.read_csv(path, low_memory=False)
    features = feature_columns(frame)
    if frame["request_id"].isna().any() or frame["request_id"].duplicated().any():
        raise ValueError("request_id mancanti o duplicati nel CSV")
    if set(frame["machine_tag"].unique()) != {"amd64", "arm64"}:
        raise ValueError("Attesi solo i machine_tag amd64 e arm64")
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Le 14 feature contengono NaN, infinito o valori non numerici")
    if check_counts:
        counts = frame.groupby(["function_name", "machine_tag"]).size()
        if len(counts) != EXPECTED_FUNCTIONS * 2 or not counts.eq(SAMPLES_PER_ARCH).all():
            raise ValueError("Campagna incompleta: servono 100 warm per funzione su ogni architettura")
        if not pd.to_numeric(frame["configured_cpus"], errors="coerce").eq(1).all():
            raise ValueError("CPU configurate diverse da 1 nel dataset")
        # Evitiamo di unire campioni di configurazioni RAM diverse come se fossero uguali.
        if frame.groupby("function_name")["configured_memory_mb"].nunique().gt(1).any():
            raise ValueError("Memoria configurata differente fra le architetture della stessa funzione")
    return frame, features


def raw_selected_durations(dataset: pd.DataFrame, dest: Path) -> pd.DataFrame:
    """Abbina le durate del raw ESATTAMENTE ai RequestID selezionati dal collector.

    NON impiega i cold start ne' i campioni warm scartati. Usa l'indice di
    provenienza creato dall'exporter, evitando accoppiamenti per ordine casuale.
    """
    provenance = pd.read_csv(dest / "paper-rf-provenance.csv", dtype=str, keep_default_na=False)
    expected_cols = {"arch", "function", "request_id", "block_id", "round"}
    if not expected_cols.issubset(provenance):
        raise ValueError("Intestazione paper-rf-provenance.csv inattesa")
    if len(provenance) != len(dataset):
        raise ValueError("Provenienza e dataset hanno lunghezze differenti")
    for a, b in (("request_id", "request_id"), ("function_name", "function"), ("machine_tag", "arch")):
        if dataset[a].astype(str).tolist() != provenance[b].tolist():
            raise ValueError(f"Provenienza non allineata al dataset: {a}")

    durations = {}
    for (arch, fn, block_id, round_text), subset in provenance.groupby(
        ["arch", "function", "block_id", "round"], sort=False
    ):
        if not round_text.isdecimal() or any("/" in x or ".." in x or "\\" in x for x in (arch, fn, block_id)):
            raise ValueError("Identificatore di blocco non sicuro")
        raw_path = dest / "blocks" / arch / f"round_{int(round_text):03d}" / fn / "raw.jsonl"
        if not raw_path.is_file():
            raise ValueError(f"Raw del blocco mancante: {raw_path}")
        wanted = set(subset["request_id"])
        found = set()
        with raw_path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                rid = entry.get("request_id")
                if rid not in wanted:
                    continue
                if rid in found:
                    raise ValueError(f"Raw con RequestID ripetuto: {rid}")
                if entry.get("schema_version") != 6 or entry.get("function_name") != fn or entry.get("machine_tag") != arch:
                    raise ValueError(f"Raw incoerente: {raw_path}:{lineno}")
                # InvocationSample.Timing.DurationMs -> JSON timing.duration_ms.
                timing = entry.get("timing") or {}
                try:
                    duration = float(timing["duration_ms"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"timing.duration_ms non disponibile: {raw_path}:{lineno}") from exc
                if not np.isfinite(duration) or duration <= 0:
                    raise ValueError(f"DurationMs non valida: {raw_path}:{lineno}")
                durations[rid] = duration
                found.add(rid)
        if found != wanted:
            raise ValueError(f"Raw non contiene tutti i RequestID selezionati: {raw_path}; mancanti={len(wanted-found)}")

    result = provenance[["function", "arch", "request_id"]].copy()
    result.columns = ["function_name", "machine_tag", "request_id"]
    result["duration_ms"] = result["request_id"].map(durations)
    if result["duration_ms"].isna().any():
        raise ValueError("Durate mancanti dopo la verifica")
    return result


def make_paper_labels(durations: pd.DataFrame, *, threshold: float = PAPER_THRESHOLD) -> pd.DataFrame:
    """Etichette ternarie da mediana warm per funzione, non da predizioni.

    Con un'unica configurazione per funzione (diversamente dai 40 step del
    paper) la mediana su 100 misure e' una scelta dichiarata e riproducibile.
    """
    if threshold != PAPER_THRESHOLD:
        raise ValueError("La replica principale usa la soglia fissa del paper: 15%")
    stats = durations.groupby(["function_name", "machine_tag"])["duration_ms"].median().unstack()
    if stats.isna().any().any() or not {"amd64", "arm64"}.issubset(stats.columns):
        raise ValueError("Mancano durate x86 o ARM per alcune funzioni")
    if (stats[["amd64", "arm64"]] <= 0).any().any():
        raise ValueError("Mediane non positive")
    out = stats.rename(columns={"amd64": "x86_median_duration_ms", "arm64": "arm_median_duration_ms"}).reset_index()
    out["delta_percent"] = 100.0 * (out["arm_median_duration_ms"] / out["x86_median_duration_ms"] - 1.0)
    out["paper_class"] = np.select(
        [out["delta_percent"] < -100 * threshold - 1e-9,
         out["delta_percent"] > 100 * threshold + 1e-9],
        ["ARM-faster", "ARM-slower"], default="ARM-similar",
    )
    out["serverledge_class"] = out["paper_class"].map(CLASS_TO_SERVERLEDGE)
    return out.sort_values("function_name").reset_index(drop=True)


def make_model(trees: int, seed: int) -> RandomForestClassifier:
    """Configurazione sobria: impostazioni sklearn standard, seed e CPU espliciti.

    Il paper non pubblica iperparametri completi: non inventiamo un tuning.
    """
    if trees < 1:
        raise ValueError("Il numero di alberi deve essere positivo")
    return RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=-1)


def majority(predictions: list[str]) -> str:
    """Voto plurale; parita' risolta in ordine fisso e registrabile."""
    if not predictions:
        raise ValueError("Non e' possibile votare con zero campioni")
    counts = Counter(predictions)
    return min(CLASSES, key=lambda item: (-counts[item], CLASSES.index(item)))


def evaluate(samples: pd.DataFrame, selected: list[str], *, trees: int = 100,
             seed: int = 42, method: str = "holdout") -> dict:
    """Train/test separati per intera famiglia di funzione, mai per invocazione."""
    x = samples[selected].to_numpy(dtype=float)
    y = samples["paper_class"].to_numpy(dtype=str)
    groups = samples["function_name"].astype(str).map(family).to_numpy()
    if len(np.unique(groups)) < 4:
        raise ValueError("Poche famiglie indipendenti per valutazione")
    if len(set(y)) != 3:
        raise ValueError("Non ci sono esempi di tutte le tre classi originali del paper")
    if method == "holdout":
        splitter = GroupShuffleSplit(n_splits=1, test_size=PAPER_TEST_FRACTION, random_state=seed)
    elif method == "lofo":
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError("Valutazione non riconosciuta")
    predictions = np.full(len(samples), "", dtype=object)
    split_info = []
    for train_idx, test_idx in splitter.split(x, y, groups):
        missing_train = sorted(set(CLASSES) - set(y[train_idx]))
        if missing_train:
            raise ValueError(f"Classi assenti nel training: {missing_train}; valutazione non valida")
        model = make_model(trees, seed)
        model.fit(x[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(x[test_idx])
        split_info.append({
            "train_functions": sorted(set(samples.iloc[train_idx]["function_name"])),
            "test_functions": sorted(set(samples.iloc[test_idx]["function_name"])),
            "train_families": len(set(groups[train_idx])),
            "test_families": len(set(groups[test_idx])),
        })
    valid = predictions != ""
    truth = y[valid]
    pred = predictions[valid].astype(str)
    heldout = samples.loc[valid, ["function_name", "paper_class"]].copy()
    heldout["prediction"] = pred
    per_fn = []
    for fn, grp in heldout.groupby("function_name", sort=True):
        row = {"function_name": fn, "true": str(grp["paper_class"].iloc[0]),
               "n_samples": int(len(grp))}
        for n in (1, 10, 100):
            if len(grp) >= n:
                voted = majority(grp["prediction"].iloc[:n].tolist())
                row[f"predicted_vote_{n}"] = voted
                row[f"correct_vote_{n}"] = voted == row["true"]
            else:
                row[f"predicted_vote_{n}"] = None
                row[f"correct_vote_{n}"] = None
        per_fn.append(row)
    votes = {}
    for n in (1, 10, 100):
        vals = [f[f"correct_vote_{n}"] for f in per_fn if f[f"correct_vote_{n}"] is not None]
        votes[f"vote_{n}_function_accuracy"] = float(np.mean(vals)) if vals else None
        votes[f"vote_{n}_functions"] = len(vals)
    return {
        "evaluation_protocol": "paper-inspired unseen-function holdout (~11/18 train, 7/18 test)" if method == "holdout" else "leave-one-family-out",
        "sample_accuracy": float(accuracy_score(truth, pred)),
        "sample_count_test": int(valid.sum()),
        "test_functions": len(per_fn),
        "confusion_labels": list(CLASSES),
        "sample_confusion_matrix": confusion_matrix(truth, pred, labels=list(CLASSES)).tolist(),
        "sample_classification_report": classification_report(truth, pred, labels=list(CLASSES), output_dict=True, zero_division=0),
        "voting": votes, "per_function": per_fn, "splits": split_info,
    }


def train(args: argparse.Namespace) -> None:
    """Verifica export e raw, genera etichette paper, valuta e allena il modello finale."""
    path = Path(args.dataset)
    manifest = verify_final_dataset(path)
    data, features = load_dataset(path)
    if len(data) != TARGET_ROWS:
        raise ValueError("Numero di righe diverso dal manifest")
    durations = raw_selected_durations(data, path.parent)
    labels = make_paper_labels(durations)
    if len(labels) != EXPECTED_FUNCTIONS:
        raise ValueError("Numero di funzioni nelle etichette inatteso")
    reference = data.loc[data["machine_tag"] == REFERENCE].copy()
    reference = reference.merge(labels[["function_name", "paper_class"]], on="function_name",
                                how="left", validate="many_to_one")
    if reference["paper_class"].isna().any():
        raise ValueError("Ground truth mancante dopo il merge")
    evaluation = evaluate(reference, features, trees=args.trees, seed=args.seed, method=args.evaluation)
    full_model = make_model(args.trees, args.seed)
    full_model.fit(reference[features].to_numpy(dtype=float), reference["paper_class"])
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    labels.to_csv(dest / "paper_ground_truth.csv", index=False)
    report = {
        "paper": "Chen et al. UCC 2024 — ARM-speed ternary classification (adapted)",
        "runtime_commit": PIN,
        "source_dataset_sha256": manifest["dataset_sha256"],
        "source_provenance_sha256": manifest["provenance_sha256"],
        "class_threshold_percent": 15,
        "duration_ground_truth": "median of 100 selected warm timing.duration_ms per function and architecture",
        "feature_architecture": REFERENCE, "features": features,
        "feature_count": len(features), "paper_original_feature_count": 21,
        "trees": args.trees, "seed": args.seed, "evaluation_method": args.evaluation,
        "train_samples_final": len(reference), "function_count": len(labels),
        "class_distribution_functions": labels["paper_class"].value_counts().to_dict(),
        "evaluation": evaluation,
        "adaptations": ["14 Serverledge feature columns instead of 21 SAAF features",
                        "one workload configuration per function instead of 40",
                        "100 warm runs per function/arch; 67 functions rather than 18",
                        "grouped train/test split to prevent function and twin leakage",
                        "sklearn default RF hyperparameters except explicit trees/seed/n_jobs"],
    }
    (dest / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    joblib.dump({"model": full_model, "features": features, "reference": REFERENCE,
                 "runtime_commit": PIN, "paper_classes": list(CLASSES)}, dest / "model.joblib")
    print("PASS: classificatore paper-inspired addestrato su CPU")
    print("Classi:", report["class_distribution_functions"])
    print("Accuratezza singolo campione TEST:", round(evaluation["sample_accuracy"], 4))
    print("Voto TEST (10/100):", evaluation["voting"])
    print("Ground truth:", dest / "paper_ground_truth.csv")
    print("Valutazione:", dest / "evaluation.json")
    print("Modello:", dest / "model.joblib")


def predict(args: argparse.Namespace) -> None:
    """Inferenza con soli profili x86; produce classi originali e mappatura separata."""
    payload = joblib.load(args.model)  # Caricare esclusivamente file joblib affidabili.
    if payload.get("runtime_commit") != PIN or payload.get("reference") != REFERENCE:
        raise ValueError("Modello non compatibile con runtime o reference architecture")
    incoming = pd.read_csv(args.samples)
    features = feature_columns(incoming)
    if features != payload["features"]:
        raise ValueError("Schema feature del modello e dei nuovi campioni differente")
    only_x86 = incoming.loc[incoming["machine_tag"] == REFERENCE].copy()
    if only_x86.empty:
        raise ValueError("Nessun campione x86 disponibile")
    if only_x86["request_id"].isna().any() or only_x86["request_id"].duplicated().any():
        raise ValueError("RequestID vuoti o duplicati nei nuovi campioni")
    for name in features:
        only_x86[name] = pd.to_numeric(only_x86[name], errors="raise")
    x = only_x86[features].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Feature x86 non finite")
    only_x86["predicted_paper_class"] = payload["model"].predict(x)
    result = []
    for fn, grp in only_x86.groupby("function_name", sort=True):
        row = {"function_name": fn, "total_x86_samples": len(grp)}
        for n in (1, 10, 100):
            if len(grp) >= n:
                klass = majority(grp["predicted_paper_class"].iloc[:n].tolist())
                row[f"vote_{n}"] = {"paper_class": klass,
                                    "serverledge_mapping": CLASS_TO_SERVERLEDGE[klass]}
        row["votes_all"] = dict(Counter(grp["predicted_paper_class"].tolist()))
        result.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runtime_commit": PIN, "reference_architecture": REFERENCE,
                                  "predictions": result}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS: predizioni paper e mappatura Serverledge:", output)


def inspect(args: argparse.Namespace) -> None:
    """Controlla l'intestazione di un selected.csv gia' raccolto, senza lock."""
    raw = pd.read_csv(args.dataset, nrows=25)
    names = feature_columns(raw)
    if raw.empty:
        raise ValueError("selected.csv vuoto")
    for name in names:
        numeric = pd.to_numeric(raw[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Feature non finita in selected.csv: {name}")
    print(f"PASS: schema 6 reale (8 metadati, 14 feature numeriche; {len(raw)} righe verificate)")
    for name in names:
        print("  ", name)


def inspect_raw(args: argparse.Namespace) -> None:
    """Verifica la presenza delle durate nei raw locali, senza scrivere niente."""
    with Path(args.input).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            timing = sample.get("timing") or {}
            if "duration_ms" in timing:
                duration = float(timing["duration_ms"])
                if not np.isfinite(duration) or duration <= 0:
                    raise ValueError("duration_ms non valida")
                print("PASS: timing.duration_ms presente nel raw:", duration)
                return
    raise ValueError("Nessun campione con timing.duration_ms trovato nel raw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Legge lo schema di un selected.csv locale")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=inspect)
    p = sub.add_parser("inspect-raw", help="Verifica campo duration su un raw.jsonl locale")
    p.add_argument("--input", required=True)
    p.set_defaults(func=inspect_raw)
    p = sub.add_parser("train", help="Training e validazione solo CPU a campagna conclusa")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evaluation", choices=("holdout", "lofo"), default="holdout")
    p.set_defaults(func=train)
    p = sub.add_parser("predict", help="Inferenza da CSV schema6 x86")
    p.add_argument("--model", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
