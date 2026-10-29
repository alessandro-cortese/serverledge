#!/usr/bin/env python3
"""Valutazione di conformita' al criterio di exact-match ternario del paper.

NON modifica alcun modello. Per Serverledge, che predice soltanto ARM/x86,
le predizioni su target realmente neutri contano come errori. Questa e' una
valutazione ESTERNA con tre classi reali, NON un classificatore Serverledge
ternario: non vengono inventate predizioni ARM-similar.

Eseguire dalla root del repository:
    python3 analysis/profiling/verifica_accuracy_ternaria_serverledge_59.py

Richiede l'output precedentemente prodotto da valuta_jolly_professore_59.py.
Opzionalmente aggiungere --output <file.csv> per esportare il riepilogo.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

DEFAULT = Path('data/architecture-comparison/comparison/professor-ternary-jolly-v1')
METHODS = {
    15.0: ['paper_RF_15_original', 'majority_diagnostic', 'weighted_frozen'],
    2.5: ['paper_RF_2p5_relabelled', 'majority_diagnostic', 'weighted_frozen'],
}
SEEDS = {11, 23, 37, 41, 53}
DIRECTIONAL = {'arm-preferred', 'x86-preferred'}
NEUTRAL = 'architecture-independent'
ABSTAIN = 'ABSTAIN'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, default=DEFAULT)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    input_dir = args.input_dir
    path = input_dir/'jolly_per_function_seed.csv'
    manifest = input_dir/'manifest.json'
    if not path.is_file() or not manifest.is_file():
        raise SystemExit('Manca jolly_per_function_seed.csv o manifest.json: eseguire prima valuta_jolly_professore_59.py run')
    doc = json.loads(manifest.read_text(encoding='utf-8'))
    saved = doc.get('outputs', {}).get(path.name, {}).get('sha256')
    if not saved or digest(path) != saved:
        raise SystemExit('ERRORE: SHA256 del CSV non coincide con il manifest congelato.')

    df = pd.read_csv(path)
    required = {'function_name', 'seed', 'method', 'threshold_percent_EVAL_ONLY',
                'true_preference_EVAL_ONLY', 'predicted_preference'}
    if missing := required - set(df.columns):
        raise SystemExit(f'Colonne mancanti: {sorted(missing)}')
    df['seed'] = pd.to_numeric(df.seed, errors='raise').astype(int)
    df['threshold_percent_EVAL_ONLY'] = pd.to_numeric(df.threshold_percent_EVAL_ONLY, errors='raise')
    if set(df.seed.unique()) != SEEDS:
        raise SystemExit('Seed inattesi.')
    if df.duplicated(['seed', 'threshold_percent_EVAL_ONLY', 'method', 'function_name']).any():
        raise SystemExit('Duplicati funzione-metodo-seed-soglia.')
    permitted = DIRECTIONAL | {NEUTRAL, ABSTAIN}
    if not set(df['predicted_preference'].unique()) <= permitted:
        raise SystemExit('Predizioni non riconosciute.')
    if not set(df['true_preference_EVAL_ONLY'].unique()) <= DIRECTIONAL | {NEUTRAL}:
        raise SystemExit('Etichette ground truth non riconosciute.')

    rows = []
    for threshold, methods in METHODS.items():
        for method in methods:
            group = df[(df.threshold_percent_EVAL_ONLY == threshold) & (df.method == method)]
            if len(group) != 59*len(SEEDS):
                raise SystemExit(f'Numero righe inatteso per {threshold}/{method}: {len(group)}')
            per_seed = []
            for seed, q in group.groupby('seed'):
                if len(q) != 59 or q.function_name.nunique() != 59:
                    raise SystemExit(f'Fold non valido: {threshold}/{method}/{seed}')
                if int(q.true_preference_EVAL_ONLY.eq(NEUTRAL).sum()) != (40 if threshold == 15 else 9):
                    raise SystemExit('Distribuzione target cambiata rispetto al dataset congelato.')
                correct = int(q.predicted_preference.eq(q.true_preference_EVAL_ONLY).sum())
                per_seed.append(correct)
            rows.append({
                'soglia_percento': threshold,
                'metodo': method,
                'accuracy_exact_label_percento': round(100*sum(per_seed)/(59*len(SEEDS)), 6),
                'corrette_medie_su_59': round(sum(per_seed)/len(SEEDS), 6),
                'numero_funzioni': 59,
                'seed': 5,
                'nota': ('RF: vera predizione ternaria'
                         if method.startswith('paper_RF')
                         else 'Serverledge: output binario forzato a confronto con verita ternaria'),
            })
    output = pd.DataFrame(rows)
    print('PASS: hash verificato; 59 funzioni per seed; cinque seed; nessun riaddestramento')
    print(output[['soglia_percento','metodo','corrette_medie_su_59','accuracy_exact_label_percento']].to_string(index=False))
    print('\nATTENZIONE: Serverledge non predice ARM-similar; i target neutrali sono inevitabilmente errati nel criterio ternario esatto.')
    if args.output is not None:
        if args.output.exists():
            raise SystemExit('Rifiuto di sovrascrivere output esistente.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print('Creato:', args.output)

if __name__ == '__main__':
    main()
