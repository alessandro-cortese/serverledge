import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurazione stile
sns.set_theme(style="whitegrid")
# Palette colori ad alto contrasto per distinguere le policy
custom_palette = {"MAB_LinUCB": "#E63946", "RoundRobin": "#457B9D"}

def analyze_locust_experiment(file_path):
    # 1. Caricamento
    df = pd.read_csv(file_path)
    df = df.sort_values(by='timestamp')

    # 2. Preprocessing Temporale
    # Calcoliamo il tempo relativo per ogni policy in modo indipendente
    # Così partono entrambe da t=0 per il confronto
    dfs = []
    for policy, group in df.groupby('policy'):
        group = group.copy()
        start_time = group['timestamp'].min()
        group['relative_time'] = group['timestamp'] - start_time

        # Cumulativo GLOBALE per la policy (aggregato)
        group['cumulative_total'] = range(1, len(group) + 1)

        # Cumulativo PER FUNZIONE
        # Ordiniamo per funzione e tempo per fare il cumcount corretto
        group_func = group.sort_values(by=['function', 'relative_time'])
        group_func['cumulative_func'] = group_func.groupby('function').cumcount() + 1

        dfs.append(group_func)

    df_processed = pd.concat(dfs)

    # ---------------------------------------------------------
    # GRAFICO 1: Throughput Separato (Fairness per Funzione)
    # ---------------------------------------------------------
    # Usiamo 'line' plot. Assi Y indipendenti (sharey=False) fondamentali
    # perché primenum è molto più veloce di linpack.
    g = sns.relplot(
        data=df_processed,
        x="relative_time",
        y="cumulative_func",
        hue="policy",
        style="policy",
        col="function",
        kind="line",
        palette=custom_palette,
        height=5,
        aspect=1.2,
        linewidth=2.5,
        facet_kws={'sharey': False, 'sharex': True}
    )
    g.fig.suptitle('Confronto A: Velocità per singola Funzione', y=1.03, fontsize=16)
    g.set_axis_labels("Tempo Trascorso (s)", "Richieste Completate")
    plt.show()

    # ---------------------------------------------------------
    # GRAFICO 2: Throughput Aggregato (La visione d'insieme)
    # ---------------------------------------------------------
    # Qui ordiniamo per tempo relativo per avere una linea crescente pulita
    df_agg = df_processed.sort_values(by=['policy', 'relative_time'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_agg,
        x='relative_time',
        y='cumulative_total',
        hue='policy',
        style='policy',
        palette=custom_palette,
        linewidth=2.5
    )
    plt.title('Confronto B: Throughput Aggregato (Sistema Completo)', fontsize=16)
    plt.xlabel('Tempo Trascorso (s)')
    plt.ylabel('Totale Richieste Completate (Somma delle funzioni)')
    plt.legend(title='Policy', loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    # ---------------------------------------------------------
    # GRAFICO 3: Analisi Architetturale (MAB sta imparando?)
    # ---------------------------------------------------------
    # Visualizziamo quale architettura viene scelta nel tempo
    g = sns.FacetGrid(df_processed, col="function", row="policy", height=4, aspect=1.5, sharex=True)
    g.map_dataframe(sns.scatterplot, x="relative_time", y="node_arch", alpha=0.6, s=30)
    g.fig.suptitle('Strategia di Assegnazione: Chi esegue cosa?', y=1.02, fontsize=16)
    g.set_axis_labels("Tempo (s)", "Architettura")
    plt.show()

    # ---------------------------------------------------------
    # 4. Statistiche "Fair"
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("ANALISI PRESTAZIONI (Metrics)")
    print("="*60)

    # Calcoliamo il Throughput Medio (Richieste / Durata Totale)
    policy_stats = df_processed.groupby('policy').agg(
        total_req=('response_time_ms', 'count'),
        duration=('relative_time', 'max'),
        avg_exec_time=('response_time_ms', 'mean'),
        p95_exec_time=('response_time_ms', lambda x: x.quantile(0.95))
    )

    policy_stats['throughput_req_per_sec'] = policy_stats['total_req'] / policy_stats['duration']

    print("--- Riepilogo Generale ---")
    print(policy_stats[['total_req', 'duration', 'throughput_req_per_sec']].round(2))

    print("\n--- Dettaglio Latenza (Esecuzione Server) ---")
    print(policy_stats[['avg_exec_time', 'p95_exec_time']].round(2))

    print("\n--- Dettaglio per Funzione ---")
    func_stats = df_processed.groupby(['policy', 'function'])['response_time_ms'].mean().unstack().round(2)
    print(func_stats)

if __name__ == "__main__":
    analyze_locust_experiment('experiment_results.csv')
