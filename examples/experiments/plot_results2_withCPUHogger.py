import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.patches as mpatches

# Configurazione stile
sns.set_theme(style="whitegrid")
# Palette colori ad alto contrasto per distinguere le policy
custom_palette = {"MAB_LinUCB": "#E63946", "MAB_UCB1": "#457B9D", "RoundRobin": "#26ac26"}

def analyze_locust_experiment(file_path):
    # 1. Caricamento Dati Principali
    df = pd.read_csv(file_path)
    df = df.sort_values(by='timestamp')

    # 1.1 Caricamento Dati CPU Hogger (Modificato per intervalli multipli)
    hog_intervals = [] # Lista di tuple (start, end)
    if os.path.exists("cpu_hogger.csv"):
        try:
            # Leggiamo il CSV che ha colonne start_ts e end_ts
            df_hog = pd.read_csv("cpu_hogger.csv")
            
            # Iteriamo sulle righe per salvare tutti gli intervalli
            for _, row in df_hog.iterrows():
                # Assumiamo che le colonne si chiamino 'start_ts' e 'end_ts'
                if pd.notna(row['start_ts']) and pd.notna(row['end_ts']):
                    hog_intervals.append((row['start_ts'], row['end_ts']))
                    
        except Exception as e:
            print(f"Nota: Impossibile leggere o processare cpu_hogger.csv ({e})")

    # 2. Preprocessing Temporale
    # Calcoliamo il tempo relativo per ogni policy in modo indipendente
    dfs = []
    
    # Dizionario per salvare l'inizio assoluto di ogni policy
    # Serve per mappare gli intervalli assoluti del hogger nel tempo relativo della policy
    policy_start_times = {} 
    
    # Salviamo anche la durata massima per evitare di disegnare hogging fuori dal grafico
    policy_durations = {}

    for policy, group in df.groupby('policy'):
        group = group.copy()
        start_time = group['timestamp'].min()
        policy_start_times[policy] = start_time
        
        group['relative_time'] = group['timestamp'] - start_time
        policy_durations[policy] = group['relative_time'].max()

        # Cumulativo GLOBALE per la policy (aggregato)
        group['cumulative_total'] = range(1, len(group) + 1)

        # Cumulativo PER FUNZIONE
        # Ordiniamo per funzione e tempo per fare il cumcount corretto
        group_func = group.sort_values(by=['function', 'relative_time'])
        group_func['cumulative_func'] = group_func.groupby('function').cumcount() + 1

        dfs.append(group_func)

    df_processed = pd.concat(dfs)

    # Funzione helper per disegnare l'area colorata su un asse specifico
    def highlight_hog_region(ax):
        if not hog_intervals:
            return
        
        # Iteriamo su tutti gli intervalli trovati nel CSV
        for h_start_abs, h_end_abs in hog_intervals:
            # Controlliamo per ogni policy (per gestire il tempo relativo corretto)
            for policy, p_start in policy_start_times.items():
                rel_h_start = h_start_abs - p_start
                rel_h_end = h_end_abs - p_start
                
                # REGOLE DI SOVRAPPOSIZIONE:
                # 1. L'intervallo deve finire DOPO l'inizio dell'esperimento (rel_h_end > 0)
                # 2. L'intervallo deve iniziare PRIMA della fine dell'esperimento (rel_h_start < duration)
                p_duration = policy_durations[policy]
                
                if rel_h_end > 0 and rel_h_start < p_duration:
                    # Disegna la banda verticale (gialla semitrasparente)
                    # max(0, ...) serve per tagliare l'inizio a 0 se l'evento è iniziato prima
                    ax.axvspan(max(0, rel_h_start), rel_h_end, color='#FFD700', alpha=0.2, lw=0, zorder=0)

    # Creiamo un patch per la legenda personalizzata
    hog_patch = mpatches.Patch(color='#FFD700', alpha=0.2, label='CPU Hogging Active')

    # Funzione per aggiornare la legenda includendo il patch del hogging
    def update_legend(ax):
        if not hog_intervals:
            return
        # Recupera la legenda esistente
        handles, labels = ax.get_legend_handles_labels()
        # Aggiunge il nostro patch se non c'è già
        if 'CPU Hogging Active' not in labels:
            handles.append(hog_patch)
            ax.legend(handles=handles, title='Legenda', loc='upper left')

    # ---------------------------------------------------------
    # GRAFICO 1: Throughput Separato (Fairness per Funzione)
    # ---------------------------------------------------------
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
    
    # Applichiamo l'evidenziazione su tutti i subplot
    for ax in g.axes.flatten():
        highlight_hog_region(ax)
    
    # Aggiungiamo la voce alla legenda (basta farlo sull'ultimo asse o riconfigurare la legenda del FacetGrid)
    # Per Seaborn FacetGrid, spesso è meglio ri-aggiungere la legenda globale
    g._legend.remove() # Rimuoviamo la legenda automatica esterna
    # Creiamo una legenda unificata prendendo gli handle dal primo asse
    handles, labels = g.axes[0,0].get_legend_handles_labels()
    if hog_intervals:
        handles.append(hog_patch)
    g.fig.legend(handles=handles, title="Policy / Status", loc='center right', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.85) # Facciamo spazio per la legenda esterna

    g.fig.suptitle('Confronto A: Velocità per singola Funzione', y=1.03, fontsize=16)
    g.set_axis_labels("Tempo Trascorso (s)", "Richieste Completate")
    plt.show()

    # ---------------------------------------------------------
    # GRAFICO 2: Throughput Aggregato (La visione d'insieme)
    # ---------------------------------------------------------
    df_agg = df_processed.sort_values(by=['policy', 'relative_time'])

    plt.figure(figsize=(10, 6))
    ax2 = sns.lineplot(
        data=df_agg,
        x='relative_time',
        y='cumulative_total',
        hue='policy',
        style='policy',
        palette=custom_palette,
        linewidth=2.5
    )
    
    highlight_hog_region(ax2)
    update_legend(ax2) # Aggiorna la legenda di questo plot specifico

    plt.title('Confronto B: Throughput Aggregato (Sistema Completo)', fontsize=16)
    plt.xlabel('Tempo Trascorso (s)')
    plt.ylabel('Totale Richieste Completate (Somma delle funzioni)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    # ---------------------------------------------------------
    # GRAFICO 3: Analisi Architetturale (MAB sta imparando?)
    # ---------------------------------------------------------
    g = sns.FacetGrid(df_processed, col="function", row="policy", height=4, aspect=1.5, sharex=True)
    g.map_dataframe(sns.scatterplot, x="relative_time", y="node_arch", alpha=0.6, s=30)
    
    for ax in g.axes.flatten():
        highlight_hog_region(ax)

    # Gestione Legenda per FacetGrid (qui non c'è hue automatica complessa, quindi possiamo inserirla manualmente in uno dei plot o fuori)
    # Creiamo una legenda personalizzata solo per indicare il colore giallo, dato che le policy sono divise per righe
    if hog_intervals:
        g.add_legend(legend_data={'CPU Hogging Active': hog_patch}, title="Eventi")

    g.fig.suptitle('Strategia di Assegnazione: Chi esegue cosa?', y=1.02, fontsize=16)
    g.set_axis_labels("Tempo (s)", "Architettura")
    plt.show()

    # ---------------------------------------------------------
    # 4. Statistiche "Fair"
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("ANALISI PRESTAZIONI (Metrics)")
    print("="*60)

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
