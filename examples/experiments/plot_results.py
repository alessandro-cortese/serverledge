import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

df = pd.read_csv("experiment_results.csv")

df = df.dropna(subset=['response_time_ms', 'node_arch'])
df = df[df['node_arch'] != 'unknown']

# Separate policies
policies = df['policy'].unique()
if len(policies) < 2:
    print("WARNING: Only one policy found in the CSV. Cannot make comparisons.")
    # Continue anyway for debugging

# --- STEP 1: Data Alignment (Makespan Logic) ---
# Find the minimum number of requests executed between the two policies to trim the data
min_requests = df.groupby('policy').size().min()
print(f"Trimming data to the first {min_requests} requests to make the comparison fair.")

df_trimmed = pd.DataFrame()
for p in policies:
    # Take the first N requests for each policy
    subset = df[df['policy'] == p].sort_values('timestamp').head(min_requests).copy()
    # Recalculate a progressive index (Request ID) from 1 to N
    subset['request_id'] = range(1, len(subset) + 1)
    # Calculate cumulative
    subset['cumulative_time_s'] = subset['response_time_ms'].cumsum() # they're already seconds, actually
    df_trimmed = pd.concat([df_trimmed, subset])

# --- GRAPH 1: Cumulative Time (Monotonic Curves) ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_trimmed, x='request_id', y='cumulative_time_s', hue='policy', linewidth=2.5)

plt.xlabel('Numero di Richieste Completate')
plt.ylabel('Tempo Totale Cumulativo (secondi)')
plt.title('Confronto Velocità: Baseline vs MAB (A parità di carico)')
plt.legend(title='Strategia')
plt.tight_layout()
plt.savefig('grafico_cumulativo.png')
plt.show()

# --- GRAPH 2: Architecture Distribution (Bar Chart) ---
# We want to see: For each Function -> For each Policy -> How much x86 vs ARM?

# Aggregate data: Count requests by (Policy, Function, Arch)
# Use the original (untrimmed) df to see the full behavior
count_data = df.groupby(['policy', 'function', 'node_arch']).size().reset_index(name='count')

g = sns.catplot(
    data=count_data,
    kind="bar",
    x="function",
    y="count",
    hue="node_arch",
    col="policy", # Creates two separate side-by-side graphs: one for RR, one for MAB
    palette="muted",
    height=5,
    aspect=1
)

# Add labels with numbers above the bars
g.set_axis_labels("Function", "Number of Executions")
g.fig.suptitle('Distribution of Architectural Choices (x86 vs ARM)', y=1.05)

for ax in g.axes.flat:
    for p in ax.containers:
        ax.bar_label(p, label_type='edge')

plt.savefig('grafico_distribuzione.png')
plt.show()

# --- GRAPH 3: Architecture Choice Timeline (by policy, colored by function) ---

# Ordiniamo temporalmente
df_timeline = df.sort_values('timestamp').copy()

# Indice temporale separato per policy
df_timeline['decision_step'] = (
    df_timeline
    .groupby('policy')
    .cumcount() + 1
)

# Mappatura architetture reali
arch_mapping = {'arm64': 0, 'amd64': 1}
df_timeline = df_timeline[df_timeline['node_arch'].isin(arch_mapping)]
df_timeline['arch_numeric'] = df_timeline['node_arch'].map(arch_mapping)

# Grafico: pannelli separati per policy, colori per funzione
g = sns.relplot(
    data=df_timeline,
    x='decision_step',
    y='arch_numeric',
    hue='function',
    col='policy',
    kind='scatter',
    height=4,
    aspect=1.5,
    s=45
)

# Etichette e titoli
g.set_axis_labels("Decision step (ordine temporale)", "Architettura scelta")
g.set_titles("Policy: {col_name}")

for ax in g.axes.flat:
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['ARM64', 'AMD64'])

plt.suptitle("Cronologia delle decisioni architetturali nel tempo", y=1.05)
plt.tight_layout()
plt.savefig("grafico_timeline_decisioni.png")
plt.show()


# --- Textual Statistics ---
print("\n=== Final Statistics ===")
for p in policies:
    sub = df[df['policy'] == p]
    total_time = sub['response_time_ms'].sum() / 1000.0
    count = len(sub)
    print(f"Policy: {p}")
    print(f"  - Total Requests: {count}")
    print(f"  - Total Cumulative Time: {total_time:.2f} s")
    print(f"  - Average Time per Request: {sub['response_time_ms'].mean():.2f} ms")
    print("-" * 30)

# --- Average Execution Time per Function, Architecture and Policy ---
print("\n=== Average Execution Time per Function and Architecture (by Policy) ===")

avg_func_arch_policy = (
    df
    .groupby(['policy', 'function', 'node_arch'])['response_time_ms']
    .mean()
    .reset_index()
    .sort_values(by=['policy', 'function', 'node_arch'])
)

current_policy = None
current_function = None

for _, row in avg_func_arch_policy.iterrows():
    policy = row['policy']
    func = row['function']
    arch = row['node_arch']
    avg_time = row['response_time_ms']

    # Nuova policy
    if policy != current_policy:
        print(f"\nPolicy: {policy}")
        current_policy = policy
        current_function = None

    # Nuova funzione (dentro la policy)
    if func != current_function:
        print(f"  Function: {func}")
        current_function = func

    print(f"    - {func} su {arch.upper()}: {avg_time:.2f} ms")


