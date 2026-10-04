"""
BilliBench - App5_MlTraining / Func18_kmeans

Adaptation for Serverledge:
- GCS download/upload removed.
- Dataset is generated once during module initialization (cold phase).
- Warm handler executes the original KMeans training kernel.
- Official BilliBench "medium" workload is used.
- Dataset random_state=777 is fixed only to guarantee identical
  input generation on x86 and ARM.

Original KMeans estimator does not specify random_state, therefore
the estimator is intentionally left unchanged.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


N_SAMPLES = 50_000
N_FEATURES = 40
N_CLUSTERS = 60
DATASET_RANDOM_STATE = 777


X, _ = make_blobs(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    centers=N_CLUSTERS,
    center_box=(-32, 32),
    shuffle=True,
    random_state=DATASET_RANDOM_STATE,
)


def handler(params, context):
    model = KMeans(
        n_clusters=N_CLUSTERS,
        max_iter=50,
    )

    model.fit(X)

    return {
        "benchmark": "billibench-kmeans",
        "size": "medium",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_clusters": N_CLUSTERS,
        "iterations": int(model.n_iter_),
        "inertia": float(model.inertia_),
        "centers_checksum": float(model.cluster_centers_.sum()),
    }
