"""
BilliBench - App5_MlTraining / Func14_knn

Adaptation for Serverledge:
- GCS download/upload removed.
- Dataset is generated once during module initialization (cold phase).
- Warm handler executes the original KNN training kernel.
- Official BilliBench "medium" workload is used.

Original medium workload:
    samples       = 500000
    features      = 40
    informative   = 40
    classes       = 15
    random_state  = 777

Original estimator:
    KNeighborsClassifier(
        n_neighbors=3,
        weights="uniform",
        algorithm="kd_tree",
        metric="euclidean",
    )
"""

import os

# Serverledge profiles functions with configured CPU = 1.
# Prevent BLAS/OpenMP oversubscription.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier


N_SAMPLES = 500_000
N_FEATURES = 40
N_INFORMATIVE = 40
N_CLASSES = 15
RANDOM_STATE = 777


# Dataset preparation belongs to the cold initialization phase.
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_repeated=0,
    n_redundant=0,
    n_classes=N_CLASSES,
    random_state=RANDOM_STATE,
)


def handler(params, context):
    model = KNeighborsClassifier(
        n_neighbors=3,
        weights="uniform",
        algorithm="kd_tree",
        metric="euclidean",
    )

    model.fit(X, y)

    return {
        "benchmark": "billibench-knn",
        "size": "medium",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_classes": len(model.classes_),
        "checksum": int(model.classes_.sum()),
    }
