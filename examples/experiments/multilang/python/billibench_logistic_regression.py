"""
BilliBench - App5_MlTraining / Func16_logisticReg

Adaptation for Serverledge:
- GCS download/upload removed.
- Dataset is generated once during module initialization (cold phase).
- Warm handler executes the original LogisticRegression training kernel.
- Official BilliBench "medium" workload is used.
- multi_class="auto" removed for compatibility with scikit-learn 1.8.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


N_SAMPLES = 500_000
N_FEATURES = 40
N_INFORMATIVE = 40
N_CLASSES = 15
RANDOM_STATE = 777


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
    model = LogisticRegression(
        C=1.0,
        fit_intercept=True,
        verbose=False,
        tol=1e-10,
        max_iter=100,
        solver="lbfgs",
        l1_ratio=0,
    )

    model.fit(X, y)

    return {
        "benchmark": "billibench-logistic-regression",
        "size": "medium",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_classes": len(model.classes_),
        "iterations": int(model.n_iter_.max()),
        "coef_checksum": float(model.coef_.sum()),
    }
