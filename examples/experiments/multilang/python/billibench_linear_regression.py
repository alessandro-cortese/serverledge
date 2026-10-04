"""
BilliBench - App5_MlTraining / Func15_linearReg

Adaptation for Serverledge:
- GCS download/upload removed.
- Dataset is generated once during module initialization (cold phase).
- Warm handler executes the original LinearRegression training kernel.
- Official BilliBench "medium" workload is used.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state


N_SAMPLES = 500_000
N_FEATURES = 40
N_INFORMATIVE = 40
RANDOM_STATE = 777


# Preserve BilliBench dataset-generation procedure.
_rs = check_random_state(RANDOM_STATE)

X, y = make_regression(
    n_targets=1,
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    bias=_rs.normal(0, 3),
    random_state=_rs,
)


def handler(params, context):
    model = LinearRegression(
        fit_intercept=True,
    )

    model.fit(X, y)

    return {
        "benchmark": "billibench-linear-regression",
        "size": "medium",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "coef_checksum": float(model.coef_.sum()),
        "intercept": float(model.intercept_),
    }
