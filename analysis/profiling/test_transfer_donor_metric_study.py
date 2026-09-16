import unittest

from analysis.profiling.transfer_donor_metric_study import (
    first_true_run,
    safe_scale,
)


class TransferDonorMetricStudyTest(unittest.TestCase):

    def test_first_true_run(self):
        values = [False, True, True, True, False]
        self.assertEqual(
            first_true_run(values, 3),
            4,
        )

    def test_first_true_run_none(self):
        values = [True, False, True, True]
        self.assertIsNone(
            first_true_run(values, 3)
        )

    def test_safe_scale_uses_positive_finite_values(self):
        import numpy as np

        values = np.asarray(
            [0.0, 2.0, 4.0, float("inf")]
        )
        self.assertEqual(
            safe_scale(values),
            3.0,
        )

    def test_safe_scale_fallback(self):
        import numpy as np

        values = np.asarray([0.0, 0.0])
        self.assertEqual(
            safe_scale(values),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
