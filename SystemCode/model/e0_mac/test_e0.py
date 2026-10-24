"""Small metric/aggregation regression tests; no weights or torch required."""
import unittest
import numpy as np
from run_e0 import binary_metrics, legacy_uint8, group_metrics


class E0MetricsTests(unittest.TestCase):
    def test_known_overlap(self):
        pred = np.array([[1, 1], [0, 0]], dtype=bool)
        gt = np.array([[1, 0], [1, 0]], dtype=bool)
        m = binary_metrics(pred, gt)
        self.assertEqual((m['tp'], m['fp'], m['fn']), (1, 1, 1))
        self.assertEqual(m['dice'], 0.5)
        self.assertAlmostEqual(m['iou'], 1/3)

    def test_empty_conventions_are_explicit(self):
        empty = np.zeros((2, 2), dtype=bool)
        self.assertEqual(binary_metrics(empty, empty)['dice'], 1)
        self.assertEqual(binary_metrics(empty, empty, empty_score=0)['dice'], 0)
        self.assertEqual(binary_metrics(~empty, empty)['iou'], 0)

    def test_minmax_is_not_a_probability(self):
        p = np.array([[0.10, 0.20]], dtype=np.float32)
        self.assertEqual(int((p >= .5).sum()), 0)
        self.assertEqual(int((legacy_uint8(p) / 255 >= .5).sum()), 1)

    def test_global_average_is_image_weighted(self):
        rows = []
        for ds, val in [('Kvasir', 1.), ('Kvasir', 1.), ('CVC-300', 0.)]:
            rows.append(dict(dataset=ds, dice=val, iou=val, mae=0., forward_ms=1.))
        groups = group_metrics(rows, False)
        self.assertAlmostEqual(groups[0]['mean_dice'], 2/3)
        self.assertEqual(groups[0]['count'], 3)
        self.assertEqual(groups[1]['mean_dice'], 1.)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            binary_metrics(np.zeros((2, 2)), np.zeros((2, 1)))


if __name__ == '__main__':
    unittest.main()
