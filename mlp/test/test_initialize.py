import unittest
from initialize import initialize_weights
import numpy as np


class TestInitialize(unittest.TestCase):
    def test_zero_init(self):
        # GIVEN
        in_dim, out_dim = 2, 3

        # WHEN
        weight = initialize_weights(in_dim, out_dim, "zero")
        expected_weight = np.array([[0, 0, 0],
                                    [0, 0, 0]])

        # THEN
        self.assertEqual(weight.shape, (in_dim, out_dim))
        self.assertTrue(np.array_equal(weight, expected_weight))

    def test_normal_init(self):
        # GIVEN
        in_dim, out_dim = 2, 3

        # WHEN
        weight = initialize_weights(in_dim, out_dim, "normal")

        # THEN
        self.assertEqual(weight.shape, (in_dim, out_dim))
