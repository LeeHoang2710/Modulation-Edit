"""CPU tests for two-subject SharedAdaLN token alignment."""
import unittest

import torch

from tools.spatial_modulation import (
    compose_horizontal_modulation,
    horizontal_region_weights,
    scale_offset,
)


class HorizontalRegionWeightTests(unittest.TestCase):
    def test_even_width_is_exact_left_right_partition(self):
        weights = horizontal_region_weights((1, 2, 4)).flatten()
        self.assertTrue(torch.equal(weights, torch.tensor([1., 1., 0., 0., 1., 1., 0., 0.])))

    def test_coarsest_one_token_scale_blends_both_subjects(self):
        weights = horizontal_region_weights((1, 1, 1)).flatten()
        self.assertTrue(torch.equal(weights, torch.tensor([0.5])))

    def test_odd_width_only_blends_boundary_column(self):
        weights = horizontal_region_weights((1, 1, 3)).flatten()
        self.assertTrue(torch.allclose(weights, torch.tensor([1., 0.5, 0.])))


class CompositionTests(unittest.TestCase):
    def test_all_six_vectors_follow_the_same_spatial_alignment(self):
        left = torch.arange(6., dtype=torch.float32).view(1, 1, 6, 1)
        right = left + 100
        result = compose_horizontal_modulation(left, right, (1, 1, 4))
        self.assertEqual(result.shape, (1, 4, 6, 1))
        self.assertTrue(torch.equal(result[:, :2], left.expand(1, 2, 6, 1)))
        self.assertTrue(torch.equal(result[:, 2:], right.expand(1, 2, 6, 1)))

    def test_scale_offset_supports_shared_per_scale_and_missing_scales(self):
        shared = torch.ones(1, 1, 6, 2)
        self.assertIs(scale_offset(shared, 10), shared)
        per_scale = torch.stack((shared[0], 2 * shared[0]), dim=0)
        self.assertTrue(torch.equal(scale_offset(per_scale, 1), 2 * shared))
        self.assertTrue(torch.equal(scale_offset(per_scale, 3), torch.zeros_like(shared)))


if __name__ == "__main__":
    unittest.main()
