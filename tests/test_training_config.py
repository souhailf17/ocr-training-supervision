import unittest

from training_config import accumulation_steps


class AccumulationConfigTests(unittest.TestCase):
    def test_existing_global_batch_configuration(self):
        self.assertEqual(accumulation_steps(384, 4, 3), 32)

    def test_microbatch_four_preserves_global_batch(self):
        self.assertEqual(4 * 4 * accumulation_steps(384, 4, 4), 384)
        self.assertEqual(accumulation_steps(384, 4, 4), 24)

    def test_non_divisible_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            accumulation_steps(384, 4, 5)

    def test_invalid_dimensions_are_rejected(self):
        for values in ((0, 4, 1), (384, 0, 1), (384, 4, 0), (384.0, 4, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                accumulation_steps(*values)


if __name__ == "__main__":
    unittest.main()
