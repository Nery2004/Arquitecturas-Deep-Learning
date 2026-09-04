"""Pruebas pequeñas: no generan ni duplican el Dataset Version 1."""

import unittest
from dataclasses import replace

from src.data_generator import (
    GeneratorConfig,
    dataset_fingerprint,
    generate_transactions,
    validate_dataset,
)


class DataGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = GeneratorConfig.small(42)
        cls.rows, _ = generate_transactions(cls.config)

    def test_small_dataset_is_valid(self) -> None:
        validation = validate_dataset(self.rows, self.config)
        self.assertTrue(validation["amount_ranges_overlap"])

    def test_same_seed_has_same_hash(self) -> None:
        repeated, _ = generate_transactions(self.config)
        self.assertEqual(dataset_fingerprint(self.rows), dataset_fingerprint(repeated))

    def test_different_seed_has_different_hash(self) -> None:
        changed, _ = generate_transactions(replace(self.config, random_seed=43))
        self.assertNotEqual(dataset_fingerprint(self.rows), dataset_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
