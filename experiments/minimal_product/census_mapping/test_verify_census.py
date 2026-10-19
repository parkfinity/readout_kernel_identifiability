"""Negative controls for the index checker; no public files are changed."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import verify_census as vc


class IndexChecks(unittest.TestCase):
    def test_current_mapping_and_table_agree(self):
        actual = vc.build(vc.DEFAULT_PACKAGE, vc.HERE.parent / "manuscript")
        self.assertEqual(actual, vc.read(vc.HERE / "census_54_mapping.json"))
        self.assertEqual(len(actual["settings"]), 54)
        self.assertEqual(actual["summary"]["extension_nonempty_modular_minors"], 54)
        self.assertEqual(actual["summary"]["nonempty_rational_minors"], 6)

    def test_changed_pilot_rank_rejected(self):
        real_read = vc.read
        target = "chebyshev_rx_n2_l1_local_xyz.json"

        def altered(path):
            value = copy.deepcopy(real_read(path))
            if path.name == target and "minor" in value:
                value["minor"]["lower"] += 1
            return value

        with patch.object(vc, "read", altered):
            with self.assertRaisesRegex(ValueError, "unclosed pilot rank"):
                vc.build(vc.DEFAULT_PACKAGE, vc.HERE.parent / "manuscript")

    def test_zero_containing_interval_rejected(self):
        real_read = vc.read
        target = "chebyshev_rx_n2_l1_local_xyz.json"

        def altered(path):
            value = copy.deepcopy(real_read(path))
            if path.name == target and "minor" in value:
                value["minor"]["determinant_bounds"] = ["-1", "1"]
            return value

        with patch.object(vc, "read", altered):
            with self.assertRaisesRegex(ValueError, "zero-containing minor"):
                vc.build(vc.DEFAULT_PACKAGE, vc.HERE.parent / "manuscript")

    def test_noninvertible_denominator_rejected(self):
        info = {"weights": ["1/5"], "program": []}
        record = {"theta_star": {"zeta": [[3, 4, 5]]}}
        with self.assertRaisesRegex(ValueError, "Noninvertible denominator"):
            vc.denominator_check(info, record, 5)

    def test_false_rational_circle_rejected(self):
        info = {"weights": [1], "program": []}
        record = {"theta_star": {"zeta": [[2, 4, 5]]}}
        with self.assertRaisesRegex(ValueError, "Invalid rational-circle"):
            vc.denominator_check(info, record, 7)

    def test_composite_modulus_rejected(self):
        self.assertFalse(vc.is_prime(134217687))
        self.assertTrue(vc.is_prime(134217689))


if __name__ == "__main__":
    unittest.main()
