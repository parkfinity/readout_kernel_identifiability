"""Acceptance/rejection controls for the integrated scientific comparator."""
import unittest
from recovery_extension import compare, check_source_errors, SOURCE_ERROR_KEYS

class ComparatorTests(unittest.TestCase):
    def test_roundoff_accepted(self):
        compare({'rank':4,'x':[1.0,0.0]},{'rank':4,'x':[1.0+1e-12,1e-12]})
    def test_rank_change_rejected(self):
        with self.assertRaises(ValueError): compare({'rank':4},{'rank':3})
    def test_material_float_change_rejected(self):
        with self.assertRaises(ValueError): compare(0.01,0.02)
    def test_missing_field_rejected(self):
        with self.assertRaises(ValueError): compare({'rank':4},{})
    def test_type_change_rejected(self):
        with self.assertRaises(ValueError): compare(4,4.0)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): compare(float('nan'),float('nan'))
    def test_original_error_threshold_accepts_roundoff(self):
        check_source_errors({key:1e-10 for key in SOURCE_ERROR_KEYS})
    def test_original_error_threshold_rejects_failure(self):
        with self.assertRaises(ValueError):
            check_source_errors({key:1e-7 for key in SOURCE_ERROR_KEYS})

if __name__=='__main__': unittest.main()
