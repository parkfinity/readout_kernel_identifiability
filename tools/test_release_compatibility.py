"""Regression tests: the compatibility exception must not hide source changes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/'core_reproduction/core'
spec = importlib.util.spec_from_file_location('release_comparator', CORE/'compare_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads((CORE/'reference_results/kyriienko_scaling/20261003T121202Z/provenance.json').read_text())['code']
        self.current = dict(self.reference)
        self.current['kyriienko_scaling.py'] = hashlib.sha256(
            (CORE/'journal_extension/kyriienko_scaling/kyriienko_scaling.py').read_bytes()).hexdigest()

    def test_original_hashes_pass(self):
        self.assertTrue(module.kyriienko_code_matches(self.reference, self.reference))

    def test_actual_patched_source_passes(self):
        self.assertNotEqual(self.current, self.reference)
        self.assertTrue(module.kyriienko_code_matches(self.reference, self.current))

    def test_unknown_scaling_patch_fails(self):
        self.current['kyriienko_scaling.py'] = '0'*64
        self.assertFalse(module.kyriienko_code_matches(self.reference, self.current))

    def test_other_source_change_fails(self):
        key = next(k for k in self.current if k != 'kyriienko_scaling.py')
        self.current[key] = '0'*64
        self.assertFalse(module.kyriienko_code_matches(self.reference, self.current))

    def test_missing_source_fails(self):
        self.current.pop(next(k for k in self.current if k != 'kyriienko_scaling.py'))
        self.assertFalse(module.kyriienko_code_matches(self.reference, self.current))

    def test_extra_source_fails(self):
        self.current['unexpected.py'] = '0'*64
        self.assertFalse(module.kyriienko_code_matches(self.reference, self.current))

    def test_wrong_historical_source_fails(self):
        self.reference['kyriienko_scaling.py'] = '0'*64
        self.assertFalse(module.kyriienko_code_matches(self.reference, self.current))

    def test_missing_provenance_fails(self):
        self.assertFalse(module.kyriienko_code_matches(None, None))

if __name__ == '__main__':
    unittest.main()
