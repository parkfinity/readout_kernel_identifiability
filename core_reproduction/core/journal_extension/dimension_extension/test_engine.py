import copy
import json
from pathlib import Path
import unittest
import numpy as np
from mpmath import iv
import exact_engine as ex
import interval_engine as ie

ROOT = Path(__file__).resolve().parent


def fixture(angles, obs='Z', data=('a0',), count=1):
    return {'n': 1, 'count': count, 'observables': [obs], 'weights': [1],
            'data_expressions': list(data), 'program': angles}


def rot(word, terms): return {'op': 'R', 'pauli': word, 'terms': terms}


class Tests(unittest.TestCase):
    def setUp(self): iv.dps = 50

    def test_exact_input_independent_rx(self):
        a = fixture([rot('X', [[1, 0, -1]])]); self.assertEqual(ex.certify(a)[0]['inactive'], [0])

    def test_overlift_rejected(self):
        a = fixture([rot('X', [[1, 0, -1]])])
        w = {'source_sha256': ie.info_hash(a), 'space': 'independent_angles', 'dps': 50,
             'theta': ['1/3'], 'x': ['0', '0'], 'xprime': ['1/2', '0'], 'parameter': 0}
        with self.assertRaises(ValueError): ie.check_witness(a, w)
        w['space'] = 'original_inputs'
        with self.assertRaises(ValueError): ie.check_witness(a, w)

    def test_data_only_bias_active(self):
        a = fixture([rot('X', [[1, 0, -1], [1, -1, 0]])])
        self.assertFalse(ex.certify(a)[0]['inactive'])
        self.assertTrue(ie.nonzero(ie.derivative(a, ['1/3'], ['1/5', '0'], ['-2/5', '0'])[1][0]))

    def test_shared_live_occurrence(self):
        a = fixture([rot('X', [[1, 0, 0]]), rot('Z', [[1, 0, -1]])])
        self.assertFalse(ex.certify(a)[0]['inactive'])

    def test_shared_inverse_data(self):
        a = fixture([rot('X', [[1, 0, 0]]), rot('X', [[-1, 0, 1]])], data=('acos(a0)', 'acos(a0)'))
        self.assertEqual(ex.certify(a)[0]['inactive'], [0])

    def test_inverse_pair(self):
        a = fixture([rot('X', [[1, 0, -1]]), rot('Y', [[1, 1, -1]]),
                     rot('Y', [[-1, 1, -1]]), rot('X', [[-1, 0, -1]])], count=2)
        self.assertTrue(ex.certify(a)[0]['distance_is_zero'])

    def test_noninteger_not_snapped(self):
        a = fixture([rot('X', [[1.5707963267948966, -1, -1]])])
        with self.assertRaises(ValueError): ex.certify(a)

    def test_dependent_data_not_freed(self):
        a = fixture([rot('X', [[1, 0, 0], [-1, 0, 1]])], data=('a0', 'a0**2'))
        with self.assertRaises(ValueError): ex.certify(a)
        self.assertTrue(ie.nonzero(ie.derivative(a, ['1/3'], ['1/5', '0'], ['-2/5', '0'])[1][0]))

    def test_h_gate_exact_expectation(self):
        a = fixture([{'op': 'H', 'wires': [0]}], obs='X', count=0)
        f, _ = ie.features(a, [], ['1/3', '1/4']); self.assertIn(1, f[0])
        self.assertEqual(ex.feature(a, 'X', [None]), {(): ex.ONE})

    def test_source_hash_rejected(self):
        a = fixture([rot('X', [[1, 0, 0]])])
        with self.assertRaises(ValueError): ie.check_witness(a, {'source_sha256': 'tampered'})

    def test_domain_rejected(self):
        with self.assertRaises(ValueError): ie.features(fixture([]), ['1/3'], ['1', '0'])

    def test_interval_minor(self):
        self.assertTrue(ie.nonzero(ie.determinant([[ie.rational(1), ie.rational(2)],
                                                 [ie.rational(3), ie.rational(5)]])))
        self.assertIsNone(ie.determinant([[ie.rational(1), ie.rational(1)],
                                         [ie.rational(1), ie.rational(1)]]))

    def test_product_gradient_vs_closed_form(self):
        a = fixture([rot('Y', [[1, 0, 0]])])
        d, g = ie.derivative(a, ['2/3'], ['1/5', '0'], ['-2/5', '0'])
        t, x, xp = 2/3, 1/5, -2/5
        expected = 2*(np.cos(t*x)-np.cos(t*xp))*(-x*np.sin(t*x)+xp*np.sin(t*xp))
        self.assertAlmostEqual(ie.mid(g[0]), expected, places=14)

    def test_param_z_layers_distinguished(self):
        a = fixture([{'op': 'H', 'wires': [0]}, rot('Z', [[1, 0, 0]])])
        self.assertEqual(ex.certify(a)[0]['inactive'], [0])
        b = copy.deepcopy(a); b['count'] = 2
        b['program'] += [{'op': 'H', 'wires': [0]}, rot('Z', [[1, 1, 0]])]
        self.assertEqual(ex.certify(b)[0]['inactive'], [1])

    def test_duplicate_observable_rejected(self):
        a = fixture([]); a['observables'] = ['Z', 'Z']; a['weights'] = [1, 1]
        with self.assertRaises(ValueError): ex.certify(a)

    def test_clifford_conjugation_dense(self):
        mats = {'I': np.eye(2), 'X': np.array([[0,1],[1,0]]),
                'Y': np.array([[0,-1j],[1j,0]]), 'Z': np.diag([1,-1])}
        from itertools import product
        gate = {'op': 'CX', 'wires': [0, 1]}
        U = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        for a,b in product('IXYZ', repeat=2):
            sign, word = ex.conjugate(a+b, gate)
            np.testing.assert_array_equal(U.T @ np.kron(mats[a],mats[b]) @ U,
                                          sign*np.kron(mats[word[0]],mats[word[1]]))


if __name__ == '__main__': unittest.main()
