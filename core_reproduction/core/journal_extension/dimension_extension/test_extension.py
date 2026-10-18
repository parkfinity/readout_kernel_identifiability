import copy
import unittest
from unittest import mock
import signal
from fractions import Fraction as F
import exact_engine as ex
import interval_engine as ie
from test_engine import fixture,rot
from experiment import resource_audit,optimize,bounded_identity

class ExtensionTests(unittest.TestCase):
    def test_budget_handler_disarms_alarm(self):
        with mock.patch('exact_engine.certify',side_effect=RuntimeError('budget')):
            proof,reason=bounded_identity({})
        self.assertIsNone(proof)
        self.assertEqual(reason,'budget')
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)

    def test_half_coefficient_preserved(self):
        a=fixture([rot('Y',[[.5,0,0]])])
        self.assertEqual(ex.canonical_angles(a)[1][0],((0,0,F(1,2)),))
        self.assertEqual(ex.certify(a)[0]['r_upper'],1)

    def test_near_half_not_rounded(self):
        with self.assertRaises(ValueError):ex.certify(fixture([rot('Y',[[.50000000001,0,0]])]))

    def test_half_inverse_cancels(self):
        a=fixture([rot('Y',[[.5,0,0]]),rot('Y',[[-.5,0,0]])])
        self.assertEqual(ex.certify(a)[0]['inactive'],[0])

    def test_commutant_product_is_baseline(self):
        a=fixture([{'op':'H','wires':[0]},rot('Z',[[1,0,0]])]);a['ids']=[0]
        r=resource_audit(a,[0])
        self.assertEqual(r['metrics']['global_isometry']['trainable_angle_tasks'],1)
        self.assertEqual(r['metrics']['global_plus_commutant']['trainable_angle_tasks'],0)
        self.assertEqual(r['incremental']['trainable_angle_tasks'],0)

    def test_active_product_not_removed(self):
        a=fixture([rot('Y',[[1,0,0]])]);a['ids']=[0]
        b=optimize(a,True)
        self.assertEqual(b['ids'],[0])
        self.assertTrue(ie.nonzero(ie.derivative(b,['2/3'],['1/5','0'],['-2/5','0'])[1][0]))

if __name__=='__main__':unittest.main()
