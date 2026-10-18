import unittest
import numpy as np
from survey import readout,isometry_tail,make_info,angle_tasks,shared_gradient,maxerr
from structural_core import Gate,cx
from certificates import reduce_circuit
from engines import objective_gradient,kernel


class SurveyTests(unittest.TestCase):
    def test_readouts(self):
        self.assertEqual(len(readout(2,'pauli_weight_le_2')[0]),15)
        self.assertEqual(len(readout(4,'pauli_weight_le_2')[0]),66)

    def test_full_pauli_entangling_tail_is_baseline(self):
        obs,w=readout(2,'pauli_weight_le_2')
        c=(Gate('R','XX','param',0),Gate('R','ZZ','fixed',value=.4))
        tail,removed=isometry_tail(c,obs,w)
        self.assertEqual(removed,[0])

    def test_data_not_a_common_suffix(self):
        obs,w=readout(2,'pauli_weight_le_2')
        c=(Gate('R','XI','param',0),Gate('R','ZI','data',feature=0))
        self.assertEqual(isometry_tail(c,obs,w)[1],[])
        self.assertEqual(len(reduce_circuit(c,obs,w)[0]),2)

    def test_readout_changes_tail(self):
        c=(Gate('R','XI','param',0),)
        self.assertEqual(isometry_tail(c,*readout(2,'local_z'))[1],[])
        self.assertEqual(isometry_tail(c,*readout(2,'local_xyz'))[1],[0])

    def test_entangler_cannot_hide_initially_disjoint_data(self):
        c=(Gate('R','XI','param',0),cx(0,1),Gate('R','IZ','data',feature=0))
        self.assertEqual(isometry_tail(c,*readout(2,'pauli_weight_le_2'))[1],[])

    def test_shared_angle_chain_and_tied_occurrences(self):
        c=(Gate('R','XI','param',0),Gate('R','XI','product',1,feature=0),
           cx(0,1),Gate('R','YZ','param',0),Gate('R','ZI','data',feature=0))
        info=make_info(c,2,2,['a0'],*readout(2,'local_xyz'))
        E=np.array([[-.6],[-.1],[.3],[.7]]);y=np.array([1.,-1.,.3,.5]);t=np.array([.43,.67])
        a=shared_gradient(info,E,y,t);b=objective_gradient(info,E,y,t,'batched')
        self.assertLess(maxerr(a[1],b[1]),1e-10)
        self.assertEqual(a[2]['angle_derivatives'],2)
        self.assertEqual(a[2]['feature_sweeps'],5)


if __name__=='__main__':unittest.main()
