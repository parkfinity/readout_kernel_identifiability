import unittest
import numpy as np
from structural_core import Gate
from pipeline import configurations,local_tail,boundary_transport,readout,prepare,program
from certificates import projected_kernel,reduce_circuit
from engines import objective_gradient,features,alignment,gram
from adapter import encode

class IncrementalTests(unittest.TestCase):
    def test_disjoint_local_tail(self):
        c=(Gate('R','XI','param',0),Gate('R','IZ','data',feature=0),Gate('R','YI','param',1))
        obs,w=readout(2);r,p=local_tail(c,obs,w)
        self.assertEqual(p['removed_indices'],[0,2]);self.assertEqual(len(r),1)

    def test_data_and_entangler_block(self):
        obs,w=readout(2)
        for suffix in [Gate('R','XI','data',feature=0),Gate('CX',wires=(0,1))]:
            c=(Gate('R','XI','param',0),suffix);self.assertEqual(local_tail(c,obs,w)[0],c)

    def test_local_tail_rejects_wrong_readout(self):
        for kind in ['z','anisotropic']:
            obs,w=readout(2,kind)
            with self.assertRaises(ValueError):local_tail((Gate('R','XI','param',0),),obs,w)

    def test_shared_parameter_retained(self):
        c=(Gate('R','X','product',0,feature=0),Gate('R','Y','param',0))
        obs,w=readout(1);r,_=local_tail(c,obs,w)
        self.assertEqual({g.parameter for g in r},{0})

    def test_zero_active_and_cliffords(self):
        c=(Gate('H',wires=(0,)),Gate('R','YI','data',feature=0),
           Gate('CX',wires=(0,1)),Gate('R','IY','param',0))
        obs,w=readout(2);red,_=local_tail(c,obs,w)
        rng=np.random.RandomState(2);X=rng.normal(size=(6,1));y=rng.normal(size=6);theta=np.array([.7])
        info={'program':program(red),'n':2,'ids':[],'count':1,'observables':obs,'weights':w}
        K=projected_kernel(c,X,theta,2,obs,w)[0];score,_=alignment(K,y)
        for engine in ['serial','batched','adjoint']:
            s,g=objective_gradient(info,X,y,theta,engine)
            self.assertAlmostEqual(s,score,places=12);np.testing.assert_array_equal(g,[0.])

    def test_compiler_precedes_both_methods(self):
        cfg=configurations()[1];a=prepare(cfg,'boundary');b=prepare(cfg,'transport')
        self.assertEqual(a['pre_certificate_gates'],b['pre_certificate_gates'])
        self.assertTrue(set(b['inactive_parameters'])>set(a['inactive_parameters']))
        self.assertEqual(a['local_proof'],b['local_proof'])
        self.assertEqual(a['inactive_parameters'],[7]);self.assertEqual(b['inactive_parameters'],[2,3,7])

    def test_boundary_does_not_cross_later_data(self):
        obs,w=readout(1);c=(Gate('R','X','param',0),Gate('R','X','data',feature=0))
        r,_=boundary_transport(c,obs,w,[{'native_gate':0},{'native_gate':1}]);self.assertEqual(r,c)
        r,_=boundary_transport(c,obs,w,[{'native_gate':0},{'native_gate':0}]);self.assertEqual(len(r),1)

    def test_fused_shared_product_and_fixed(self):
        info={'n':1,'count':2,'ids':[0,1],'observables':['X','Y','Z'],'weights':[1.,2.,3.],
            'program':[{'op':'H','wires':[0]},
             {'op':'R','pauli':'Z','terms':[[.8,0,0],[-.2,0,-1],[.4,1,-1],[.3,-1,-1]]},
             {'op':'R','pauli':'X','terms':[[1.,-1,0],[.2,1,-1]]}]}
        rng=np.random.RandomState(55);X=rng.normal(size=(7,1));y=rng.normal(size=7);t=np.array([.4,-.7])
        reference=objective_gradient(info,X,y,t,'serial')
        for engine in ['batched','adjoint']:
            s,g=objective_gradient(info,X,y,t,engine)
            np.testing.assert_allclose(g,reference[1],atol=1e-12,rtol=0)
            for j in range(2):
                plus=t.copy();minus=t.copy();plus[j]+=1e-6;minus[j]-=1e-6
                def value(v):return alignment(gram(features(info,X,v)[0],info['weights']),y)[0]
                self.assertAlmostEqual(g[j],(value(plus)-value(minus))/2e-6,places=7)

    def test_adjacent_rotation_refusion(self):
        c=(Gate('R','X','product',0,feature=0),Gate('R','X','param',1),Gate('H',wires=(0,)))
        p=program(c,fuse_adjacent=True)
        self.assertEqual(len(p),2);self.assertEqual(len(p[0]['terms']),2)

if __name__=='__main__':unittest.main()
