import json
import unittest
import numpy as np
from design import *

class Tests(unittest.TestCase):
    def st(self):
        return fg.Setting(json.loads((HERE/'frozen_inputs/yz_cx_encoding_circuit_n2_l1_local_xyz.json').read_text()))
    def test_metrics_spd_trace(self):
        for M in catalogue(8,42).values():
            self.assertAlmostEqual(np.trace(M),1)
            self.assertGreater(np.linalg.eigvalsh(M).min(),0)
            self.assertLessEqual(np.linalg.cond(M),4+1e-12)
    def test_dense_and_tangents(self):
        st=self.st(); p=probes(st,19,1,8)[0]
        self.assertLess(np.max(abs(dense_features(st,p['theta'],p['phi'])-p['f'])),1e-12)
        M=catalogue(len(st.obs),18)['random_dense_00']; J=jac(p,M)
        h=1e-5
        for k in range(st.K):
            v=np.eye(st.K)[k]*h
            vals=[]
            for th in (p['theta']+v,p['theta']-v):
                f=dense_features(st,th,p['phi']); d=f[:,:8]-f[:,8:]
                vals.append(np.einsum('op,oq,qp->p',d,M,d))
            self.assertLess(np.max(abs((vals[0]-vals[1])/(2*h*np.sqrt(8))-J[:,k])),1e-9)
    def test_pair_difference_translation(self):
        st=self.st(); p=probes(st,29,1,8)[0]
        old=jac(p,np.eye(len(st.obs)))
        q=dict(p); q['f']=p['f']+3; q['df']=p['df']+2
        q['delta']=q['f'][:,:8]-q['f'][:,8:];q['ddelta']=q['df'][:,:,:8]-q['df'][:,:,8:]
        self.assertTrue(np.allclose(old,jac(q,np.eye(len(st.obs)))))
    def test_basis_marginals(self):
        st=self.st();p=probes(st,39,1,4)[0];psi=dense_states(st,p['theta'],p['phi'])
        ints=np.arange(st.N)
        for b in cover(st.obs):
            pr=basis_probabilities(psi,b)
            for i,o in enumerate(st.obs):
                if compatible(o,b):
                    parity=sum((ints>>q)&1 for q,a in enumerate(o) if a!='I')
                    self.assertLess(np.max(abs(pr@(1-2*(parity%2))-p['f'][i])),1e-12)
    def test_cover(self):
        for n in (2,4):
            obs=words(n);self.assertEqual(len(measured(obs,cover(obs))),len(obs))
    def test_joint_covariance(self):
        # Bell state: ZI and IZ estimates are identical in every replicate.
        psi=np.array([[1,0,0,1]],complex)/np.sqrt(2)
        hs,used=estimates(psi,['ZI','IZ','ZZ'],['ZZ'],128,100,np.random.default_rng(9))
        for h in hs:
            self.assertTrue(np.array_equal(h[:,0],h[:,1]));self.assertTrue(np.all(h[:,2]==1))
        self.assertEqual(used,128)
    def test_unbiased_distance(self):
        psi=np.array([[1,0],[np.sqrt(.7),np.sqrt(.3)]],complex)
        hs,_=estimates(psi,['Z'],['Z'],128,5000,np.random.default_rng(12))
        d=unbiased_dist(hs,np.ones((1,1)))[:,0]
        self.assertLess(abs(d.mean()-.36),5*d.std()/np.sqrt(len(d)))
    def test_no_spurious_rank(self):
        self.assertEqual(rank_stat(np.zeros((30,4)))['rank'],0)
        self.assertEqual(rank_stat(np.eye(4))['rank'],4)

if __name__=='__main__': unittest.main()
