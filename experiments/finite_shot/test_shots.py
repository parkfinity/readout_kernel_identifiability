import unittest
from core import *

class Tests(unittest.TestCase):
    def test_canonical_sign(self):
        A=np.array([[1.,0.],[2.,1.],[0.,3.]])
        self.assertTrue(np.allclose(canonical_span(A),canonical_span(-A)))
        q=canonical_span(A);self.assertTrue(np.allclose(q.T@q,np.eye(2)))
    def test_bell_joint_covariance(self):
        psi=np.array([[1,0,0,1]],complex)/np.sqrt(2)
        c=outcome_covariance(psi,['ZI','IZ','ZZ'],['ZZ'])[0]
        self.assertTrue(np.allclose(c,[[1,1,0],[1,1,0],[0,0,0]]))
    def test_variance_monte_carlo(self):
        rng=np.random.default_rng(421)
        psi=rng.normal(size=(2,4))+1j*rng.normal(size=(2,4));psi/=np.linalg.norm(psi,axis=1)[:,None]
        obs=['ZI','IZ','ZZ'];bases=['ZZ'];M=np.array([[2.,.2,0],[.2,1.,.1],[0,.1,1.]])/4
        f=np.stack([np.real(np.sum(psi.conj()*(psi@matrix(w).T),axis=1)) for w in obs])
        delta=f[:,:1]-f[:,1:];c=outcome_covariance(psi,obs,bases)
        v,used=distance_variance(delta,c,M,256,bases)
        h,u=estimates(psi,obs,bases,256,12000,rng);s=unbiased_dist(h,M)[:,0]
        self.assertEqual(u,used)
        self.assertLess(abs(s.var(ddof=1)/v[0]-1),.06)
        truth=float(delta[:,0]@M@delta[:,0])
        self.assertLess(abs(s.mean()-truth),5*np.sqrt(v[0]/len(s)))
    def test_shot_accounting(self):
        st=setting('yz');p=probes(st,13,1,2)[0];psi=dense_states(st,p['theta'],p['phi'])
        _,used=estimates(psi,st.obs,cover(st.obs),4096,2,np.random.default_rng(5))
        self.assertEqual(used,4092)
    def test_same_joint_bits_z_zz(self):
        a=setting('hub_z');b=setting('hub_zz');p=probes(a,12,1,2)[0]
        psi=dense_states(a,p['theta'],p['phi'])
        ha,_=estimates(psi,a.obs,['ZZZZ'],256,16,np.random.default_rng(91))
        hb,_=estimates(psi,b.obs,['ZZZZ'],256,16,np.random.default_rng(91))
        for x,y in zip(ha,hb):self.assertTrue(np.array_equal(x,y[:,:4]))
    def test_metrics(self):
        for M in catalogue(10,1).values():
            e=np.linalg.eigvalsh(M);self.assertGreater(e.min(),0)
            self.assertLessEqual(e.max()/e.min(),4.0000001);self.assertAlmostEqual(np.trace(M),1)
    def test_full_target_not_just_old_null(self):
        st=setting('hub_z');p=probes(st,5,1,16)[0];Q,L=target_spaces(p)
        self.assertEqual(Q.shape,(8,4));self.assertEqual(L.shape,(8,2))
        self.assertTrue(np.allclose(Q.T@Q,np.eye(4)))
    def test_covariance_psd(self):
        st=setting('kyri');p=probes(st,7,1,2)[0];psi=dense_states(st,p['theta'],p['phi'])
        c=outcome_covariance(psi,st.obs,cover(st.obs))
        self.assertGreater(np.linalg.eigvalsh(c).min(),-1e-12)
    def test_dense_engine(self):
        for key in CASES:
            st=setting(key);p=probes(st,8,1,2)[0]
            self.assertTrue(np.allclose(p['f'],dense_features(st,p['theta'],p['phi']),atol=1e-12))
    def test_frozen_split_and_design(self):
        if not (HERE/'plan.json').exists():self.skipTest('Maintainer preparation not run')
        plan=load_plan()
        for panel in plan['panels']:
            for group in panel['groups'].values():
                for a in group['design']:
                    for b in group['test']:
                        self.assertFalse(np.allclose(a['theta'],b['theta']))
                for p in group['design']+group['test']:
                    Q=np.array(p['Q']);self.assertTrue(np.allclose(Q.T@Q,np.eye(Q.shape[1]),atol=1e-12))
                    for v in Q.T:self.assertGreater(v[np.argmax(abs(v))],0)
            for spec in panel['cases']:
                self.assertEqual(spec['arms']['rank_selected'],selector(spec['design_scores'],False))
                self.assertEqual(spec['arms']['shot_selected'],selector(spec['design_scores'],True))

if __name__=='__main__':unittest.main()
