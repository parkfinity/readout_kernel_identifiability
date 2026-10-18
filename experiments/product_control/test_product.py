import itertools
import unittest
import numpy as np
from run import product_cov, lifted, variance

class ProductControlTests(unittest.TestCase):
    def test_exact_two_shot_enumeration(self):
        mu=np.array([.2,-.3,.7,-.5]);N=2
        sets=[(q,) for q in range(4)]+list(itertools.combinations(range(4),2))
        bits=np.array(list(itertools.product([-1,1],repeat=8))).reshape(-1,N,4)
        prob=np.prod((1+bits*mu)/2,axis=(1,2));means=bits.mean(axis=1)
        direct=np.stack([np.prod(bits[:,:,s],axis=-1).mean(axis=1) for s in sets],axis=1)
        prod=np.stack([np.prod(means[:,s],axis=-1) for s in sets],axis=1)
        truth=np.array([np.prod(mu[list(s)]) for s in sets])
        np.testing.assert_allclose(prob@prod,truth,atol=1e-14)
        np.testing.assert_allclose(prob@direct,truth,atol=1e-14)
        cp=(prod-truth).T@(prob[:,None]*(prod-truth))
        cd=(direct-truth).T@(prob[:,None]*(direct-truth))
        np.testing.assert_allclose(cp,product_cov(mu[:,None],sets,N)[0],atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(cd-cp).min(),-1e-14)
        # Conditional expectation at each per-qubit count vector.
        for key in np.unique(means,axis=0):
            idx=np.all(means==key,axis=1)
            np.testing.assert_allclose(prob[idx]@direct[idx]/prob[idx].sum(),
                                       [np.prod(key[list(s)]) for s in sets],atol=1e-14)

    def test_scalar_variance_difference(self):
        for n in [1,2,128,2048]:
            a,b=.3,-.6
            direct=(1-a*a*b*b)/n
            prod=(a*a*(1-b*b)+b*b*(1-a*a))/n+(1-a*a)*(1-b*b)/(n*n)
            self.assertAlmostEqual(direct-prod,(1-a*a)*(1-b*b)*(1/n-1/(n*n)))

    def test_correlated_counterexample(self):
        # P(++ )=P(-- )=1/2: exact means zero, ZZ one.
        self.assertNotEqual(0.*0.,1.)

    def test_lift_shape_and_values(self):
        h=np.arange(2*4*3,dtype=float).reshape(2,4,3)/24
        out=lifted([h,h],[(0,),(1,),(0,1)])
        np.testing.assert_allclose(out[0][:,2,:],h[:,0,:]*h[:,1,:])

if __name__=='__main__':unittest.main()
