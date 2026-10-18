import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
import numpy as np
from normalizer import normalize, signed_word, coefficient_rank
from experiment import matrix, phase_error
from checker import conjugate, check
from normalizer import delete_common_isometries
import copy, itertools


class NormalizerTests(unittest.TestCase):
    def test_checker_signs_exhaustive(self):
        for n in (2,3):
            for word in map(''.join,itertools.product('IXYZ',repeat=n)):
                for g in [{'op':'H','wires':[0]},{'op':'CX','wires':[0,n-1]}]:
                    a,b=conjugate(word,g)
                    self.assertEqual((b,a),signed_word(word,[g],n,'s'))

    def test_checker_tamper_rejected(self):
        info={'n':1,'count':1,'data_expressions':[],'observables':['X','Y','Z'],'weights':[1,1,1],
              'program':[{'op':'H','wires':[0]},{'op':'R','pauli':'Y','terms':[[1.,0,-1]]}], 'ids':[0]}
        out,proof=normalize(info);pruned,d=delete_common_isometries(out)
        arms={'compiler':info,'global_unitary':out,'global_kernel':pruned}
        record={'normalization':proof,'deletions':d}
        self.assertTrue(check(arms,record))
        bad=copy.deepcopy(record);bad['normalization']['rewrites'][0]['sign']*=-1
        with self.assertRaises(AssertionError):check(arms,bad)
        badarms=copy.deepcopy(arms);badarms['global_kernel']['program'].append({'op':'H','wires':[0]})
        with self.assertRaises(AssertionError):check(badarms,record)

    def test_h_y_sign(self):
        self.assertEqual(signed_word('Y',[{'op':'H','wires':[0]}],1,'h'),('Y',-1))

    def test_noncommuting_barrier(self):
        p=[{'op':'R','pauli':w,'terms':[[1.,i,-1]]} for i,w in enumerate(['X','Z','X'])]
        a,_=normalize({'n':1,'count':3,'program':p})
        self.assertEqual(len(a['program']),3)

    def test_random_signed_unitaries(self):
        rng=np.random.RandomState(532)
        for n in (2,3,4):
            for repeat in range(30):
                p=[];count=0
                for j in range(20):
                    choice=rng.randint(3)
                    if choice==0:p.append({'op':'H','wires':[int(rng.randint(n))]})
                    elif choice==1:p.append({'op':'CX','wires':rng.choice(n,2,replace=False).tolist()})
                    else:
                        word=''.join(rng.choice(list('IXYZ'),n))
                        p.append({'op':'R','pauli':word,'terms':[[float(rng.choice([-1,1])),count,-1]]});count+=1
                info={'n':n,'count':count,'program':p}; b,_=normalize(info)
                theta=rng.normal(size=count);E=np.zeros((1,0))
                self.assertLess(phase_error(matrix(info,E,theta),matrix(b,E,theta)),1e-12)

    def test_nonadjacent_merge(self):
        p=[{'op':'R','pauli':w,'terms':[[1.,i,-1]]} for i,w in enumerate(['XI','IX','XI'])]
        out,_=normalize({'n':2,'count':3,'program':p})
        self.assertEqual(len(out['program']),2)


if __name__=='__main__':unittest.main()
