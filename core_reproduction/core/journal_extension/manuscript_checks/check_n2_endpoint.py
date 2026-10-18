"""Exact rational Jacobian checks for the n=2 shared-endpoint exception.

Standard library only. The rational coordinates t=tan(theta/2) have
nonzero dtheta/dt, so their Jacobian has the same rank as the angle Jacobian.
Based on the archived local-block calculation, made self-contained here.
This checks a lower-bound witness, not the universal gauge upper-bound proof.
"""
from fractions import Fraction as F
from itertools import product
import json

def mm(a, b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]

def rotation(t, axis, derivative=False):
    d=1+t*t
    s,c=2*t/d,(1-t*t)/d
    if derivative:
        s,c=2*(1-t*t)/(d*d),-4*t/(d*d)
    one=0 if derivative else 1
    if axis=='Z': return [[c,-s,0],[s,c,0],[0,0,one]]
    return [[one,0,0],[0,c,-s],[0,s,c]]

def euler(t):
    a,b,c=t
    A,B,C=rotation(a,'Z'),rotation(b,'X'),rotation(c,'Z')
    return mm(C,mm(B,A)), [mm(C,mm(B,rotation(a,'Z',True))),
        mm(C,mm(rotation(b,'X',True),A)),mm(rotation(c,'Z',True),mm(B,A))]

def rank_q(rows):
    a=[[F(x) for x in row] for row in rows]
    r=0
    for c in range(len(a[0])):
        p=next((i for i in range(r,len(a)) if a[i][c]),None)
        if p is None: continue
        a[r],a[p]=a[p],a[r]
        pivot=a[r][c]
        a[r]=[v/pivot for v in a[r]]
        for i in range(r+1,len(a)):
            fac=a[i][c]
            a[i]=[u-fac*v for u,v in zip(a[i],a[r])]
        r+=1
        if r==len(a): break
    return r

def jacobian(features, point):
    r0,d0=euler(point[:3]); r1,d1=euler(point[3:])
    modes=list(product((0,2),repeat=2)); axes={'X':0,'Y':1,'Z':2}
    def coeff(a,b):
        return [[a[axes[f[0]]][i]*b[axes[f[1]]][j] for i,j in modes] for f in features]
    C=coeff(r0,r1)
    DC=[coeff(d,r1) for d in d0]+[coeff(r0,d) for d in d1]
    return [[sum(d[f][i]*C[f][j]+C[f][i]*d[f][j] for f in range(len(features)))
             for d in DC] for i in range(4) for j in range(i,4)]

def check():
    point=[F(1,3),F(2,5),F(3,7),F(1,2),F(4,9),F(2,7)]
    cases=[(['XX','YX','ZY','ZZ'],4),(['XX','YX','ZZ'],5),(['XX','ZY','ZZ'],5)]
    result=[]
    for features,expected in cases:
        j=jacobian(features,point); r=rank_q(j)
        assert r==expected,(features,r,expected)
        result.append({'features':features,'rank':r,'expected':expected,
                       'jacobian':[[str(v) for v in row] for row in j]})
    return {'passed':True,'coordinate':'t=tan(theta/2)',
            'point':[str(v) for v in point],'cases':result,
            'scope':'Exact pointwise lower witnesses; analytic endpoint gauges provide upper bounds.'}

if __name__=='__main__':
    print(json.dumps(check(),indent=2))
