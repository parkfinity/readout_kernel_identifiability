"""Small exact rewrite checker. Standard library only; no normalizer imports.

Checks the recorded IR rewrite, not the source adapter, theorem proof, numerical
training, or completeness. Paulis include signs; coefficients use decimal rationals.
"""
import argparse, copy, json, time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def multiply(a,b):
    phase=1;out=[]
    pairs={'XY':(1j,'Z'),'YX':(-1j,'Z'),'YZ':(1j,'X'),
           'ZY':(-1j,'X'),'ZX':(1j,'Y'),'XZ':(-1j,'Y')}
    for x,y in zip(a,b):
        if x=='I':out.append(y)
        elif y=='I':out.append(x)
        elif x==y:out.append('I')
        else:
            s,z=pairs[x+y];phase*=s;out.append(z)
    return phase,''.join(out)


def conjugate(word,g):
    n=len(word)
    if g['op']=='H':
        q=g['wires'][0];w=list(word);a=w[q];w[q]={'I':'I','X':'Z','Y':'Y','Z':'X'}[a]
        return (-1 if a=='Y' else 1),''.join(w)
    assert g['op']=='CX'
    c,t=g['wires'];phase=1;result='I'*n
    for q,a in enumerate(word):
        if a=='I':continue
        parts=['X','Z'] if a=='Y' else [a]
        if a=='Y':phase*=1j # Y = i X Z
        for axis in parts:
            w=['I']*n;w[q]=axis
            if axis=='X' and q==c:w[t]='X'
            if axis=='Z' and q==t:w[c]='Z'
            s,result=multiply(result,''.join(w));phase*=s
    assert phase in (-1,1)
    return int(phase.real),result


def commutes(a,b):return sum(x!='I' and y!='I' and x!=y for x,y in zip(a,b))%2==0


def collect(terms):
    d=defaultdict(Fraction)
    for c,t,x in terms:d[t,x]+=Fraction(str(c))
    assert all(Fraction(str(float(v)))==v for v in d.values())
    return [[float(v),t,x] for (t,x),v in sorted(d.items()) if v]


def check(arms,record):
    original=arms['compiler'];out=arms['global_unitary'];n=original['n']
    proof=record['normalization'];frame=[];r=[];logs=iter(proof['rewrites'])
    for i,g in enumerate(original['program']):
        if g['op'] in ('H','CX'):frame.append(g);continue
        log=next(logs);assert log['input_index']==i and log['frame']==frame and log['input_word']==g['pauli']
        word=g['pauli'];sign=1
        for h in reversed(frame):s,word=conjugate(word,h);sign*=s
        assert (word,sign)==(log['output_word'],log['sign'])
        terms=[[sign*c,t,x] for c,t,x in g['terms']];j=log['merge_target']
        if j is None:r.append({'op':'R','pauli':word,'terms':terms})
        else:
            assert 0<=j<len(r) and r[j]['pauli']==word
            assert all(commutes(word,h['pauli']) for h in r[j+1:])
            r[j]['terms']+=terms
    assert next(logs,None) is None
    for g in r:g['terms']=collect(g['terms'])
    r=[g for g in r if g['terms']]
    if proof['frame_identity_up_to_phase']:
        for q in range(n):
            for a in ('X','Z'):
                start='I'*q+a+'I'*(n-q-1);w=start;sign=1
                for h in frame:s,w=conjugate(w,h);sign*=s
                assert (sign,w)==(1,start)
        tail=[]
    else:tail=frame
    assert proof['trailing_frame']==tail and out['program']==r+tail
    for key in ('n','count','data_expressions','observables','weights'):assert original[key]==out[key]
    p=copy.deepcopy(out['program']);weights=dict(zip(out['observables'],out['weights']))
    assert len(weights)==len(out['observables']) and all(v>0 for v in weights.values())
    seen=set()
    for deletion in record['deletions']:
        i=deletion['rotation_index'];assert i not in seen;seen.add(i)
        g=p[i];assert g['op']=='R';word=g['pauli']
        for h in p[i+1:]:
            if h['op']=='R':assert commutes(word,h['pauli'])
            else:_,word=conjugate(word,h)
        assert word==deletion['output_word']
        for o,v in weights.items():
            if not commutes(word,o):
                _,partner=multiply(word,o);assert weights.get(partner)==v
        assert deletion['terms']==[t for t in g['terms'] if t[2]==-1]
        g['terms']=[t for t in g['terms'] if t[2]!=-1]
    p=[g for g in p if g['op']!='R' or g['terms']]
    assert p==arms['global_kernel']['program']
    for arm in ('global_unitary','global_kernel'):
        ids=sorted({t for g in arms[arm]['program'] if g['op']=='R' for c,t,x in g['terms'] if c and t>=0})
        assert arms[arm]['ids']==ids
    for key in ('n','count','data_expressions','observables','weights'):assert original[key]==arms['global_kernel'][key]
    return True


def audit(folder):
    start=time.perf_counter();count=0
    for p in sorted(folder.glob('*/programs.json')):
        check(json.loads(p.read_text()),json.loads((p.parent/'rewrite_record.json').read_text()));count+=1
    assert count==12
    return {'passed':True,'checked_settings':count,'checker_seconds':time.perf_counter()-start,
            'scope':'Exact supplied-IR normalization/deletion records; not source-adapter or theorem verification'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.folder),indent=2))
