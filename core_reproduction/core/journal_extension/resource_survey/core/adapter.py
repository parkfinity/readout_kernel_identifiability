"""Symbolic Pauli rotation adapter with explicit nonlinear data feature slots."""
from dataclasses import asdict
import numpy as np
import sympy as sp
from qiskit import transpile
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import Statevector,Pauli
from structural_core import Gate,pauli,h,cx

class UnsupportedExpression(ValueError):pass

def adapt(qc,x,t):
    n=qc.num_qubits
    native=transpile(qc,optimization_level=0,basis_gates=['h','cx','rx','ry','rz','rxx','ryy','rzz','rzx'],seed_transpiler=71,num_processes=1)
    xs=list(sp.symbols('a0:2'));ts=list(sp.symbols('t0:'+str(len(t))))
    names={str(p):s for p,s in list(zip(x,xs))+list(zip(t,ts))}
    gates=[];expressions=[];provenance=[]
    for index,item in enumerate(native.data):
        op=item.operation;w=[native.find_bit(q).index for q in item.qubits]
        if op.name=='barrier':continue
        if op.name in ('h','cx'):
            gates.append(h(w[0]) if op.name=='h' else cx(*w));provenance.append({'native_gate':index,'operation':op.name});continue
        if op.name not in ['rx','ry','rz','rxx','ryy','rzz','rzx']:raise UnsupportedExpression('gate '+op.name)
        chars=['I']*n
        # Qiskit RZX acts with Z on its first wire and X on its second wire.
        # Our Pauli word is in ascending wire order, not Qiskit's display order.
        for q,axis in zip(w,op.name[1:].upper()):chars[q]=axis
        word=''.join(chars);raw=op.params[0]
        expr=sp.sympify(raw.sympify()) if isinstance(raw,ParameterExpression) else sp.Float(float(raw))
        if any(str(s) not in names for s in expr.free_symbols):raise UnsupportedExpression('Unknown symbol '+str(expr))
        expr=sp.expand(expr.xreplace({s:names[str(s)] for s in expr.free_symbols}))
        for term in sp.Add.make_args(expr):
            if term==0:continue
            parameters=[s for s in ts if s in term.free_symbols]
            if len(parameters)>1:raise UnsupportedExpression('Mixed parameter product '+str(term))
            if parameters:
                symbol=parameters[0];coefficient=sp.simplify(term/symbol)
                if coefficient.free_symbols.intersection(ts):raise UnsupportedExpression('Nonlinear parameter '+str(term))
                pi=ts.index(symbol)
                if coefficient.free_symbols:
                    if not coefficient.free_symbols.issubset(set(xs)):raise UnsupportedExpression(str(term))
                    expressions.append(coefficient);g=Gate('R',word,'product',pi,feature=len(expressions)-1)
                else:g=Gate('R',word,'param',pi,scale=float(coefficient))
            elif term.free_symbols:
                if not term.free_symbols.issubset(set(xs)):raise UnsupportedExpression(str(term))
                expressions.append(term);g=Gate('R',word,'data',feature=len(expressions)-1)
            else:g=Gate('R',word,'fixed',value=float(term))
            gates.append(g);provenance.append({'native_gate':index,'angle_term':str(term)})
    serialized=[str(e) for e in expressions]
    return tuple(gates),serialized,{'native_operations':len(native.data),'gates':[asdict(g) for g in gates],
                                    'gate_provenance':provenance,'data_expressions':serialized,
                                    'global_phase_ignored':str(native.global_phase)}

def encode(expressions,X):
    symbols=list(sp.symbols('a0:2'));columns=[]
    for expression in expressions:
        function=sp.lambdify(symbols,sp.sympify(expression,locals={str(s):s for s in symbols}),'numpy')
        values=np.asarray(function(X[:,0],X[:,1]),dtype=float)
        columns.append(np.broadcast_to(values,(len(X),)))
    out=np.column_stack(columns) if columns else np.zeros((len(X),0))
    if not np.isfinite(out).all():raise ValueError('Nonfinite encoded data')
    return out

def source_states(obj,X,theta):
    return np.stack([Statevector.from_instruction(obj.get_circuit(x,theta)).data for x in X])

def source_features(states,observables):
    return np.array([[float(Statevector(s).expectation_value(Pauli(o[::-1])).real) for o in observables] for s in states])

def gram(features,weights):
    return np.exp(-.5*np.sum(weights*(features[:,None,:]-features[None,:,:])**2,axis=-1))
