"""Pauli-level (symplectic) analysis of the HEA even-then-odd CNOT entangler E.
Verifies: dim S_E (gauge), which single-qubit Paulis stay weight-1 under Ad_E
(=> N_q diagonal eigenvalues), and light-cone supports of pulled-back readout."""
import numpy as np

def cnot_pairs(n):
    pairs=[]
    for start in (0,1):
        for i in range(start,n-1,2):
            pairs.append((i,i+1))
    return pairs  # order = chronological (even layer then odd layer)

def adE(xz, pairs):
    """Ad_E(P)=E P E^dag. Apply conj rules in gate order g1..gL."""
    x,z = xz[0].copy(), xz[1].copy()
    for (c,t) in pairs:
        x[t]^=x[c]; z[c]^=z[t]
    return (x,z)

def adEdag(xz, pairs):
    """Ad_{E^dag}(P)=E^dag P E. Apply in reverse gate order."""
    x,z = xz[0].copy(), xz[1].copy()
    for (c,t) in reversed(pairs):
        x[t]^=x[c]; z[c]^=z[t]
    return (x,z)

def weight(xz):
    x,z=xz
    return int(np.sum((x|z)!=0))

def supp(xz):
    x,z=xz
    return tuple(i for i in range(len(x)) if (x[i]|z[i]))

def single(n,q,P):
    x=np.zeros(n,dtype=int); z=np.zeros(n,dtype=int)
    if P in ('X','Y'): x[q]=1
    if P in ('Z','Y'): z[q]=1
    return (x,z)

print("=== dim S_E : single-qubit Paulis staying weight-1 under Ad_E ===")
for n in range(2,13):
    pairs=cnot_pairs(n)
    keep=[]
    for q in range(n):
        for P in 'XYZ':
            if weight(adE(single(n,q,P),pairs))==1:
                keep.append(f"{P}{q}")
    print(f"n={n:2d}  dim S_E={len(keep):2d}  generators={keep}  pairs={pairs}")

print("\n=== N_q eigenvalue pattern: [X,Y,Z weight-1 after Ad_E?] per qubit (n=6) ===")
n=6; pairs=cnot_pairs(n)
for q in range(n):
    ind=[1 if weight(adE(single(n,q,P),pairs))==1 else 0 for P in 'XYZ']
    print(f"  qubit {q}: (aX,aY,aZ)={tuple(ind)}  (weights X,Y,Z = "
          +str([weight(adE(single(n,q,P),pairs)) for P in 'XYZ'])+")")

print("\n=== light-cone supports of pulled-back readout P_i = E^dag O_i E (n=6) ===")
for q in range(n):
    row=[]
    for P in 'XYZ':
        s=supp(adEdag(single(n,q,P),pairs))
        row.append(f"{P}{q}->{s}")
    print("  "+ "  ".join(row))

print("\n=== forward supports supp(Ad_E(single)) (n=6), for comparison ===")
for q in range(n):
    row=[f"{P}{q}->{supp(adE(single(n,q,P),pairs))}" for P in 'XYZ']
    print("  "+"  ".join(row))
