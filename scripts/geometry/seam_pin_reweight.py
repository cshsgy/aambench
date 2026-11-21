"""Test the 'compact re-weighting' claim: replace the 0.5/0.5 face interpolation at seam faces by a
linear-exact least-squares interpolation over a wider compact stencil (own cell, mirror neighbour,
and the along-seam neighbours of both) and recompute the pinned force C K = V^-1 G^T dpsi."""
import numpy as np
from seam_pin import build, analyse
def analyse_w(g, stencil):
    cen=g['cen']; area=g['area']; N=g['N']; pid=g['pid']
    z=cen[:,2]; psi=1-z**2
    gexact=-2*z[:,None]*(np.array([0,0,1.0])[None,:]-z[:,None]*cen)
    # neighbour table (shared faces)
    nb={c:[] for c in range(len(cen))}
    for cL,cR,l,mid,n in g['faces']: nb[cL].append(cR); nb[cR].append(cL)
    force=np.zeros_like(cen)
    for cL,cR,l,mid,n in g['faces']:
        seam = pid[cL]!=pid[cR]
        if stencil=='2pt' or not seam:
            S=[cL,cR]; w=np.array([0.5,0.5])
        else:
            S=sorted(set([cL,cR]+nb[cL]+nb[cR]))
            # tangent-plane coords at face midpoint
            t1=np.cross(mid,n); t1/=np.linalg.norm(t1); t2=n
            X=np.array([[1,(cen[c]-mid)@t1,(cen[c]-mid)@t2] for c in S])   # rows: cells
            # minimum-norm weights with constraints X^T w = (1,0,0)
            w=X@np.linalg.solve(X.T@X, np.array([1.0,0,0]))
        for c,wc in zip(S,w):
            force[c]+=wc*l*(psi[cR]-psi[cL])*n/area[c]
    ref=np.linalg.norm(gexact,axis=1).max()   # max_c |grad psi_c| = Omega a
    err=np.linalg.norm(force-gexact,axis=1)/ref
    d=g['dist']; return [err[d==k].max() for k in range(3)]+[err[d>=3].max()]
print("  N stencil  [row0 max, row1 max, row2 max, interior max] / (Omega a); polar axis, psi = Omega s^2")
for N in [8,16,32]:
    g=build(N)
    print(N,'2pt     ',np.round(analyse_w(g,'2pt'),4))
    print(N,'LSQ-lin ',np.round(analyse_w(g,'lsq'),4))
