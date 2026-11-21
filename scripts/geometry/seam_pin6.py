"""Pinned force C K = V^-1 G^T dpsi for a 6-point cp5-central face mass flux (snapy's weno5 with
shock:false reconstructs velocity with linear cp5 coefficients; central part = 0.5(L+R)).
(G^T dpsi)_c = sum_f w_fc (psi_R - psi_L)_f n_f l_f, with psi_R/psi_L at the two adjacent cells.
Panel-edge ghosts are filled by linear or cubic interpolation along the edge of the neighbour panel;
their weights are pushed back (transposed) onto the neighbour cells. Reports error of V^-1 G^T dpsi
against grad(psi) on solid-body rotation, normalised by max|grad psi| = Omega a."""
import numpy as np
from seam_pin import build

CP5 = {-2: 2/60, -1: -13/60, 0: 47/60, 1: 27/60, 2: -3/60}   # left state at i+1/2 from cells i-2..i+2
W6 = {}                                                       # central = 0.5(L+R) over offsets -2..3
for k,w in CP5.items():
    W6[k] = W6.get(k,0)+0.5*w          # left-biased, cells i+k
    W6[1-k] = W6.get(1-k,0)+0.5*w      # right-biased mirror about the face
TWO = {0: 0.5, 1: 0.5}

def analyse(g, ghost='linear', stencil='6', axis=np.array([0,0,1.0])):
    N=g['N']; cen=g['cen']; area=g['area']; pid=g['pid']; ij=g['ij']; P=g['P']; h=g['h']
    z=cen@axis; psi=1-z**2
    gexact=-2*z[:,None]*(axis[None,:]-z[:,None]*cen)
    W = W6 if stencil=='6' else TWO
    def cid(p,i,j): return (p*N+i)*N+j
    def pt(p,al,be):
        n,e1,e2=P[p]; v=n+np.tan(al)*e1+np.tan(be)*e2; return v/np.linalg.norm(v)
    cache={}
    def ghost_weights(p,i,j):
        key=(p,i,j)
        if key in cache: return cache[key]
        X=pt(p,-np.pi/4+(i+0.5)*h,-np.pi/4+(j+0.5)*h)
        for q,(n,e1,e2) in enumerate(P):
            d=X@n
            if d<=0: continue
            al=np.arctan(X@e1/d); be=np.arctan(X@e2/d)
            if abs(al)<=np.pi/4+1e-12 and abs(be)<=np.pi/4+1e-12:
                fi=(al+np.pi/4)/h-0.5; fj=(be+np.pi/4)/h-0.5
                # the ghost lies inside panel q at fractional index along the seam; the other index is the
                # row inside q (0 or N-1 for the first ghost layer, deeper for the second)
                if abs(fi-round(fi))>abs(fj-round(fj)): x=fi; row=int(round(fj)); along='i'
                else: x=fj; row=int(round(fi)); along='j'
                row=min(max(row,0),N-1)
                def cell(k):
                    k=min(max(k,0),N-1)
                    return cid(q,k,row) if along=='i' else cid(q,row,k)
                k0=int(np.floor(x)); t=x-k0
                if ghost=='linear': out={cell(k0):1-t, cell(k0+1):t}
                else:
                    ks=[k0-1,k0,k0+1,k0+2]; xs=np.array(ks,float); out={}
                    for l in range(4):
                        w=np.prod([(x-xs[m])/(xs[l]-xs[m]) for m in range(4) if m!=l])
                        out[cell(ks[l])]=out.get(cell(ks[l]),0)+w
                cache[key]=out; return out
        raise RuntimeError
    def weights(p,i,j):
        if 0<=i<N and 0<=j<N: return {cid(p,i,j):1.0}
        return ghost_weights(p,i,j)
    force=np.zeros_like(cen)
    for cL,cR,l,mid,n in g['faces']:
        p=pid[cL]; i,j=ij[cL]
        nn,e1,e2=P[p]; d=mid@nn; al=np.arctan(mid@e1/d); be=np.arctan(mid@e2/d)
        if abs(abs(al)-np.pi/4)<1e-9 and abs(be)<np.pi/4-1e-9: di,dj=(1 if al>0 else -1),0
        elif abs(abs(be)-np.pi/4)<1e-9: di,dj=0,(1 if be>0 else -1)
        elif pid[cR]==p and cR==cL+N: di,dj=1,0
        else: di,dj=0,1
        if di<0 or dj<0:   # express the face as the +face of the cell on its other side (in this panel)
            i,j,di,dj=i+di,j+dj,-di,-dj     # (i,j) is now a ghost index; stencil still panel-p based
        jump=(psi[cR]-psi[cL])            # n is outward from cL; closure uses psi at the adjacent cells
        for k,w in W.items():
            for c,wc in weights(p,i+k*di,j+k*dj).items():
                force[c]+= w*wc*jump*l*n/area[c]
    ref=np.linalg.norm(gexact,axis=1).max()   # max_c |grad psi_c| = Omega a
    err=np.linalg.norm(force-gexact,axis=1)/ref
    d=g['dist']
    return [err[d==0].max(), np.sqrt((err[d==0]**2).mean()), err[d==1].max(), err[d==2].max(), err[d>=3].max()]

if __name__=='__main__':
    print(f"{'N':>3} {'sten':>4} {'ghost':>6} {'row0max':>8} {'row0rms':>8} {'row1':>8} {'row2':>8} {'int':>8}")
    for N in [8,16,32,64]:
        g=build(N)
        for st,gh in (('2','linear'),('6','linear'),('6','cubic')):
            r=analyse(g,gh,st)
            print(f"{N:3d} {st:>4} {gh:>6} {r[0]:8.4f} {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f} {r[4]:8.4f}")
