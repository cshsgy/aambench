"""Lane P check: structure and moments of the pinned-force defect e = V^-1 G^T dpsi - grad(psi) at the seams.

Each of the 12 physical seams gets a common edge coordinate u = (x_c - m) . t (m the mid-edge point, t the
unit tangent of the seam great circle, cross(n_p, n_q)) and the across-seam direction n_perp = t x x_c
(odd between the two sides: it points from panel p into panel q on one side and the reverse on the other,
so a couple shows up as the same sign of e.n_perp on both sides in this convention? no: n_perp is a fixed
direction in space, so a couple is the OPPOSITE sign on the two sides). Per stencil (2-point central;
6-point cp5-central with linear or cubic panel-edge ghosts), axis and N, the script reports
  row0 max   max over the row-0 cells of |e|                                   / (Omega a)
  Cn r0      max over the row-0 cells of |e . n_perp| (across the seam)        / (Omega a)
  Ct r0      max over the row-0 cells of |e . t|      (along the seam)         / (Omega a)
  even/odd   share of the row-0 across-seam profile e.n_perp(u) that is even in u on one side (0..1)
  couple     |sum over both sides of V_c e.n_perp sign| / sum |V_c e.n_perp|: 0 = exact couple
  |Z|, |F|   rms over the 12 seams of the zeroth (sum V_c e_c) and first (sum V_c e_c u_c) moments
             over all cells within 3 rows of the seam on both sides, / (Omega a)
Everything is per unit (Omega a); the solid-body state psi = Omega s^2 with a polar or a 30 deg tilted axis.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seam_pin import build
import seam_pin6 as s6

def defect(g, ghost, stencil, axis, only_seam=False, psi=None, gexact=None):
    """per-cell force V^-1 G^T dpsi (same construction as seam_pin6.analyse) and grad psi.
    only_seam: keep only the contributions of the panel-seam faces (the footprint check)."""
    N=g['N']; cen=g['cen']; area=g['area']; pid=g['pid']; ij=g['ij']; P=g['P']; h=g['h']
    if psi is None:
        z=cen@axis; psi=1-z**2
        gexact=-2*z[:,None]*(axis[None,:]-z[:,None]*cen)
    W = s6.W6 if stencil=='6' else s6.TWO
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
        if only_seam and pid[cR]==p: continue
        nn,e1,e2=P[p]; d=mid@nn; al=np.arctan(mid@e1/d); be=np.arctan(mid@e2/d)
        if abs(abs(al)-np.pi/4)<1e-9 and abs(be)<np.pi/4-1e-9: di,dj=(1 if al>0 else -1),0
        elif abs(abs(be)-np.pi/4)<1e-9: di,dj=0,(1 if be>0 else -1)
        elif pid[cR]==p and cR==cL+N: di,dj=1,0
        else: di,dj=0,1
        if di<0 or dj<0: i,j,di,dj=i+di,j+dj,-di,-dj
        jump=(psi[cR]-psi[cL])
        for k,w in W.items():
            for c,wc in weights(p,i+k*di,j+k*dj).items():
                force[c]+= w*wc*jump*l*n/area[c]
    return force, gexact


def linear_error(g, ghost, stencil, axis):
    """E_c = (V^-1 G^T d - grad)(a . x) for the three basis directions a, assembled with the same weights
    (mirror-pair jumps, ghost-interpolated deposits) as the defect: the stencil's error on linear functions"""
    cen=g['cen']; E=np.zeros((len(cen),3,3))
    for k in range(3):
        a=np.zeros(3); a[k]=1.0
        psi=cen@a; glin=a[None,:]-(cen@a)[:,None]*cen
        f,_=defect(g,ghost,stencil,axis,psi=psi,gexact=glin)
        E[:,:,k]=f-glin
    return E

def seams_of(g):
    """physical seams: (p,q) -> (mid-edge point m, unit tangent t); and cell -> (seam key, row) for rows <= 3"""
    N=g['N']; cen=g['cen']; pid=g['pid']; ij=g['ij']; P=g['P']
    mids={}
    for cL,cR,l,mid,n in g['faces']:
        p,q=pid[cL],pid[cR]
        if p!=q: mids.setdefault(tuple(sorted((p,q))),[]).append(mid)
    geo={}
    for key,ms in mids.items():
        m=np.mean(ms,axis=0); m/=np.linalg.norm(m); t=np.cross(P[key[0]][0],P[key[1]][0]); t/=np.linalg.norm(t); geo[key]=(m,t)
    edge={0:(-np.pi/4,0),1:(np.pi/4,0),2:(0,-np.pi/4),3:(0,np.pi/4)}
    cells={}
    for c in range(len(cen)):
        p=pid[c]; i,j=ij[c]; dist,side=min([(i,0),(N-1-i,1),(j,2),(N-1-j,3)])
        if dist>3: continue
        n0,e1,e2=P[p]; al,be=edge[side]; v=n0+np.tan(al)*e1+np.tan(be)*e2; v/=np.linalg.norm(v)
        key=min(geo, key=lambda k: np.linalg.norm(geo[k][0]-v)); cells[c]=(key,dist)
    return geo,cells

def table(g, force, gexact, ref, E=None):
    e=(force-gexact)/ref; area=g['area']; cen=g['cen']; pid=g['pid']
    rad=np.abs(np.einsum('ij,ij->i',e,cen)); d=g['dist']; rad0=rad[d==0].max(); radi=rad[d>=3].max()
    mech=(float('nan'),float('nan'),float('nan'))
    if E is not None:
        devn=np.linalg.norm(E.reshape(len(cen),-1),axis=1)
        elin=np.einsum('nab,nb->na',E,gexact)/ref
        res=np.linalg.norm(e-elin,axis=1)
        mech=(devn[d==0].max(), devn[d>=3].max(), res[d==0].max())
    geo,cells=seams_of(g)
    r0max=0.0; cn=0.0; ct=0.0; even=[]; couple=[]; Z=[]; F=[]
    for key,(m,t) in geo.items():
        zs=np.zeros(3); fs=np.zeros(3); prof={key[0]:[], key[1]:[]}; cpl_num=0.0; cpl_den=0.0
        for c,(k,dist) in cells.items():
            if k!=key: continue
            u=(cen[c]-m)@t; nperp=np.cross(t,cen[c]); nperp/=np.linalg.norm(nperp)
            zs+=area[c]*e[c]; fs+=area[c]*e[c]*u
            if dist==0:
                r0max=max(r0max,np.linalg.norm(e[c])); cn=max(cn,abs(e[c]@nperp)); ct=max(ct,abs(e[c]@t))
                prof[pid[c]].append((u,e[c]@nperp)); cpl_num+=area[c]*(e[c]@nperp); cpl_den+=area[c]*abs(e[c]@nperp)
        Z.append(np.linalg.norm(zs)); F.append(np.linalg.norm(fs)); couple.append(abs(cpl_num)/cpl_den if cpl_den>0 else 0.0)
        side=sorted(prof[key[0]]); u=np.array([x[0] for x in side]); v=np.array([x[1] for x in side])
        ve=0.5*(v+v[::-1]); vo=0.5*(v-v[::-1]); even.append((ve@ve)/((ve@ve)+(vo@vo)) if (ve@ve)+(vo@vo)>0 else 0.0)
    rms=lambda a: float(np.sqrt(np.mean(np.square(a))))
    return r0max, cn, ct, float(np.mean(even)), float(np.mean(couple)), rms(Z), rms(F), rad0, radi, mech

if __name__=='__main__':
    axes={'polar':np.array([0,0,1.0]), 'tilt30':np.array([0,-np.sin(np.pi/6),np.cos(np.pi/6)])}
    print("unit: max |grad psi| over the cells = Omega a (= 1 here); psi = Omega s^2 on the unit sphere; E = the stencil's error tensor on linear functions (a.x)")
    print(f"{'axis':>6} {'N':>3} {'sten':>4} {'ghost':>6} {'row0max':>8} {'Cn r0':>8} {'Ct r0':>8} {'even':>6} {'couple':>7} {'rms|Z|':>9} {'rms|F|':>9} {'rad r0':>7} {'rad int':>8} {'|E| r0':>9} {'|E| int':>10} {'|e-E g| r0':>14}")
    for name,axis in axes.items():
        for N in [8,16,32,64]:
            g=build(N); seamZ={}; seamF={}
            for st,gh in (('2','linear'),('6','linear'),('6','cubic')):
                force,gexact=defect(g,gh,st,axis); ref=np.linalg.norm(gexact,axis=1).max()
                E=linear_error(g,gh,st,axis); r=table(g,force,gexact,ref,E)
                m=r[9]
                print(f"{name:>6} {N:3d} {st:>4} {gh:>6} {r[0]:8.4f} {r[1]:8.4f} {r[2]:8.4f} {r[3]:6.3f} {r[4]:7.4f} {r[5]:9.5f} {r[6]:9.5f} {r[7]:7.4f} {r[8]:8.4f} {m[0]:9.4f} {m[1]:10.5f} {m[2]:14.4f}", flush=True)
                fs,_=defect(g,gh,st,axis,only_seam=True); V=g['area']; cen=g['cen']
                seamZ[(st,gh)]=(V[:,None]*fs).sum(0); seamF[(st,gh)]=np.einsum('n,na,nb->ab',V,fs,cen)
            z2=seamZ[('2','linear')]; f2=seamF[('2','linear')]
            for key in (('6','linear'),('6','cubic')):
                dz=np.abs(seamZ[key]-z2).max(); df=np.abs(seamF[key]-f2).max()
                print(f"{name:>6} {N:3d} seam-face footprint moments vs 2-point, {key[0]}-point {key[1]}: |dZ| {dz:.1e} (|Z| {np.abs(z2).max():.1e}), |dF| {df:.1e} (|F| {np.abs(f2).max():.1e})", flush=True)
