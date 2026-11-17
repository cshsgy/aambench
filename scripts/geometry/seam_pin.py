"""Independent numpy reproduction (紫薇, 2026-11-16) of the cubed-sphere seam-row defect.
Pinned force on solid-body rotation under (a) zero work + (b) exact AAM closure against a
two-point central face mass flux:  C K = V^{-1} G^T dpsi  = Green-Gauss gradient of psi=Omega s^2
built on centre-to-centre links.  Compare with the exact grad psi (pointwise Coriolis force on
solid-body rotation).  Also measure the neighbour-centre offset across seams.
Grid: six-panel gnomonic equiangular, N cells per edge, unit sphere, Omega=1, omega=1."""
import numpy as np, sys

def panels():
    ex,ey,ez=np.eye(3)
    # (normal, e1, e2) right-handed so that e1 x e2 = normal
    return [( ex, ey, ez), ( ey,-ex, ez), (-ex,-ey, ez), (-ey, ex, ez), ( ez, ey,-ex), (-ez, ey, ex)]

def build(N):
    P=panels(); h=(np.pi/2)/N
    a=-np.pi/4+(np.arange(N)+0.5)*h          # centres
    ac=-np.pi/4+np.arange(N+1)*h             # corners
    ncell=6*N*N
    cen=np.zeros((ncell,3)); area=np.zeros(ncell)
    pid=np.zeros(ncell,int); ij=np.zeros((ncell,2),int)
    def pt(p,al,be):
        n,e1,e2=P[p]; v=n+np.tan(al)*e1+np.tan(be)*e2; return v/np.linalg.norm(v)
    def tri_area(A,B,C):
        num=np.dot(A,np.cross(B,C)); den=1+np.dot(A,B)+np.dot(B,C)+np.dot(C,A); return 2*np.arctan2(num,den)
    faces=[]  # (cL, cR, length, midpoint, n_LtoR)
    corner=np.zeros((6,N+1,N+1,3))
    for p in range(6):
        for i in range(N+1):
            for j in range(N+1): corner[p,i,j]=pt(p,ac[i],ac[j])
    for p in range(6):
        for i in range(N):
            for j in range(N):
                c=(p*N+i)*N+j; pid[c]=p; ij[c]=(i,j); cen[c]=pt(p,a[i],a[j])
                A,B,C,D=corner[p,i,j],corner[p,i+1,j],corner[p,i+1,j+1],corner[p,i,j+1]
                area[c]=tri_area(A,B,C)+tri_area(A,C,D)
    def cell_of(X):
        for q,(n,e1,e2) in enumerate(P):
            d=X@n
            if d<=0: continue
            al=np.arctan(X@e1/d); be=np.arctan(X@e2/d)
            if abs(al)<=np.pi/4+1e-12 and abs(be)<=np.pi/4+1e-12:
                i=min(max(int((al+np.pi/4)/h),0),N-1); j=min(max(int((be+np.pi/4)/h),0),N-1)
                return (q*N+i)*N+j
        raise RuntimeError
    def add_face(cL,P1,P2):
        mid=P1+P2; mid/=np.linalg.norm(mid)
        length=np.arccos(np.clip(P1@P2,-1,1))
        t=P2-P1; t-= (t@mid)*mid; t/=np.linalg.norm(t)
        n=np.cross(t,mid); n/=np.linalg.norm(n)
        if n@(cen[cL]-mid)>0: n=-n          # outward from cL
        return [cL,None,length,mid,n]
    seam_faces=[]
    for p in range(6):
        for i in range(N):
            for j in range(N):
                c=(p*N+i)*N+j
                # +i face
                f=add_face(c,corner[p,i+1,j],corner[p,i+1,j+1])
                if i<N-1: f[1]=c+N
                else: f[1]=cell_of(f[3]+1e-6*f[4]); seam_faces.append(len(faces))
                faces.append(f)
                # +j face
                f=add_face(c,corner[p,i,j+1],corner[p,i+1,j+1])
                if j<N-1: f[1]=c+1
                else: f[1]=cell_of(f[3]+1e-6*f[4]); seam_faces.append(len(faces))
                faces.append(f)
                if i==0:
                    f=add_face(c,corner[p,0,j],corner[p,0,j+1]); f[1]=cell_of(f[3]+1e-6*f[4]); faces.append(f); seam_faces.append(len(faces)-1)
                if j==0:
                    f=add_face(c,corner[p,i,0],corner[p,i+1,0]); f[1]=cell_of(f[3]+1e-6*f[4]); faces.append(f); seam_faces.append(len(faces)-1)
    # seam faces got added twice (once from each side) -> dedupe by midpoint
    keep=[];seen={}
    for k,f in enumerate(faces):
        key=tuple(np.round(f[3],9))
        if key in seen: continue
        seen[key]=k; keep.append(f)
    faces=keep
    dist=np.array([min(ij[c][0],ij[c][1],N-1-ij[c][0],N-1-ij[c][1]) for c in range(ncell)])
    return dict(N=N,h=h,cen=cen,area=area,pid=pid,ij=ij,faces=faces,dist=dist,P=P)

def analyse(g,axis=np.array([0,0,1.0])):
    cen=g['cen']; area=g['area']; N=g['N']
    z=cen@axis; s2=1-z**2            # s^2 on unit sphere
    psi=s2                           # Omega=1
    # exact grad psi (tangent): 2 s grad s ; grad s = -(z/s) * grad z_tangent ... simpler: grad(1-z^2) = -2 z grad z, grad z tangent = axis - z*cen
    gexact=-2*z[:,None]*(axis[None,:]-z[:,None]*cen)
    gGG=np.zeros_like(cen); gGG0=np.zeros_like(cen)
    for cL,cR,l,mid,n in g['faces']:
        d=0.5*l*(psi[cR]-psi[cL])
        gGG[cL]+=d*n/area[cL]; gGG[cR]+=d*n/area[cR]      # n is outward from L = inward to R; for R: 0.5 l (psi_L-psi_R)*(-n)=same
        gGG0[cL]+=0.5*l*(psi[cL]+psi[cR])*n/area[cL]; gGG0[cR]+=0.5*l*(psi[cL]+psi[cR])*(-n)/area[cR]
    ref=np.abs(gexact).max(axis=1).max()  # = Omega*a
    err=np.linalg.norm(gGG-gexact,axis=1)/ref
    err0=np.linalg.norm(gGG0-gexact,axis=1)/ref
    out={}
    for d in range(4):
        m=g['dist']==d
        if m.any(): out[d]=(err[m].max(), np.sqrt((err[m]**2).mean()), err0[m].max())
    m=g['dist']>=4
    out['int']=(err[m].max() if m.any() else np.nan, np.sqrt((err[m]**2).mean()) if m.any() else np.nan, err0[m].max() if m.any() else np.nan)
    return out,err

def seam_offsets(g):
    """neighbour centre across a seam: offset relative to the mirror of own centre across the face plane,
    and relative to the extension of own grid line (both in units of h)."""
    cen=g['cen']; h=g['h']; res=[]
    for cL,cR,l,mid,n in g['faces']:
        if g['pid'][cL]==g['pid'][cR]: continue
        for c,o in ((cL,cR),(cR,cL)):
            # mirror of c across the great-circle plane containing the face (plane normal = n x mid ... the face great circle plane normal m = t x? ) use: face plane normal q = cross(P? ) approximate by cross(mid, t) where t = tangent along face
            t=np.cross(mid,n); q=np.cross(mid,t); q/=np.linalg.norm(q)   # q: normal of great-circle plane of the face
            mirror=cen[c]-2*(cen[c]@q)*q
            # grid-line extension: continue the great circle from cen[c] through mid by the same arc length
            ang=np.arccos(np.clip(cen[c]@mid,-1,1))
            tt=mid-(mid@cen[c])*cen[c]; tt/=np.linalg.norm(tt)
            ext=np.cos(2*ang)*cen[c]+np.sin(2*ang)*tt
            # along-seam coordinate of the face midpoint in panel-local beta of cell c
            res.append((np.linalg.norm(cen[o]-mirror)/h, np.linalg.norm(cen[o]-ext)/h, mid))
    return res

if __name__=='__main__':
    print(f"{'N':>3} {'row0 max':>9} {'row0 rms':>9} {'row1 max':>9} {'row2 max':>9} {'row3 max':>9} {'int max':>9} | mirror-off(h) max  ext-off(h) max/mean")
    for N in [8,16,32,64]:
        g=build(N); out,err=analyse(g)
        so=seam_offsets(g)
        mo=max(r[0] for r in so); eo=[r[1] for r in so]
        r0,r1,r2,r3,ri=out[0],out.get(1,(np.nan,)*3),out.get(2,(np.nan,)*3),out.get(3,(np.nan,)*3),out['int']
        print(f"{N:3d} {r0[0]:9.4f} {r0[1]:9.4f} {r1[0]:9.4f} {r2[0]:9.4f} {r3[0]:9.4f} {ri[0]:9.4f} | {mo:10.2e}  {max(eo):8.4f} {np.mean(eo):8.4f}")
