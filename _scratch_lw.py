import numpy as np

def lw(X):
    t,n = X.shape
    Xc = X - X.mean(0)
    S = Xc.T@Xc/t
    var = np.diag(S).reshape(-1,1); sd = var**0.5
    corr = S/np.outer(sd.ravel(),sd.ravel())
    rbar = (corr.sum()-n)/n/(n-1)
    F = rbar*np.outer(sd.ravel(),sd.ravel()); np.fill_diagonal(F,var.ravel())
    y = Xc**2
    phi_mat = (y.T@y)/t - S**2
    phi = phi_mat.sum()
    theta = ((Xc**3).T@Xc)/t - var*S
    np.fill_diagonal(theta,0)
    rho = np.diag(phi_mat).sum() + rbar*(np.outer(1/sd.ravel(), sd.ravel())*theta).sum()
    gamma = ((F-S)**2).sum()
    return float(np.clip((phi-rho)/(gamma*t),0,1)), rbar, S, F

rng=np.random.default_rng(1)
N=12;T=250
B=rng.normal(0,1,(N,3)); Fm=np.diag([0.6,0.3,0.15]); D=np.diag(rng.uniform(0.2,0.6,N))
Sigma=B@Fm@B.T+D; L=np.linalg.cholesky(Sigma)
orc=[];est=[];fs=[];fl=[]
for rep in range(400):
    X=rng.standard_normal((T,N))@L.T
    S=np.cov(X,rowvar=False,ddof=0)
    v=np.diag(S); s_=np.sqrt(v); c=S/np.outer(s_,s_); rb=(c.sum()-N)/(N*(N-1))
    Ft=rb*np.outer(s_,s_); np.fill_diagonal(Ft,v)
    d_orc=np.clip(((Sigma-S)*(Ft-S)).sum()/((Ft-S)**2).sum(),0,1)
    d,_,_,_=lw(X)
    orc.append(d_orc); est.append(d)
    fs.append(((S-Sigma)**2).sum()); fl.append((((1-d)*S+d*Ft-Sigma)**2).sum())
orc=np.array(orc);est=np.array(est)
print("oracle mean %.4f sd %.4f"%(orc.mean(),orc.std()))
print("est    mean %.4f sd %.4f corr %.3f mae %.4f"%(est.mean(),est.std(),np.corrcoef(est,orc)[0,1],np.abs(est-orc).mean()))
print("frob sample %.5f  lw %.5f"%(np.mean(fs),np.mean(fl)))
# second truth: exact constant correlation (delta should hit 1)
C=np.full((N,N),0.35); np.fill_diagonal(C,1.0); sd=rng.uniform(0.008,0.02,N)
Sig2=np.outer(sd,sd)*C; L2=np.linalg.cholesky(Sig2)
ds=[];fs2=[];fl2=[]
for rep in range(200):
    X=rng.standard_normal((T,N))@L2.T
    d,_,S,Ft=lw(X); ds.append(d)
    fs2.append(((S-Sig2)**2).sum()); fl2.append((((1-d)*S+d*Ft-Sig2)**2).sum())
print("const-corr case: est delta mean %.3f ; frob sample %.5f lw %.5f"%(np.mean(ds),np.mean(fs2),np.mean(fl2)))
