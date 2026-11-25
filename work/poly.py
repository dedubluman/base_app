import numpy as np,cv2
from skimage.color import rgb2lab,deltaE_ciede2000
X=np.load("work/X.npy");Y=np.load("work/Y.npy")
def feats(x,deg=3):
    r,g,b=x[:,0],x[:,1],x[:,2];cols=[np.ones_like(r)]
    for d in range(1,deg+1):
        for i in range(d+1):
            for j in range(d+1-i):
                k=d-i-j;cols.append(r**i*g**j*b**k)
    return np.stack(cols,1)
def fit(deg,lam):
    F=feats(X,deg);A=F.T@F+lam*np.eye(F.shape[1]);return np.linalg.solve(A,F.T@Y)
if __name__=="__main__":
    idx=np.random.RandomState(0).permutation(len(X));tr,te=idx[:len(X)//2],idx[len(X)//2:]
    for deg in (2,3,4):
        for lam in (1e-4,1e-3,1e-2):
            F=feats(X[tr],deg);W=np.linalg.solve(F.T@F+lam*np.eye(F.shape[1]),F.T@Y[tr])
            p=np.clip(feats(X[te],deg)@W,0,1)
            de=deltaE_ciede2000(rgb2lab(p[None]),rgb2lab(Y[te][None]))[0]
            print(deg,lam,"dE mean %.2f p90 %.2f"%(de.mean(),np.percentile(de,90)))
    de0=deltaE_ciede2000(rgb2lab(X[None]),rgb2lab(Y[None]))[0];print("identity dE",de0.mean())
