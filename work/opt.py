import numpy as np,cv2,sys
sys.path.insert(0,"work")
from grade_model import *
from scipy.optimize import least_squares
from skimage.color import rgb2lab
W,H=216,384
def load(k):
    f=lambda p:cv2.resize(cv2.imread(p),(W,H),interpolation=cv2.INTER_AREA)[:,:,::-1].astype(np.float32)/255
    return f(f"work/src{k}.png"),f(f"work/ref{k}.png")
data=[load(0),load(1)]
def lab(x):return rgb2lab(np.clip(x,0,1).astype(np.float64))
labs=[lab(r) for s,r in data]
def resid(p,idx,reg=0.02):
    out=[]
    for i in idx:
        g=grade(data[i][0],p.astype(np.float32),sigma=3)
        d=lab(g)-labs[i]
        out.append((d*np.array([0.7,1,1])).ravel()[::7]/10)
    out.append(reg*(p-P0)/(HI-LO)*10)
    return np.concatenate(out)
def run(idx,p0=P0):
    r=least_squares(resid,p0,args=(idx,),bounds=(LO,HI),max_nfev=60,diff_step=1e-2)
    return r.x
def dE(p,i):
    g=grade(data[i][0],p.astype(np.float32),sigma=3);return np.sqrt(((lab(g)-labs[i])**2).sum(-1)).mean()
if __name__=="__main__":
    for tr,te in (([0],1),([1],0)):
        p=run(tr);print("train",tr,"test",te,"dE train %.2f test %.2f"%(dE(p,tr[0]),dE(p,te)),"identity test %.2f"%dE(P0,te))
    p=run([0,1]);print("both dE",dE(p,0),dE(p,1))
    np.save("work/p_both.npy",p)
    for n,v in zip(NAMES,p):print(n,round(float(v),3),end=" | ")
