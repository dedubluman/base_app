import numpy as np,cv2,sys,json
sys.path.insert(0,"/home/user/base_app");sys.path.insert(0,"/home/user/base_app/work")
from reef_grade import *
from scipy.optimize import least_squares
from skimage.color import rgb2lab
W,H=162,288
def load(k):
    f=lambda p:cv2.resize(cv2.imread(p),(W,H),interpolation=cv2.INTER_AREA)[:,:,::-1].astype(np.float32)/255
    return f(f"/home/user/base_app/work/src{k}.png"),f(f"/home/user/base_app/work/ref{k}.png")
data=[load(0),load(1)]
bl=lambda x:cv2.GaussianBlur(x,(0,0),2)
labs=[rgb2lab(bl(r).astype(np.float64)) for s,r in data]
# fit params: (name, index-in-tuple or None, lo, hi)
SP=[("wb",0,.9,1.6),("wb",1,.6,1.1),("wb",2,.7,1.3),("bp",None,0,.12),("wp",None,.6,1.0),("con",None,0,.8),("gam",None,.8,1.5),
    ("teal_amt",None,0,1),("teal_ab",0,-.07,0),("teal_ab",1,-.06,0),("warm_rot",None,0,.9),("warm_c",None,1,2.2),("warm_sig",None,20,70),
    ("split_amt",None,0,1),("split_ab",0,0,.08),("split_ab",1,0,.1),("mid_cool",None,0,1),("mid_ab",0,-.04,0),("mid_ab",1,-.04,0.0),
    ("sat",None,.8,1.5),("sand_desat",None,0,.9),("warm_c0",None,.005,.03)]
def build(v):
    P=dict(P_DEFAULT);P["clar_big"]=0;P["clar_mic"]=0
    for (n,i,lo,hi),x in zip(SP,v):
        if i is None:P[n]=float(x)
        else:
            t=list(P[n]);t[i]=float(x);P[n]=tuple(t)
    return P
def v0():
    out=[]
    for n,i,lo,hi in SP:out.append(P_DEFAULT[n] if i is None else P_DEFAULT[n][i])
    return np.array(out)
LO=np.array([s[2] for s in SP]);HI=np.array([s[3] for s in SP])
def resid(v,idx,reg=0.05):
    P=build(v);out=[]
    for i in idx:
        g=grade(data[i][0],P);d=(rgb2lab(bl(g).astype(np.float64))-labs[i])
        out.append((d*np.array([0.6,1,1])).ravel()[::5]/10)
    out.append(reg*(v-v0())/(HI-LO))
    return np.concatenate(out)
def dE(v,i):
    g=grade(data[i][0],build(v));return float(np.sqrt(((rgb2lab(bl(g).astype(np.float64))-labs[i])**2).sum(-1)).mean())
if __name__=="__main__":
    x0=v0();print("start dE",dE(x0,0),dE(x0,1))
    r=least_squares(resid,x0,args=([0,1],),bounds=(LO,HI),max_nfev=40,diff_step=2e-2,x_scale=(HI-LO))
    print("final dE",dE(r.x,0),dE(r.x,1))
    P=build(r.x);json.dump({k:(list(v) if isinstance(v,tuple) else v) for k,v in P.items()},open("/home/user/base_app/work/P_fit.json","w"),indent=1)
    for (n,i,_,_),x in zip(SP,r.x):print(n,i,round(float(x),3))
