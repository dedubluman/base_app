import numpy as np,cv2
LUMA=np.array([0.2126,0.7152,0.0722],np.float32)
NAMES=(["m%d%d"%(i,j) for i in range(3) for j in range(3)]+["lift_r","lift_g","lift_b","gam_r","gam_g","gam_b","gain_r","gain_g","gain_b","con","sat","vib","lc","wk_r","wk_b","wk_t","wk_w"])
P0=np.array([1,0,0,0,1,0,0,0,1, 0,0,0, 1,1,1, 1,1,1, 0.0, 1.0, 0.0, 0.0, 0.0,0.0,0.9,0.08],np.float32)
LO=np.array([0.4,-.6,-.6,-.6,0.4,-.6,-.6,-.6,0.4, -.08,-.08,-.08, .5,.5,.5, .5,.5,.5, 0,.5,-1,0, -.2,-.3,.6,.02],np.float32)
HI=np.array([2.2,.6,.6,.6,2.2,.6,.6,.6,2.2, .12,.12,.12, 2,2,2, 2.5,2.5,2.5, 1,1.8,1.5,1.2, .8,.5,1.3,.4],np.float32)
def grade_color(x,p):
    """per-pixel part. x: ...x3 float32 gamma-encoded 0..1 -> same"""
    M=p[:9].reshape(3,3);lift=p[9:12];gam=p[12:15];gain=p[15:18];con,sat,vib=p[18:21]
    y=np.clip(x@M.T,0,1)
    wk_r,wk_b,wk_t,wk_w=p[22:26]
    ratio=y[...,0]/(y[...,1]+1e-3)
    ww=1/(1+np.exp(-(ratio-wk_t)/wk_w))
    y=np.stack([y[...,0]*(1+wk_r*ww),y[...,1],y[...,2]*(1-wk_b*ww)],-1)
    y=np.clip(y,0,1)
    y=np.clip(lift+(gain-lift)*np.power(y,gam),0,1)
    # S curve on luma-preserving blend
    if con>0:
        s=y*y*(3-2*y);y=y*(1-con)+s*con
    L=y@LUMA
    w=np.clip((y[...,0]-y[...,2])*3,0,1)[...,None]
    y=L[...,None]+(y-L[...,None])*(sat+vib*w)
    return np.clip(y,0,1)
def local_contrast(y,a,sigma):
    if a<=0:return y
    L=y@LUMA;B=cv2.GaussianBlur(L,(0,0),sigma)
    d=(L-B)[...,None]
    # amplify detail, keep chroma ratio
    return np.clip(y+a*d,0,1)
def grade(x,p,sigma=6):
    return local_contrast(grade_color(x,p),p[21],sigma)
