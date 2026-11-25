import cv2,numpy as np,itertools,sys
U="/root/.claude/uploads/0dd24014-fa8f-5e0a-bfc3-d451d46b78e2/"
refs=[cv2.imread(U+"2224de72-image.png"),cv2.imread(U+"98f3fca1-image.png")]
srcs=[cv2.imread(U+"0fe1ee7c-image.jpg"),cv2.imread(U+"1e1a7b3e-image.jpg")]
def prep(s,r):
    rr=cv2.resize(r,(s.shape[1],s.shape[0]),interpolation=cv2.INTER_AREA)
    g=lambda x:cv2.GaussianBlur(cv2.cvtColor(cv2.resize(x,None,fx=.25,fy=.25),cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),1.5)
    a,b=g(s),g(rr);a=(a-a.mean())/a.std();b=(b-b.mean())/b.std()
    W=np.eye(2,3,dtype=np.float32)
    cc,W=cv2.findTransformECC(a,b,W,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-6),None,5)
    W[:,2]*=4
    rw=cv2.warpAffine(rr,W,(s.shape[1],s.shape[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
    return rw
X=[];Y=[]
for s,r in zip(srcs,refs):
    rw=prep(s,r)
    # heavy blur at low-res => compare colour not detail
    f=lambda x:cv2.GaussianBlur(cv2.resize(x,(270,480),interpolation=cv2.INTER_AREA),(0,0),2)[:,:,::-1].reshape(-1,3)/255.
    X.append(f(s));Y.append(f(rw))
X=np.vstack(X);Y=np.vstack(Y)
np.save("work/X.npy",X);np.save("work/Y.npy",Y)
print("mean src",X.mean(0),"mean ref",Y.mean(0))
print("p1/p50/p99 src",np.percentile(X,[1,50,99],0).round(3).tolist())
print("p1/p50/p99 ref",np.percentile(Y,[1,50,99],0).round(3).tolist())
