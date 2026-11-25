import sys,cv2,numpy as np,json
sys.path.insert(0,"/home/user/base_app")
import importlib,reef_grade;importlib.reload(reef_grade)
from reef_grade import *
LS=0.4
def run(over,tag):
    P=dict(P_DEFAULT);P.update(over)
    rows=[]
    for k in (0,1):
        s=cv2.imread(f"/home/user/base_app/work/src{k}.png");r=cv2.imread(f"/home/user/base_app/work/ref{k}.png")
        x=s[:,:,::-1].astype(np.float32)/255;km=(0.43/frame_lmed(cv2.resize(x,(108,192),interpolation=cv2.INTER_AREA)))**LS;g=grade(x,P,lmul=km)
        g8=(g[:,:,::-1]*255).astype(np.uint8)
        t=lambda a:cv2.resize(a,(300,533),interpolation=cv2.INTER_AREA)
        rows.append(np.hstack([t(s),t(g8),t(r)]))
        for n,im in (("mine",g8),("ref",r)):
            x=im[:,:,::-1].reshape(-1,3)/255.
            print(k,n,"p5",np.percentile(x,5,0).round(2),"p50",np.percentile(x,50,0).round(2),"p95",np.percentile(x,95,0).round(2))
    cv2.imwrite(f"/tmp/claude-0/-home-user-base-app/0dd24014-fa8f-5e0a-bfc3-d451d46b78e2/scratchpad/{tag}.jpg",np.vstack(rows) if False else np.hstack(rows))
if __name__=="__main__":
    run(json.loads(sys.argv[1]) if len(sys.argv)>1 else {},"g1")
