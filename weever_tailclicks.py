import cv2,numpy as np,json,sys
sys.path.insert(0, '.')
from weever_spotline import spotline
clicks=[]
for f in range(463,481):
    A,ra,rb,ext,_=spotline(f)
    s=cv2.imread(f'masks/weever__fish/fine_v2/{f:05d}.png',0)>127
    dt=cv2.distanceTransform(s.astype(np.uint8),cv2.DIST_L2,5)
    neg=[]
    for r in range(ra,rb,4):
        row=np.nonzero(s[r])[0]
        sx=np.polyval(A,r)
        if len(row) and sx-row.min()>=44:
            x=int(round(sx-40))
            if dt[r,x]>=4: neg.append((x,r))
    if neg:
        for q in (0.2,0.5,0.8):
            x,r=neg[int(q*(len(neg)-1))]; clicks.append([f,float(x),float(r),0])
    for r in np.linspace(ra+10,rb-10,3):
        clicks.append([f,float(np.polyval(A,r)+10),float(r),1])
    ys,xs=np.nonzero(s); y0,y1=ys.min(),ys.max()
    for b in range(8):
        a_,b_=int(y0+(y1-y0)*b/8),int(y0+(y1-y0)*(b+1)/8)
        band=dt[a_:b_]; yy,xx=np.unravel_index(np.argmax(band),band.shape)
        clicks.append([f,float(xx),float(yy+a_),1])
    g=cv2.GaussianBlur(cv2.imread(f'.cache/frames_weever/{f:05d}.jpg',0).astype(np.float32),(0,0),2)
    up=(dt>=3)&(g>0.85*np.median(g[s]))&(np.arange(s.shape[0])[:,None]<y0+0.35*(y1-y0))
    uy,ux=np.nonzero(up); mid=np.median(ux)
    for i in (np.argmin(ux),np.argmax(ux)):
        if abs(ux[i]-mid)>25: clicks.append([f,float(ux[i]),float(uy[i]),1])
json.dump(clicks,open('output/clicks_tail.json','w'))
print(len(clicks),[(c[0],c[3]) for c in clicks if c[3]==0])
# preview
tiles=[]
for f in range(463,481):
    img=cv2.imread(f'.cache/frames_weever/{f:05d}.jpg'); s=cv2.imread(f'masks/weever__fish/fine_v2/{f:05d}.png',0)>127
    cn,_=cv2.findContours(s.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(img,cn,-1,(0,0,255),1)
    for _,x,y,l in [c for c in clicks if c[0]==f]: cv2.circle(img,(int(x),int(y)),4,(0,255,0) if l else (255,0,255),-1)
    ys,xs=np.nonzero(s); t=img[ys.max()-330:ys.max()+20,xs.min()-20:xs.min()+280]
    tiles.append(cv2.resize(t,None,fx=1.6,fy=1.6))
cv2.imwrite('output/qc/tailclicks.jpg',np.hstack(tiles))
