import cv2,numpy as np,sys
def spotline(f):
    img=cv2.imread(f'.cache/frames_weever/{f:05d}.jpg',0).astype(np.float32)
    s=cv2.imread(f'masks/weever__fish/fine/{f:05d}.png',0)>127
    ys,xs=np.nonzero(s); y0,y1=ys.min(),ys.max(); L=y1-y0
    ra,rb=int(y1-0.55*L),int(y1-0.18*L)          # peduncle band
    er=cv2.erode(s.astype(np.uint8),np.ones((7,7),np.uint8))>0
    dark=(cv2.GaussianBlur(img,(0,0),1.5)<45)&er
    dark[:ra]=False; dark[rb:]=False
    n,lab,st,cen=cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    pts=[cen[i] for i in range(1,n) if st[i,4]>25]
    if len(pts)<2: return None
    P=np.array(pts); A=np.polyfit(P[:,1],P[:,0],1)   # x = a*y+b
    ext=[]
    for r in range(ra,rb):
        row=np.nonzero(s[r])[0]
        if len(row): ext.append(np.polyval(A,r)-row.min())
    return A,ra,rb,np.median(ext),len(pts)
if __name__=='__main__':
    for f in [int(x) for x in sys.argv[1:]]:
        r=spotline(f); print(f, None if r is None else 'left extension %.0f px (spots %d)'%(r[3],r[4]))
