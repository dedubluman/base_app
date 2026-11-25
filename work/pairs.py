import cv2,numpy as np
exec(open("work/fit.py").read().split("X=[];Y=[]")[0])
out={}
for k,(s,r) in enumerate(zip(srcs,refs)):
    rw=prep(s,r)
    cv2.imwrite(f"work/src{k}.png",s);cv2.imwrite(f"work/ref{k}.png",rw)
