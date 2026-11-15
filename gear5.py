"""Watch (GEAR) detector: the strap is the darkest thing ON the wrist, with skin on both sides of it.
A forearm shadow is only slightly darker than the skin; the strap is as dark as the water."""
import cv2
import numpy as np

H, W = 1078, 1232


def luma(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255


def candidates(img, skin, person, head, lo=70, hi=1300):
    g = luma(cv2.GaussianBlur(img, (0, 0), 0.8))
    ls = float(np.median(g[cv2.erode(skin.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0])) if skin.sum() > 500 else 0.45
    thr = min(0.27, 0.58 * ls)
    dark = (g < thr) & (cv2.dilate(person.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    dark &= ~(cv2.dilate(head.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    num, cc, st, cen = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    out = []
    k5 = np.ones((11, 11), np.uint8)
    for k in range(1, num):
        a = int(st[k, cv2.CC_STAT_AREA])
        if a < lo or a > hi:
            continue
        m = cc == k
        ring = (cv2.dilate(m.astype(np.uint8), k5) > 0) & ~m
        fs = float(skin[ring].mean()) if ring.sum() else 0
        w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        if fs < 0.38:
            continue
        # strap-like: not a thin sliver (forearm shadow edge) -> solidity of the blob
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        hull = cv2.contourArea(cv2.convexHull(cnt[0])) if cnt else 1
        sol = a / max(hull, 1)
        mean_g = float(g[m].mean())
        out.append(dict(mask=m, area=a, cen=(float(cen[k][0]), float(cen[k][1])), ring_skin=fs, sol=sol, g=mean_g, ls=ls))
    return out
