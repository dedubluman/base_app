#!/usr/bin/env python3
"""Fit the reference-matched underwater grade from the aligned source/reference pairs.

Reads grade_data/reference/ref{1,2}.png (graded targets) and grade_data/reference/src_still_{a,b}.jpg (the matching
untouched stills). Writes grade_data/poly_coef.npy (RGB->RGB cubic polynomial, fitted on
background pixels only) and grade_data/lumacurve.npy (residual luma tone curve).
Re-run after swapping reference images to revise the look."""
import itertools
import cv2, numpy as np
W_DIR = "grade_data/reference"

def feats(X):
    cols = [np.ones(len(X))]
    for d in (1, 2, 3):
        for c in itertools.combinations_with_replacement(range(3), d):
            cols.append(np.prod([X[:, k] for k in c], axis=0))
    return np.stack(cols, 1)

def pair(i, s, diver_box, sigma):
    o = cv2.imread(f"{W_DIR}/src_still_{s}.jpg"); r = cv2.imread(f"{W_DIR}/ref{i}.png")
    r = cv2.resize(r, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_AREA)
    if i == 1:  # generative reference is slightly offset: align on gradient magnitude
        def g(im):
            x = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)
            return cv2.magnitude(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))
        w = np.eye(2, 3, dtype=np.float32)
        _, w = cv2.findTransformECC(g(o), g(r), w, cv2.MOTION_TRANSLATION,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        r = cv2.warpAffine(r, w, (o.shape[1], o.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
    o = cv2.cvtColor(o, cv2.COLOR_BGR2RGB) / 255.; r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB) / 255.
    o = cv2.GaussianBlur(o, (0, 0), sigma); r = cv2.GaussianBlur(r, (0, 0), sigma)
    H, W = o.shape[:2]; m = np.ones((H, W), bool)
    x0, y0, x1, y1 = diver_box; m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = False
    m[:int(0.02 * H)] = False; m[-int(0.02 * H):] = False
    return o[m], r[m]

if __name__ == "__main__":
    Xa, Ya = pair(2, "b", (0.24, 0.37, 0.57, 0.58), 2)
    Xb, Yb = pair(1, "a", (0.55, 0.50, 1.0, 0.78), 4)
    rng = np.random.default_rng(0)
    ia = rng.choice(len(Xa), 150000, replace=False); ib = rng.choice(len(Xb), 100000, replace=False)
    X = np.vstack([Xa[ia], Xb[ib]]); Y = np.vstack([Ya[ia], Yb[ib]])
    # ridge-regularised least squares keeps the cubic well-behaved outside the sampled gamut
    F = feats(X); lam = 1e-3
    coef = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]) * len(F) * 1e-3, F.T @ Y)
    np.save("grade_data/poly_coef.npy", coef)
    P = np.clip(F @ coef, 0, 1)
    print("fit mean abs err", np.abs(P - Y).mean(0).round(4))
    luma = lambda x: x @ np.array([.2126, .7152, .0722])
    q = np.linspace(0, 1, 129)
    xp = np.maximum.accumulate(np.quantile(luma(P), q) + np.arange(129) * 1e-7)
    np.save("grade_data/lumacurve.npy", np.stack([xp, np.quantile(luma(Y), q)]))
