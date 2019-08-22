import numpy as np 
from utils import Skew

'''
Minimum Solver for Fundamental Matrix: currently using skimage.transform.FundamentalMatrixTransform
DLT for Fundamental Matrix: direct linear transform for fundamental matrix
Simple triangulation (midpoint method)
Optimal triangulation (with adjusted 2D points)
'''

def DLT(x1, x2, normalize=True):
    # Inputs:
    #    x1 - inhomogeneous inlier correspondences in image 1
    #    x2 - inhomogeneous inlier correspondences in image 2
    #    normalize - if True, apply data normalization to x1 and x2
    #
    # Outputs:
    #    F - the DLT estimate of the fundamental matrix  
    
    # data normalization
    x1, x2 = np.matrix(x1), np.matrix(x2)
    if normalize:
        x1, T1 = Normalize(x1)
        x2, T2 = Normalize(x2)
    else:
        x1 = Homogenize(x1)
        x2 = Homogenize(x2)
    A = np.zeros((0, 9))
    for i in range(x1.shape[1]):
        Ai = np.kron(x2[:, i].T, x1[:, i].T)
        A = np.vstack((A, Ai))
    u, d, vt = np.linalg.svd(A)
    f = vt[-1, :].T
    F = f.reshape(3, 3)
    u, d, vt = np.linalg.svd(F)
    d[2] = 0
    F = u @ np.diag(d) @ vt
    # data denormalization
    if normalize:
        F = T2.T @ F @ T1
    F = F / np.linalg.norm(F)
    return F


def get_P(F):
    U, D, Vt = np.linalg.svd(F)
    W = np.matrix([[0, 1, 0],
                   [-1, 0, 0],
                   [0, 0, 0]])
    Z = np.matrix([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
    s, t = np.asscalar(D[0]), np.asscalar(D[1])
    D = np.diag(np.array([s, t, (s + t) / 2]))
    M = U @ Z @ D @ Vt
    assert np.linalg.matrix_rank(M) == 3
    S = U @ W @ U.T
    e = np.matrix([[S.item((2, 1)), S.item((0, 2)), S.item(1, 0)]]).T
    assert M.shape == (3, 3)
    # P = [I | 0], P' = [M | e]
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = np.hstack((M, e))
    assert P1.shape == (3, 4)
    assert P2.shape == (3, 4)
    return P1, P2

def get_F(P):
    e = P[:, -1]
    F = Skew(e) @ P[:, :3] 
    return F / np.linalg.norm(F)

'''
optimal triangulation
'''
import sympy
from sympy import Symbol
from sympy.functions import re

def two_view_optimal_triangulation(x1, x2, F):
    assert x1.shape == (3, 1)
    assert x2.shape == (3, 1)
    T1, T2 = np.matrix(np.eye(3) * np.asscalar(x1[2, 0])), np.matrix(np.eye(3) * np.asscalar(x2[2, 0]))
    T1[:2, 2] = -x1[:2, 0]
    T2[:2, 2] = -x2[:2, 0]
    Fs = np.linalg.inv(T2).T @ F @ np.linalg.inv(T1)
    # calculate the epipole e and e_prime of Fs
    _, _, vt = np.linalg.svd(Fs)
    e1 = vt[-1, :].T
    e1 /= np.linalg.norm(e1[:2, 0])
    _, _, vt = np.linalg.svd(Fs.T)
    e2 = vt[-1, :].T
    e2 /= np.linalg.norm(e2[:2, 0])
    e1, e2 = np.array(e1), np.array(e2)
    assert np.allclose(e2[0] ** 2 + e2[1] ** 2, 1)
    assert np.allclose(e1[0] ** 2 + e1[1] ** 2, 1)
    R1 = np.matrix([[e1.item(0), e1.item(1), 0],
                    [-e1.item(1), e1.item(0), 0],
                    [0, 0, 1]])
    R2 = np.matrix([[e2.item(0), e2.item(1), 0],
                    [-e2.item(1), e2.item(0), 0],
                    [0, 0, 1]])
    Fs = R2 @ Fs @ R1.T
    a, b, c, d = np.asscalar(Fs[1, 1]), np.asscalar(Fs[1, 2]), np.asscalar(Fs[2, 1]), np.asscalar(Fs[2, 2])
    f, fp = e1.item(2), e2.item(2)
    t = Symbol('t')
    expression = t * ((a * t + b)**2 + fp**2 * (c * t + d)**2)**2 - (a * d - b * c) * (1 + f**2 * t**2)**2 * (a * t + b) * (c * t + d)
    coeffs = sympy.Poly(expression).all_coeffs()
    ts = np.roots(coeffs)
    assert len(ts) == 6
    min_t = np.inf
    min_cost = 1 / f**2 + c**2 / (a**2 + fp**2 * c**2)
    for sol in ts:
        sol = np.real(sol)
        cost = sol ** 2 / (1 + f**2 * sol**2) + (c * sol + d)**2 / ((a * sol + b)**2 + fp**2 * (c * sol + d)**2)
        if min_cost == 0 or cost < min_cost:
            min_cost = cost
            min_t = sol
    x1_correct = np.matrix([
        [f * min_t**2],
        [min_t],
        [min_t**2*f**2 + 1]
    ])
    x2_correct = np.matrix([
        [fp * (c * min_t + d)**2],
        [-(a * min_t + b) * (c * min_t + d)],
        [fp**2 * (c * min_t + d)**2 + (a * min_t + b)**2]
    ])
    if min_t == np.inf:
        x1_correct = np.matrix([
            [f, 0, f**2]
        ]).T
        x2_correct = np.matrix([
            [fp * c**2, a * c, fp**2 * c**2 + a**2]
        ]).T
    x1_correct = np.linalg.inv(T1) @ R1.T @ x1_correct
    x2_correct = np.linalg.inv(T2) @ R2.T @ x2_correct
    return x1_correct, x2_correct

def get_3D_pt(x1, x2, F, P2):
    x1_correct, x2_correct = two_view_optimal_triangulation(x1, x2, F)
    assert x1_correct.shape == (3, 1)
    assert x2_correct.shape == (3, 1)
#     _, P2 = get_P(F)
    l1 = np.array(F @ x1_correct)
    x2_correct = np.array(x2_correct)
    l_orth = np.matrix([[-l1.item(1) * x2_correct.item(2), 
                         l1.item(0) * x2_correct.item(2), 
                         l1.item(1) * x2_correct.item(0) - l1.item(0) * x2_correct.item(1)]]).T
    pi = P2.T @ l_orth
    assert pi.shape == (4, 1)
    n, d = pi[:3, 0], pi[-1, 0]
    X = np.vstack((d * x1_correct, -n.T @ x1_correct))
    assert X.shape == (4, 1)
    return X

def get_3D_pts(x1, x2, F, P):
    Xs = np.zeros((4, 0))
    for i in range(x1.shape[1]):
        Xs_i = get_3D_pt(x1[:, i], x2[:, i], F, P)
        Xs = np.hstack((Xs, Xs_i))
    assert Xs.shape == (4, x1.shape[1])
    return Xs