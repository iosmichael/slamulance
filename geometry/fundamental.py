import numpy as np 
from .triangulation import SimpleTriangulation
from .utils import *

'''
Minimum Solver for Fundamental Matrix: currently using skimage.transform.FundamentalMatrixTransform
DLT for Fundamental Matrix: direct linear transform for fundamental matrix
'''
def DLT_F(x1, x2, normalize=True):
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

def DLT_E(x1, x2, K, normalize=True):
    # Inputs:
    #    x1 - inhomogeneous inlier correspondences in image 1
    #    x2 - inhomogeneous inlier correspondences in image 2
    #    normalize - if True, apply data normalization to x1 and x2
    #
    # Outputs:
    #    F - the DLT estimate of the essential matrix  
    
    # points normalization with calibration matrix
    x1, x2 = np.matrix(x1.T), np.matrix(x2.T)
    # shape == (2, n)
    x1, x2 = NormalizePoints(Homogenize(x1), K), NormalizePoints(Homogenize(x2), K)
    x1, x2 = Dehomogenize(x1), Dehomogenize(x2)
    # data normalization
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
    E = f.reshape(3, 3)
    u, d, vt = np.linalg.svd(E)
    d[2] = 0
    E = u @ np.diag(d) @ vt
    # data denormalization
    if normalize:
        E = T2.T @ E @ T1
    E = E / np.linalg.norm(E)
    return E

'''
P = [I | 0], P' = [M | e]
'''
def decompose_F(F):
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
Decompose Essential Matrix into four solutions
- inputs:
    - x1, x2 are one 2d correspondence
    - E is essential matrix that is ready to be decomposed
- ouput:
    - P1=[I|0], P2
'''
def decompose_E(E, x1, x2):
    U, d, Vt = np.linalg.svd(E)
    Z = np.matrix([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
    R1, R2 = U @ Z @ Vt, U @ Z.T @ Vt
    t1, t2 = U[:, -1], -U[:, -1]
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    # Canonical Camera Matrix
    P0 = np.hstack((np.eye(3), np.zeros((3,1))))
    # Four Possible Solutions
    P1 = np.concatenate((R1, t1), axis=1)
    P2 = np.concatenate((R1, t2), axis=1)
    P3 = np.concatenate((R2, t1), axis=1)
    P4 = np.concatenate((R2, t2), axis=1)
    for P in [P1, P2, P3, P4]:
        # testing chirality of Triangulation
        X = SimpleTriangulation(x1, x2, P0, P)
        P0_front = chirality(P0, X)
        P_front = chirality(P, X)
        if P0_front and P_front:
            return P0, P
        else:
            print('not good solution')
    print("could not find a valid decomposition of essential matrix")
    return P0, P1

'''
X is the reconstructed homogeneous 3D point
Chirality tests whether the reconstructed point is in front of the camera pose
'''
def chirality(P, X):
    assert X.shape == (4, 1)
    w = P[2, :] @ X
    return w * X[-1,0] * np.linalg.det(P[:, :3]) > 0
