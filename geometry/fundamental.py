import numpy as np 
'''
Minimum Solver for Fundamental Matrix: currently using skimage.transform.FundamentalMatrixTransform
DLT for Fundamental Matrix: direct linear transform for fundamental matrix
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

'''
P = [I | 0], P' = [M | e]
'''
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
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = np.hstack((M, e))
    assert P1.shape == (3, 4)
    assert P2.shape == (3, 4)
    return P1, P2

def get_F(P):
    e = P[:, -1]
    F = Skew(e) @ P[:, :3] 
    return F / np.linalg.norm(F)