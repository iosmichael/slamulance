import numpy as np 
from .triangulation import LinearTriangulation
from .utils import *

'''
Minimum Solver for Essential Matrix: currently using skimage.transform.EssentialMatrixTransform
DLT for Essential Matrix: direct linear transform for essential matrix
'''
def DLT_E(norm_x1, norm_x2, normalize=True):
    # Inputs:
    #    x1 - homogeneous normalized correspondences in image 1
    #    x2 - homogeneous normalized correspondences in image 2
    #    normalize - if True, apply data normalization to x1 and x2
    #
    # Outputs:
    #    E - the DLT estimate of the essential matrix  
    
    # points normalization with calibration matrix
    print("normalized points shape: {}".format(norm_x1.shape))
    assert norm_x1.shape[0] == 3
    x1, x2 = np.matrix(Dehomogenize(norm_x1)), np.matrix(Dehomogenize(norm_x2))
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
    f = RightNull(A)
    E = f.reshape(3, 3)
    u, d, vt = np.linalg.svd(E)
    d[2] = 0
    a, b = d[0], d[1]
    d[0] = d[1] = (a+b)/2
    E = u @ np.diag(d) @ vt
    # data denormalization
    if normalize:
        E = T2.T @ E @ T1
    E_norm = E / np.linalg.norm(E)
    return E_norm

'''
Decompose Essential Matrix into four solutions
- inputs:
    - x1, x2 are one 2d correspondence
    - E is essential matrix that is ready to be decomposed
- ouput:
    - P1=[I|0], P2
'''
def Decompose_Essential(E, x1, x2):
    '''
    X is the reconstructed homogeneous 3D point
    Cheirality (Richard I. Harley) tests whether the reconstructed point is in front of the camera pose
    '''
    def cheirality(P, X):
        assert X.shape == (4, 1)
        w = P[2, :] @ X
        return w * X[-1,0] * np.linalg.det(P[:, :3]) > 0

    U, d, Vt = np.linalg.svd(E)
    print("d: {}".format(d))
    Z = np.matrix([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
    R1, R2 = U @ Z @ Vt, U @ Z.T @ Vt
    print('R1: {}, R2: {}'.format(R1, R2))
    print('det(R1): {}, det(R2): {}'.format(np.linalg.det(R1), np.linalg.det(R2)))
    t1, t2 = U[:, -1], -U[:, -1]
    print('t1: {}, t2: {}'.format(t1, t2))
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    # Canonical Camera Matrix
    P0 = np.hstack((np.eye(3), np.zeros((3,1))))
    P_prime = None
    not_essential_count = 0
    # Four Possible Solutions
    P1 = np.concatenate((R1, t1), axis=1)
    P2 = np.concatenate((R1, t2), axis=1)
    P3 = np.concatenate((R2, t1), axis=1)
    P4 = np.concatenate((R2, t2), axis=1)
    for P in [P1, P2, P3, P4]:
        # testing chirality of Triangulation
        # X_geo = GeometricTriangulation(x1[:, 0], x2[:, 0], P0, P)
        X = LinearTriangulation(x1[:, 0], x2[:, 0], P0, P)
        print('Triangulation points shape: {}'.format(X.shape))
        if cheirality(P0, X) and cheirality(P, X):
            P_prime = P
        else:
            not_essential_count += 1
    assert not_essential_count == 3
    assert P_prime is not None
    return P0, P_prime

def Compose_Essential(P1, P2):
    R1, t1 = P1[:, :3], P1[:, -1]
    R2, t2 = P2[:, :3], P2[:, -1]
    R, t = R2 @ R1.T, t2.reshape(-1,1) - t1.reshape(-1,1)
    E = Skew(t.reshape(-1, 1)) @ R
    return E