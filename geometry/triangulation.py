'''
Visual Monocular SLAM Implementation
Created: Sept 10, 2019
Author: Michael Liu (GURU AI Group, UCSD)
'''

import numpy as np 
from .utils import *

def Triangulation(x1s, x2s, P1, P2, option='linear', verbose=False):
    assert x1s.shape[1] == x2s.shape[1]
    Xs = []
    for i in range(x1s.shape[1]):
        X = LinearTriangulation(x1s[:, i], x2s[:, i], P1, P2, verbose)
        Xs.append(X)
    Xs = np.hstack(Xs)
    assert Xs.shape[0] == 4
    return Xs

'''
Linear Triangulation (triangulation via orthogonal regression) MVP: Richard I. Harley
'''
def LinearTriangulation(x1, x2, P1, P2, verbose=False):
    '''
    input:
        - x1: inhomogeneous normalized point u, v in image 1
        - x2: inhomogeneous normalized point u, v in image 2
        - P1: projection matrix in image 1
        - P2: projection matrix in image 2
    '''
    '''
    DOUBLE CHECK: GOOD
    v1P1^3T - P1^2T
    u1P1^3T - P1^1T  == X, perform SVD to minimize algebraic error
    v2P2^3T - P2^2T
    u2P2^3T - P2^1T
    ''' 
    # x1 shape == (3, n)
    A = []
    x1, x2 = Dehomogenize(x1), Dehomogenize(x2)
    # start with one point
    A.append(x1[0] * P1[-1, :] - P1[0, :])
    A.append(x1[1] * P1[-1, :] - P1[1, :])
    A.append(x2[0] * P2[-1, :] - P2[0, :])
    A.append(x2[1] * P2[-1, :] - P2[1, :])
    A = np.vstack(A)
    # perform singular value decomposition
    U, d, Vt = np.linalg.svd(A)
    X = Vt[-1, :].T
    assert X.shape == (4, 1)
    # calculating the re-projection error
    x1_est, x2_est = P1 @ X, P2 @ X
    if verbose:
        print('reporjection error: {}'.format(np.sqrt(np.sum(Dehomogenize(x1_est) - x1.reshape(-1, 1)) ** 2 + np.sum(Dehomogenize(x2_est) - x2.reshape(-1,1)) ** 2)))
    return X
