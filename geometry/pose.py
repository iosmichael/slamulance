import numpy as np 
from utils import Homogenize, Dehomogenize, Normalize

'''
3D to 2D absolute camera pose estimation
EPnP: efficient PnP pose estimation
'''

def control_points_world(X):
    '''
    input:
        - inhomogeneous 3D points, 3 x n
    output:
        - four control points, C_1, C_2, C_3, C_4
    '''
    cov = np.cov(X)
    # assert covariance matrix of X is (3x3)
    assert cov.shape == (3, 3)
    _, d, v_t = np.linalg.svd(cov)
    s = np.sqrt(np.sum(d) / 3)
    # return s and V
    return np.asscalar(s), v_t.T

def control_points_cam(I):
    # find the right null space of I
    u, d, v = np.linalg.svd(I)
    return np.matrix(v.T)[:, -1]
    
def parameterize_3D_world(X, s, V):
    A_inv = V.T / s
    print(A_inv)
    # mean of X is the first control point C_1
    mu_X = np.mean(X, axis=1)
    b = X - mu_X
    alpha = A_inv @ b
    assert alpha.shape == (3, X.shape[1])
    alpha = np.vstack((1-np.sum(alpha, axis=0), alpha))
    return alpha

def scale_3D_cam(X, X_cam):
    mean = np.mean(X_cam, axis=1)
    _, d, _ = np.linalg.svd(np.cov(X))
    _, d_cam, _ = np.linalg.svd(np.cov(X_cam))
    beta = np.sign(mean[2]) * np.sqrt(np.sum(d) / np.sum(d_cam))
    return np.asscalar(beta) * X_cam

def EPnP(x, X, K):
    # Inputs:
    #    x - 2D inlier points
    #    X - 3D inlier points
    # Output:
    #    P - normalized camera projection matrix
    
    # shape of x_hat is 3 x n
    x_hat = np.linalg.inv(K) @ Homogenize(x)
    
    # parameterize 3D point in the world coordinate frame
    X = np.matrix(X)
    s, V = control_points_world(X)
    
    # shape of alpha is 4 x n
    alpha = parameterize_3D_world(X, s, V)
    C_world = np.hstack((np.zeros((3, 1)), s * V)) + np.mean(X, axis=1)
    assert C_world.shape == (3, 4)
    X_world = C_world @ alpha
    
    # parameterize 3D point in the camera coordinate frame
    # construct matrix I
    I = np.zeros((0, 12))
    for i in range(x.shape[1]):
        sub_I = np.zeros((2, 0))
        for j in range(4):
            mat = np.array([[alpha[j, i], 0, -alpha[j, i] * x_hat[0, i] / x_hat[2, i]], 
                            [0, alpha[j, i], -alpha[j, i] * x_hat[1, i] / x_hat[2, i]]])
            sub_I = np.hstack((sub_I, mat))
        I = np.vstack((I, sub_I))
    assert I.shape == (2 * x_hat.shape[1], 12)
    # find the right null space of matrix I
    C_cams = control_points_cam(I)
    assert C_cams.shape == (12, 1)
    # the shape of C_matrix is 3 x 4
    C_matrix = C_cams.reshape(4,3).T
    # deparameterize 3D points in camera coordinate frame
    X_cam = C_matrix @ alpha 
    assert X_cam.shape == (3, X.shape[1])
    X_cam = scale_3D_cam(X, X_cam)
    # 3D euclidean transformation
    # calculate the translation and rotation from the world coordinate frame to camera coordinate frame
    X_cam_prime = X_cam - np.mean(X_cam, axis=1)
    X_world_prime = X_world - np.mean(X_world, axis=1)
    # scatter matrix S to calculate the Rotation Matrix: S = C.T @ B
    S = X_cam_prime @ X_world_prime.T
    u, d, v = np.linalg.svd(S)
    R = np.eye(3) 
    if np.linalg.det(u) * np.linalg.det(v) < 0:
        d = np.eye(3)
        d[2, 2] = -1
        R = u @ d @ v
    else:
        R = u @ v
    t = np.mean(X_cam, axis=1) - R @ np.mean(X_world, axis=1)
    P = np.concatenate((R, t), axis=1)
    return P

'''
DLT for pose estimation, when there is no outliers, all points are clean
'''
def DLT(x, X, normalize=True):
    # Inputs:
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    normalize - if True, apply data normalization to x and X
    #
    # Output:
    #    P - the (3x4) DLT estimate of the camera projection matrix
    P = np.eye(3,4)+np.random.randn(3,4)/10
    P, x, X = np.matrix(P), np.matrix(x), np.matrix(X)
    # data normalization
    if normalize:
        x, T = Normalize(x)
        X, U = Normalize(X)
    else:
        x = Homogenize(x)
        X = Homogenize(X)
    '''
        compute A matrix, which give rises to systems of linear equations: A p = 0
        shape of x: 3, 50
        shape of X: 4, 50
    '''
    x_dims, x_nums = x.shape
    X_dims, X_nums = X.shape
    assert x_nums == X_nums
    # dummy first row of the A matrix
    A = np.zeros((0,12))
    for i in range(x_nums):
        x_pt, X_pt = x[:, i], X[:, i]
        l_null = left_null(x_pt, dims=3)
        assert l_null.shape == (2, 3)
        assert X_pt.T.shape == (1, 4)
        A = np.vstack((A, np.kron(l_null, X_pt.T)))
    assert A.shape == (2 * x_nums, 12)
    
    # solve for p using SVD, and reverse vectorize p matrix
    u, sigma, vh = np.linalg.svd(A)
    # taking the last row of v transpose
    p = vh[-1,:].T
    # reverse vectorize p
    P = p.reshape(3,4)
    # data denormalize
    if normalize:
        P = np.linalg.inv(T) @ P @ U    
        P = P / np.linalg.norm(P, ord='fro')
    print("frobineous norm of P: {}".format(np.linalg.norm(P, ord='fro')))
    return P