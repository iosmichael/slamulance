import numpy as np

def Homogenize(pts):
	return np.vstack((pts, np.ones((1, pts.shape[1]))))

def Dehomogenize(pts):
	return pts[:-1]/pts[-1]

def Normalize(pts):
    # data normalization of n dimensional pts
    #
    # Input:
    #    pts - is in inhomogeneous coordinates
    # Outputs:
    #    pts - data normalized points
    #    T - corresponding transformation matrix
    means, var = np.mean(pts, axis=1), np.var(pts, axis=1)
    s = np.sqrt(pts.shape[0] / np.sum(var))
    s_means = -means * s
    T = np.matrix(np.eye(pts.shape[0]+1)) * s
    T[-1, -1] = 1
    T[:-1, -1] = s_means
    pts = T @ Homogenize(pts)
    return pts, T

def Skew(w):
    assert w.shape == (3, 1)
    # Returns the skew-symmetrix represenation of a vector
    w_skew = np.matrix([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])
    return w_skew

def LeftNull(x, dims=3):
    '''
    finding left null space of x using Householder transformation
    assume the shape of x is: dims x 1
    '''
    e = np.matrix(np.eye(dims))[:, 0]
    v = x + np.sign(x[0,0]) * np.linalg.norm(x) * e
    assert v.shape == x.shape
    H_v = np.eye(dims) - 2 * (v @ v.T) / (v.T @ v)
    # return H_v matrix with the omission of the first row
    return H_v[1:, :]

'''
Multiple View Geometry: p88
Only use for small A matrix
'''
def RightNull(A):
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :].reshape(-1, 1)
    assert np.allclose(A @ h, np.zeros((A @ h).shape))
    return h

# testing units
def test_LeftNull(x, dims=3):
    l_null = LeftNull(x, dims)
    print(l_null)
    print(l_null @ x)

test_LeftNull(np.matrix([1,2,3]).T, dims=3)