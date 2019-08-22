import numpy as np 

'''
Finstwalder's Solution for P3P: Calibrated Camera
---
P3P: minimal solver with Finsterwalder's Solution [https://link.springer.com/article/10.1007%2FBF02028352]
with correction (from UCSD Ben Ochoa): [https://cseweb.ucsd.edu/~bochoa/notes/haralick94corrections.pdf]
'''
def P3P(x, X, K):
    # x: 3x3 points
    x, X = np.matrix(x), np.matrix(X)
    x_hat = np.linalg.inv(K) @ Homogenize(x)
    f = K[0, 0]
    distances, angles, angles2, js = parameters_P3P(x_hat, X, f)
    lams = get_lambda(distances, angles, angles2)
    j1, j2, j3 = js
    a, b, c = distances
    alpha, beta, gamma = angles
    alpha2, beta2, gamma2 = angles2
    # choose the smallest real lambda
    lams_filter = np.isreal(lams)
    if lams[lams_filter].shape[0] == 0:
        print("no valid lambda, P3P algorithm exits")
        return None
    lam = np.min(lams[lams_filter])
    # solve u and v as parameters
    us, vs = get_uv(lam.real, distances, angles, angles2)
    if len(us) == 0:
        print("no valid u,v solution found")
        return None
    uv = zip(us, vs)
    # get u and v
    models = []
    for u, v in uv:
#         print("u, v: {} {}".format(u, v))
        s1 = np.sqrt(a / (u ** 2 + v ** 2 - 2 * u * v * alpha))
        s2 = u * s1
        s3 = v * s1
        p1, p2, p3 = s1 * j1, s2 * j2, s3 * j3
        # get three points in the 3D camera coordinate systems
        assert p1.shape == (3,1)
        # C is camera coordinates (3, n)
        C = np.hstack((p1, p2, p3))
        C_norm = C - np.mean(C, axis=1)
        X_norm = X - np.mean(X, axis=1)
        # X is the world coordinates (3, n)
        S = C_norm @ X_norm.T
        U_mat, d, V_mat = np.linalg.svd(S)
        R = np.eye(3) 
        if np.linalg.det(U_mat) * np.linalg.det(V_mat) < 0:
            d = np.eye(3)
            d[2, 2] = -1
            R = U_mat @ d @ V_mat
        else:
            R = U_mat @ V_mat
        t = np.mean(C, axis=1) - R @ np.mean(X, axis=1)
        P = np.hstack((R, t))
        assert R.shape == (3, 3)
        models.append(P)
    return models

def parameters_P3P(x, X, f):
    # shape of X is (3, 3)
    a = np.linalg.norm(X[:, 1] - X[:, 2])
    b = np.linalg.norm(X[:, 0] - X[:, 2])
    c = np.linalg.norm(X[:, 0] - X[:, 1])
    # calculate j1, j2, j3 (unit vector)
    # shape of x is (2, 3)
    j1 = np.matrix([x[0, 0] / x[2, 0] * f, 
                    x[1, 0] / x[2, 0] * f, 
                    f]).T
    j2 = np.matrix([x[0, 1] / x[2, 1] * f, 
                    x[1, 1] / x[2, 1] * f, 
                    f]).T
    j3 = np.matrix([x[0, 2] / x[2, 2] * f, 
                    x[1, 2] / x[2, 2] * f, 
                    f]).T
    j1, j2, j3 = j1 / np.linalg.norm(j1), \
                 j2 / np.linalg.norm(j2), \
                 j3 / np.linalg.norm(j3)
    assert j1.shape == (3,1)
    # calculate cos_alpha cos_beta cos_gamma
    cos_alpha = j2.T @ j3
    cos_beta = j1.T @ j3
    cos_gamma = j1.T @ j2
    # package parameters
    distances = (np.asscalar(a ** 2), np.asscalar(b ** 2), np.asscalar(c ** 2))
    angles = (np.asscalar(cos_alpha), np.asscalar(cos_beta), np.asscalar(cos_gamma))
    angles2 = (np.asscalar(cos_alpha ** 2), np.asscalar(cos_beta ** 2), np.asscalar(cos_gamma ** 2))
    unit_vecs = (j1, j2, j3)
    return distances, angles, angles2, unit_vecs 

def get_lambda(distances, angles, angles2):
    a, b, c = distances
    # alpha is cos_alpha ^ 2
    alpha, beta, gamma = angles
    alpha2, beta2, gamma2 = angles2
    s_alpha, s_gamma, s_beta = 1 - alpha2, 1 - gamma2, 1 - beta2
    G = c * (c * s_beta - b * s_gamma)
    H = b * (b - a) * s_gamma + \
                    c * (c + 2 * a) * s_beta + \
                    2 * b * c * (-1 + alpha * beta * gamma)
    I = b * (b - c) * s_alpha + \
                    a * (a + 2 * c) * s_beta + \
                    2 * a * b * (-1 + alpha * beta * gamma)
    J = a * (a * s_beta - b * s_alpha)
    return np.roots([G, H, I, J])

def get_uv(lam, distances, angles, angles2):
    # four pairs of u and v
    # construct a list of u and v
    a, b, c = distances
    alpha, beta, gamma = angles
    alpha2, beta2, gamma2 = angles2
    # calculate coefficients that depend on lambda
    A = 1 + lam
    B = -alpha
    C = (b - a) / b - lam * c / b 
    D = -lam * gamma
    E = (a / b + lam * c / b ) * beta
    F = -a / b + lam * (b - c ) / b
    # solve p, q
    p = np.sqrt(B ** 2 - A * C)
    q = np.sign(B * E - C * D) * np.sqrt(E ** 2 - C * F)
    # solve m, n (two sets)
    m1, n1 = (-B + p) / C, (-(E - q)) / C
    m2, n2 = (-B - p) / C, (-(E + q)) / C
    us, vs = [], []
    # solve u11 u12
    A1 = b - m1 ** 2 * c
    B1 = c * (beta - n1) * m1 - b * gamma
    C1 = -c * n1 ** 2 + 2 * c * n1 * beta + b - c
    if B1 ** 2 - A1 * C1 >= 0:
        u11 = -np.sign(B1) / A1 * (np.abs(B1) + np.sqrt(B1 ** 2 - A1 * C1))
        u12 = C1 / (A1 * u11)
        v11, v12 = u11 * m1 + n1, u12 * m1 + n1
        us += [u11, u12]
        vs += [v11, v12]
    # solve u21 u22
    A2 = b - m2 ** 2 * c
    B2 = c * (beta - n2) * m2 - b * gamma
    C2 = -c * n2 ** 2 + 2 * c * n2 * beta + b - c
    if B2 ** 2 - A2 * C2 >= 0:
        u21 = -np.sign(B2) / A2 * (np.abs(B2) + np.sqrt(B2 ** 2 - A2 * C2))
        u22 = C2 / (A2 * u21)
        # solve v11
        v21, v22 = u21 * m2 + n2, u22 * m2 + n2
        us += [u21, u22]
        vs += [v21, v22]
    # return four sets of u, v points
    return us, vs
