
# optimal triangulation (Richard I. Harley)

# import sympy
# from sympy import Symbol
# from sympy.functions import re

def Triangulation(x1, x2, F, P):
    Xs = np.zeros((4, 0))
    for i in range(x1.shape[1]):
        Xs_i = OptimalTriangulation(x1[:, i], x2[:, i], F, P)
        Xs = np.hstack((Xs, Xs_i))
    assert Xs.shape == (4, x1.shape[1])
    return Xs

def OptimalTriangulation(x1, x2, F, P2):
    x1_correct, x2_correct = OptimalTriangulationCorrection(x1, x2, F)
    assert x1_correct.shape == (3, 1)
    assert x2_correct.shape == (3, 1)
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

def OptimalTriangulationCorrection(x1, x2, F):
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