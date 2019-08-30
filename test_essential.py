import numpy as np
import os
from geometry.utils import *
from geometry.fundamental import *

def main():
	# calibration matrix
	WIDTH, HEIGHT = 621, 187
	K = np.array([[525, 0, WIDTH//2],
				  [0, 525, HEIGHT//2],
				  [0, 0, 1]])
	'''
	import two testing sets of points
	'''
	pts1, pts2 = np.load('./test/pts1.npy'), np.load('./test/pts2.npy')
	print('points from frame 1 shape: {}'.format(pts1.shape))
	print('points from frame 2 shape: {}'.format(pts2.shape))
	assert pts1.shape == pts2.shape
	pts1, pts2 = Homogenize(pts1.T), Homogenize(pts2.T)
	n = pts1.shape[1]
	norm_pts1, norm_pts2 = NormalizePoints(pts1, K), NormalizePoints(pts2, K)
	# use the normalized points to estimate essential matrix
	E = DLT_E(norm_pts1, norm_pts2)
	print("Essential matrix: {}".format(E))
	# decompose the essential matrix into two projective matrices
	# P1 = [I | 0] -> pts1, P2 = [R | t]

	# norm_pts1.shape == (3, n)
	decompose_essential(E, norm_pts1, norm_pts2)

def decompose_essential(E, x1, x2):
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
	# Four Possible Solutions
	P1 = np.concatenate((R1, t1), axis=1)
	P2 = np.concatenate((R1, t2), axis=1)
	P3 = np.concatenate((R2, t1), axis=1)
	P4 = np.concatenate((R2, t2), axis=1)
	for P in [P1, P2, P3, P4]:
		# testing chirality of Triangulation
		X = LinearTriangulation(x1[:, 0], x2[:, 0], P0, P)
		print('Triangulation points shape: {}'.format(X.shape))
		P0_front = chierality(P0, X)
		P_front = chierality(P, X)
		if P0_front and P_front:
			return P0, P
		else:
			print('not essential')
	print("could not find a valid decomposition of essential matrix")
	return P0, P1

def LinearTriangulation(x1, x2, P1, P2):
	'''
	input:
		- x1: inhomogeneous normalized point u, v in image 1
		- x2: inhomogeneous normalized point u, v in image 2
		- P1: projection matrix in image 1
		- P2: projection matrix in image 2
	'''
	'''
	DOUBLE CHECK
	v1P1^3T - P1^2T
	u1P1^3T - P1^1T  == X, perform SVD to minimize algebraic error
	v2P2^3T - P2^2T
	u2P2^3T - P2^1T
	'''	
	# x1 shape == (3, n)
	A = []
	x1, x2 = Dehomogenize(x1), Dehomogenize(x2)
	# start with one point
	A.append(x1[0] * P1[-1, :] - P1[1, :])
	A.append(x1[1] * P1[-1, :] - P1[0, :])
	A.append(x2[0] * P2[-1, :] - P2[1, :])
	A.append(x2[1] * P2[-1, :] - P2[0, :])
	A = np.vstack(A)
	# perform singular value decomposition
	U, d, Vt = np.linalg.svd(A)
	X = Vt[-1, :].T
	assert X.shape == (4, 1)
	# calculating the re-projection error
	x1_est, x2_est = P1 @ X, P2 @ X
	print('reporjection error: {}'.format(np.sqrt(np.sum(Dehomogenize(x1_est) - x1.reshape(-1, 1)) ** 2 + np.sum(Dehomogenize(x2_est) - x2.reshape(-1,1)) ** 2)))
	return X
	
'''
X is the reconstructed homogeneous 3D point
Chirality tests whether the reconstructed point is in front of the camera pose
'''
def chierality(P, X):
	assert X.shape == (4, 1)
	w = P[2, :] @ X
	return w * X[-1,0] * np.linalg.det(P[:, :3]) > 0

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

if __name__ == '__main__':
	main()