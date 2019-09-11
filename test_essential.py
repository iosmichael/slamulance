'''
Visual Monocular SLAM Implementation
Created: Sept 8, 2019
Author: Michael Liu (GURU AI Group, UCSD)
'''

import numpy as np
import os
from geometry.utils import *
from geometry.triangulation import Triangulation
from geometry.essential import *

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
	I, P2 = Decompose_Essential(E, norm_pts1, norm_pts2, mode='note')
	Xs1 = Triangulation(norm_pts1, norm_pts2, I, P2, verbose=False)
	Rt = generate_Rt(30)

	P1, P3 = Project_Essential(I, P2, Rt)
	Xs2 = Triangulation(norm_pts1, norm_pts2, P1, P3, verbose=False)
	E_compose = Compose_Essential(P1, P3)
	print("rank of E: {}".format(np.linalg.matrix_rank(E_compose)))
	print("difference: {}".format(np.linalg.det(E - E_compose)))

def generate_Rt(theta):
	R = np.array([[np.cos(theta), -np.sin(theta), 0],
				  [np.sin(theta), np.cos(theta), 0],
				  [0, 0, 1]])
	assert np.linalg.det(R) == 1
	t = np.random.random((3,1)) * 100
	print('R: {}'.format(R))
	print('t: {}'.format(t))
	return np.hstack((R, t))

def GeometricTriangulation(x1, x2, P1, P2):
	'''
	Bad result currently
	'''
	R1, t1 = P1[:, :3], P1[:, -1]
	R2, t2 = P2[:, :3], P2[:, -1]
	R, t = R2 @ R1.T, t2.reshape(-1,1) - t1.reshape(-1,1)
	E = Skew(t.reshape(-1, 1)) @ R
	'''
	x1 and x2 has to be normalized points

	backproject P1 as 3D line and P2 as 3D plane to perform intersection
	1. generate F from two camera projection matrices
	'''
	if x1.shape == (2,1):
		x1, x2 = Homogenize(x1), Homogenize(x2)
	x1, x2 = x1.reshape(3,1), x2.reshape(3,1)
	assert x1.shape == (3, 1)
	if np.allclose(t1, np.zeros((3,1))):
		C1 = np.zeros((4,1))
		C1[-1, 0] = 1
		sudoP1 = P1.T
	else:
		C1 = RightNull(P1) # camera center from P1, shape = 4, 1
		sudoP1 = np.linalg.inv(P1.T @ P1) @ P1.T
	# project the point from first image to the line in the second image
	assert E.shape == (3, 3)
	l2 = E @ x1 # l2 is (a, b, c)
	l2_orth = np.matrix([-l2.item(1) * x2.item(2), 
						l2.item(0) * x2.item(2), 
						l2.item(1) * x2.item(0) - l2.item(0) * x2.item(1)]).reshape(3,1)
	# backproject 3D plane into the space
	pi = P2.T @ l2_orth # shape = (4, 1), (a, b, c, d)
	# backproject 3D line into the space
	X_inf = sudoP1 @ x1
	# intersection between 3D line and 3D plane formula
	piC1, piX_inf = np.sum(np.multiply(pi, C1)) - np.multiply(pi,C1), np.multiply(pi, X_inf) - np.sum(np.multiply(pi, X_inf))
	X = np.multiply(X_inf, piC1) - np.multiply(C1, piX_inf)
	assert X.shape == (4, 1)
	x1_est, x2_est = P1 @ X, P2 @ X
	print('reporjection error: {}'.format(np.sqrt(np.sum(Dehomogenize(x1_est) - Dehomogenize(x1).reshape(-1, 1)) ** 2 + np.sum(Dehomogenize(x2_est) - Dehomogenize(x2).reshape(-1,1)) ** 2)))
	return X

if __name__ == '__main__':
	main()
	# generate_Rt(39)