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
from test import loader

from feature import FeatureExtractor

def main():
	stereo_camera_gt, stereo_images, stereo_image_files = loader.get_stereo_data()
	# pick two consecutive images
	K = stereo_camera_gt[stereo_image_files[0]]["K"]
	print("calibration matrix: {}".format(K))
	ind_1, ind_2 = 0, 1
	img1_key, img2_key = stereo_image_files[ind_1], stereo_image_files[ind_2]
	detector = FeatureExtractor()
	kp1, des1 = detector.feature_detecting(stereo_images[img1_key], mode='orb')
	kp2, des2 = detector.feature_detecting(stereo_images[img2_key], mode='orb')
	kp1, kp2 = np.array([item.pt for item in kp1]), np.array([item.pt for item in kp2])

	kp1_inliers, kp2_inliers = detector.feature_matching(kp1, des1, kp2, des2, K)
	assert len(kp1_inliers) == len(kp2_inliers)
	print("{} matches found.".format(len(kp1_inliers)))
	pts1, pts2 = kp1[kp1_inliers], kp2[kp2_inliers]
	pts1, pts2 = Homogenize(pts1.T), Homogenize(pts2.T)
	n = pts1.shape[1]
	norm_pts1, norm_pts2 = NormalizePoints(pts1, K), NormalizePoints(pts2, K)
	# use the normalized points to estimate essential matrix
	F = DLT_F(pts1, pts2)
	E = K.T @ F @ K
	print('rank of essential: {}'.format(np.linalg.matrix_rank(E)))
	print("Essential Matrix: {}".format(E))

	# decompose the essential matrix into two projective matrices
	# P1 = [I | 0] -> pts1, P2 = [R | t]
	I, P2 = Decompose_Essential(E, norm_pts1, norm_pts2, mode='matrix')

	Rt1 = np.hstack((stereo_camera_gt[img1_key]["R"], stereo_camera_gt[img1_key]["t"]))
	Rt2 = np.hstack((stereo_camera_gt[img2_key]["R"], stereo_camera_gt[img2_key]["t"]))
	P1, P3 = Project_Essential(I, P2, Rt1)
	print("Rt2: {}".format(Rt2))
	print("P3: {}".format(P3))
	E_compose = Compose_Essential(Rt1, Rt2)
	print('rank of essential: {}'.format(np.linalg.matrix_rank(E_compose)))
	U, d, Vt = np.linalg.svd(E_compose)
	print(d)
	print("Ground Truth Essential: {}".format(E_compose))

	assert np.allclose(E, E_compose)
	print("PASS")

# def main():
# 	# calibration matrix
# 	WIDTH, HEIGHT = 621, 187
# 	K = np.array([[525, 0, WIDTH//2],
# 				  [0, 525, HEIGHT//2],
# 				  [0, 0, 1]])
# 	'''
# 	import two testing sets of points
# 	'''
# 	pts1, pts2 = np.load('./test/data/pts1.npy'), np.load('./test/data/pts2.npy')
# 	print('points from frame 1 shape: {}'.format(pts1.shape))
# 	print('points from frame 2 shape: {}'.format(pts2.shape))
# 	assert pts1.shape == pts2.shape
# 	pts1, pts2 = Homogenize(pts1.T), Homogenize(pts2.T)
# 	n = pts1.shape[1]
# 	norm_pts1, norm_pts2 = NormalizePoints(pts1, K), NormalizePoints(pts2, K)
# 	# use the normalized points to estimate essential matrix
# 	E = DLT_E(norm_pts1, norm_pts2)
# 	print("Essential matrix: {}".format(E))
# 	# decompose the essential matrix into two projective matrices
# 	# P1 = [I | 0] -> pts1, P2 = [R | t]

# 	# norm_pts1.shape == (3, n)
# 	I, P2 = Decompose_Essential(E, norm_pts1, norm_pts2, mode='note')
# 	Xs1 = Triangulation(norm_pts1, norm_pts2, I, P2, verbose=False)
# 	Rt = generate_Rt(30)

# 	P1, P3 = Project_Essential(I, P2, Rt)
# 	Xs2 = Triangulation(norm_pts1, norm_pts2, P1, P3, verbose=False)
# 	E_compose = Compose_Essential(P1, P3)
# 	print("rank of E: {}".format(np.linalg.matrix_rank(E_compose)))
# 	print("difference: {}".format(np.linalg.det(E - E_compose)))

# def generate_Rt(theta):
# 	R = np.array([[np.cos(theta), -np.sin(theta), 0],
# 				  [np.sin(theta), np.cos(theta), 0],
# 				  [0, 0, 1]])
# 	assert np.linalg.det(R) == 1
# 	t = np.random.random((3,1)) * 100
# 	print('R: {}'.format(R))
# 	print('t: {}'.format(t))
# 	return np.hstack((R, t))

if __name__ == '__main__':
	main()