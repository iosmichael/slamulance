'''
this test demonstrates the pipeline of initial stereo pose estimation against stanford vision database

Sept 16, 2019
Created: Michael Liu
'''
import numpy as np
import os
import cv2
from geometry.utils import *
from geometry.triangulation import Triangulation
from geometry.essential import *

from view import *
from test import loader

from feature import FeatureExtractor

def main():
	# visualization
	# view_2d, view_3d = SLAMView2D(640, 480), SLAMView3D()
	# view_3d = SLAMView3D()
	# view_2d = SLAMView2D()
	stereo_camera_gt, stereo_images, stereo_image_files = loader.get_stereo_data()
	# pick two consecutive images
	K = stereo_camera_gt[stereo_image_files[0]]["K"]
	print("calibration matrix: {}".format(K))
	ind_1, ind_2 = 9, 10
	img1_key, img2_key = stereo_image_files[ind_1], stereo_image_files[ind_2]

	# test visualization


	# cv2.imshow("images", np.concatenate((stereo_images[img1_key], stereo_images[img2_key]), axis=0))
	# cv2.waitKey(0)

	detector = FeatureExtractor()
	kp1, des1 = detector.feature_detecting(stereo_images[img1_key], mode='feat')
	kp2, des2 = detector.feature_detecting(stereo_images[img2_key], mode='feat')
	kp1, kp2 = np.array([item.pt for item in kp1]), np.array([item.pt for item in kp2])

	kp1_inliers, kp2_inliers = detector.feature_matching(kp1, des1, kp2, des2, K)

	matches = np.stack((kp1[kp1_inliers], kp2[kp2_inliers]), axis=0)
	disp = stereo_images[img2_key]
	# view_2d.draw_2d_matches(disp, matches)

	assert len(kp1_inliers) == len(kp2_inliers)
	print("{} matches found.".format(len(kp1_inliers)))
	pts1, pts2 = kp1[kp1_inliers], kp2[kp2_inliers]

	pts1, pts2 = Homogenize(pts1.T), Homogenize(pts2.T)
	n = pts1.shape[1]
	norm_pts1, norm_pts2 = NormalizePoints(pts1, K), NormalizePoints(pts2, K)
	# use the normalized points to estimate essential matrix
	E = DLT_E(norm_pts1, norm_pts2)
	# E = K.T @ F @ K

	Rt1 = np.hstack((stereo_camera_gt[img1_key]["R"], stereo_camera_gt[img1_key]["t"]))
	Rt2 = np.hstack((stereo_camera_gt[img2_key]["R"], stereo_camera_gt[img2_key]["t"]))
	E_compose = Compose_Essential(Rt1, Rt2)

	print('rank of essential: {}'.format(np.linalg.matrix_rank(E)))
	print("Essential Matrix: {}".format(E))
	print("Ground Truth Essential: {}".format(E_compose))

	# decompose the essential matrix into two projective matrices
	# P1 = [I | 0] -> pts1, P2 = [R | t]
	I, P2 = Decompose_Essential(E, norm_pts1, norm_pts2, mode='note')

	P1, P3 = Project_Essential(I, P2, Rt1)
	cameras = [P1, P3]
	Xs = Triangulation(norm_pts1, norm_pts2, Rt1, Rt2, verbose=False)
	Xs1 = Triangulation(norm_pts1, norm_pts2, I, P2, verbose=True)
	points = [Xs[:, i].reshape(-1,1) for i in range(Xs.shape[1])]
	colors = [np.ones((3,1)) for i in range(len(points))]
	# view_3d.draw_cameras_points(cameras, points, colors)

	# cv2.imshow("matches", disp)
	# cv2.waitKey(0)

	print("Rt2: {}".format(Rt2))
	print("P3: {}".format(P3))
	print('rank of essential: {}'.format(np.linalg.matrix_rank(E_compose)))
	U, d, Vt = np.linalg.svd(E_compose)
	print(d)
	print("Ground Truth Essential: {}".format(E_compose))

	assert np.allclose(E, E_compose)
	print("PASS")


if __name__ == '__main__':
	main()