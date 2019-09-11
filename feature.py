'''
Visual Monocular SLAM Implementation
Created: Sept 10, 2019
Author: Michael Liu (GURU AI Group, UCSD)
'''

import cv2
import numpy as np 
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform
from geometry.utils import *


class FeatureExtractor:
	'''
	George Hotz's Feature Stack:
	- Good Features to Track: Keypoint Detections
	- ORB: Descriptor
	- BFMatcher: Descriptor matching
	- SKImage: RANSAC
	'''
	def feature_detecting(self, frame, mode='feat'):
		orb = cv2.ORB_create()
		kps = []
		if mode == 'feat':
			pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
			if pts is None:
				return None, None
			kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
		else:
			kps = orb.detect(frame,None)
		kps, des = orb.compute(frame, kps)
		return kps, des

	def within_window(self, kp1, kp2, window=100):
		return (kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2 < window**2

	def feature_matching(self, kp1, des1, kp2, des2, K):
		# BFMatcher with default params
		# Referred from OpenCV Website
		if des1 is None or des2 is None:
			return np.zeros((0, 0, 2))

		bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		matches = bf.knnMatch(des1, des2, k=2)
		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				if m.distance < 30:
					# clipping with window
					if self.within_window(kp1[m.queryIdx], kp2[m.trainIdx], window=30):
						good.append((m.queryIdx, m.trainIdx))
		# use tuple to represent
		good = np.array(good)
		# kp1, kp2 = np.array([item.pt for item in kp1]), np.array([item.pt for item in kp2])
		kp1_ransac = kp1[good[:, 0]]
		kp2_ransac = kp2[good[:, 1]]
		# Apply Ransac
			# shape = (n, 2)
		# model, inliers = ransac((kp1_ransac, kp2_ransac), EssentialMatrixTransform, min_samples=8, residual_threshold=0.02, max_trials=1000)
		if True:
			# essential
			kp1_ransac, kp2_ransac = Dehomogenize(NormalizePoints(Homogenize(kp1_ransac.T), K)).T, Dehomogenize(NormalizePoints(Homogenize(kp2_ransac.T), K)).T
			model, inliers = ransac((kp1_ransac, kp2_ransac),
            	            EssentialMatrixTransform, min_samples=8,
                	        residual_threshold=0.02, max_trials=100)
		'''
		TODO: add a window for inlier
		'''
		kp1_inliers = good[:, 0][inliers]
		kp2_inliers = good[:, 1][inliers]
		return kp1_inliers, kp2_inliers