import cv2
import numpy as np 
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from geometry.utils import *


class FeatureExtractor:
	'''
	George Hotz's Feature Stack:
	- Good Features to Track: Keypoint Detections
	- ORB: Descriptor
	- BFMatcher: Descriptor matching
	- SKImage: RANSAC
	'''
	def feature_detecting(self, frame):
		orb = cv2.ORB_create()
		pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
		if pts is None:
			return None, None
		kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
		kps, des = orb.compute(frame, kps)
		return kps, des

	def feature_matching(self, kp1, des1, kp2, des2):
		# BFMatcher with default params
		# Referred from OpenCV Website
		if des1 is None or des2 is None:
			return np.zeros((0, 0, 2))

		# this is just for test
		WIDTH, HEIGHT = 621, 187
		K = np.array([[525, 0, WIDTH//2],
				  	  [0, 525, HEIGHT//2],
				  	  [0, 0, 1]])

		bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		matches = bf.knnMatch(des1, des2, k=2)
		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.75*n.distance:
				if m.distance < 32:
					good.append((m.queryIdx, m.trainIdx))
		# use tuple to represent
		good = np.array(good)
		# kp1, kp2 = np.array([item.pt for item in kp1]), np.array([item.pt for item in kp2])
		kp1_ransac = kp1[good[:, 0]]
		kp2_ransac = kp2[good[:, 1]]
		# Apply Ransac
			# shape = (n, 2)
		kp1_ransac, kp2_ransac = Dehomogenize(NormalizePoints(Homogenize(kp1_ransac.T), K)).T, Dehomogenize(NormalizePoints(Homogenize(kp2_ransac.T), K)).T
		model, inliers = ransac((kp1_ransac, kp2_ransac),
                        EssentialMatrixTransform, min_samples=8,
                        residual_threshold=0.02, max_trials=1000)
		kp1_inliers = good[:, 0][inliers]
		kp2_inliers = good[:, 1][inliers]
		return kp1_inliers, kp2_inliers