import cv2
import numpy as np 
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


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
		bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		matches = bf.knnMatch(des1, des2, k=2)
		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.75*n.distance:
				if m.distance < 32:
					good.append((kp1[m.queryIdx], kp2[m.trainIdx]))

		kp1 = np.array([item[0].pt for item in good])
		kp2 = np.array([item[1].pt for item in good])
		# Apply Ransac
		model, inliers = ransac((kp1, kp2),
                        FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=1000)
		kp1 = kp1[inliers]
		kp2 = kp2[inliers]
		return np.stack((kp1, kp2), axis=0).astype(int)