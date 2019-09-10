import numpy as np
import cv2

from view import SLAMView2D, SLAMView3D
from feature import FeatureExtractor
from models import Frame, Point3D, Pose
from geometry.utils import *
from geometry.essential import *
from geometry.triangulation import *
from geometry.EPnP import *

'''
Controller class that manages the data structure and view models
- settings
- algorithm banks
- camera intrinsics
'''

class SLAMController:

	def __init__(self, cap):
		self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
		self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
		self.total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		print('WIDTH: {}, HEIGHT: {}, FRAME_COUNT: {}'.format(self.frame_width, self.frame_height, self.total_frame))

		self.view2d = SLAMView2D()
		self.feature_extractor = FeatureExtractor()

		self.frame_idx = 0

		# frame: keypoints, poses and 3D points
		self.frames = []
		# point: keypoints index, frame index
		self.points = []
		self.K = np.array([[525, 0, self.frame_width//2],
						   [0, 525, self.frame_height//2],
						   [0, 0, 1]])

	def __str__(self):
		return "Controller: frames: width {} height {} total {}".format(self.frame_width, self.frame_height, self.total_frame)

	def process_frame(self, frame):
		'''
		main controller function that does basically everything
		'''
		# do nothing if it is the first frame
		
		image, curr_frame = self.preprocessing_frame(frame)
		self.frames.append(curr_frame)

		if self.frame_idx - 1 < 0:
			self.view2d.draw_2d_frame(image)
			self.frame_idx += 1
			return
		if self.frame_idx >= self.total_frame:
			# TODO: throw exceptions
			print("current frame out of bounds")
			return

		prev_frame = self.frames[self.frame_idx - 1]
		# if we can find keypoints for both frames
		if prev_frame.kps is not None and curr_frame.kps is not None:
			
			# indices for matched keypoints
			curr_inliers, prev_inliers = self.feature_extractor.feature_matching(curr_frame.kps, curr_frame.des, prev_frame.kps, prev_frame.des, self.K)
			
			# update connection graph between the two frames
			prev_frame.rightInliers = prev_inliers
			curr_frame.leftInliers = curr_inliers
			
			if prev_frame.pose is None:
				# use matches to calculate fundamental matrix
				# perform triangulation with P = [I | 0] and P' = [M | v]
				self.TwoViewPoseEstimation(curr_frame, prev_frame)
			else:
				# find the 3D points in the previous frame
				# EPnP for pose estimation to only update current frame's camera pose				
				self.AbsolutePoseEstimation(curr_frame, prev_frame)

			# shape of matches: 2 x n x 2
			# post-processing the keypoints data
			kp1 = curr_frame.kps[curr_inliers]
			kp2 = prev_frame.kps[prev_inliers]
			matches = np.stack((kp1, kp2), axis=0)
			self.view2d.draw_2d_matches(image, matches)
		
			# clear keypoints and descriptors from the previous model after matching, memory efficiency
			# prev_model.clear()
		self.view2d.draw_2d_frame(image)
		self.frame_idx += 1

	# any preprocessing functionality here
	def preprocessing_frame(self, frame):
		frame_resize = cv2.resize(frame, dsize=(self.frame_width, self.frame_height))
		kps, des = self.feature_extractor.feature_detecting(frame_resize)
		kps = np.array([item.pt for item in kps])
		print('changing keypoints to np array')
		model = Frame(kps, des)
		return frame_resize, model

	# DLT estimation for Projective Matrix P, 
	# given 3D points from previous frames and 2D points in current frames
	def AbsolutePoseEstimation(self, curr_frame, prev_frame):
		assert(len(prev_frame.rightInliers) == len(curr_frame.leftInliers)) 
		poseIdx, triIdx = self.find_union_intersection(curr_frame, prev_frame)
		pts3D = prev_frame.get_3D_points([idx[0] for idx in poseIdx])
		X = np.hstack([item.get_data().reshape(-1,1) for item in pts3D])
		# find the 2d points that are related to 3d points
		pts2D = curr_frame.kps[[idx[1] for idx in poseIdx]]
		x = np.hstack([item.reshape(-1,1) for item in pts2D])
		x_hat = NormalizePoints(Homogenize(x), self.K)
		P = EPnP(x_hat, Dehomogenize(X))
		for i, item in enumerate(poseIdx):
			pts3D[i].add_observation(point=curr_frame.kps[item[1]].reshape(-1,1), frame_idx=self.frame_idx)
			curr_frame.add_3D_point(item[1], pts3D[i])
			
		curr_frame.pose = Pose(P[:, :3], P[:, -1])

		pts1, pts2 = prev_frame.kps[[idx[0] for idx in triIdx]], curr_frame.kps[[idx[1] for idx in triIdx]]
		pts1, pts2 = Homogenize(pts1.T), Homogenize(pts2.T)
		n = pts1.shape[1]
		norm_pts1, norm_pts2 = NormalizePoints(pts1, self.K), NormalizePoints(pts2, self.K)
		Xs = Triangulation(norm_pts1, norm_pts2, prev_frame.pose.P(), curr_frame.pose.P(), option='linear', verbose=False)
		assert Xs.shape[1] == len(triIdx)
		for i, item in enumerate(triIdx):
			p3d = Point3D(Xs[:, i].reshape(-1,1))
			# add new 3d point
			self.points.append(p3d)
			p3d.add_observation(point=prev_frame.kps[item[0]].reshape(-1,1), frame_idx=self.frame_idx-1)
			p3d.add_observation(point=curr_frame.kps[item[1]].reshape(-1,1), frame_idx=self.frame_idx)
			prev_frame.add_3D_point(item[0], p3d)
			curr_frame.add_3D_point(item[1], p3d)
		assert curr_frame.has_all(curr_frame.leftInliers)

	def find_union_intersection(self, curr_frame, prev_frame):
		poseIdx, triIdx = [], []
		for i, item in enumerate(prev_frame.rightInliers):
			if item in prev_frame.leftInliers:
				poseIdx.append((item, curr_frame.leftInliers[i]))
			else:
				triIdx.append((item, curr_frame.leftInliers[i]))
		assert len(poseIdx) + len(triIdx) == len(curr_frame.leftInliers)
		return poseIdx, triIdx

	def TwoViewPoseEstimation(self, curr_frame, prev_frame):
		# creation of essential matrix and 3D points assuming the first pose (f2) is [I | 0], the second pose (f1) is [R | t]
		# save for testing
		pts1, pts2 = prev_frame.kps[prev_frame.rightInliers], curr_frame.kps[curr_frame.leftInliers]
		pts1, pts2 = Homogenize(pts1.T), Homogenize(pts2.T)
		n = pts1.shape[1]
		norm_pts1, norm_pts2 = NormalizePoints(pts1, self.K), NormalizePoints(pts2, self.K)
		# use the normalized points to estimate essential matrix
		E = DLT_E(norm_pts1, norm_pts2)
		P1, P2 = Decompose_Essential(E, norm_pts1, norm_pts2)
		print('First camera R: {} t: {}'.format(P1[:, :3], P1[:, -1]))
		print('Second camera R: {} t: {}'.format(P2[:, :3], P2[:, -1]))
		prev_frame.pose = Pose(P1[:, :3], P1[:, -1])
		curr_frame.pose = Pose(P2[:, :3], P2[:, -1])
		Xs = Triangulation(norm_pts1, norm_pts2, P1, P2, option='linear', verbose=False)
		assert Xs.shape[1] == n
		for i in range(n):
			p3d = Point3D(Xs[:, i].reshape(-1,1))
			# add new 3d point
			self.points.append(p3d)
			p3d.add_observation(point=prev_frame.kps[prev_frame.rightInliers[i]].reshape(-1,1), frame_idx=self.frame_idx-1)
			p3d.add_observation(point=curr_frame.kps[curr_frame.leftInliers[i]].reshape(-1,1), frame_idx=self.frame_idx)
			prev_frame.add_3D_point(prev_frame.rightInliers[i], p3d)
			curr_frame.add_3D_point(curr_frame.leftInliers[i], p3d)
		assert n == len(curr_frame.leftInliers)

	def Optimization(self):
		pass

