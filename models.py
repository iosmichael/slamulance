import numpy as np 

'''
each frame has many features, [many to many]
- it holds the inliers of matches
- it holds the keypoints and descriptors for only the next interations
'''
class Frame:

	def __init__(self, kps, des):
		self.keyframe = False
		self.kps = kps
		self.des = des
		self.features = []
		self.pose = None

	def set_keyframe(self, is_keyframe=False):
		self.keyframe = is_keyframe

	def add_feature(self, feature):
		self.features.append(feature)

	def set_pose(self, Rt):
		self.pose = Rt

	def clear(self):
		del self.kps
		del self.des

'''
camera pose model
- holds the Rotation and Translation parameters
- holds the frame it is representing
'''
class Pose:

	def __init__(self, R, t):
		self.R = R
		self.t = t

	def Rt(self):
		return np.concatenate((R, t))

'''
3D feature model
- holds the pts and their respective frame id
- holds a triangulation value (3d homogeneous coordinates)
'''
class Feature:

	def __init__(self):
		self.triangulation = None
		'''
		points = [x ...]
				 [y ...]
		'''
		self.points = np.zeros((2, 0))
		self.frame_indices = []

	def add_2D_point(self, point, frame_idx):
		# points are represented by numpy data structure for computing efficiency
		assert point.shape == (2, 1)
		self.points = np.stack((self.points, point), axis=1)
		self.frame_indices.append(frame_idx)
		assert len(self.frame_indices) == self.points.shape[1]

	def set_triangulation(self, point):
		self.triangulation = point