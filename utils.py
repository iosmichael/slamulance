import numpy as np

class EssentialCamera:

	def __init__(self, frame_width, frame_height, scale=1):
		'''
		camera intrinsic: F = [ f, 0, Cx
								0, f, Cy
								0, 0, 1  ]
		usually focal length is equivalent as the diagonal of the images
		'''
		Cx, Cy = frame_width // 2, frame_height // 2
		f = np.sqrt((frame_width/scale) ** 2 + (frame_height/scale) ** 2)
		K = np.array([[f, 0, Cx],
			[0, f, Cy],
			[0, 0, 1]])
		self.K = K
	
	def homogenize(self, pts):
		return np.vstack((pts, np.ones((1, pts.shape[1]))))

	def dehomogenize(self, pts):
		return pts[:-1]/pts[-1]

	def normalize(self, pts):
		return np.dot(np.linalg.inv(self.K), self.homogenize(pts))

	def denormalize(self, pts):
		return self.dehomogenize(np.dot(self.K, pts))

if __name__ == '__main__':
	'''
	normalization test
	'''
	pts = np.ones((2, 20))
	camera = EssentialCamera(100, 100, scale=1)
	print(camera.normalize(pts))
	print(camera.normalize(pts).shape)

	print(camera.denormalize(camera.normalize(pts)))
	print(camera.denormalize(camera.normalize(pts)).shape)