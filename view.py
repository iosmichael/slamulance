import cv2
import numpy as np 
from multiprocessing import Process, Queue
import pangolin
import OpenGL.GL as gl
import numpy as np


class SLAMView:
	def __init__(self):
		self.state = None
		self.q = Queue()
		self.vp = Process(target=self.viewer_thread, args=(self.q,))
		self.vp.daemon = True
		print('3D viewer started...')
		self.vp.start()

	def viewer_thread(self, q):
		self.viewer_init(1024, 768)
		while True:
		  self.viewer_refresh(q)

	def viewer_init(self, w, h):
		pangolin.CreateWindowAndBind('Map Viewer', w, h)
		gl.glEnable(gl.GL_DEPTH_TEST)

		self.scam = pangolin.OpenGlRenderState(
		  pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
		  pangolin.ModelViewLookAt(0, -10, -8,
								   0, 0, 0,
								   0, -1, 0))
		self.handler = pangolin.Handler3D(self.scam)

		# Create Interactive View in window
		self.dcam = pangolin.CreateDisplay()
		self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
		self.dcam.SetHandler(self.handler)
		# hack to avoid small Pangolin, no idea why it's *2
		self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
		self.dcam.Activate()


	def viewer_refresh(self, q):
		while not q.empty():
		  self.data = q.get()

		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glClearColor(0.0, 0.0, 0.0, 1.0)
		self.dcam.Activate(self.scam)

		if self.data is not None:
			if self.data[0].shape[0] >= 2:
				# draw poses
				gl.glColor3f(0.0, 1.0, 0.0)
				pangolin.DrawCameras(self.data[0][:-1])

			if self.data[0].shape[0] >= 1:
				# draw current pose as yellow
				gl.glColor3f(1.0, 1.0, 0.0)
				pangolin.DrawCameras(self.data[0][-1:])

			if self.data[1].shape[0] != 0:
				# draw keypoints
				gl.glPointSize(5)
				gl.glColor3f(1.0, 0.0, 0.0)
				pangolin.DrawPoints(self.data[1], self.data[2])
		pangolin.FinishFrame()

	def draw_3d(self, poses, points):
		if self.q is None:
			return

		poses, pts, colors = [], [], []
		for f in mapp.frames:
			# invert pose for display only
			poses.append(np.linalg.inv(f.pose))
		for p in mapp.points:
			pts.append(p.pt)
			colors.append(p.color)
		self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))

	'''
	View class that contains all 2D cv2 drawing functionality in addition to 3D drawing
	'''
	def draw_2d_frame(self, frame, window_name='Driving POV'):
		cv2.imshow(window_name, frame)

	def draw_2d_matches(self, frame, matches):
		print("{} matches".format(matches.shape[1]))
		matches = matches.astype(int)
		for i in range(matches.shape[1]):
			cv2.circle(frame, (matches[0, i, 0], matches[0, i, 1]), radius=3, color=(0, 255.0, 0))
			cv2.line(frame, (matches[0, i, 0], matches[0, i, 1]), (matches[1, i, 0], matches[1, i, 1]), (255, 0, 0), thickness=1)

