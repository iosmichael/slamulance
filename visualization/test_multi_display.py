import sys

sys.path.append('../lib')

import OpenGL.GL as gl
import pangolin
import multiprocessing as mp
# import threading

import numpy as np
import time


from multiprocessing import Process, Queue
import pangolin
import OpenGL.GL as gl
import numpy as np

class Display3D(object):
	def __init__(self):
		self.state = None
		mp.set_start_method('spawn')
		self.q = mp.Queue()
		self.vp = mp.Process(target=self.viewer_thread, args=(self.q,))
		self.vp.daemon = True
		self.vp.start()

	def viewer_thread(self, q):
		self.viewer_init(1024, 768)
		while True:
			self.viewer_refresh(q)
			# print(self.vp.is_alive())

	def viewer_init(self, w, h):
		# print(self.vp.is_alive())
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
			self.state = q.get()

		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glClearColor(0.0, 0.0, 0.0, 1.0)
		self.dcam.Activate(self.scam)
		gl.glLineWidth(3)
		gl.glColor3f(0.0, 1.0, 0.0)
		pangolin.DrawCamera(np.identity(4))

		# if self.state is not None:
		# 	if self.state[0].shape[0] >= 2:
		# 		# draw poses
		# 		gl.glColor3f(0.0, 1.0, 0.0)
		# 		pangolin.DrawCameras(self.state[0][:-1])

		# 	if self.state[0].shape[0] >= 1:
		# 		# draw current pose as yellow
		# 		gl.glColor3f(1.0, 1.0, 0.0)
		# 		pangolin.DrawCameras(self.state[0][-1:])

		# 	if self.state[1].shape[0] != 0:
		# 		# draw keypoints
		# 		gl.glPointSize(5)
		# 		gl.glColor3f(1.0, 0.0, 0.0)
		# 		pangolin.DrawPoints(self.state[1], self.state[2])

		pangolin.FinishFrame()

# def main():
# 	# display = PangolinDisplay()
# 	# while True:
# 	# 	print("main thread")
# 	# 	time.sleep(0.2)

if __name__ == '__main__':
	# main()
	display = Display3D()
	while True:
		print("main thread")
		time.sleep(0.2)