'''
Visual Monocular SLAM Implementation
Created: Sept 10, 2019
Author: Michael Liu (GURU AI Group, UCSD)
'''

import cv2
import numpy as np 
import multiprocessing as mp
import pangolin
import OpenGL.GL as gl

# import pygame
# from pygame.locals import DOUBLEBUF
import numpy as np
import time
from geometry.utils import *


class SLAMView2D:

	# def __init__(self, W, H):
	# 	pygame.init()
	# 	self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
	# 	self.surface = pygame.Surface(self.screen.get_size()).convert()

	# def paint(self, img):
	# 	# junk
	# 	for event in pygame.event.get():
	# 		pass

	# 	# draw
	# 	pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [2,1,0]])

	# 	# RGB, not BGR (might have to switch in twitchslam)
	# 	pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [0,1,2]])
	# 	self.screen.blit(self.surface, (0,0))

	# 	# blit
	# 	pygame.display.flip()

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

class SLAMView3D:
	
	'''
	taken from george hotz's twitchslam display file
	- https://github.com/geohot/twitchslam/display.py
	'''

	def __init__(self):
		self.data = None

		# create a new context for 3d visualization process

		mp.set_start_method('spawn')
		self.q = mp.Queue()
		self.vp = mp.Process(target=self.viewer_thread, args=(self.q,))
		self.poses = []
		self.vp.daemon = True
		print('3D viewer started...')
		self.vp.start()

	def viewer_thread(self, q):
		self.viewer_init(1024, 768)
		while not pangolin.ShouldQuit():
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

		# drawing begins here

		# self.pose[2, 3] = self.pose[2, 3] - 0.1
		# self.poses.append(np.linalg.inv(self.pose))

		if self.data is not None:
			gl.glLineWidth(3)
			gl.glColor3f(0.0, 1.0, 0.0)
			pangolin.DrawCameras(self.data[0])
			if len(self.data) > 1:
				gl.glPointSize(5)
				points = self.data[1]
				colors = self.data[2]
				pangolin.DrawPoints(points, colors)
			
		pangolin.FinishFrame()

	def draw_3d(self, frames, points):
		if self.q is None:
			return

		poses = []
		pts = []
		colors = []
		for f in frames:
			# invert pose for display only
			pose = f.pose.P()
			pose = np.vstack((pose, np.zeros((1,4))))
			pose[-1, -1] = 1
			poses.append(np.array(np.linalg.inv(pose)))
		
		for p in points:
			pts.append(p.get_data())
			colors.append(p.get_color())
		# print(poses)
		print(np.stack(poses, axis=0).shape)
		if len(pts) == 0:
			self.q.put([np.stack(poses, axis=0)])
		else:
			self.q.put([np.stack(poses, axis=0), Dehomogenize(np.hstack(pts)).T, np.hstack(colors).T])

	# stereo testing
	def draw_cameras_points(self, cameras, points, colors):
		if self.q is None:
			return

		poses, pts = [], points
		for cam in cameras:
			Rt = cam
			Rt = np.vstack((Rt, np.zeros((1,4))))
			Rt[-1, -1] = 1
			poses.append(np.array(np.linalg.inv(Rt)))

		if len(pts) == 0:
			self.q.put([np.stack(poses, axis=0)])
		else:
			self.q.put([np.stack(poses, axis=0), Dehomogenize(np.hstack(pts)).T, np.hstack(colors).T])
