# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/HelloPangolin

import sys

sys.path.append('../lib')

import OpenGL.GL as gl
import pangolin

import numpy as np
import time


def main():
	w, h = 640, 480
	pangolin.CreateWindowAndBind('Main', 640, 480)
	gl.glEnable(gl.GL_DEPTH_TEST)

	# Create Interactive View in window
	scam = pangolin.OpenGlRenderState(
	  pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
	  pangolin.ModelViewLookAt(0, -10, -8,
							   0, 0, 0,
							   0, -1, 0))
	handler = pangolin.Handler3D(scam)

	# Create Interactive View in window
	dcam = pangolin.CreateDisplay()
	dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
	dcam.SetHandler(handler)
	# hack to avoid small Pangolin, no idea why it's *2
	dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
	dcam.Activate()

	poses = []
	# pose = np.hstack((np.identity(3), np.zeros((3,1))))
	pose = np.identity(4)
	poses.append(np.linalg.inv(pose))
	while not pangolin.ShouldQuit():
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glClearColor(0.0, 0.0, 0.0, 1.0)
		dcam.Activate(scam)

		pose[2, 3] = pose[2, 3] - 1
		poses.append(np.linalg.inv(pose))

		print(poses[-1])

		gl.glLineWidth(3)
		gl.glColor3f(0.0, 1.0, 0.0)
		pangolin.DrawCameras(poses)
		# pangolin.DrawCamera(pose)
		time.sleep(0.2) 
		# pangolin.DrawCameras(np.linalg.inv(poses[-1]))
		# pangolin.DrawCameras(np.stack(poses, axis=0))
		# print(np.stack(poses,axis=2).shape)
		pangolin.FinishFrame()

if __name__ == '__main__':
	main()
