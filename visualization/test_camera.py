# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/HelloPangolin

import sys

sys.path.append('../lib')

import OpenGL.GL as gl
import pangolin

import numpy as np
import time


def main():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)
    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(2, 2, 2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)

    pose = np.identity(4)
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)


        pose[2, 3] = pose[2, 3] + 1
        gl.glLineWidth(5)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCamera(pose, 1, 0.75, 0.8)
        time.sleep(0.2) 
        pangolin.FinishFrame()

if __name__ == '__main__':
    main()
