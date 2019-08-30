Inspired by George Hotz's twitch slam

Personal implementation of SLAM, hopefully I can implement some design principles to make it more elegant...

# Feature Extraction
- GoodFeatureToTrack for Keypoint
- ORB for Descriptor
- BFMatcher for Feature Matching

# Basic Design

*Controller*
- DLT, Triangulation
- Finsterwalder Solution for 3D absolute pose estimation
- Bundle Adjustment, g2o

*View*
- 2D OpenCV Display
- 3D PyGame Reconstruction Display

*Model*
- Frame -> isKeyframe [(one-to-many)features, (one-to-one)pose]
- 2D / 3D Features [poses], Camera Pose [features]

# Progress

## First Phase:

* Build Data Structure for Point and Pose Maps. Use Dummy Imagining Geometry
* Use RANSAC outline rejections for initial matching.
* DLT for absolute pose estimation and fundamental matrix estimation
* Complete linear triangulation with essential matrix decomposition for creating 3D feature points
* Bundle Adjustment

## Second Phase:

* RANSAC for absolute pose estimation
* Add Keyframes and Loop Closure
* 3D Visualization
* Essential Matrix

# Library
[Pangolin (for 3D Visualization Library)]

[G2O (for bundle adjustment)](https://github.com/RainerKuemmerle/g2o)

*MAC build problem for g2opy:*

[change "python/core/hyper\_graph.h"](https://github.com/uoip/g2opy/issues/1)

line 71: "id"\_a=HyperGraph::UnassignedId -> "id"\_a=-1

line 82: "id"\_a=HyperGraph::InvalidId -> "id"\_a=-2

Also make sure to manually install Eigen version=3.3.4, it will generated old DataType declaration files in the /usr/lib/eigen3/Unsupported/... which will solve compilation errors caused by g2o

compiled libraries of g2o, pangolin are saved in slam/libs, they are for Python 3.7

- slam/lib/g2o.cpython-37m-darwin.so
- slam/lib/pangolin.cpython-37m-darwin.so

# ScreenShot
![Screenshot](./screenshots/screen.png "Feature Extraction and Matching")
