Inspired by George Hotz's twitch slam

Personal implementation of SLAM, hopefully I can do better...

# Feature Extraction
- GoodFeatureToTrack for Keypoint
- ORB for Descriptor
- BFMatcher for Feature Matching

Controller
- DLT, Triangulation
- Bundle Adjustment, g2o

Model
- Frame -> isKeyframe [features, pose]
- 2D / 3D Features [poses], Camera Pose [features]

View
- Display