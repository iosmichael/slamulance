# Usage of Library g2o

## Important Data Container Classes:
- g2o.SE3Quant  # class that wraps Rotation and Translation Matrices to ensure their properties
- g2o.CameraParameters # class that holds the focal length and principal point of the camera parameters

Usage of Data Containers:

```
pose = g2o.SE3Quat(R, t)
camera = g2o.CameraParameters(focal_length, principal_point, baseline)
optimizer.add_parameter(camera)
```


## Main graph initialization:

```python
optimizer = g2o.SparseOptimizer()
solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
solver = g2o.OptimizationAlgorithmLevenberg(solver)
optimizer.set_algorithm(solver)
```

Essentially different solvers that utilizes different special properties of the inverting matrix. Here we initialize the optimizer using levenberg-marquadt optimization.

## Vertex
- g2o.VertexSBAPointXYZ  # class that holds 3-d inhomogeneous points as vertices in the optimization graph, *reconstructed points in our case*
- g2o.VertexSE3Expmap    # class that wraps the camera pose as vertex in the optimization graph

Usage of Vertex: (Putting points and poses into the graph)
```python
# initialization of points
v_point = g2o.VertexSBAPointXYZ()
v_point.set_id(point_id) # use for query vertex from the optimization graph, optimizer.vertex(point_id)
v_point.set_marginalized(True)
v_point.set_estimate(point) # set initial 3d np.array data
optimizer.add_vertex(v_point)
```

```python
v_se3 = g2o.VertexSE3Expmap()
v_se3.set_id(i)
v_se3.set_estimate(pose)
optimizer.add_vertex(v_se3)
```

## Edge
- g2o.EdgeProjectXYZ2UV   # class that builds an edge connecting 3d inhomogeneous point to camera poses

```python
edge = g2o.EdgeProjectXYZ2UV()
# first vertex is XYZ vertex
edge.set_vertex(0, v_point)
# second vertex is Camera Pose vertex
edge.set_vertex(1, v_se3)
# reprojection is calculated inside the edge, now we need to set measurement points, which are in our case, 2d ground-truth image coordinates
edge.set_measurement(gt_point) # gt_point is np.array(), with shape (2, 1)
edge.set_information(np.identity(2)) # covariance of the points, assume to be identity in this case
if optional:
    edge.set_robust_kernel(g2o.RobustKernelHuber())
edge.set_parameter_id(0, 0)
optimizer.add_edge(edge)
```

## G2O Graph (Optimizer)
**Run the Graph for Optimization**

```python
optimizer.initialize_optimization()
optimizer.set_verbose(True)
optimizer.optimize(10) # run 10 iterations
```

**To Get Optimized Vertices**
```python
# To get vertices and edges
print(optimizer.vertices())
print(optimizer.edges())
# To obtain optimized information from the graph at vertex i
print(optimizer.vertex(i).estimate())
```

