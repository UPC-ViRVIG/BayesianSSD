# BayesianSSD
Official Code for SGP 2025 paper [Bayesian 3D Shape Reconstruction from Noisy Points and Normals](https://doi.org/10.1111/cgf.70201).

BayesianSSD is a reconstruction algorithm from noisy point clouds. The method integrates the concept of uncertainty in the input and output. The inputs are a set of points with their corresponding isotropic variances. Optionally, the user can also specify the normals and their corresponding variances. Otherwise, they are computed using a Bayesian PCA. Even when the PCA computes the normals, it is important to give an approximated normal to specify its orientation.

![Intro Image](https://github.com/user-attachments/assets/10f90163-f18e-4dcb-a7d8-91cc0ad7d4be)

## Compiling

All the project dependencies are downloaded automatically during the CMake execution.

## How to use

The project has two main targets: `recon_2d` and `recon_3d`. 

Both targets use the same configuration files. The config file is a JSON that contains all the important information for a reconstruction. The configurable fields are:

- `pointCloudName`: The file name of the point cloud. The program will look at the folder `{CurrentPath}/data`.
- `outputName`: The file name of the reconstruction outputs. All the outputs will be written in the folder `{CurrentPath}/output`.
- `bbMargin`: Margin between the models BB and the octree BB. The margin is expressed regarding the size of the model BB, e.g., a value of 0.2 is a 10% margin on each side. 
- `octreeMaxDepth`: The octree maximum depth.
- `octreeSubRuleInVoxels`: Specifies the subdivision rule. This value defines the minimum distance that a node has to be from any point to stop being subdivided. The magnitude of the values is expressed in terms of the voxel length at maximum depth.
- `gradientStd`: Multiplicative factor applied to the gradient standard deviation. This value helps control the values of the PCA result when the final field has a gradient that is too small or too large.
- `smoothnessStd`: This value is the standard deviation of the field curvature prior.
- `computeVariance`: Boolean that specifies whether the algorithm should compute the output variances.
- `inverseMethod`: Specifies the method to invert the matrix required to compute the variance of the solution. It can have three options: `"full"`, `"octree_red"`, `"base_red"`. With `"full"`, the inverse of the matrix is computed without any approximation (can be expensive for big problems). With `"octree_red"`, the variances are calculated by simplifying the octree (using an octree of lower depth to compute the variance). Finally, with `"base_red"`, the inverse is calculated using a low rank approximation using the Laplacian eigenvalues to get the more meaningful values.
- `baseRedRank`: When the option `"base_red"` is selected, this value specifies the rank of the matrix to invert.
- `octreeRedDepth`: When the option `"octree_red"` is selected, this value is the maximum depth of the simplified octree used to compute the variances.
- `mulPointsStd`: Multiplicative factor applied to the standard deviation of all the points.
- `computeNormals`: Boolean specifying if the normals should be computed using the Bayesian PCA.
- `normalsNumNearPoints`: Number of neighbour points to use for the Bayesian PCA.
- `normalsDistanceFactor`: The noise factor added to the normals computations (beta value in the paper).
- `defaultNormalsStd`: Default normals variance value when Bayesian PCA is not used.

Example:
```
{
    "pointCloudName": "MyHand",
    "outputName": "MyHand",
    "bbMargin": 0.1,
    "octreeMaxDepth": 8,
    "octreeSubRuleInVoxels": 1.66,
    "gradientStd": 2.0,
    "smoothnessStd": 0.004,
    "computeVariance": true,
    "inverseMethod": "octree_red",
    "baseRedRank": 1024,
    "octreeRedDepth": 5,
    "mulPointsStd": 1.0,
    "computeNormals": true,
    "normalsNumNearPoints": 20,
    "normalsDistanceFactor": 0.5,
    "defaultNormalsStd": 0.2
}
```

Next, we have a description of the inputs and outputs of each one.

### 2D Reconstruction

The target `recon_2d` makes a 2D reconstruction. The inputs are a point cloud specified in text format and the configuration file. The outputs are PNG images of the reconstruction showing different properties.

The executable expects the path of the JSON config file as an input argument:

```
{PathToExecutable}/recon_2d.exe ./path/to/config.json
```

##### Point cloud description

In the 2D version, the input point cloud is encoded in a .txt file. The first row has a number specifying the number of points. The following line defines the properties of each point (each property is separated by spaces). Next, we have a format example where the tags between parentheses should be replaced by the corresponding values.

```
[Number of points]
[Position x] [Position y] [Normal x] [Normal y] [Variance]
...
```
Example:
```
1
69.3 30.4 -0.9 -0.43 2.4
```

These files should be placed inside the **data** folder.

##### Outputs

In the 2D case, the outputs of the method are four PNG images when the variance computation is enabled, which are:

- `{CurrentPath}/output/[outputName]_mu.png`: Image showing the final field corresponding to the mean values.
- `{CurrentPath}/output/[outputName]_pIn.png`: Image showing the probability of being inside or outside using the viridis palette.
- `{CurrentPath}/output/[outputName]_pSur.png`: Image showing the confidence of the surface being there using the magma palette.
- `{CurrentPath}/output/[outputName]_std.png`: The output variances using the viridis palette.

### 3D Reconstruction

The target `recon_3d` makes a 3D reconstruction. The inputs are a point cloud specified as a PLY with some custom properties and the configuration file. The outputs are a PLY of the final mesh and two binary files that store the reconstructed field in a custom format. These binary files can be used in the 3D viewer (explained below) to observe different properties.

The executable expects the path of the JSON config file as an input argument:

```
{PathToExecutable}/recon_3d.exe ./path/to/config.json
```

##### Point cloud description
In the 3D version, the input point clouds are expected in PLY files. Each point, represented as a vertex in the PLY, must contain seven properties:

- `x`, `y`, `z`: The position of the point.
- `nx`, `ny`, `nz`: The normal of the point. If the normals are computed using PLY, this field is still necessary to compute the orientation of the final solution.
- `noise_std`: The points' position standard deviation. The variance is only a scalar value because we only work with isotropic Gaussians for the positions.

##### Outputs
The outputs of the model are a mesh and two binaries. The mesh is a PLY file with path `{CurrentPath}/output/[outputName].ply`. The vertices of the mesh have a special property called `noise_std`, which represents the standard deviation of the solution at that point. 

The two binary files `{CurrentPath}/output/[outputName].bin` and `{CurrentPath}/output/[outputName]_var.bin` contain the values of the final octree reconstruction. These two results can be visualized using the target `Viewer`. The viewer expects the output name specified in the configuration file:

```
{PathToExecutable}/Viewer.exe [outputName]
```

It is important to execute both programs in the same directory; otherwise, the viewer will not find the output folder with the results.

## Limitations

As mentioned in the paper's conclusions, there is a relationship between the input parameters. If we are forcing a high smoothness of the field and the normals have a high variance, then the final field gradient might not have magnitude one near the points. Also, if the variance of the gradient is too low, this can produce instabilities in the system, which will result in more solver iterations and less accurate results. Both reconstruction executables print a `Mean gradient magnitude at the points: 0.96`, which can be used to diagnose these two cases. This value should be between [0.9, 0.98]. This value can be changed by playing with the configuration parameter `gradientStd` without affecting the resulting uncertainty critically. 

TODO: An interesting addition to the algorithm could be selecting this value automatically.




