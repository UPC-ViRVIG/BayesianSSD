# BayesianSSD
Official Code for SGP 2025 paper [Bayesian 3D Shape Reconstruction from Noisy Points and Normals](https://doi.org/10.1111/cgf.70201).

BayesianSSD is a reconstruction algorithm from noisy point clouds. The method integrates the concept of uncertainty in the input and output. The inputs are a set of points with their corresponding isotropic variances. Optionally, the user can also specify the normals and their corresponding variances. Otherwise, they are computed using a Bayesian PCA. Even when the PCA computes the normals, it is important to give an approximated normal to specify its orientation.

![Intro Image](image.png)

## Compiling

All the project dependencies are downloaded automatically during the CMake execution.

## How to use

The project has two main targets: `recon_2d` and `recon_3d`. Next, we have a description of each one.

### 2D Reconstruction

The target `recon_2d` makes a 2D reconstruction. The inputs are a point cloud specified in text format and a configuration JSON file. The outputs are PNG images of the reconstruction showing different properties.

##### Point cloud description


##### Reconstruction config


##### Outputs

### 3D Reconstruction

The target `recon_3d` makes a 3D reconstruction. The inputs are a point cloud specified as a PLY with some custom properties and a configuration JSON file. The outputs are a PLY of the final mesh and two binary files that store the reconstructed field in a custom format. These binary files can be used in the 3D viewer (explained below) to observe different properties.

##### Point cloud description

##### Reconstruction config

##### Outputs

### 3D Viewer






