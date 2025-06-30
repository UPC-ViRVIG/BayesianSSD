import math
import random
import pathlib
import numpy as np

# dx = np.linspace(-2.0, 2.0, 100)
# std = 0.7
# dy = 1.6 * np.exp(-0.5 * dx * dx / (std * std))
# pData = []
# for i in range(dx.shape[0]):
#     pData.append((dx[i], dy[i]))
# polyline(pData)

dx = np.linspace(0.0, 0.99, 100)
std = 0.7
dy = 0.5 * 0.01 / (-np.sqrt(0.985 * dx) + 1.0)
pData = []
for i in range(dx.shape[0]):
    pData.append((dx[i], dy[i]))
pData.append((0.99, 0.0))
polygon(pData)
# polyline(pData)


grid_size = np.array([17, 17])
grid_spacing = np.array([2.44089296, 2.44089296])
minB = np.array([20.11188547, 73.08231803])
sizeB = grid_spacing * (grid_size - 1)

# Draw grid
# grid = group()
# for i in range(grid_size[0]):
#     off = i * grid_spacing
#     grid.append(line(minB + np.array([off, 0.0]), minB + np.array([off, sizeB[1]])))

# for j in range(grid_size[1]):
#     off = j * grid_spacing
#     grid.append(line(minB + np.array([0.0, off]), minB + np.array([sizeB[0], off])))

# Draw vector field
# mean_normals = np.load("C:/Users/user/Documents/reconstruction/test/myVectors.npy")
# arr = mean_normals.astype("float64")
# print(mean_normals.shape)
# points = group()
# normals = group()
# for j in range(grid_size[1]):
#     for i in range(grid_size[0]):
#         p = minB + grid_spacing[0] * np.array([i, j])
#         p = np.array([float(p[0]), float(p[1])])
#         gx = float(mean_normals[j * grid_size[0] + i + 0, 0])
#         gy = float(mean_normals[j * grid_size[0] + i + 1 * grid_size[0] * grid_size[1], 0])
#         g = np.array([gx, gy])
#         points.append(circle(p, 0.001))
#         normals.append(line(tuple(p), tuple(p + g * 2.5)))
#         # normals.append(line(p, p + 0.05 * np.array([1.0, 0.0])))
