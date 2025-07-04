# from gpytoolbox import stochastic_poisson_surface_reconstruction
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import scipy
import gpytoolbox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# def read_matrix_from_file(filename):
#     """Reads a matrix from a binary file written by the C++ code."""
#     try:
#         with open(filename, "rb") as f:
#             rows = np.fromfile(f, dtype=np.int32, count=1)[0]
#             cols = np.fromfile(f, dtype=np.int32, count=1)[0]
#             matrix = np.fromfile(f, dtype=np.float64, count=rows * cols).reshape(cols, rows)
#             return matrix.T
#     except FileNotFoundError:
#         print(f"Error: File not found: {filename}")
#         return None
#     except Exception as e:  # Catch other potential errors during file reading
#         print(f"An error occurred: {e}")
#         return None


# mat = -read_matrix_from_file("../PSRtestSvalue.bin")
# final = np.empty((mat.shape[0], mat.shape[1], 3))
# final[..., 0] = mat
# final[..., 1] = mat
# final[..., 2] = mat
# print(final.shape)
# final = final.astype(np.float32)
# cv2.imwrite("./grayScaleImage2SSD.exr", final)
# exit()


def read_text_file(file_path):
    """Reads a text file with three floats per row, separated by spaces.

    Args:
      file_path: The path to the text file.

    Returns:
      A list of lists, where each inner list contains three floats.
    """

    data = []
    first = True
    with open(file_path, "r") as f:
        for line in f:
            if first:
                first = False
                continue
            row = np.array([float(x) for x in line.split()])
            data.append(row)
    return np.array(data)


# data = read_text_file("../data/myshape.txt")
data = read_text_file("../data/PSRtest.txt")
P = data[:, :2]
N = -data[:, 2:4]
DEPTH = 6
# DEPTH = 4
IMAGE_SIZE = 2056

min = np.min(P, axis=0)
max = np.max(P, axis=0)
max_size = np.max(max - min)
center = 0.5 * (min + max)

minB = center - np.ones(center.shape) * 0.5 * 1.5 * max_size
maxB = center + np.ones(center.shape) * 0.5 * 1.5 * max_size

grid_size = np.ones(center.shape) * (np.power(2, DEPTH) + 1)

grid_spacing = (maxB - minB) / (grid_size - 1)
print(grid_size)
print(grid_spacing)
print(minB)


scalar_mean, scalar_variance, grid_vertices = gpytoolbox.stochastic_poisson_surface_reconstruction(
    P,
    -N,
    gs=grid_size.astype(np.int32),
    h=grid_spacing,
    corner=minB,
    output_variance=True,
    sigma_n=0.3,
    sigma=0.6,
)

# mean_values = gpytoolbox.stochastic_poisson_surface_reconstruction(
#     P,
#     N,
#     gs=grid_size.astype(np.int32),
#     h=grid_spacing,
#     corner=minB,
#     output_variance=True,
#     sigma_n=0.2,
#     sigma=0.8,
# )

# np.save("./myVectors", mean_values)


scalar_mean = np.reshape(scalar_mean, grid_vertices[0].shape)
scalar_variance = np.reshape(scalar_variance, grid_vertices[0].shape)

image_range = np.power(2, DEPTH) * np.arange(0.0, 1.0, 1 / IMAGE_SIZE)
xs, ys = np.meshgrid(image_range, image_range)
fxs = np.floor(xs).astype(np.int32)
fys = np.floor(ys).astype(np.int32)

rxs = xs - fxs
rys = ys - fys


def getImage(scalars):
    p00 = scalars[fys, fxs]
    p01 = scalars[fys, fxs + 1]
    p10 = scalars[fys + 1, fxs]
    p11 = scalars[fys + 1, fxs + 1]

    d0 = p00 * (1.0 - rxs) + p01 * rxs
    d1 = p10 * (1.0 - rxs) + p11 * rxs

    return d0 * (1.0 - rys) + d1 * rys


image_mean = getImage(scalar_mean)
image_variance = getImage(scalar_variance)

image_shape = image_mean.shape
image_mean = np.ndarray.flatten(image_mean)
image_variance = np.ndarray.flatten(image_variance)

surface_density = scipy.stats.norm.pdf(image_mean, 0.0, np.sqrt(image_variance))
inside_prob = scipy.stats.norm.cdf(image_mean, 0.0, np.sqrt(image_variance))

surface_density = np.reshape(surface_density, image_shape)
inside_prob = np.reshape(inside_prob, image_shape)

viridisPalette = np.array(
    [[68, 1, 84], [65, 68, 135], [42, 120, 142], [34, 163, 132], [122, 209, 81], [253, 231, 37]]
).astype(np.float32)

tmp = np.copy(viridisPalette[:, 0])
viridisPalette[:, 0] = viridisPalette[:, 2]
viridisPalette[:, 2] = tmp


def getColorImage(img, palette, vmin, vmax):
    norm_img = (img - vmin) / (vmax - vmin)
    idx_img = np.minimum(np.maximum((palette.shape[0] - 1.0) * norm_img, 0.0), palette.shape[0] - 1.001)
    t_img = np.modf(idx_img)[0]
    i_img = np.floor(idx_img).astype(np.int32)
    final = palette[i_img]
    col1 = palette[i_img]
    col2 = palette[i_img + 1]
    final[..., 0] = col1[..., 0] * (1.0 - t_img) + col2[..., 0] * t_img
    final[..., 1] = col1[..., 1] * (1.0 - t_img) + col2[..., 1] * t_img
    final[..., 2] = col1[..., 2] * (1.0 - t_img) + col2[..., 2] * t_img
    return final


grayPalette = np.array([[0, 0, 0], [255, 255, 255]]).astype(np.float32)

# print(np.min(image_mean))
# print(np.max(image_mean))
# image_mean = np.reshape(image_mean, image_shape)
# final = np.empty((image_shape[0], image_shape[1], 3))
# final[..., 0] = image_mean
# final[..., 1] = image_mean
# final[..., 2] = image_mean
# print(final.shape)
# final = final.astype(np.float32)
# cv2.imwrite("./grayScaleImage.exr", final)

cv2.imwrite(
    "./quadtreePSurface.png",
    getColorImage(surface_density, viridisPalette, 0.0, 1.0).astype(np.uint8),
)
cv2.imwrite(
    "./quadtreePInside.png",
    getColorImage(inside_prob, viridisPalette, 0.0, 1.0).astype(np.uint8),
)

# with open("data.txt", "w") as f:
#     for row in scalar_mean:
#         line = str(row) + "\n"
#         f.write(line)

# gx = grid_vertices[0]
# gy = grid_vertices[1]

# print(gx[0, 0])
# print(gx[-1, -1])

# print(gy[0, 0])
# print(gy[-1, -1])

# # Plot mean and variance side by side with colormap
# fig, ax = plt.subplots(1, 1)
# m0 = ax.pcolormesh(
#     gx,
#     gy,
#     np.reshape(scalar_mean, gx.shape),
#     cmap="RdBu",
#     shading="gouraud",
#     vmin=-np.max(np.abs(scalar_mean)),
#     vmax=np.max(np.abs(scalar_mean)),
# )
# ax.scatter(P[:, 0], P[:, 1], 3 + 0 * P[:, 0])
# # q0 = ax[0].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
# ax.set_title("Mean")
# # divider = make_axes_locatable(ax[0])
# # cax = divider.append_axes('right', size='5%', pad=0.05)
# # fig.colorbar(m0, cax=cax, orientation='vertical')

# plt.show()
