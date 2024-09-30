# from gpytoolbox import stochastic_poisson_surface_reconstruction
import gpytoolbox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_text_file(file_path):
  """Reads a text file with three floats per row, separated by spaces.

  Args:
    file_path: The path to the text file.

  Returns:
    A list of lists, where each inner list contains three floats.
  """
  
  data = []
  first = True
  with open(file_path, 'r') as f:
    for line in f:
      if first:
        first = False
        continue
      row = np.array([float(x) for x in line.split()])
      data.append(row)
  return np.array(data)


data = read_text_file("../data/capsule.txt")
P = data[:, :2]
N = data[:, 2:]
S = 65
scalar_mean, grid_vertices = gpytoolbox.stochastic_poisson_surface_reconstruction(P, N, gs=np.array([S,S]), h=np.array([1/64, 1/64]), corner=np.array([0, 0]))

print(len(scalar_mean))

with open('data.txt', 'w') as f:
    for row in scalar_mean:
      line = str(row) + '\n'
      f.write(line)

gx = grid_vertices[0]
gy = grid_vertices[1]

print(gx[0, 0])
print(gx[-1, -1])

print(gy[0, 0])
print(gy[-1, -1])

# Plot mean and variance side by side with colormap
fig, ax = plt.subplots(1,1)
m0 = ax.pcolormesh(gx,gy,np.reshape(scalar_mean,gx.shape), cmap='RdBu',shading='gouraud', vmin=-np.max(np.abs(scalar_mean)), vmax=np.max(np.abs(scalar_mean)))
ax.scatter(P[:,0],P[:,1],3 + 0*P[:,0])
#q0 = ax[0].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
ax.set_title('Mean')
# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(m0, cax=cax, orientation='vertical')

plt.show()