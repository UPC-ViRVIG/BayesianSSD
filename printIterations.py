import numpy as np
import matplotlib.pyplot as plt

def read_array_from_file(filename, dtype=float):
  """
  Reads an array of the specified data type from a binary file.

  Args:
      filename (str): The name of the file to read from.
      dtype (dtype, optional): The data type of the elements in the array.
          Defaults to float.

  Returns:
      numpy.ndarray: The array read from the file.
  """

  with open(filename, "rb") as file:
    # Read the size of the array
    size = np.fromfile(file, dtype=int, count=1)[0]

    # Read the array elements
    data = np.fromfile(file, dtype=dtype, count=size)

  return data

P = 0.9

valueIter = read_array_from_file("solverIterValues.bin", np.float32)
stepIter = read_array_from_file("solverIterSteps.bin", np.float32)

n = int(P * valueIter.shape[0])

x = 10 * np.arange(n)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x, valueIter[-n:])
ax1.set_ylabel("Loss value")

ax2.plot(x, stepIter[-n:])
ax2.set_ylabel("Step value")

# Adjust layout to prevent clipping labelss
plt.tight_layout()

# Show the plot
plt.show()
