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

values = read_array_from_file("eigenValues.bin", np.float32)
values = np.sort(np.abs(values))


# plt.scatter(np.arange(values.shape[0]), values, s=10)
plt.scatter(np.arange(values.shape[0]), np.log(values), s=10)
# plt.scatter(np.arange(values[:100].shape[0]), values[:100], s=10)
plt.show()
