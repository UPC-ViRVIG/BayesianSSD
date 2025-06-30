import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, bicg, splu
import struct
import time


def load_sparse_matrix_binary(filename):
    """
    Reads a sparse matrix from a binary file and loads it into a csr_matrix.

    Args:
        filename (str): The path to the binary file containing the sparse matrix data.
                         The file should have the format (binary):
                         - num_rows (int)
                         - num_cols (int)
                         - num_non_zero (uint32)
                         - (row_index_1, col_index_1, value_1) (int, int, double)
                         - (row_index_2, col_index_2, value_2) (int, int, double)
                         - ...

    Returns:
        csr_matrix: The loaded sparse matrix in CSR format.
    """
    rows = []
    cols = []
    data = []
    with open(filename, "rb") as f:
        num_rows = struct.unpack("<i", f.read(4))[0]  # '<i' for little-endian integer
        num_cols = struct.unpack("<i", f.read(4))[0]
        num_non_zero = struct.unpack("<I", f.read(4))[0]  # '<I' for little-endian unsigned int

        print(f"{num_rows} {num_cols} {num_non_zero}")

        for _ in range(num_non_zero):
            row = struct.unpack("<i", f.read(4))[0]
            col = struct.unpack("<i", f.read(4))[0]
            value = struct.unpack("<d", f.read(8))[0]  # '<d' for little-endian double
            rows.append(row)
            cols.append(col)
            data.append(value)

    return csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))


def load_vector(filename):
    """
    Reads a sparse matrix from a binary file and loads it into a csr_matrix.

    Args:
        filename (str): The path to the binary file containing the sparse matrix data.
                         The file should have the format (binary):
                         - num_rows (int)
                         - num_cols (int)
                         - num_non_zero (uint32)
                         - (row_index_1, col_index_1, value_1) (int, int, double)
                         - (row_index_2, col_index_2, value_2) (int, int, double)
                         - ...

    Returns:
        csr_matrix: The loaded sparse matrix in CSR format.
    """
    data = []
    with open(filename, "rb") as f:
        num_val = struct.unpack("<i", f.read(4))[0]  # '<i' for little-endian integer
        print(f"{num_val}")

        for _ in range(num_val):
            value = struct.unpack("<d", f.read(8))[0]  # '<d' for little-endian double
            data.append(value)

    return np.array(data)


mat = load_sparse_matrix_binary("sparse_matrix.bin")
b = load_vector("vectorB.bin")

t0 = time.time()
mean_scalar, info = bicg(mat, b, atol=1e-10)
print(f"Time: {time.time() - t0}")
