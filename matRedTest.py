import os
import numpy as np
import json
import subprocess
import csv
import shutil

# import matplotlib.pyplot as plt
import numpy as np

# import scipy.sparse as sparse


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


# fig, axs = plt.subplots(nrows=1, ncols=7)

# invGT = read_matrix_from_file("./mats/invSVDmat.bin")
# invGT = np.sqrt(np.maximum(invGT, 0.0))
# for i, k in enumerate([128, 256, 512, 1024, 2048, 4096, 0]):
#     for mode in [1]:
#         if k == 0:
#             inv = np.diag(invGT)
#         else:
#             inv = np.diag(
#                 np.sqrt(np.maximum(read_matrix_from_file(f"./mats/invSVDmat_M{mode}_K{k}.bin"), 0.0))
#             )
#         print(i)
#         axs[i].hist(inv, bins=512)
#         axs[i].set_xlim(0, np.max(np.diag(invGT)))
#         axs[i].set_ylim(0, 400)

# PATH = "./config/normalsTest/"

configFile = "./config/handCov.json"
with open(configFile, "r") as f:
    data = json.load(f)

pid = 0

for mode in [1, 2, 3, 0]:
    # for mode in [2]:
    for k in [128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096]:
        data["invRedMatRank"] = k
        with open(configFile, "w") as f:
            json.dump(data, f, indent=4)

        print("")
        print(f"Executing for K={k} mode={mode}")
        process = subprocess.run(
            os.path.normpath(f"./build/Release/recon_3d.exe {configFile} {mode}"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,  # Ensures output is treated as text (strings)
            shell=True,  # allows command to be a string.
        )

        print(process.stdout)

        # shutil.move("./invSVDmat.bin", f"./mats/invSVDmat_M4_K{k}.bin")


print("")
print("Other")
process = subprocess.run(
    os.path.normpath(f"./build/Release/recon_3d.exe ./config/handCov1.json 0"),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,  # Ensures output is treated as text (strings)
    shell=True,  # allows command to be a string.
)
print(process.stdout)
