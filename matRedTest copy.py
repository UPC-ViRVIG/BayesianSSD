import os
import numpy as np
import json
import subprocess
import csv
import shutil
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

# PATH = "./config/normalsTest/"

# configFile = "./config/handCov.json"
# with open(configFile, "r") as f:
#     data = json.load(f)

# pid = 0

# for mode in [1, 2, 3, 0]:

for configFile in ["fox1cm0005.json", "fox1cm002.json", "fox1cm005.json"]:
    print("")
    print(f"{configFile}")
    process = subprocess.run(
        os.path.normpath(f"./build/Release/recon_3dv1.exe ./config/foxModel/{configFile} 5"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,  # Ensures output is treated as text (strings)
        shell=True,  # allows command to be a string.
    )
    print(process.stdout)
