import os
import numpy as np
import json
import subprocess
import csv

PATH = "./config/normalsTestFin/"


def read_array_from_file(filename):
    with open(filename, "rb") as file:
        # Read the size of the array
        size = np.fromfile(file, dtype=np.int32, count=1)[0]

        # Read the array elements
        data = np.fromfile(file, dtype=np.float32, count=size)
    return data


configFiles = []
for f in os.listdir(PATH):
    fpath = os.path.join(PATH, f)
    if os.path.isfile(fpath):
        configFiles.append(fpath)

res = [[] for _ in range(4 * 4)]
for i, cfile in enumerate(configFiles):
    with open(cfile, "r") as f:
        data = json.load(f)

    pid = 0
    # for k in [20, 35, 50, 65]:
    for k in [30, 60, 90]:
        for a, b in [(0.0, 0.0), (5.0, 0.0)]:
            # for a, b in [(0.0, 0.0), (0.5, 0.0), (0.0, 1.0), (0.5, 1.0)]:
            # for a, b in [(0.0, 1.0), (0.5, 1.0)]:
            data["normalsDistanceFactor"] = a
            data["normalsPointsCorrelation"] = b
            data["normalsNumNearPoints"] = k
            with open(cfile, "w") as f:
                json.dump(data, f, indent=4)

            process = subprocess.run(
                os.path.normpath(f"./build/Release/recon_3d.exe {cfile}"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # Ensures output is treated as text (strings)
                shell=True,  # allows command to be a string.
            )

            print(process.stderr)
            print(f"m:{data['outputName']} df: {a}, corr: {b}, K: {k}")
            lines = process.stdout.splitlines()
            if len(lines) < 4:
                print("ERROR")
                print(process.stdout)
                pid += 1
                continue
            print(lines[1])
            print(f"Time: {lines[2]}")
            print(f"Angle error: {lines[3]}")
            xye = read_array_from_file(f"./output/stats/{data['outputName']}_xyErrors.bin")
            print(f"Mean xy: {np.nanmean(xye)} std xy: {np.nanstd(xye)}")

            res[pid].append([data["outputName"], lines[2], lines[3], np.nanstd(xye), lines[1]])
            pid += 1

with open("./normalsTest.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    pid = 0
    # for k in [20, 35, 50, 65]:
    for k in [30, 60, 90]:
        for a, b in [(0.0, 0.0), (5.0, 0.0)]:
            writer.writerow(["df", a, "coeff", b, "K", k])
            writer.writerows(res[pid])
            writer.writerow([])
            writer.writerow([])
            pid += 1

with open("./normalsSummary.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["df", "corr", "K", "noise_level", ""])
    pid = 0
    # for k in [20, 35, 50, 65]:
    for k in [30, 60, 90]:
        for a, b in [(0.0, 0.0), (5.0, 0.0)]:
            # for noise_level in ["0.005", "0.0005"]:
            # for noise_level in ["0.001", "0.0001"]:
            for noise_level in ["0.032", "0.016", "0.008", "0.004", "0.002", "0.001"]:
                # for noise_level in ["0.002", "0.001", "0.0005", "0.00025", "0.000125", "6.25e-05"]:
                res_filtred = list(filter(lambda arr: arr[0].split("_")[2] == noise_level, res[pid]))
                res_filtred = np.array(
                    list(map(lambda arr: (float(arr[1]), float(arr[2]), arr[3]), res_filtred))
                )
                if res_filtred.shape[0] == 0:
                    continue
                writer.writerow(
                    [
                        a,
                        b,
                        k,
                        noise_level,
                        np.mean(res_filtred[:, 0]),
                        np.mean(res_filtred[:, 1]),
                        np.mean(res_filtred[:, 2]),
                    ]
                )
            pid += 1
