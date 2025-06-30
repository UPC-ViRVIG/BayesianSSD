import math
import random
import pathlib
import numpy as np

IN_FILE = "C:/Users/user/Documents/reconstruction/output/initFig_input.txt"
NUM_PROPS = 7
NORM_LENGTH = 6.0


def normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


with open(IN_FILE, "r") as file:
    content = file.read()
    numbers = np.array([float(num) for num in content.split()])
    numPoints = int(numbers[0])
    style(stroke_width=0.5)
    style(fill="#ffc427")
    varObj = group()
    pointsObj = group()
    normals1 = group()
    normals2 = group()
    style(stroke_width=0.0)
    for n in range(numPoints):
        pInfo = numbers[1 + n * NUM_PROPS : 1 + (n + 1) * NUM_PROPS]
        pos = pInfo[:2]
        grad = pInfo[2:4]
        var = pInfo[4]
        gradVar = pInfo[5:7]
        # grad = radial_gradient(pos, 4.5 * np.sqrt(var))
        # for i in range(10 + 1):
        #     x = i / 10
        #     stdX = 0.39
        #     a = 0.85 * np.exp(-(x**2) / (2 * stdX**2)) / (np.sqrt(2 * np.pi) * stdX)
        #     grad.add_stop(i / 10, "#ffa217", a)
        # grad.add_stop(1, "#ffa217", 0)
        # varObj.append(circle(pos, 4.5 * np.sqrt(var), fill=grad))
        push_defaults()
        style(fill="#fc3d3d")
        style(stroke_width=0.0)
        pointsObj.append(circle(pos, 0.45))
        pop_defaults()

        # Print normals
        push_defaults()
        style(stroke_width=0.35)
        style(stroke="#238845")
        normals1.append(line(pos, pos + grad * NORM_LENGTH))
        stdX = np.sqrt(gradVar[0])
        gTan = np.array([-grad[1], grad[0]])
        style(stroke_width=0.2)
        if stdX > 1:
            continue
        N = int(1 + stdX / 0.5 * 6)
        style(stroke="#74C476")
        for i in range(-N, N + 1):
            x = i / N * 2 * stdX
            if x == 0:
                continue
            x += np.random.normal(scale=0.1 * 2 * stdX / N)
            if x >= 0.99999 or x <= -0.99999:
                continue
            a = 0.1 + 0.75 * np.exp(-(x**2) / (2 * stdX**2)) / (np.sqrt(2 * np.pi) * stdX)
            style(opacity=a)
            pGrad = gTan * x + grad * np.sqrt(1 - x**2)
            normals2.append(line(pos, pos + pGrad * NORM_LENGTH))
        pop_defaults()
