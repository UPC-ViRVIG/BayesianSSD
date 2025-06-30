import math
import random
import pathlib
import numpy as np

SAMPLE_POINTS = 40
# POINTS_STD = 0.8
POINTS_STD = 0.008
OUT_FILE = "C:/Users/user/Documents/reconstruction/data/SGPTextTest.txt"
SAMPLE_RANDOM = False


def sum(a, b):
    return [a[0] + b[0], a[1] + b[1]]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1]]


def mul(a, b):
    return [a[0] * b[0], a[1] * b[1]]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def dist(a, b):
    l = sub(a, b)
    return math.sqrt(l[0] * l[0] + l[1] * l[1])


def norm(a):
    l = dist(a, [0, 0])
    if abs(l) < 1e-8:
        return [0, 0]
    return [a[0] / l, a[1] / l]


def store_points(points):
    f = open(OUT_FILE, "w")
    f.write(str(len(points)))
    f.write("\n")
    for p in points:
        first = True
        for v in p:
            f.write(("" if first else " ") + str(v))
            first = False
        f.write("\n")
    f.close()


class BezierSubPath:
    def __init__(self, p0, p1, p2, p3):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.sNum = 20
        prevPoint = self.p0
        self.sLen = []
        self.length = 0
        for i in range(1, self.sNum + 1):
            t = i / self.sNum
            nP = self.eval(t)
            self.length += dist(prevPoint, nP)
            self.sLen.append(self.length)
            prevPoint = nP

        self.startPoint = self.p0
        self.endPoint = self.p3

    def eval(self, t):
        mt = 1 - t
        c0 = mt * mt * mt
        c1 = 3 * t * mt * mt
        c2 = 3 * t * t * mt
        c3 = t * t * t
        return [
            c0 * self.p0[0] + c1 * self.p1[0] + c2 * self.p2[0] + c3 * self.p3[0],
            c0 * self.p0[1] + c1 * self.p1[1] + c2 * self.p2[1] + c3 * self.p3[1],
        ]

    def evalGrad(self, t):
        mt = 1 - t
        c0 = 3 * mt * mt
        c1 = 6 * t * mt
        c2 = 3 * t * t
        return [
            c0 * (self.p1[0] - self.p0[0]) + c1 * (self.p2[0] - self.p1[0]) + c2 * (self.p3[0] - self.p2[0]),
            c0 * (self.p1[1] - self.p0[1]) + c1 * (self.p2[1] - self.p1[1]) + c2 * (self.p3[1] - self.p2[1]),
        ]

    def tangent(self, t):
        grad = self.evalGrad(t)
        return [grad[1], -grad[0]]

    def sample(self, s):
        s *= self.length
        index = 0
        lastLen = 0
        while index < self.sNum and s > self.sLen[index]:
            lastLen = self.sLen[index]
            index += 1

        s = (s - lastLen) / (self.sLen[index] - lastLen)

        t = (1 - s) * index / self.sNum + s * (index + 1) / self.sNum

        return self.eval(t), self.tangent(t)


class ArcPath:
    def __init__(self, p0, rx, ry, alpha, fa, fs, p1):
        alpha = math.radians(alpha)
        self.alpha = alpha
        a = list(map(lambda x: 0.5 * x, sub(p0, p1)))
        a = [dot(a, [math.cos(alpha), math.sin(alpha)]), dot(a, [-math.sin(alpha), math.cos(alpha)])]
        delta = a[0] * a[0] / (rx * rx) + a[1] * a[1] / (ry * ry)
        if delta > 1 + 1e-9:
            sqrtDelta = math.sqrt(delta)
            rx = sqrtDelta * rx
            ry = sqrtDelta * ry
        self.rx = rx
        self.ry = ry
        rx2 = rx * rx
        ry2 = ry * ry
        ax2 = a[0] * a[0]
        ay2 = a[1] * a[1]
        v = math.sqrt((rx2 * ry2 - rx2 * ay2 - ry2 * ax2) / (rx2 * ay2 + ry2 * ax2))
        if fa == fs:
            v = -v
        c = [v * rx * a[1] / ry, -v * ry * a[0] / rx]
        self.center = [
            dot(c, [math.cos(alpha), -math.sin(alpha)]),
            dot(c, [math.sin(alpha), math.cos(alpha)]),
        ]
        self.center = sum(self.center, list(map(lambda x: 0.5 * x, sum(p0, p1))))
        u = [1, 0]
        v = mul(sub(a, c), [1.0 / rx, 1.0 / ry])

        def f(u, v):
            val = math.sqrt(dot(u, u)) * math.sqrt(dot(v, v))
            val = math.acos(dot(u, v) / val)
            if u[0] * v[1] - u[1] * v[0] < 0:
                val = -val
            return val

        self.startAngle = f(u, v)
        u = v
        v = mul(sub([-a[0], -a[1]], c), [1.0 / rx, 1.0 / ry])
        self.diffAngle = f(u, v) % math.radians(360.0)
        if fs == 0 and self.diffAngle > 0:
            self.diffAngle = self.diffAngle - math.radians(360.0)
        elif fs == 1 and self.diffAngle < 0:
            self.diffAngle = self.diffAngle + math.radians(360.0)

        self.sNum = 20
        self.sLen = []
        self.length = 0
        prevPoint = p0
        for i in range(1, self.sNum + 1):
            t = i / self.sNum
            nP = self.eval(t)
            self.length += dist(prevPoint, nP)
            self.sLen.append(self.length)
            prevPoint = nP

        self.startPoint = p0
        self.endPoint = p1

    def eval(self, t):
        t = self.startAngle + t * self.diffAngle
        r = [self.rx * math.cos(t), self.ry * math.sin(t)]
        res = [
            dot(r, [math.cos(self.alpha), -math.sin(self.alpha)]),
            dot(r, [math.sin(self.alpha), math.cos(self.alpha)]),
        ]
        return sum(self.center, res)

    def evalGrad(self, t):
        t = self.startAngle + t * self.diffAngle
        r = [-self.rx * math.sin(t) / self.diffAngle, self.ry * math.cos(t) / self.diffAngle]
        res = [
            dot(r, [math.cos(self.alpha), -math.sin(self.alpha)]),
            dot(r, [math.sin(self.alpha), math.cos(self.alpha)]),
        ]
        return res

    def tangent(self, t):
        grad = self.evalGrad(t)
        return [grad[1], -grad[0]]

    def sample(self, s):
        s *= self.length
        index = 0
        lastLen = 0
        while index < self.sNum and s > self.sLen[index]:
            lastLen = self.sLen[index]
            index += 1

        s = (s - lastLen) / (self.sLen[index] - lastLen)
        t = (1 - s) * index / self.sNum + s * (index + 1) / self.sNum

        return self.eval(t), self.tangent(t)


objs = selected_shapes()
if len(objs) > 0 and type(objs[0]) == SimplePathObject:
    obj = objs[0]
    print(type(obj))
    path = obj.svg_get("d", False)

    cPoint = [0, 0]
    subpaths = []
    lastCode = "M"
    index = 0
    print(path)
    while index < len(path):

        def getCoord():
            global index
            res = [float(path[index]), float(path[index + 1])]
            index += 2
            return res

        def getFloat():
            global index
            res = float(path[index])
            index += 1
            return res

        def createLine(start, end):
            mp = mul(sum(start, end), [0.5, 0.5])
            c = BezierSubPath(start, mp, mp, end)
            subpaths.append(c)

        code = path[index]
        if (
            code == "M"
            or code == "m"
            or code == "L"
            or code == "l"
            or code == "H"
            or code == "h"
            or code == "V"
            or code == "v"
            or code == "C"
            or code == "c"
            or code == "A"
            or code == "a"
            or code == "Z"
            or code == "Z"
            or code == "z"
        ):
            index += 1
        else:
            code = lastCode
        lastCode = code

        if code == "M":
            cPoint = getCoord()
        elif code == "m":
            cPoint = sum(cPoint, getCoord())
        elif code == "H":
            end = [getFloat(), cPoint[1]]
            createLine(cPoint, end)
            cPoint = end
        elif code == "h":
            end = [cPoint[0] + getFloat(), cPoint[1]]
            createLine(cPoint, end)
            cPoint = end
        elif code == "V":
            end = [cPoint[0], getFloat()]
            createLine(cPoint, end)
            cPoint = end
        elif code == "v":
            end = [cPoint[0], cPoint[1] + getFloat()]
            createLine(cPoint, end)
            cPoint = end
        elif code == "L":
            end = getCoord()
            createLine(cPoint, end)
            cPoint = end
        elif code == "l":
            end = sum(cPoint, getCoord())
            createLine(cPoint, end)
            cPoint = end
        elif code == "C":
            c = BezierSubPath(cPoint, getCoord(), getCoord(), getCoord())
            subpaths.append(c)
            cPoint = c.p3
        elif code == "c":
            c = BezierSubPath(
                cPoint, sum(cPoint, getCoord()), sum(cPoint, getCoord()), sum(cPoint, getCoord())
            )
            subpaths.append(c)
            cPoint = c.p3
        elif code == "A" or code == "a":
            rx, ry = getCoord()
            rot = getFloat()
            fa, fs = getCoord()
            end = getCoord()
            if code == "a":
                end = sum(end, cPoint)
            a = ArcPath(cPoint, rx, ry, rot, fa, fs, end)
            subpaths.append(a)
            cPoint = end
        elif code == "Z" or code == "z":
            end = subpaths[0].startPoint
            createLine(cPoint, end)
            cPoint = end

    totalLength = 0
    for p in subpaths:
        totalLength += p.length

    print(totalLength)

    style(fill="#d7eef4")
    style(stroke_width=0.4)
    points = []
    for i in range(SAMPLE_POINTS):
        print("Start")
        if SAMPLE_RANDOM:
            v = random.random()
        else:
            v = i / SAMPLE_POINTS
        v *= totalLength
        index = 0
        cLength = 0
        print(v)
        while index < len(subpaths) and v > cLength + subpaths[index].length:
            cLength += subpaths[index].length
            index += 1

        print(index)
        print(subpaths[index].length)
        print(cLength)

        v = (v - cLength) / subpaths[index].length
        print(v)
        point, tan = subpaths[index].sample(v)
        tan = norm(tan)
        tan = [-tan[0], -tan[1]]  # Inv normal vector
        c = circle(point, 3 * POINTS_STD)
        c.svg_set("opt-tx", tan[0])
        c.svg_set("opt-ty", tan[1])
        rad = [POINTS_STD * POINTS_STD]
        # circle(sum(point, tan), POINTS_SIZE * 0.4)
        points.append(point + tan + rad)
    store_points(points)
else:
    print("hi")
    points = []
    for obj in all_shapes():
        print("sel")
        if type(obj) == SimpleObject:
            print("simple")
            p = [obj.svg_get("cx", False), obj.svg_get("cy", False)]
            tan = [obj.svg_get("opt-tx", False), obj.svg_get("opt-ty", False)]
            rad = obj.svg_get("r", False)
            # if rad != None:
            #     rad *= 1.1
            # The radius represents the 89% of the points (3*std)
            rad = [POINTS_STD * POINTS_STD] if rad == None else [rad * rad / 9]
            if p[0] != None and p[1] != None and tan[0] != None and tan[1] != None:
                tan[0] = tan[0]
                tan[1] = tan[1]
                points.append(p + tan + rad)
    points = np.array(points)
    # Add noise
    points[..., 0] = np.random.normal(loc=points[..., 0], scale=np.sqrt(points[..., 4]))
    points[..., 1] = np.random.normal(loc=points[..., 1], scale=np.sqrt(points[..., 4]))
    # Print points
    pnts = group()
    # pntsVar = group()
    normals = group()
    for p, g, var in zip(points[..., :2], points[..., 2:4], points[..., 4]):
        pnts.append(circle(p, 0.01))
        # pntsVar.append(circle(p, np.sqrt(var) * 2))
        normals.append(line(p, p + g * 0.6))
    print(points.shape)
    store_points(points)
    # print("hi12")
    # points = []
    # for obj in all_shapes():
    #     print("sel")
    #     if type(obj) == SimpleObject:
    #         print("simple")
    #         x1 = [obj.svg_get("x1", False), obj.svg_get("y1", False)]
    #         x2 = [obj.svg_get("x2", False), obj.svg_get("y2", False)]
    #         if x1[0] != None and x1[1] != None and x2[0] != None and x2[1] != None:
    #             x1 = np.array(x1)
    #             x2 = np.array(x2)
    #             points.append((x1, x1 + 0.5 * (x2 - x1)))
    # normals = group()
    # for a, b in points:
    #     normals.append(line(a, b))
