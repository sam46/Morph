import aggdraw
import Tkinter
import math
from PIL import Image, ImageTk
from math import pi
import numpy as np
import itertools
from random import random


class App(object):
    def __init__(self):
        self.tk = Tkinter.Tk()
        self.canvas = Tkinter.Canvas(width=800, height=600)
        self.canvas.pack()

        def callback(event):
            self.transition()

        self.canvas.bind("<Button-1>", callback)
        self.size = 800, 600
        width, height = self.size
        # color lerping:
        self.colVec = np.array([255, 0, 0])
        self.colVecFin = np.array([0, 0, 255])
        self.dcol = 0.0
        self.brush = aggdraw.Brush((self.colVec[0], self.colVec[1], self.colVec[2]))
        # self.pen = aggdraw.Pen((255,0,0))

        self.std = np.arange(0, 1.001, 0.001)
        randPts = np.random.rand(1001, 2) * 2 * self.size - self.size
        self.pts = randPts
        self.targets, self.targetsFuzz = [], []
        self.dt = 0.0
        self.newPts = np.array(self.pts)

        line = np.dstack((self.std * width / 2 - width / 4, np.zeros(1001)))[0]
        self.targets.extend([line] * 5)

        rad = 100
        twopi = self.mapping(lambda x: pi * 2 * x, self.std)
        xs = self.mapping(lambda t: math.cos(pi * 2 * t) * rad, self.std)
        ys = self.mapping(lambda t: math.sin(pi * 2 * t) * rad, self.std)
        circle = np.dstack((xs, ys))[0]
        self.targets.append(self.interpolate(line, circle, 0.25))
        self.targets.append(self.interpolate(line, circle, 0.5))
        self.targets.append(self.interpolate(line, circle, 0.75))
        self.targets.extend([circle] * 3)

        square = []
        sideL = 100  # half side length
        _c_ = math.cos(pi / 4) * rad  # coordinates of point on the circle where x==y (1st quadrant)
        for p in circle:
            x, y = p
            ax, ay = abs(x), abs(y)
            if ax < ay:
                fx = self.lerp(-_c_, _c_, -sideL, sideL, ax) * np.sign(x)
                fy = sideL * np.sign(y)
            else:
                fy = self.lerp(-_c_, _c_, -sideL, sideL, ay) * np.sign(y)
                fx = sideL * np.sign(x)
            square.append([fx, fy])
        self.targets.extend([self.rotate(np.array(square), pi / 4)] * 3)
        self.targets.extend([self.interpolate(circle, self.rotate(np.array(square), pi / 4), 1.6)] * 3)
        self.targets.extend([self.fuzzify(self.rotate(np.array(square), pi / 4), 3, 13)] * 3)

        # square2 = []
        # sideL = 70
        # for p in circle:
        #     x, y = p
        #     ax, ay = abs(x), abs(y)
        #     if ax < ay:
        #         fx = self.lerp(-_c_,_c_,-sideL,sideL,ax)*np.sign(x)
        #         fy = sideL*np.sign(y)
        #     else:
        #         fy = self.lerp(-_c_,_c_,-sideL,sideL,ay)*np.sign(y)
        #         fx = sideL*np.sign(x)
        #     square2.append([fx,fy])
        # self.targets.append(self.split(self.rotate(np.array(square2), pi/4), 50,5))
        # self.targets.append(self.split(self.rotate(np.array(square2), pi/4), 50,5))

        self.targets.extend([circle * 0.01] * 4)  # remove redundant append call

        self.targets.extend([self.split(line, 50, 5)] * 2)  # remove redundant append call
        self.targets.extend([line] * 2)  # remove redundant append call

        amp = 100
        period = 1.5  # how many half period to display
        ntime = 2  # reptitions
        step = 0.1  # [0,1]
        splits = itertools.cycle([0, 1, 2, 3, 4, 5, 4, 3, 2, 1])
        for th in np.arange(0, ntime + step, step):
            sinus = np.dstack((line[:, 0], self.mapping(lambda t: amp * math.sin(period * 2 * pi * t / 1000 - pi * th), line[:, 0])))[0]
            sinus = self.split(sinus, 35, splits.next())
            self.targets.append(np.array(sinus))

        for i in range(1, 10):
            self.targets.append(self.interpolate(line, circle, i * 0.1))
        self.targets.append(self.fuzzify(circle, 1, 1))
        self.targets.append(self.fuzzify(circle, 2, 5))
        self.targets.append(self.fuzzify(circle, 3, 10))
        self.targets.append(circle)

        self.targets.extend([randPts] * 3)

        self.targets.append(circle * 10)
        self.targets.extend([circle * 0.01] * 6)

        #### 3d scences:
        circle = self.to3D(circle)
        circleShuffle = np.array(circle)
        np.random.shuffle(circleShuffle)
        sphere = []  # will contain 5005 vectors
        sphere.append(np.array(circleShuffle))
        [sphere.append(np.array([self.rotateY(vec, i * pi / 5) for vec in circleShuffle])) for i in range(1, 5)]  # optimized line of code
        sphereReduced = []  # will contain 1001 vectors
        for i in range(1001):
            if i in range(1, 1001 / 5 + 1):
                sphereReduced.append(np.array(sphere[0][i]))
            elif i in range(1001 / 5, 2 * 1001 / 5 + 1):
                sphereReduced.append(np.array(sphere[1][i]))
            elif i in range(2 * 1001 / 5, 3 * 1001 / 5 + 1):
                sphereReduced.append(np.array(sphere[2][i]))
            elif i in range(3 * 1001 / 5, 4 * 1001 / 5 + 1):
                sphereReduced.append(np.array(sphere[3][i]))
            else:
                sphereReduced.append(np.array(sphere[4][i]))

        th = 1.57
        sphereInit = np.array([self.rotateX(self.rotateY(vec, th), th) for vec in sphereReduced])
        self.targets.extend([sphereInit] * 3)

        for th in np.arange(0.1, pi, 0.1):
            if random() < 0.4 or th - 3.1 < 0.000001:
                self.targets.append(self.fuzzify(np.array([self.rotateX(self.rotateY(vec, -th), -th) for vec in sphereInit]), 10, 10, 10) * self.lerp(0.1, 3.1, 1, 2.3, th))
            else:
                self.targets.append(np.array([self.rotateX(self.rotateY(vec, -th), -th) for vec in sphereInit]) * self.lerp(0.1, 3.1, 1, 2.3, th))
            # self.targets.append(self.swapDim(np.array([self.rotateY(vec, pi/th) for vec in circle]), 1,2))
        self.targets.extend([np.array(self.targets[-1])] * 4)

        th = 3.15
        self.targets.append(np.array([self.rotateX(self.rotateY(vec, -th), -th) for vec in sphereInit]) * 1.3)
        self.targets.append(self.fuzzify(np.array([self.rotateX(self.rotateY(vec, -th), -th) for vec in sphereInit]), 10, 10, 10) * 1.3)
        self.targets.append(np.array([self.rotateX(self.rotateY(vec, -th), -th) for vec in sphereInit]) * 1.3)
        prevState = np.array(self.targets[-1])
        th = -1.57
        self.targets.extend([np.array([self.rotateX(self.rotateY(vec, -th), -th) for vec in prevState]) * 1.3] * 2)
        self.targets.append(self.fuzzify(np.array([self.rotateX(self.rotateY(vec, -th), -th) for vec in prevState]), 10, 10, 10) * 1.3)
        self.targets.extend([np.array([self.rotateX(self.rotateY(vec, -th), -th) for vec in prevState]) * 1.3] * 2)

        self.cycle = itertools.cycle(self.targets)
        self.cycleFuzz = itertools.cycle(self.targetsFuzz)
        self.target = self.pts
        self.targetFuzz = self.pts

    def to3D(self, vec):
        return np.hstack((vec, [[0] for i in range(len(vec))]))

    def fuzzify(self, arr, xamp, yamp, zamp=None):
        return np.array([i + (xamp * random(), yamp * random()) for i in arr]) if not zamp else np.array([i + (xamp * random(), yamp * random(), zamp * random()) for i in arr])

    def swapDim(self, arr, d1, d2):
        ret = []
        for v in arr:
            temp = list(v)
            temp[d1], temp[d2] = temp[d2], temp[d1]
            ret.append(np.array(temp))
        return np.array(ret)

    def split(self, arr, dist, n):
        if n == 0:
            return np.array(arr)
        lower = arr[:len(arr) / 2, :] - (dist, 0)
        upper = arr[len(arr) / 2:, :] + (dist, 0)
        return np.vstack((self.split(lower, dist / 2, n - 1), self.split(upper, dist / 2, n - 1)))

    def rotate(self, arr, rot):
        matRot = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        return np.array([matRot.dot(v) for v in arr])

    def interpolate(self, v1, v2, t):
        if v1.shape != v2.shape:
            if v1.shape[1] == 2:
                v1 = self.to3D(v1)
            else:
                v2 = self.to3D(v2)
        return np.array(v1 + (v2 - v1) * t)

    def transition(self):
        self.dt = 0
        self.pts = self.newPts
        self.target = self.cycle.next()
        # self.targetFuzz = self.cycleFuzz.next()

    # map x in [a,b] to y in [c,d]
    def lerp(seld, a, b, c, d, x):
        m = (c - d) / (a - b)
        p = c - m * a
        return m * x + p

    def update(self):
        self.canvas.delete('all')
        self.photo = self.draw()
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.tk.after(1000 / 60, self.update)

    def mapping(self, func, arr):
        vfunc = np.vectorize(func)
        return vfunc(arr)

    def rotateX(self, vec, th):
        y = math.cos(th) * vec[1] - math.sin(th) * vec[2];
        z = math.sin(th) * vec[1] + math.cos(th) * vec[2];
        return np.array([vec[0], y, z]);

    def rotateY(self, vec, th):
        x = math.cos(th) * vec[0] - math.sin(th) * vec[2];
        z = math.sin(th) * vec[0] + math.cos(th) * vec[2];
        return np.array([x, vec[1], z]);

    def rotateZ(self, vec, th):
        y = math.cos(th) * vec[1] - math.sin(th) * vec[0];
        x = math.sin(th) * vec[1] + math.cos(th) * vec[0];
        return np.array([x, y, vec[2]]);

    def bezier(self, ctx, p0, p1, p2, p3):
        ax, ay = p0
        bx, by = p1
        cx, cy = p2
        dx, dy = p3
        pathstring = " m{:f},{:f} c{:f},{:f},{:f},{:f},{:f},{:f}".format(ax, ay, bx, by, cx, cy, dx, dy)
        symbol = aggdraw.Symbol(pathstring)
        ctx.symbol((0, 0), symbol, aggdraw.Pen((255, 0, 0)))

    def drawArr(self, ctx, arr):
        for v in arr:
            x, y = v[:2]
            ctx.ellipse((x + 1, y + 1, x, y), self.brush)

    def draw(self):
        self.dcol += 0.0006
        colCur = self.interpolate(self.colVec, self.colVecFin, self.dcol)
        if 1.0 - self.dcol < 0.0000001:
            self.dcol = 0.0
            self.colVec, self.colVecFin = self.colVecFin, self.colVec
        self.brush = aggdraw.Brush((int(round(colCur[0])), int(round(colCur[1])), int(round(colCur[2]))))

        speed = 0.1  # interpolation speed. in range [0,1]
        img = Image.new('RGBA', self.size, "black")
        ctx = aggdraw.Draw(img)
        ctx.settransform((self.size[0] / 2, self.size[1] / 2))

        self.newPts = self.interpolate(self.pts, self.target, self.dt)
        self.dt = min(self.dt + speed, 1.0)
        self.drawArr(ctx, self.newPts)
        if 1.0 - self.dt < 0.0000001:
            self.transition()
            self.dt += speed

        ctx.flush()
        return ImageTk.PhotoImage(img)

    def run(self):
        self.update()
        self.tk.mainloop()


if __name__ == '__main__':
    App().run()
