import PIL.Image as Image
import math
import random
import jittor as jt
import jittor.nn as F
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


class Line:

    def __init__(self, pa, pb) -> None:
        self.beginPoint = pa
        self.endPoint = pb

        self.dir = (self.endPoint - self.beginPoint).unit()
        self.maxT = (self.endPoint - self.beginPoint).mod()

    def onLine(self, p):
        return (p - self.beginPoint).mod() <= self.maxT

    def onSegment(self, p):
        return self.onLine(p) and (p - self.beginPoint).dot(self.dir) >= 0

    def getPointY(self, y):
        if y < min(self.beginPoint.y, self.endPoint.y) or y > max(
                self.endPoint.y, self.beginPoint.y):
            return None
        if self.dir.y == 0:
            return Point(self.beginPoint.x, y)
        return Point(
            self.beginPoint.x +
            (y - self.beginPoint.y) * self.dir.x / self.dir.y, y)

    def getPointX(self, x):
        if x < min(self.beginPoint.x, self.endPoint.x) or x > max(
                self.endPoint.x, self.beginPoint.x):
            return None
        if self.dir.x == 0:
            return Point(x, self.beginPoint.y)
        return Point(
            x, self.beginPoint.y +
            (x - self.beginPoint.x) * self.dir.y / self.dir.x)

    def __str__(self):
        return f"[{self.beginPoint}, {self.endPoint}]"


class Point:

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        if type(other) == float:
            return Point(self.x / other, self.y / other)

    def __mul__(self, other):
        if type(other) == float:
            return Point(self.x * other, self.y * other)
        else:
            return self.x * other.x + self.y * other.y

    def __xor__(self, other):
        return self.x * other.y - self.y * other.x

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def mod(self):
        return (self.x**2 + self.y**2)**0.5

    def unit(self):
        return self / self.mod()

    def inGrid(self, lx, ly, rx, ry):
        return lx <= self.x <= rx and ly <= self.y <= ry


class AvaEdge:

    def __init__(self, topY, x, dx):
        self.topY = topY
        self.x = x
        self.dx = dx

    def __lt__(self, other):
        return self.x < other.x

    def __str__(self):
        return "topY:" + str(self.topY) + " x:" + str(self.x) + " dx:" + str(
            self.dx)


class Grid:

    def __init__(self, n, m, x, y, grid=None) -> None:
        self.n = n
        self.m = m
        self.x = x
        self.y = y
        self.dX = x / n
        self.dY = y / m
        self.grid = [[0.0 for _ in range(m)] for __ in range(n)]

        if grid != None:
            self.grid = [[grid[i][j] for j in range(m)] for i in range(n)]

    def sum(self):
        res = 0
        for i in range(self.n):
            for j in range(self.m):
                res += self.grid[i][j]
        return res

    def copy(self):
        return Grid(self.n, self.m, self.x, self.y, self.grid)

    def overlap(self, other):
        resGrid = Grid(self.n, self.m, self.x, self.y)
        for i in range(self.n):
            for j in range(self.m):
                if (self.grid[i][j] + other.grid[i][j] > 0):
                    resGrid.grid[i][j] = 1
        return resGrid

    def checkOverlap(self, other):
        res = 0.0
        for i in range(self.n):
            for j in range(self.m):
                if self.grid[i][j] + other.grid[i][j] > 1:
                    res += 1
                else:
                    res += self.grid[i][j] + other.grid[i][j]
        return res

    def add(self, other):
        resGrid = Grid(self.n, self.m, self.x, self.y)
        for i in range(self.n):
            for j in range(self.m):
                resGrid.grid[i][j] = self.grid[i][j] + other.grid[i][j]
        return resGrid

    def standerize(self):
        resGrid = Grid(self.n, self.m, self.x, self.y)
        for i in range(self.n):
            for j in range(self.m):
                resGrid.grid[i][j] = 1 if self.grid[i][j] > 0 else 0
        return resGrid

    def sumGreaterThanOne(self):
        res = 0
        for i in range(self.n):
            for j in range(self.m):
                res += max(0, self.grid[i][j] - 1)
        return res

    def fillGrid(self, bx, by, tx, ty):
        # print(self.grid, bx, by, tx, ty)
        for i in range(bx, tx + 1):
            for j in range(by, ty + 1):
                # print(i, j)
                self.grid[i][j] = 1

    def outputGrid(self):
        for y in range(len(self.grid[0]) - 1, -1, -1):
            for x in range(len(self.grid)):
                print(self.grid[x][y], end=' ')
            print()

    def outputGridToFile(self, fileName):
        with open(fileName, 'w') as f:
            for y in range(len(self.grid[0]) - 1, -1, -1):
                for x in range(len(self.grid)):
                    f.write(str(self.grid[x][y]))
                f.write('\n')
            f.close()

    def transformation(self, x, y):
        newGrid = [[0 for _ in range(self.m)] for __ in range(self.n)]
        for i in range(self.n):
            for j in range(self.m):
                newX = int(i + x + 0.5)
                newY = int(j + y + 0.5)
                if newX < 0 or newX >= self.n or newY < 0 or newY >= self.m:
                    continue
                newGrid[newX][newY] = self.grid[i][j]
        return Grid(self.n, self.m, self.x, self.y, newGrid)

    def rotation(self, cx, cy, angle):
        newGrid = [[0 for _ in range(self.m)] for __ in range(self.n)]
        for i in range(self.n):
            for j in range(self.m):
                newX = int((i - cx) * math.cos(angle) -
                           (j - cy) * math.sin(angle) + cx + 0.5)
                newY = int((i - cx) * math.sin(angle) +
                           (j - cy) * math.cos(angle) + cy + 0.5)
                if newX < 0 or newX >= self.n or newY < 0 or newY >= self.m:
                    continue
                newGrid[newX][newY] = self.grid[i][j]
        return Grid(self.n, self.m, self.x, self.y, newGrid)

    def transformBilinearInterpolation(self, cx, cy, dx, dy, angle):
        newGrid = [[0.0 for _ in range(self.m)] for __ in range(self.n)]
        for i in range(self.n):
            for j in range(self.m):
                newX = (i - dx - cx) * math.cos(-angle) - \
                    (j - dy - cy) * math.sin(-angle) + cx
                newY = (i - dx - cx) * math.sin(-angle) + \
                    (j - dy - cy) * math.cos(-angle) + cy
                # newX = i - dx
                # newY = j - dy
                tx1 = int(newX)
                tx2 = tx1 + 1 if newX >= 0 else tx1 - 1
                ty1 = int(newY)
                ty2 = ty1 + 1 if newY >= 0 else ty1 - 1

                newGrid[i][j] = 0.0
                if tx1 >= 0 and tx1 < self.n and ty1 >= 0 and ty1 < self.m:
                    newGrid[i][j] += (1.0 - math.fabs(newX - tx1)) * \
                        (1 - math.fabs(newY - ty1)) * self.grid[tx1][ty1]
                if tx1 >= 0 and tx1 < self.n and ty2 >= 0 and ty2 < self.m:
                    newGrid[i][j] += (1.0 - math.fabs(newX - tx1)) * \
                        (1 - math.fabs(newY - ty2)) * self.grid[tx1][ty2]
                if tx2 >= 0 and tx2 < self.n and ty1 >= 0 and ty1 < self.m:
                    newGrid[i][j] += (1.0 - math.fabs(newX - tx2)) * \
                        (1 - math.fabs(newY - ty1)) * self.grid[tx2][ty1]
                if tx2 >= 0 and tx2 < self.n and ty2 >= 0 and ty2 < self.m:
                    newGrid[i][j] += (1.0 - math.fabs(newX - tx2)) * \
                        (1 - math.fabs(newY - ty2)) * self.grid[tx2][ty2]
        return Grid(self.n, self.m, self.x, self.y, newGrid)

    def toArray(self):
        return self.grid


class SweepLine:

    def __init__(self, points) -> None:
        self.points = points

    def solve(self, n, m, x, y, oldCrossPoints):
        dX = x / n
        dY = y / m
        crossPoints = []

        topY = 0
        for point in self.points:
            topY = max(topY, point.y)

        nowY = 0.0
        nowRow = 0
        while (nowY + dY) <= topY:
            crossPoints.append(oldCrossPoints[nowRow])
            for i in range(0, len(self.points)):
                ne = i + 1
                la = i - 1
                if ne == len(self.points):
                    ne = 0
                if la == -1:
                    la = len(self.points) - 1
                # print(i, len(self.points), ne, self.points[i], self.points[ne])
                line = Line(self.points[i], self.points[ne])
                
                if line.dir.y == 0:
                    continue
                crossPoint = line.getPointY(nowY)
                if crossPoint is not None:
                    if crossPoint.y == max(self.points[i].y,
                                           self.points[ne].y):
                        continue
                    crossPoint.x = int(crossPoint.x / dX + 0.5)
                    crossPoint.y = nowRow
                    crossPoints[nowRow].append(crossPoint)
            nowY += dY
            nowRow += 1
        while nowRow < m:
            crossPoints.append(oldCrossPoints[nowRow])
            nowRow += 1
        return crossPoints


class Polygon:

    def readPolygon(self, fileName):
        contour = []
        with open(fileName, 'r') as f:
            minY =  1000000000
            maxX = -1000000000
            minX =  1000000000
            maxY = -1000000000
            lines = f.readlines()
            block = int(lines[0])
            nowLine = 1
            for i in range(0, block):
                points = []
                pointCnt = int(lines[nowLine])
                for j in range(0, pointCnt):
                    nowLine += 1
                    x, y = lines[nowLine].split()
                    points.append(Point(float(x), float(y)))
                contour.append(points)
                nowLine += 1
            f.close()
        return contour, maxX - minX, maxY - minY
    
    def readPolygonApp(self, fileName):
        contourList = []
        contour = []
        with open(fileName, 'r') as f:
            lines = f.readlines()
            block = int(lines[0])
            nowLine = 1
            minY =  1000000000
            maxX = -1000000000
            minX =  1000000000
            maxY = -1000000000
            for i in range(0, block):
                points = []
                pointCnt = int(lines[nowLine])
                for j in range(0, pointCnt):
                    nowLine += 1
                    x, y = lines[nowLine].split()
                    points.append([float(x) * 10, float(y) * 10])
                contourList.append(points)
                nowLine += 1
            for points in contourList:
                pointsApp = cv.approxPolyDP(np.array(points, dtype=np.float32), 5, True).squeeze(1).tolist()
                pointsApp = points
                nowContour = []
                # print(pointsApp)
                for point in pointsApp:
                    nowContour.append(Point(point[0], point[1]))
                # print(nowContour)
                contour.append(nowContour)
            f.close()
        return contour, maxX - minX, maxY - minY
 
    def __init__(self, fileName, app=True) -> None:
        if app:
            self.contour, self.x, self.y = self.readPolygonApp(fileName)
        else:
            self.contour, self.x, self.y = self.readPolygon(fileName)
        maxPointCnt = 0
        maxId = 0
        for points in self.contour:
            if len(points) > maxPointCnt:
                maxPointCnt = len(points)
                maxId = self.contour.index(points)
        self.mainContour = self.contour[maxId]
        self.grid = []
        self.maxSize = max(self.x, self.y)
        
    def getMaxContour(self):
        points = []
        for point in self.mainContour:
            points.append([point.x, point.y])
        return points
    
    def getMaxContourCenter(self):
        minX = self.mainContour[0].x
        minY = self.mainContour[0].y
        maxX = self.mainContour[0].x
        maxY = self.mainContour[0].y
        for point in self.mainContour:
            minX = min(minX, point.x)
            minY = min(minY, point.y)
            maxX = max(maxX, point.x)
            maxY = max(maxY, point.y)
        return (minX + maxX) / 2, (minY + maxY) / 2

    def solve(self, n, m, x, y):
        self.grid = Grid(n, m, x, y)
        crossPoints = []
        for i in range(0, m):
            crossPoints.append([])
        for contour in self.contour:
            if len(contour) <= 1:
                continue
            sweepLine = SweepLine(contour)
            crossPoints = sweepLine.solve(n, m, x, y, crossPoints)
            
        for row in crossPoints:
            row.sort(key=lambda point: point.x)
            for i in range(0, len(row), 2):
                self.grid.fillGrid(row[i].x, row[i].y, row[i + 1].x,
                                   row[i + 1].y + 1)
        # self.grid.outputGrid()
        return self.grid

    def outputPolygon(self):
        for points in self.contour:
            for point in points:
                print(point.x, point.y)
            print()


def outputGrid(grid):
    for y in range(len(grid[0]) - 1, -1, -1):
        for x in range(len(grid)):
            print(grid[x][y], end=' ')
        print()


def outputGridFloat(grid):
    for y in range(len(grid[0]) - 1, -1, -1):
        for x in range(len(grid)):
            print("%.1f" % grid[x][y], end=' ')
        print()


def genGrid(id, n=400, m=400, x=192, y=120):
    poly = Polygon(f"../data/bd{id}.txt")
    if poly == None or poly.x > x and poly.y > y:
        return None
    try:
        grid = poly.solve(n, m, x, y)
    except:
        return None
    else:
        return grid
    
def genGridCenter(id, n=400, m=400, x=192, y=120):
    poly = Polygon(f"../data/bd{id}.txt", app=False)
    if poly == None or poly.x > x and poly.y > y:
        return None
    #try:
    grid = poly.solve(n, m, x, y)
    #except:
    #    return None
    #else:
    cx, cy = getSize(grid)
    grid = grid.transformation(int(n / 2 - cx / 2), int(m / 2 - cy / 2))
    return grid


def getCenter(grid):
    minX = minY = 1000000
    maxX = maxY = -1000000
    for i in range(grid.n):
        for j in range(grid.m):
            if grid.grid[i][j] == 1:
                minX = min(minX, i)
                maxX = max(maxX, i)
                minY = min(minY, j)
                maxY = max(maxY, j)
    return (minX + maxX) / 2, (minY + maxY) / 2

def getSize(grid):
    minX = minY = 1000000
    maxX = maxY = -1000000
    for i in range(grid.n):
        for j in range(grid.m):
            if grid.grid[i][j] == 1:
                minX = min(minX, i)
                maxX = max(maxX, i)
                minY = min(minY, j)
                maxY = max(maxY, j)
    return maxX - minX, maxY - minY

def transformer(grid, cx, cy, dx, dy, angle):
    grid = grid.transformation(dx, dy)
    grid = grid.rotation(cx, cy, angle)
    return grid


def place(grid, poly, x, y, angle):
    poly, valid = transformer(poly, x, y, angle)
    res = Grid(grid.n, grid.m, grid.x, grid.y)
    for i in range(grid.m):
        for j in range(grid.m):
            res.grid[i][j] = grid.grid[i][j] + poly.grid[i][j]
    return res


def randomPlace(grid, placeNumber):
    for i in range(0, placeNumber):
        poly = genGrid(random.randint(0, 10), grid.n, grid.m,
                                     grid.x, grid.y)
        cx, cy = getCenter(poly)
        poly = poly.transformBilinearInterpolation(
            cx, cy,
            random.random() * 2 * math.pi,
            random.random() * grid.n,
            random.random() * grid.m)
        grid = grid.overlap(poly)
    return grid


def drawGrid(grid, fileName=None):
    if type(grid) == Grid:
        n, m = grid.n, grid.m
        grid = grid.grid
    else:
        n, m = len(grid.grid), len(grid.grid[0])
    image = Image.new("RGB", (n, m))
    for x in range(n):
        for y in range(m):
            if grid[x][y] > 0:
                image.putpixel((x, y), (0, 0, 0))
            elif grid[x][y] == 0:
                image.putpixel((x, y), (255, 255, 255))

    # image.show()
    if fileName != None:
        image.save(fileName)
    else:
        image.show()


def setRandomSeed(seed):
    random.seed(seed)
    # jt.manual_seed(seed)
    # tr.cuda.manual_seed(seed)
    # tr.cuda.manual_seed_all(seed)
    # tr.backends.cudnn.deterministic = True
    jt.misc.set_global_seed(seed)

def affineTransform(polys, affineMatrices):
    gridWeight = F.affine_grid(affineMatrices,
                               polys.size(),
                               align_corners=False)
    res = F.grid_sample(polys, gridWeight, align_corners=False)
    return res


def genRandomAffineMatrices(action, randomRange):
    randomNoise = tr.rand(action.size()) * randomRange
    if action.shape[1] == 3:
        randomNoise[:, 2] = randomNoise[:, 2] * tr.pi 
    if action.is_cuda:
        randomNoise = randomNoise.cuda()
    action = (action + randomNoise).clamp(-1, 1)

    # if rotation is not needed:
    
    if action.shape[1] == 2: theta = tr.zeros_like(action[:, 0], device=action.device, dtype=action.dtype)
    else: theta = action[:, 2] * tr.pi
    
    cosTheta = tr.cos(theta)
    sinTheta = tr.sin(theta)
    dx = cosTheta * action[:, 1] + sinTheta * action[:, 0]
    dy = -sinTheta * action[:, 1] + cosTheta * action[:, 0]

    affineMatrices = tr.cat(
        (cosTheta.unsqueeze(1), sinTheta.unsqueeze(1), -dx.unsqueeze(1),
         -sinTheta.unsqueeze(1), cosTheta.unsqueeze(1), -dy.unsqueeze(1)),
        dim=1)
    affineMatrices = affineMatrices.view(-1, 2, 3)
    return affineMatrices

# initial position of the object must be at the center of the grid
def genAffineMatrices(action):
    theta = action[:, 2] * tr.pi
    cosTheta = tr.cos(theta)
    sinTheta = tr.sin(theta)
    dx = cosTheta * action[:, 1] + sinTheta * action[:, 0]
    dy = -sinTheta * action[:, 1] + cosTheta * action[:, 0]

    affineMatrices = tr.cat(
        (cosTheta.unsqueeze(1), sinTheta.unsqueeze(1), -dx.unsqueeze(1),
         -sinTheta.unsqueeze(1), cosTheta.unsqueeze(1), -dy.unsqueeze(1)),
        dim=1)
    affineMatrices = affineMatrices.view(-1, 2, 3)
    return affineMatrices

def calLossBasic(state, action):
        grids = state[:, 0, :, :].unsqueeze(1)
        polys = state[:, 1, :, :].unsqueeze(1)
        
        lossDistance = tr.abs(action)[:, 0] + tr.abs(action)[:, 1]
        lossDistance = lossDistance.reshape(-1, 1)

        sumOriginPolys = polys.sum(3).sum(2)
        affineMatrices = genRandomAffineMatrices(action, 0)
        deformedPolys = affineTransform(polys, affineMatrices)
        lossPoly = tr.abs(deformedPolys.sum(3).sum(2) -
                          sumOriginPolys) 

        afterStates = grids + deformedPolys
        overlapAfterState = tr.where(afterStates > 1, 1, afterStates)
        lossState = tr.abs(
            overlapAfterState.sum(3).sum(2) -
            afterStates.sum(3).sum(2)) 

        return lossPoly, lossState, lossDistance

def savePoly(contour, fileNmae):
    for points in contour:
        plt.plot([p.x for p in points], [p.y for p in points])
    plt.savefig(fileNmae)
    plt.clf()

# [CONTOUR][POINTS][X, Y]
def compareApp(contours, id):
    contourApps = []
    beforePointCount = 0
    for contour in contours:
        contourArr = []
        plt.plot([p.x for p in contour], [p.y for p in contour])
        for point in contour:
            contourArr.append([point.x, point.y])
            beforePointCount += 1
        contourApp = cv.approxPolyDP(np.array(contourArr, dtype=np.float32), 1.8, True)
        contourApps.append(contourApp)

    afterPointCount = 0
    for points in contourApps:
        afterPointCount += len(points)
        plt.plot([p[0][0] for p in points], [p[0][1] for p in points])
    plt.savefig("vis/poly_%d_compare%d_%d.png" % (id, beforePointCount, afterPointCount))
    plt.clf()
    
    
def justDraw():
    fileName = f"../data/bd{52}.txt"
    contour = []
    with open(fileName, 'r') as f:
        lines = f.readlines()
        block = int(lines[0])
        nowLine = 1
        for i in range(block):
            points = []
            pointCnt = int(lines[nowLine])
            for j in range(0, pointCnt):
                nowLine += 1
                x, y = lines[nowLine].split()
                points.append(Point(float(x), float(y)))
            contour.append(points)
            nowLine += 1
        f.close()

    for points in contour:
        plt.plot([p.x for p in points], [p.y for p in points])
        plt.show()
    
    
if __name__ == "__main__":
    # justDraw()
   
    '''
    n, m, x, y = 192, 192, 192, 192
    # polyGrid, moveBackX, moveBackY = genGridCenter(108, n, m, x, y)
    # moveBackX = -moveBackX * 2 / n
    # moveBackY = -moveBackY * 2 / m
    
    
    polyGrid = genGridCenter(33, n, m, x, y)
    sx, sy = getSize(polyGrid)
    print(sx, sy)
    
    drawGrid(polyGrid, "test1.png")
    polyGrid = tr.tensor(polyGrid.grid).unsqueeze(0).unsqueeze(0)
    print(sx / n  - 1, sy / m  - 1)
    
    
    action = tr.tensor([sx / n  - 1, sy / m  - 1, 0]).unsqueeze(0)
    affineMatrices = genAffineMatrices(action)
    deformGrid = affineTransform(polyGrid, affineMatrices).reshape(n, m)
    drawGrid(deformGrid, "test2.png")
    
    '''
    for i in range(0, 109):
        print("--------------------------------------------------")
        print(i)
        poly = Polygon(f"../data/bd{i}.txt")
        polyGrid = genGridCenter(i, 128, 128, poly.maxSize * 1.05, poly.maxSize * 1.05)
        
        print(poly.maxSize)
        print(poly.maxSize / 128)
        print("--------------------------------------------------")
        drawGrid(polyGrid,      "vis/poly_" + str(i) + "_grid.png")
        savePoly(poly.contour,  "vis/poly_" + str(i) + "_poly.png")
        # outputGrid(poly.grid)
        
        # compareApp(poly.contour, i)
    '''
    grid = Grid(100, 100, 128, 96)
    grid = randomPlace(grid, 1)
    # outputGridFloat(gird.grid)
    poly, length, wide = genGrid(31, 100, 100, 96, 60)
    print(length,   wide)
    # outputGrid(grid.grid)
    cx, cy = getCenter(poly)
    sumOriginPoly = poly.sum()
    poly = poly.transformBilinearInterpolation(
        cx, cy, math.pi / 18 * 16, 16.6, 20)
    sumNewPoly = poly.sum()

    sumOriginGrid = grid.sum()
    newGird = grid.overlap(poly)
    addGrid = grid.add(poly)
    sumNewGird = grid.checkOverlap(poly)

    outputGridFloat(addGrid.grid)
    print("sumOriginPoly", sumOriginPoly)
    print("sumNewPoly", sumNewPoly)
    print("sumOriginGrid", sumOriginGrid)
    print("sumNewGird", sumNewGird)
    print("sumtotal", sumNewPoly + sumOriginGrid - sumNewGird)
    # sweepLine = SweepLine(poly.mainContour)
    # sweepLine.solve(200, 200, 192, 120)
    '''
