import PIL.Image as Image
import tools
import torch as tr
import random


class Environment: # v2, agent for arrange all the polygons
    def drawState(self, fileName):
        affineMatrices = tools.genRandomAffineMatrices(self.actions, 0)
        polys = self.inputPolys.clone()
        polys = polys.unsqueeze(1)
        deformedPolys = tools.affineTransform(polys, affineMatrices).squeeze(1)
        afterState = deformedPolys.sum(0)
        
        
        # > 1 -> -1 -> (255, 0, 0) R
        # > 0 ->  1 -> (0, 0, 0) B
        # = 0 ->  0 -> (255, 255, 255) W
        pixels = tr.where(afterState > 1, -1, afterState)
        pixels = tr.where(pixels > 0, 1, pixels)
        pixelsR = tr.where(pixels <= 0, 255, 0).unsqueeze(-1)
        pixelsG = tr.where(pixels == 0, 255, 0).unsqueeze(-1)
        pixelsB = tr.where(pixels == 0, 255, 0).unsqueeze(-1)

        pixelsRGB = tr.cat((pixelsR, pixelsG, pixelsB), -1)

        image = Image.fromarray(pixelsRGB.cpu().numpy().astype("uint8"), 'RGB')
        '''
        image = Image.new("RGB", (self.n, self.m))
        for x in range(self.n):
            for y in range(self.m):
                if afterState[x][y] == 0:
                    image.putpixel((x, y), (255, 255, 255))
                else:
                    image.putpixel((x, y), (0, 0, 0))
                if afterState[x][y] > 1:
                    image.putpixel((x, y), (255, 0, 0))
        '''
        if fileName is not None:
            image.save(fileName)
    
    def calUtil(self):
        affineMatrices = tools.genRandomAffineMatrices(self.actions, 0)
        polys = self.inputPolys.clone()
        polys = polys.unsqueeze(1)
        deformedPolys = tools.affineTransform(polys, affineMatrices).squeeze(1)
        afterStates = deformedPolys.sum(0)
        
        overlapAfterStates = tr.where(afterStates > 1, 1, afterStates)
        util = overlapAfterStates.sum(1).sum(0)
        
        return util / (self.n * self.m) * 100
    
    def __init__(self, n, m, x, y, polyNumber):
        self.n = n
        self.m = m
        self.polyNumber = polyNumber
        self.polyMaxId = 29

        # polygon init
        self.polygons = tr.zeros((self.polyMaxId, n, m), device="cuda")
        self.inputPolys = tr.zeros((self.polyNumber, n, m), device="cuda")
        self.initPolys(self.polyMaxId, n, m, x, y)
        self.polySequence = tr.tensor(
                [random.randint(0, self.polyMaxId - 1) for _ in range(self.polyNumber * 2)] 
            , dtype=tr.int64)
        
        # action init
        self.actions = tr.empty(0)
        
    def initEnv(self, n, m):
        self.n = n
        self.m = m
        self.state = tr.zeros((n, m), device="cuda")
        self.runningOn = 0
        
    def initPolys(self, polyMaxId, n, m, x, y):
        self.polygons = tr.zeros((polyMaxId, n, m), device="cuda")
        for i in range(0, polyMaxId):
            poly, length, wide = tools.genGrid(i, n, m, x, y)
            self.polygons[i] = tr.FloatTensor(poly.grid)
            
    def loadInitAction(self, actions):
        self.actions = actions.clone()
    
    def loadSequence(self, polySequence):
        self.polySequence = polySequence.clone()
            
    def reset(self, resetSequence=False, randomAction=False, polyNumber=None):
        if polyNumber is not None:
            self.polyNumber = polyNumber
        self.state = tr.zeros((self.n, self.m), device="cuda")
        self.runningOn = 0
        if resetSequence:
            self.polySequence = tr.tensor(
                [random.randint(0, self.polyMaxId - 1) for _ in range(self.polyNumber * 2)] 
            , dtype=tr.int64)
        self.inputPolys = self.polygons[self.polySequence[:self.polyNumber]].clone()
        for i in range(0, self.polyNumber):
            print(int(self.polySequence[i].item()), end=',')
        if randomAction:
            self.actions = tr.rand((self.polyNumber, 3), device="cuda") * 2 - 1
        else:
            self.actions = tr.zeros((self.polyNumber, 3), device="cuda")

    def getStates(self):
        return self.inputPolys.clone(), self.actions.clone()

    def saveAction(self, newAction):
        self.actions = newAction

    def step(self, newAction):
        self.saveAction(newAction)
        
    
if __name__ == "__main__":
    tools.setRandomSeed(0)
    env = Environment(192, 128, 64, 64, 29, 16, 1)
    batchSize = 1
    polyId = 4
    packingTimes = 1

    for i in range(0, packingTimes):
        print("running on " + str(i) + "th packing")
        # initBatch
        nowStates = env.getStates()
        print(env.beforeStates.shape)

        # doAction
        singleAction = tr.FloatTensor([0.5, -0.5, 0])
        action = singleAction.unsqueeze(0).repeat(batchSize, 1)
        print(action)
        # should be something like:
        #     action = net(input)
        env.doAction(action)
        print(env.afterStates.shape)
        print(env.deformedPolys.shape)

        # drawGrid
        env.drawGrid("grid" + str(i) + ".png")

        # calLoss
        # loss = env.calLossBasic()

        # updateState
        # env.drawState("before_update" + str(i) + ".png")
        # env.updateState()
        # env.drawState("after_update" + str(i) + ".png")