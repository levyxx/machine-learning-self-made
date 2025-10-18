import numpy as np

class NearestNeighbors():
    def __init__(self, sizex, sizey):
        self.sizex = sizex
        self.sizeLabel = sizey
        self.bufferx = []
        self.buffery = []

    def learning(self, x, y):
        self.bufferx.append(x)
        self.buffery.append(y)
        print("NearestNeighbours.learning() self.bufferx=%s" % self.bufferx)
        print("NearestNeighbours.learning() self.buffery=%s" % self.buffery)

    def getOutput(self, x):
        X = np.array(x)
        dist = 100000
        nearestNeighborIndex = -1
        print(len(self.bufferx))
        size = len(self.bufferx)
        print(size)
        for i in range((int)(size)):
            X2 = self.bufferx[i]
            each_dist = np.linalg.norm(X-X2, 2)
            if each_dist < dist:
                nearestNeighborIndex = i
                dist = each_dist
        return self.buffery[nearestNeighborIndex]
    
    
if __name__ == "__main__":
    nn = NearestNeighbors(2,1)
    x = [0,1]
    nn.learning(x, 2)
    x = [1,0]
    nn.learning(x, 3)
    x = [0,0]
    nn.learning(x, 4)
    x = [1,1]
    nn.learning(x, 5)
    y = nn.getOutput([0,0.5])
    print("answer", y)