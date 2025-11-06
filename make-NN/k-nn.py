import numpy as np

class kNearestNeighbors():
    def __init__(self, sizex, sizey, k):
        self.sizex = sizex
        self.k = k
        self.sizeLabel = sizey
        self.bufferx = []
        self.buffery = []
        self.knnIndex = []

    def learning(self, x, y):
        self.bufferx.append(x)
        self.buffery.append(y)
        print("NearestNeighbours.learning() self.bufferx=%s" % self.bufferx)
        print("NearestNeighbours.learning() self.buffery=%s" % self.buffery)

    def getOutput(self, x):
        X = np.array(x)
        for i in range(len(self.bufferx)):
            X2 = self.bufferx[i]
            each_dist = np.linalg.norm(X-X2, 2)
            D = [i, each_dist]
            if len(self.knnIndex) < self.k:
                self.knnIndex.append(D)
                print("append")
            else:
                print("replace")
                maxDistance = -1.0
                targetIndex = -1
                for j in range(self.k):
                    d = self.knnIndex[j]
                    if maxDistance < d[1]:
                        maxDistance = d[1]
                        targetIndex = j
                if D[1] < maxDistance:
                    self.knnIndex[targetIndex] = D

        print("max(self.buffery)=%s" % (max(self.buffery)))
        labels = np.zeros(max(self.buffery))
        print(len(labels))
        for d in self.knnIndex:
            labels[self.buffery[d[0]]-1] += 1
        print("labels=%s" % (labels))
        return np.argmax(labels)+1


if __name__ == "__main__":
    nn = kNearestNeighbors(2,1,3)
    x = [0,1]
    nn.learning(x, 2)
    x = [0,0.9]
    nn.learning(x, 2)
    x = [1,0]
    nn.learning(x, 3)
    x = [0.9,0]
    nn.learning(x, 3)
    x = [0,0]
    nn.learning(x, 4)
    x = [1,1]
    nn.learning(x, 5)
    y = nn.getOutput([0,0.3])
    print("answer", y)