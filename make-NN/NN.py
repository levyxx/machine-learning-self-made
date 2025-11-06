import copy
import numpy as np

class Neuron():
    def __init__(self, sizex):
        self.sizex = sizex
        self.w = np.random.rand(sizex+1)
        self.w *= 0.01
        self.y = np.ndarray([])
        self.net = 0
        self.label = ''

    def getOutput(self, x):
        X = copy.deepcopy(x)
        X.append(1)  # bias
        self.net = np.dot(self.w, X)
        self.y = self.net
        return self.y
    
    def learning(self, templateX, labelY):
        self.w = copy.deepcopy(templateX)
        normW = np.linalg.norm(templateX, 2)
        self.w.append(-0.5*normW*normW) # bias
        self.label = copy.deepcopy(labelY)
        print("Neuron.learning() self.w=%s, label=%s" % (self.w, self.label))


class NNPerceptron:
    def __init__(self, inputsize):
        self.inputsize = inputsize
        self.units = []

    def learning(self, x, y):
        ne = Neuron(self.inputsize)
        ne.learning(x,y)
        self.units.append(ne)
    
    def Reasoning(self, x):
        maxOutput = -np.inf
        for i in range(len(self.units)):
            eachUnit = self.units[i]
            out = eachUnit.getOutput(x)
            if maxOutput < out:
                maxOutput = out
                winnerUnit = eachUnit
        print("NNPerceptron.Reasoning() %s" % (out))
        return winnerUnit.label
    

if __name__ == "__main__":
    pc = NNPerceptron(2)
    X = [1,2]
    y = 'red'
    pc.learning(X, y)
    X2 = [3,4]
    y2 = 'yellow'
    pc.learning(X2, y2)
    X3 = [1,3]
    print("winner label=%s-->%s" % (X3, pc.Reasoning(X3)))