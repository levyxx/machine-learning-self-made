import numpy as np
from numpy.random import *
import copy

class Neuron():
    def __init__(self, sizex): # constructor 
        self.sizex = sizex # input size
        self.w = 0.01 * np.random.rand(sizex+1) #+1 means bias
        print("Neuron._init__ self.w=%s" %(self.w))
        self.y = np.ndarray([]) # output of this unit
        self.deltaW = np.zeros(sizex+1) # initialization of delta W
        self.net = 0# net value of this unit
        self.DeltaP = 0 # delta value from the next layer
        self.debug = False
            
    def getOutput(self, x):
        X = copy.deepcopy(x) # create X, which includes x and bias value:1.
        X.append(1)
        self.net = np.ndarray([])
        if self.debug:
            print("Neuron.getOutput() X=%s" %(X))
            print("Neuron.getOutput() self.w=%s" %(self.w))
        np.dot(self.w, X, self.net) # net value is the dotproduct of w and X.
        if self.debug:
            print("Neuron.getOutput() self.net=%s" %(self.net))
        self.y = 1/(1+np.exp(-self.net)) # calculate output value y
        if self.debug:
            print("Neuron.getOutput() y=%s" %(self.y))
        return self.y
    
    def resetDelta(self):
        for i in range(self.sizex+1):
            self.deltaW[i] = 0
        self.DeltaP = 0
        
    def addDelta(self, deltaP):
        self.DeltaP = self.DeltaP + deltaP
        
    def calcDeltaW(self, x):
        self.getOutput(x) # At first, the output value is calculated
        X=copy.deepcopy(x) # Create X = [1,x]
        X.append(1)
        if self.debug:
            print("Neuron.calcDeltaW() X=%s" %(X))
        delta = self.y * (1-self.y) # gradient of sigmoid
        self.DeltaP *= delta
        for i in range(self.sizex+1):  
            self.deltaW[i] = self.DeltaP * X[i] # calculate deltaW[]
    
    def UpdateW(self, eta): # Update W
        for i in range(self.sizex+1):
            self.w[i] = self.w[i] + eta * self.deltaW[i]
            
    def getDelta(self, i):
        return self.DeltaP * self.w[i]
    
    def display(self):
        print("Neuron.sizex=%s " %(self.sizex))
        
class NeuronLayer():
    def __init__(self, size_input, size_output):
        self.size_input = size_input
        self.size_output = size_output
        self.units = []
        self.outputs = []
        for cell in range(self.size_output):
            neuron = Neuron(self.size_input)
            self.units.append(neuron)
    
    def getOutputs(self, x):
        self.outputs = []
        for index in range(self.size_output):
            eachNeuron = self.units[index]
            self.outputs.append(eachNeuron.getOutput(x))
        return self.outputs
    
    def display(self):
        print("MyPerceptrons.NeuronLayer.display() # of units = %s" %(len(self.units)))
        
    def resetDelta(self):
        for eachCell in self.units:
            eachCell.resetDelta()
                       
    def addDelta(self, deltas):
        for cellIndex in range(len(self.units)):
            eachCell = self.units[cellIndex]
            eachCell.addDelta(deltas[cellIndex])
            
    def getBackProp(self):
        deltas = np.zeros(self.size_input)
        for i in range(self.size_input):
            for cellIndex in range(len(self.units)):
                eachCellObj = self.units[cellIndex]       
                deltas[i] += eachCellObj.getDelta(i)
        return deltas
    
    def setUpDeltaW(self, x):
        for eachUnit in self.units:
            eachUnit.calcDeltaW(x)
            
    def updateW(self, eta):
        for eachUnit in self.units:
            eachUnit.UpdateW(eta)
        
class PerceptronFC(): # full connected perceptron
    def __init__(self, sizex, sizeh, sizey, maxlayer): # size of input, size of hidden units, max layer
        self.sizex = sizex
        self.sizeh = sizeh
        self.sizey = sizey
        self.maxlayer = maxlayer 
        self.layerUnits = []
        self.debug = False
        self.outputs = []
        #allocate each layer
        for l in range(maxlayer):
            if 0 <= l < maxlayer-1 :
                if l==0:
                    layerUnit = NeuronLayer(self.sizex, self.sizeh)
                else:
                    layerUnit = NeuronLayer(self.sizeh, self.sizeh)
       
            else:
                layerUnit = NeuronLayer(self.sizeh, self.sizey)
            self.layerUnits.append(layerUnit)
                    
    def getOutputs(self, x):
        self.outputs = [] # outputs from each layer
        for layer in range(self.maxlayer):
            layerUnits = self.layerUnits[layer]
            if layer==0:
                self.outputs.append(layerUnits.getOutputs(x))
            else:
                self.outputs.append(layerUnits.getOutputs(self.outputs[layer-1]))
        return self.outputs[self.maxlayer-1]
 
    
    def learning(self, x, y):
        err = (np.array(y)-np.array(self.getOutputs(x)))
        squareErr = np.linalg.norm(err, 2)
        X=copy.deepcopy(x)
        self.resetDelta()
        upperDelta = []
        for layer in reversed(range(self.maxlayer)):
            #print("PerceptronFC.learning() layer=%s" %(layer))
            index = 0
            layerUnit = self.layerUnits[layer] 
            if layer == self.maxlayer-1:
                layerUnit.addDelta(err)
                upperDelta = layerUnit.getBackProp() 
            else:
                layerUnit.addDelta(upperDelta)
                upperDelta = layerUnit.getBackProp()
 
        for layer in range(self.maxlayer):
            layerUnit = self.layerUnits[layer]
            if layer == 0:      
                layerUnit.setUpDeltaW(X)
                layerUnit.updateW(0.1)
            else:
                layerUnit.setUpDeltaW(self.outputs[layer-1])
                layerUnit.updateW(0.1)
        return squareErr
    
    def resetDelta(self):
        for eachlayer in self.layerUnits:
                eachlayer.resetDelta()
    
        
if __name__ == "__main__":
    pc = PerceptronFC(12, 15, 1, 2)
    learningInput = []
    learningOutput = []
    sampleIndex = 0
    for line in open('./test0.dat', 'r'):
        items = line.split(' ')
        linewise = []
        linewiseOut = []
        for xindex in range(0,12):
            linewise.append(float(items[xindex]))
        learningInput.append(linewise)
        linewiseOut.append(float(items[12]))
        learningOutput.append(linewiseOut)
    
    for n in range(500):
        totalErr = 0
        for l in range(len(learningInput)):
            totalErr += pc.learning(learningInput[l], learningOutput[l])
        print("totalErr = %s" % (totalErr))

    for n in range(167):
        output = pc.getOutputs(learningInput[n])
        print("test input = %s, desired output = %s, actual output = %s" % (learningInput[n], learningOutput[n], output))