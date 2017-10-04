import scipy.io as sio
import numpy as np
import math
from matplotlib import pyplot as plt
from random import randint

class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron=connectedNeuron
        self.weight=0.1
        self.dWeight=0.0
        
class Neuron:
    eta=0.1; #learning rate
    alpha=0.9; #momentum rate
    def __init__(self,layer):
        self.dendrons=[]
        self.error=0.0
        self.gradient=0.0
        self.output=0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con=Connection(neuron)
                self.dendrons.append(con)
    
    def addError(self,err):
        self.error=self.error+err;
        
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x*1.0))
    
    def dSigmoid(self,x):
        return x*(1.0-x);
    
    def setError(self, err):
        self.error=err;

    def setOutput(self, output):
        self.output=output;

    def getOutput(self):
        return self.output;
    
    def feedForword(self):
        sumOutput=0.0;
        if(len(self.dendrons)==0):
            return
        for dendron in self.dendrons:
            #print ("output " + str(dendron.connectedNeuron.getOutput()) + str(type(dendron.connectedNeuron.getOutput())))
            #print ("Weight " + str(dendron.weight) + str(type(dendron.weight)))
            sumOutput=sumOutput+dendron.connectedNeuron.getOutput()*dendron.weight;
        self.output=self.sigmoid(sumOutput);
    
    def backPropagate(self):
        self.gradient=self.error*self.dSigmoid(self.output);
        for dendron in self.dendrons:
            dendron.dWeight=Neuron.eta*(dendron.connectedNeuron.output*self.gradient)+self.alpha*dendron.dWeight;
            dendron.weight=dendron.weight+dendron.dWeight;
            dendron.connectedNeuron.addError(dendron.weight*self.gradient);
        self.error=0;
class Net:
    def __init__(self,topology):
        self.layers=[]
        for numNeuron in topology:
            layer=[]
            for i in range(numNeuron):
                if(len(self.layers)==0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1);
            self.layers.append(layer)

    def setInput(self,inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword();

    def backPropagate(self,target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i]-self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    def getError(self,target):
        err=0
        for i in range(len(target)):
            e=(target[i]-self.layers[-1][i].getOutput())
            err=err+e*e
        err=err/len(target)
        err=math.sqrt(err)
        return err
        
        return target-self.layers[-1][0].getOutput()
    def getResults(self):
        output=[]
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        return output
    def getThResults(self):
        output=[]
        for neuron in self.layers[-1]:
            o=neuron.getOutput()
            if(o>0.5):
                o=1
            else:
                o=0
            output.append(o)
        return output

def main():
    topology=[];
    topology.append(2);
    topology.append(3);
    topology.append(5);
    topology.append(1);
    net=Net(topology);

    iteration = 0
    error_rate=[]
    no_of_iterations=[]
    while True:
        
        iteration=iteration+1
        err=0
        '''inputs=[[0,0],[0,1],[1,0],[1,1]]
        outputs=[[0],[1],[1],[1]]'''
		
        '''inputs = [[0,0],[0,1],[1,0],[1,1]]
		outputs = [[0],[0],[0],[1]]'''
		
        inputs = [[1,2],[2,5],[15,13],[6,6],[7,10],[20,11],[21,15]]
        outputs = [[1],[1],[0],[0],[1],[0],[0]]
		
        for i in range(len(inputs)):
            print ("input:" +str(inputs[i]))
            net.setInput(inputs[i])
            net.feedForword()
            net.backPropagate(outputs[i])
            print ("output:"+str(net.getResults()))
            err=err+(net.getError(outputs[i])*net.getError(outputs[i]))
	
        print ("Error : " + str(err))
        error_rate.append(err)
        no_of_iterations.append(iteration)
        if err<0.01:
            break
			
    plt.plot(no_of_iterations,error_rate)
    plt.show()
    
    while True:
        print ("Neural Network Implementation")
        print ("Inputs Entry")
        i1 = int(input("Enter the first input : "))
        i2 = int(input("Enter the second input: "))
        i = [i1,i2]
        net.setInput(i);
        net.feedForword();
        print (net.getResults())

        
if __name__=='__main__':
    main()