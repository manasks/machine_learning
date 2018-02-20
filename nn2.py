# -*- coding: utf-8 -*-
"""
@author: Namratha Basavanahalli Manjunatha
@Des   : Perceptrons
"""
#Imports
from __future__ import division
import numpy
import scipy.special
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import *

class perceptron:

    #Constructor
    def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate,momentum):
        #Setting parameters
        self.inodes=inputNodes
        self.hnodes=hiddenNodes
        self.onodes=outputNodes
        self.lrate=learningRate
        self.momentum=momentum
        #Weight between Input and Hidden Layer
        self.wih=numpy.random.uniform(-0.05,0.05,(self.inodes,self.hnodes))
        #Weight between Hidden Layer and Output
        self.who=numpy.random.uniform(-0.05,0.05,(self.hnodes+1,self.onodes))
        #Initialize previous weight value arrays
        self.delta_wih_prev = numpy.zeros(self.wih.shape)
        self.delta_who_prev = numpy.zeros(self.who.shape)
        pass
  
    def activation_function(self,dot_outputs):
        return 1/(1+numpy.exp(-dot_outputs))

    #Train Network
    def train(self,inputs_list,targets_list):
        
        #Calculate Network Outputs
        #Convert into a 2D Array
        inputs = numpy.reshape(inputs_list, (1, self.inodes))
        #Calculate signal into hidden layer
        hidden_inputs=numpy.dot(inputs,self.wih)
        #Pass final inputs through activation function
        hidden_outputs=self.activation_function(hidden_inputs)
        hidden_outputs = numpy.append(hidden_outputs,[1])
        #Calculate signal into output layer
        final_inputs=numpy.dot(hidden_outputs,self.who)
        #Pass final inputs through activation function
        final_outputs=self.activation_function(final_inputs)
        
        targets=numpy.array(targets_list,ndmin=2)
        #Calculate Error
        output_errors=targets-final_outputs
        
        #Calculating error terms
        delta_output = final_outputs*(1-final_outputs)*(output_errors)
        delta_hidden = hidden_outputs*(1-hidden_outputs)*numpy.dot(delta_output,self.who.T) 
       
        hidden_outputs=numpy.array(hidden_outputs,ndmin=2)
        #Caluting delta weights
        delta_who_current = (self.lrate*numpy.dot(hidden_outputs.T, delta_output)) + (self.momentum*self.delta_who_prev)
        delta_wih_current = (self.lrate*numpy.dot(inputs.T, delta_hidden[:,:-1])) + (self.momentum*self.delta_wih_prev)
        
        #print "delta_who_current: ",delta_who_current
        #print "delta_wih_current: ",delta_wih_current

        #Update Weights between output and input
        self.who+=delta_who_current
        self.wih+=delta_wih_current

        self.delta_who_prev = delta_who_current
        self.delta_wih_prev = delta_wih_current

        pass
    
    #Score with Network
    def query(self,inputs_list):
        #Convert input into a 2D Array
        inputs = numpy.reshape(inputs_list, (1, self.inodes))
        #Calculate signal into hidden layer
        hidden_inputs=numpy.dot(inputs,self.wih)
        #Calculate signal into output layer
        #Pass final inputs through activation function
        hidden_outputs=self.activation_function(hidden_inputs)
        hidden_outputs = numpy.append(hidden_outputs,[1])
        #Calculate signal into output layer
        final_inputs=numpy.dot(hidden_outputs,self.who)
        #Pass final inputs through activation function
        final_outputs=self.activation_function(final_inputs)
        return final_outputs
        pass

#Compute accuracy for test data
def compute_testdata_accuracy(ANetwork,confusion=None):
    #Initializing global variable access
    global test_perf_array
    global test_epoch_array
    global prediction_array
    global target_array
    global ep
    #Test scorecard initialization
    test_scorecard=[]
    for record in test_data_list:
        #Reading values from each row
        all_values=record.split(',')
        #Computing actual value
        correct_label=int(all_values[0])
        #Normalizing the values
        inputs=(numpy.asfarray(all_values[1:])/255.0)
        #Adding BIAS input to the normalized input array
        inputs = numpy.append(inputs,[1])
        #Computing output from Neural Network
        outputs=ANetwork.query(inputs)
        #Computing Neural Network output value
        #print "outputs :", outputs
        label=numpy.argmax(outputs)
        #Code to compute confusion matrix inputs
        if confusion is not None:    
            prediction_array.append(label)
            target_array.append(correct_label)
        #print "label : ", label, "correct_label: ",correct_label
        if(label==correct_label):
            test_scorecard.append(1)
        else:
            test_scorecard.append(0)
    #Code to compute the accuracy of the test data 
    #Converting list to array
    test_scorecard_array=numpy.asarray(test_scorecard)
    #Adding array contents to find number of correct predictions
    test_sumval=test_scorecard_array.sum()
    #Computing total number of input values
    test_size=test_scorecard_array.size
    #Computing Accuracy
    test_perf=float(test_sumval/test_size)*100;
    #Appending accuracy to a list for future plotting
    test_perf_array.append(test_perf)
    #Appending epoch value to a list for future plotting
    test_epoch_array.append(ep)
    print("Test Data Performance="+str(test_perf)+"%")

def compute_traindata_accuracy(ANetwork):
    #Compute accuracy for train data
    global train_perf_array
    global train_epoch_array
    train_scorecard=[]
    for record in train_data_list:
        #Reading values from each row
        all_values=record.split(',')
        #Computing actual value
        correct_label=int(all_values[0])
        #Normalizing the values
        inputs=(numpy.asfarray(all_values[1:])/255.0)
        #Adding BIAS input to the normalized input array
        inputs = numpy.append(inputs,[1])
        #Computing output from Neural Network
        outputs=ANetwork.query(inputs)
        #Computing Neural Network output value
        label=numpy.argmax(outputs)
        #Code to compute confusion matrix inputs
        if(label==correct_label):
            train_scorecard.append(1)
        else:
            train_scorecard.append(0)

    #Code to compute the accuracy of the test data 
    #Converting list to array
    train_scorecard_array=numpy.asarray(train_scorecard)
    #Adding array contents to find number of correct predictions
    train_sumval=train_scorecard_array.sum()
    #Computing total number of input values
    train_size=train_scorecard_array.size
    #Computing Accuracy
    train_perf=float(train_sumval/train_size)*100;
    #Appending accuracy to a list for future plotting
    train_perf_array.append(train_perf)
    #Appending epoch value to a list for future plotting
    train_epoch_array.append(ep)
    print("Training Data Performance="+str(train_perf)+"%")

def train_network(ANetwork,train_size=60000):
    print "train_size: ",train_size
    for sample,record in enumerate(train_data_list):
        if sample <= train_size:
            #Training the Neural Network with current input set
            #Split by comma
            all_values=record.split(",")
            #Normalize the values
            inputs=(numpy.asfarray(all_values[1:])/255.0)
            inputs = numpy.append(inputs,[1])
            #Setup target values
            targets=numpy.zeros(output_nodes)+0.1
            targets[int(all_values[0])]=0.9   
            #print "inputs: ",inputs.shape,"targets: ",targets,"target_value: ",int(all_values[0])
            ANetwork.train(inputs,targets)

#Global parameters
ANetwork=None
ep=None

#Network Parameters
input_nodes=785                     #No of input pixels
hidden_nodes_list=[20,50,100]       #No of hidden nodes in the Neural Network
output_nodes=10                     #No of perceptrons in the Neural Network
learning_rate=0.1                   #Learning rate for this homework
epoch=50                            #No of Epochs per learning rate
i=220
momentum_list=[0,0.25,0.5,0.9]      #Momentum Value
fig=1                   

#Initializing lists
test_perf_array=[]                      
test_epoch_array=[]
train_perf_array=[]
train_epoch_array=[]
prediction_array=[]
target_array=[]

#Loading Test Data from csv file
print "Loading Test data from csv file"
test_data_file=open("mnist_test.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()

#Loading Training Data from csv file
print "Loading Training data from csv file"
train_data_file=open("mnist_train.csv","r")
train_data_list=train_data_file.readlines()
train_data_file.close()

def core_code(exp,train_size=60000):
    global hidden_nodes_list
    global ep
    global momentum_list
    global test_perf_array                      
    global test_epoch_array
    global train_perf_array
    global train_epoch_array
    global prediction_array
    global target_array
    i=220 
    print "hidden_nodes_list: ", hidden_nodes_list, "momentum_list: ", momentum_list

    for hidden_nodes in hidden_nodes_list:
        for momentum in momentum_list:
            test_perf_array=[]                      
            test_epoch_array=[]
            train_perf_array=[]
            train_epoch_array=[]
            prediction_array=[]
            target_array=[]
            
            print "hidden_nodes: ", hidden_nodes, "momentum: ", momentum
            #Create a sample network
            ANetwork=perceptron(input_nodes,hidden_nodes,output_nodes,learning_rate,momentum)
            for ep in range(epoch+1):
                
                #Initializing lists
                print "Epoch: ",ep
                if ep==0:
                    #Computing Accuracies for test and train data
                    compute_testdata_accuracy(ANetwork)
                    compute_traindata_accuracy(ANetwork)
                else:
                    #Train the network
                    train_network(ANetwork,train_size)
                    
                    #Computing Accuracies for test and train data
                    compute_testdata_accuracy(ANetwork)
                    compute_traindata_accuracy(ANetwork)

            #Compute accuracy for test data
            test_scorecard=[]
            print "Computing Confusion Matrix"
            compute_testdata_accuracy(ANetwork,confusion=1)
            print(confusion_matrix(target_array, prediction_array))

            #Plotting Accuracy vs Epoch graphs for learning rates 0.1, 0.01 and 0.001
            i+=1
            plt.figure(exp,figsize=(8,6))
            plt.subplot(i)
            plt.title("Hidden Units: %s, Momentum: %s"%(hidden_nodes,momentum))
            plt.plot(test_epoch_array,test_perf_array,label='Test Data')
            plt.plot(train_epoch_array,train_perf_array,label='Training Data')
            plt.legend()
            plt.ylabel("Accuracy %")
            plt.xlabel("Epoch")
            plt.yticks(range(0,100,10))
            plt.ylim(0,100)
            plt.tight_layout()

#Experiment 1
hidden_nodes_list=[20,50,100]       #No of hidden nodes in the Neural Network
momentum_list=[0.9]                 #Momentum Value
core_code(exp=1)

#Experiment 2
hidden_nodes_list=[100]             #No of hidden nodes in the Neural Network
momentum_list=[0,0.25,0.5,0.9]      #Momentum Value
core_code(exp=2)

#Experiment 3
hidden_nodes_list=[100]             #No of hidden nodes in the Neural Network
momentum_list=[0.9]                 #Momentum Value
core_code(exp=3,train_size=15000)
core_code(exp=4,train_size=30000)

plt.show()
print "ALL DONE!"



