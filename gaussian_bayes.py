from __future__ import division
import math
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy

print "Loading data from Spambase"
spam_data=shuffle(numpy.loadtxt('spambase.data.csv', delimiter=','))
features_unscaled=spam_data[:,:-1]
features=preprocessing.StandardScaler().fit_transform(features_unscaled)
labels=spam_data[:,57]

print "Splitting SPAM data into training and test set"
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.50)


#PART I - Classification with Gaussian Naive Bayes
print "\nPART I - Classification with Gaussian Naive Bayes"

non_spam_train = numpy.nonzero(labels_train==0)[0]
spam_train = numpy.nonzero(labels_train==1)[0]
print "\nNo of non-spam datapoints in Training set: ",len(non_spam_train)
print "No of spam datapoints in Training set: ",len(spam_train)

non_spam_test = numpy.nonzero(labels_test==0)[0]
spam_test = numpy.nonzero(labels_test==1)[0]
print "No of non-spam datapoints in Test set: ",len(non_spam_test)
print "No of spam datapoints in Test set: ",len(spam_test)

print "Computing prior probabilities"
spam_prob=int(len(spam_train))/(int(len(spam_train))+int(len(non_spam_train)))
non_spam_prob=int(len(non_spam_train))/(int(len(spam_train))+int(len(non_spam_train)))
print "Spam probability: ",spam_prob
print "Non-Spam probability: ",non_spam_prob
logPclass = numpy.log([non_spam_prob,spam_prob])

indices=[numpy.nonzero(labels_train==0)[0],numpy.nonzero(labels_train)[0]]
mean=numpy.transpose([numpy.mean(features_train[indices[0],:],axis=0),numpy.mean(features_train[indices[1],:],axis=0)])
std=numpy.transpose([numpy.std(features_train[indices[0],:],axis=0),numpy.std(features_train[indices[1],:],axis=0)])
zero_std = numpy.nonzero(std==0)[0]
if (numpy.any(zero_std)):
    numpy.place(std,std==0,0.0001)

pred=[]
for i in range(0, features_test.shape[0]):
#for i in range(0,5):
    denom=math.sqrt(2*numpy.pi)*std
    index=-1*(numpy.divide(numpy.power(numpy.subtract(features_test[i,:].reshape(features_test.shape[1],1),mean), 2),2*numpy.power(std, 2)))
    num=numpy.exp(index)
    #print "index: ", index
    zero_num = numpy.nonzero(num==0)[0]
    #print "num: ",num
    #HACK TO WORK AROUND THE RUNTIME DIVIDE BY ZERO WARNING. LIMITATION DUE TO numpy.exp function giving a 0 for very high index values
    if (numpy.any(zero_num)):
        numpy.place(num,num==0,0.1e-250)
    pdf = numpy.divide(num,denom)
    # Compute class prediction for the test sample
    prediction = numpy.argmax(logPclass+numpy.sum(numpy.nan_to_num(numpy.log(pdf)), axis=0))  
    pred.append(prediction)

acc=accuracy_score(pred,labels_test)
print "\nAccuracy: ",acc
print "\nClassification Report"
print classification_report(labels_test,pred)
print "Confusion matrix"
print confusion_matrix(labels_test,pred)

#PART II - Classification with Logistic Regression
print "\nPART II - Classification with Logistic Regression"
lrm=LogisticRegression()
lrm=lrm.fit(features_train,labels_train)
pred=lrm.predict(features_test)
parameters=lrm.get_params()
print "Parameters: ",parameters
acc=accuracy_score(pred,labels_test)
#Compute Precision, Recall
print "\nClassification Report"
print classification_report(labels_test,pred)
print "Confusion matrix"
print confusion_matrix(labels_test,pred)
print "Accuracy: ",acc








