from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
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

print "Model fitting"
from sklearn.svm import SVC
#Using Linear Model for SVM
clf=SVC(kernel="linear")
#Model fitting based on training data
clf.fit(features_train,labels_train)
#Prediction based on the model
pred=clf.predict(features_test)

#Experiment 1
print "\nExperiment 1"
from sklearn.metrics import accuracy_score
#Compute accuracy
acc=accuracy_score(pred,labels_test)
print "Accuracy: ",acc
#Computer Precision, Recall
print "\nClassification Report"
print classification_report(labels_test,pred)
scores_test=clf.decision_function(features_test)
#Calculate FPR, TPR and Thresholds
fpr,tpr,thresholds=metrics.roc_curve(labels_test, scores_test)
#Calculate Area Under Curve
auc=roc_auc_score(labels_test, scores_test)

pyplot.figure(1)
pyplot.plot(fpr, tpr, label='ROC Curve\n(area under curve = %f)' %auc,  lw=2)
pyplot.plot([0, 1],[0, 1],lw=2,linestyle='--',label='Random Guess')
pyplot.xlabel('\nFalse Positive Rate\n')
pyplot.ylabel('\nTrue Positive Rate\n')
pyplot.title('\nAccuracy of SVM model trained on Spambase')
pyplot.legend(loc='lower right')

#Experiment 2
print "\nExperiment 2"
#Get weights from model
coefs=numpy.copy(clf.coef_)
print "Coefficients: "
print coefs
top_positive_coefficients = numpy.argsort(coefs[0])[-5:]
top_negative_coefficients = numpy.argsort(coefs[0])[:5]
print "\nTop Positive Features: ", top_positive_coefficients
print "Top Negative Features: ", top_negative_coefficients
i=numpy.argmax(coefs)
features_train_max=numpy.array(features_train[:,i],ndmin=2).T
features_test_max=numpy.array(features_test[:,i],ndmin=2).T
coefs[0][i]=float('-Infinity')
m_array=[]
acc_array=[]
for m in range(2,58):
    i=numpy.argmax(coefs)
    coefs[0][i]=float('-Infinity')
    features_train_max=numpy.insert(features_train_max,0,features_train[:,i],axis=1)
    features_test_max=numpy.insert(features_test_max,0,features_test[:,i],axis=1)
    clf.fit(features_train_max,labels_train)
    #Run test data throught the model
    pred=clf.predict(features_test_max)
    acc=accuracy_score(pred,labels_test)
    m_array.append(m)
    acc_array.append(acc)
    print "m: ",m,"\tAccuracy: ",acc

pyplot.figure(2)
pyplot.title("Feature Selection with Linear SVM")
pyplot.plot(m_array,acc_array)
pyplot.xlabel("m")
pyplot.ylabel("Accuracy")


m_array=[]
acc_array=[]
#Experiment 3
print "\nExperiment 3"
for m in range(2,58):
    feature_indices=numpy.random.choice(numpy.arange(57),m,replace=0)
    features_train_max=features_train[:,feature_indices]
    features_test_max=features_test[:,feature_indices]
    clf.fit(features_train_max,labels_train)
    #Run test data throught the model
    pred=clf.predict(features_test_max)
    acc=accuracy_score(pred,labels_test)
    m_array.append(m)
    acc_array.append(acc)
    print "m: ",m,"\tAccuracy: ",acc

pyplot.figure(3)
pyplot.title("Random Feature Selection")
pyplot.plot(m_array,acc_array)
pyplot.xlabel("m")
pyplot.ylabel("Accuracy")

pyplot.show()
