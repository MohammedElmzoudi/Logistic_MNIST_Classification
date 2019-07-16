import numpy as np
import pandas as pd
from mlxtend.data import loadlocal_mnist
from scipy.special import expit
import pickle


def process_data(m,select,normalize):
    DataX,DataY = loadlocal_mnist(images_path='train-images-idx3-ubyte',labels_path='train-labels-idx1-ubyte')
    DataY = np.expand_dims(DataY,axis=1)

    DataX = DataX[0:m,:]
    DataY = DataY[0:m,:]

    DataX = np.asarray(DataX)
    DataY = np.asarray(DataY)
    if normalize:
        print("Normalizing Data!")
        DataX = (DataX - np.mean(DataX))/np.std(DataX)
    if select == 0:
        return DataX,DataY
    if select == 1:
        return DataX
    if select == 2:
        return DataY
    else:
        print("Please enter valid selection values. 0 = X and Y, 1 = X, 2 = Y")

def binary_clear_values(X,y):

    Xtemp = np.ones([1,X.shape[1]])
    ytemp = np.array([1])
    for val in range(y.shape[0]):
        if y[val]== 1 or y[val]==0:
            ytemp = np.append( ytemp,y[val],axis = 0 )
            Xtemp = np.append( Xtemp,np.expand_dims(X[val],axis = 0),axis = 0 )

    np.savetxt("mnist0-1_images.csv", Xtemp, delimiter=",")
    np.savetxt("mnist0-1_labels.csv", ytemp, delimiter=",")

def multiclass_create_data(y):

    for numClass in range(10): #Want to go from 0-9
        ytemp = np.array([1])

        for val in range(y.shape[0]):
            if y[val] == numClass:
                ytemp = np.append( ytemp,np.array([1]),axis = 0 )
            else:
                ytemp = np.append( ytemp,np.array([0]),axis = 0 )
        ytemp = np.delete(ytemp,0,0)
        np.savetxt("Multiclass_data60k/mnist"+str(numClass)+"_labels.csv", ytemp, delimiter=",")
        print("Data processed for Class " + str(numClass) +"!")
        print("Data size:" + str(ytemp.shape))


def train_models(X,y,theta,saveLoc,returnError):
    a= 0.01
    m = X.shape[0]
    num_iters = 5000
    error_hist = []

    for i in range(num_iters):
        predictions = predict(X,theta)
        theta = theta - a* (1/m)* np.matmul((predictions - y).T,X )
        if returnError:
            error_hist.append(logistic_error(X,y))
    
    np.save(saveLoc,theta)
    
    if returnError:
        return theta,error_hist
    else:
        return theta

def logistic_error(y,theta):

    pred = predict(X,theta)
    return -(1/X.shape[0])* (np.matmul(y.T,np.log(pred)) + np.matmul((1-y.T),np.log(1-pred))  ) 

def multiclass_logistic_error(theta):
    X = process_data(60000,1,True)
    tot_error = 0

    for numClass in range(10): #load each y class set into m x 10 matrix
        ytemp = pd.read_csv("Multiclass_data60k/mnist" +str(numClass)+"_labels.csv",header = None )
        ytemp = np.asarray(ytemp)
        tot_error += logistic_error(ytemp,theta[numClass,:])

    return tot_error/10    
               
    
def binary_classification_train():
    X = pd.read_csv("mnist0-1_images.csv")
    X = np.asarray(X)
    y = pd.read_csv("mnist0-1_labels.csv")
    y = np.asarray(y)
    print(X.shape)
    theta = np.zeros([1,X.shape[1]])
    X = expit(X)
    
    return train_models(X,y,theta);

def multiclass_classification_train(data_size):
    #Matrix with theta values for each training class, row 1 = theta values for class 1, etc.
    
    X = process_data(data_size,1,True)
    y = np.zeros([X.shape[0],10])
    theta_values  = np.zeros([10,X.shape[1]])
    size = ""
    saveLoc = "Multiclass_weights60k 5I/multiClass_weights.npy"
    
    if(data_size == 60000):
        size = "60k"
    if(data_size == 30000):
        size = "30k"
    if(data_size == 6000):
        size = "6k"
    for numClass in range(10): #load each y class set into m x 10 matrix
        ytemp = pd.read_csv("Multiclass_data"+str(size)+"/mnist" +str(numClass)+"_labels.csv",header = None )
        ytemp = np.asarray(ytemp)
        y[:,numClass] = ytemp[:,0]
    
    print("y shape is"+str(y.shape))
    print("X shape is"+str(X.shape))
    print("Theta shape is"+str(theta_values.shape))
    theta_valuess = train_models(X,y,theta_values,saveLoc,False)
        
        

    
    

def predict(X,theta):
    pred = expit(np.matmul(X,theta.T))
    return pred
















    

