import numpy as np
from PIL import Image
#import pyside_drag_drop_image_file
import LogisticRegression
import pickle
from scipy.special import expit


def mean_std(X):
    return (X - np.mean(X))/np.std(X)
def binary_predict(np_im):
    theta = pickle.load(open("lc_trained_theta00","rb"))
    prediction = LogisticRegression.predict(np_im,theta)
    print(prediction)

def multiclass_predict(np_im):
    predictions = np.zeros([10,1])
    for numClass in range(10):
        theta = np.load("Multiclass_weights6k 10I/class"+str(numClass)+"weights.npy")
        predictions[numClass] = LogisticRegression.predict(np_im,theta)
    print(predictions)
    print("Prediction:" + str(np.where(predictions == max(predictions))[0]) ) 

def multiclass_predict_optim(np_im):
    predictions = np.zeros([10,1])
    theta = np.load("Multiclass_weights60k 5I/multiClass_weights.npy")
    print("Theta shape: "+str(theta.shape))
    print("Image shape: "+str(np_im.shape))
    predictions = LogisticRegression.predict(np_im,theta)
    print(predictions)
    largest = 0
    for i in range(predictions.shape[1]):
        if predictions[0,i] > predictions[0,largest]:
            largest = i
    print("Predicted value:" + str(largest))
    

image = Image.open("Test Images/img_337.jpg").convert('L')
np_im = np.array(image)
np_im= np.reshape(np_im,(1,np_im.shape[1]*np_im.shape[1]))
np_im = mean_std(np_im)

