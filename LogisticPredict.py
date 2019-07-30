import numpy as np
from PIL import Image
import pickle
import LogisticRegression
from scipy.special import expit


def mean_std(X):
    return (X - np.mean(X)) / np.std(X)


def binary_predict(image_loc, theta_loc):
    np_im = convert_image(image_loc)
    theta = pickle.load(open(theta_loc, 'rb'))
    prediction = LogisticRegression.predict(np_im, theta)
    print("Output:", str(prediction))
    if (prediction >= 0.5):
        print("Predicted Value:", str(1))
    else:
        print("Predicted Value:", str(0))

    return prediction

def multiclass_predict(image_loc):
    np_im = convert_image(image_loc)
    predictions = np.zeros([10, 1])
    for numClass in range(10):
        theta = np.load("Multiclass_weights6k 10I/class" + str(numClass) + "weights.npy")
        predictions[numClass] = LogisticRegression.predict(np_im, theta)
    print(predictions)
    print("Prediction:" + str(np.where(predictions == max(predictions))[0]))


def multiclass_predict_optim(image_loc, weights_loc):
    np_im = convert_image(image_loc)
    predictions = np.zeros([10, 1])
    theta = np.load(weights_loc)
    print("Theta shape: " + str(theta.shape))
    print("Image shape: " + str(np_im.shape))
    predictions = LogisticRegression.predict(np_im, theta)
    print(predictions)
    largest = 0
    for i in range(predictions.shape[1]):
        if predictions[0, i] > predictions[0, largest]:
            largest = i
    print("Predicted value:" + str(largest))
    return largest


def convert_image(image_loc):
    image = Image.open(image_loc).convert('L')
    np_im = np.array(image)
    np_im = np.reshape(np_im, (1, np_im.shape[1] * np_im.shape[1]))
    np_im = mean_std(np_im)
    return np_im
