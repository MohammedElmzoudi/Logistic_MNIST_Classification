# Logistic_MNIST_Classification
From "scratch" logistic regression handwritten digit classifier trained on MNIST dataset
## Description
- A Logistic Regression Classification model created using only numpy and other basic libraries trained on 60,000 26x26 MNIST images.
This project includes a binary classifier and a multiclass classifier using an implementation of the "One vs. All" logistic regression technique.
## (Very)Brief Explination 
Logistic Regression is a binary method of classification that can predict a value between 1 and 0 for a given set of data  features, the closer the value is to 1, the higher the probability that the positive output is correct.
For example - Predicting whether a tumor is benign(1) or malignant(0) given data such as the size of the tumor, shape, etc, the model returns a value of 0.8. This means the model predicts that the tumor has an 80% chance of being benign and a 20% chance of being malignant
  
- In this case however, Logistic Regression is used to predict the values of handwriten digits 0-9  
  Now you may be wondering, if Logistic Regression is only a binary classifier, how can it recognize digits 0-9?  
  Good question. 
- Using the "One vs. All" Logistic technique, we can accomplish this. "One vs. All" aims to solve the problem of multiclass   classification by training multiple models for each possible output value (in this example, the numbers 0-9) on a dataset   that is 'currated' for each output value.  
- This currated dataset contains all the previous data, but the output values are replaced by ones and zeros to allow for     binary classification. For example the dataset that is currated for the output of 5, would replace every instance of a 5     with the value of 1 and all other output values with a zero. By doing this you can get the predict the probability that     the number is a 5.   
- For example, the model trained on currated dataset 5 returns a value of 0.2. This means that there is a 20% chance           that the number is indeed a 5, and a 80% chance that the correct output is some number other than 5.  
  Now just repeat this process for each output value and select the prediction with the highest probability!



## Installation
1) Make sure you have the following packages installed
```
Numpy, Pandas, mlxtend, scipy, PIL
```
2) Download zip file and you're ready to go!

## Usage
### Predicting Values:    
#### To run binary model on pre-trained hyperparameters
- Place desired image in project folder 
- Run following command in project folder location: 
```
python -c 'import LogisticPredict; LogisticPredict.binary_predict('Image_Location','Weights_location')'
```
- Replace 'Image_Location' with the name of your image  
- Replace 'Weights_Location' with the location of your desired weights  
  -> Only pretrained binary wieghts are in location - 'Binary_weights10k 10I/lc_trained_theta00'
  
#### To run multi-class model on pre-trained hyperparameters
- Place desired image in project folder  
- Run following command in project folder location: 
```
python -c 'import LogisticPredict; LogisticPredict.multiclass_predict_optim('Image_Location','Weights_location')'
```
- Replace 'Image_Location' with the name of your image  
- Replace 'Weights_Location' with the location of your desired weights  
  -> The following weight locations are avaliable:  
        - Multiclass_weightsX Y  
          -- X refers to the number of training samples used  
          -- Y refers to the number of training iterations  
       'Multiclass_weights6k 5I/multiclass_weights.npy'   
       'Multiclass_weights6k 10I/multiclass_weights.npy'  
       'Multiclass_weights30k 5I/multiclass_weights.npy'  
       'Multiclass_weights60k 5I/multiclass_weights.npy'  
       
   
## Training Custom Weights
### Binary Training
#### 'Create' Binary Data
- Given a full MNIST dataset you can separate out the data corresponding to ones and zeros with the following command:
```
python -c 'import LogisticRegression; LogisticRegression.binary_clear_values('full_training_images_loc','full_training_')'
```
