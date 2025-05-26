# Module 21: Neural Network Model Report

## Overview

The purpse of this analysis is to use machine learning and neural networks to create a model that can help nonprofit foundation Alphabet Soup select its applicants for funding with the best chance of success in their ventures. We will use a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years to preprocess, compile, train and evaluate our model.

## Results

* Data Preprocessing
  * Target variable(s) for the model:
     IS_SUCCESSFUL which measures if the money was used efficiently is the target variable
    
  * Feature variable(s) for the model:
   There are many feature variables in this model namely: APPLICATION_TYPE—Alphabet Soup application type, AFFILIATION—Affiliated sector of industry, CLASSIFICATION—Government organization classification, USE_CASE—Use case for funding, ORGANIZATION—Organization type, STATUS—Active status, INCOME_AMT—Income classification, SPECIAL_CONSIDERATIONS—Special considerations for application, and ASK_AMT—Funding amount requested.
 
  * Variable(s) should be removed from the input data because they are neither targets nor features:
    EIN and NAME which are identification columns and are removed from the data 

* Compiling, Training, and Evaluating the Model

  * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    1. For the initial analysis, I used 2 hidden layers with 80 and 30 neurons respectively and the activation function was ReLU. For the output I used Sigmoid activation function. I used ReLU and Sigmod because it captures the non-linear relationship in the data and is best for our binary classification tasks. Using two hidden layers allows for the model to learn more complex patterns in the data and with ReLU activation the model can train faster. It resulted in an accuracy of 72.55%
       
    2. To optimize the model, I changed the architecture with more neurons and hidden layers. I used 3 hidden layers and 128, 64, and 32 neurons in each. The third hidden layer increases the model depth and allows it to better learn patterns. The 128 → 64 → 32 neuron design mimics the structure of many real-world patterns and is good for dimention reduction and filtering out the noise. I kept ReLU activation and Sigmod output as these are fast and stable. It slightly improved accuracy to 72.57%
   
    3. For next optimization, I used two hidden layers with 75 and 35 neurons in each. I channged the activation function to use tanh in the first layer to well-center the data (e.g., for standardization), and ReLU in the second for faster training and sparse activation, improving efficiency. I also added class_weight balancing to reduces bias toward majority class and improve fairness and recall. However, this attempt slightly decreased my accuracy to 72.36%
   
    4. For my final optimization, I used keras tuner to find the best model parameter. Keras tuner suggested the following hyperparameters are the best: {'activation': 'tanh',
 'first_units': 67,
 'num_layers': 2,
 'units_0': 11,
 'units_1': 47,
 'units_2': 85,
 'units_3': 81,
 'units_4': 7,
 'tuner/epochs': 7,
 'tuner/initial_epoch': 0,
 'tuner/bracket': 1,
 'tuner/round': 0}
   Using tanh keeps activations centered; and two hidden layers are suffient to run the model without over fiitting with 67 neurons in the first hidden layer. It also stops early at 7 epochs which saves training time. However, it only slightly improved the model accuracy to 72.72%
   

  * Were you able to achieve the target model performance?
    No, I was not able to optimize the model to achieve a target predictive accuracy higher than 75%. The best it could get was approximately 73% using keras tuner in the final attempt.

    
  * What steps did you take in your attempts to increase model performance?
    To optimize the model I tried various things like adding more neurons to a hidden layer, adding more hidden layers, and using different activation functions for the hidden layers. Finally, I also reduced the number of epochs and used optimal hyperparameters using keras tuner, but model accuracy was not greater than 75% suggesting some limitations with the data.

## Summary

To summarize, the neural network model gave approximately 73% accuracy on a dataset using two hidden layers and fewer neurons, but the input features do not contain enough predictive information to increase the model accuracy further.
Another model we can use is a tree based model which are good for tabular and structured data and also handle class imbalance well. The tree based model also capture the patterns in the data without to much preprocessing and are less sensitive to irrelavent features.

## References
The code was depuged using Xpert learning assistant. The in class resources were used to create the optimizations

