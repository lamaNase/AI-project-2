## Spam Detection Using k-NN and MLP Classifiers

This repository contains a Python implementation of a spam detection system using 
k-Nearest Neighbors (k-NN) and Multi-Layer Perceptron (MLP) classifiers. The code 
reads a dataset, preprocesses it, trains the models, and evaluates their performance.

  ### Features
    1) Data Loading: Reads a CSV file containing spam data, where each row represents 
        an email with 57 features and a label (1 for spam, 0 for not spam).
    
    2) Preprocessing: Normalizes the feature values by subtracting the mean and dividing 
        by the standard deviation.
    
    3) k-Nearest Neighbors (k-NN) Classifier: Custom implementation to predict the class 
        labels based on the k-nearest neighbors.
    
    4) Multi-Layer Perceptron (MLP) Classifier: Uses scikit-learn's MLPClassifier for 
        training and prediction.
    
    5) Evaluation: Computes accuracy, precision, recall, and F1-score to evaluate model performance.

  ### How to run the code
    you ca use the following command for running the code (make sure that bpth the code 
    and the dataset are in the same directory)
  
> python main.py ./spambase.csv

  ### Results for this dataset
  ![image](https://github.com/user-attachments/assets/14aa7337-2da6-46ce-9193-cdb62265846f)



  ![image](https://github.com/user-attachments/assets/14a27fb7-8408-4939-b85e-10ab62608443)


## Team Members 👥:

<p>
  <img src="https://img.shields.io/badge/Lama%20Naser-blue?style=for-the-badge" alt="Lama Naser">
</p>

<p>
  <img src="https://img.shields.io/badge/Maha%20Mali-yellow?style=for-the-badge" alt="Maha Mali">
</p>  
