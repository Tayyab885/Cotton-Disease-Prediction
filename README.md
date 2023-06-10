# Cotton Disease Prediction

This project aims to predict diseases in cotton plants using deep learning techniques. The goal is to assist farmers in accurately diagnosing and identifying diseases in cotton plants based on images. The model is trained on a dataset of cotton plant images and is capable of classifying images into four categories: diseased cotton leaf, diseased cotton plant, fresh cotton leaf, and fresh cotton plant.

## Data Preparation

The data for training, validation, and testing is stored in separate directories: `Train`, `Val`, and `Test`. Each directory contains images corresponding to the different categories of cotton plants. The data is loaded, preprocessed, and stored in appropriate variables for training and evaluation.

### Data Shuffling

To introduce randomness and reduce bias during training, the training dataset is shuffled using the shuffle function from the sklearn library. This ensures that the order of the images does not affect the model's learning process.

### Data Splitting

The shuffled dataset is split into training, validation, and testing sets. The training set comprises 70% of the data, while the validation and testing sets each contain 15% of the data. The data splitting is performed to ensure unbiased evaluation of the model's performance.

## Model Building

The model is built using deep learning techniques, leveraging the Keras API. It consists of convolutional layers with varying filter sizes, activation functions, and pooling layers. Dropout layers are added to prevent overfitting. The output layer consists of four neurons with a softmax activation function, representing the four possible cotton plant disease categories.

The model architecture is defined, and the necessary hyperparameters are set, including the learning rate, batch size, and number of epochs.

## Model Training

The model is trained using the training data and evaluated using the validation data. During training, the model's performance metrics, including accuracy, loss, precision, recall, and F1 score, are computed and monitored. The training process helps optimize the model's parameters to improve its disease prediction capabilities.

## Model Evaluation

After training, the model's performance is evaluated using the testing dataset that was not seen during training or validation. The accuracy, precision, recall, F1 score, and confusion matrix are calculated to assess the model's ability to predict cotton plant diseases.

## Results

The trained model achieves an accuracy of 87% in classifying cotton plant diseases. This indicates that the model can correctly classify the majority of the images with a high degree of accuracy. The precision, recall, and F1 score for each disease category are also computed to provide insights into the model's performance for individual classes.

The project includes code snippets for data preprocessing, model building, training, evaluation, and result visualization. Additionally, the necessary libraries (such as OpenCV, NumPy, TensorFlow, Keras) are imported and relevant functions are defined to ensure the code runs smoothly.
