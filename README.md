**Project Overview**
This project aims to utilize deep learning techniques to diagnose faults in optical lens systems. By employing a neural network, the system can identify deviations in lens positioning and other issues that may arise during the assembly and calibration of optical systems. The goal is to enhance the efficiency and accuracy of optical system troubleshooting, reducing the time and cost associated with traditional methods.

**Project Structure**

**1. train_resnet.py**
This script is responsible for training the neural network model using residual connections. The network is designed to identify faults in lens systems by analyzing the locations where projected rays hit a screen behind the optical system.

Key Components:
Network Structure: Defines the architecture of the neural network, including residual blocks and fully connected layers.
Data Loading: Loads and preprocesses the dataset, which consists of spot diagrams and corresponding lens deviations.
Training Loop: Implements the training process, including data normalization, forward and backward passes, and model evaluation.

**2. preprocess.py**
This script handles the preprocessing of data, including loading and stacking arrays from NPZ files. It ensures that the data is in the correct format for training the neural network.

Key Components:
Data Loading: Loads data from NPZ files and processes it into a format suitable for training.
Data Filtering: Filters out incomplete or corrupted data entries.
Data Normalization: Normalizes the data to ensure consistency during training.

**Prerequisites**
Ensure you have the following installed:

Python 3.x
PyTorch
NumPy
Pandas
Matplotlib
tqdm
