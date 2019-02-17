# IFT6134 - Assignment 1 - Practical part

Solutions to the practical problems of assignment 1 in IFT 6135 winter 2019.

Open this repository in your favorite IDE to run the code Kaggle and MLP code. Install necessary python packages.

The repository is divided into three parts, one for each problem:
- `mlp`
  - The Initialization experiment can be run with mlp/src/experiments/init_experiments.py
  - The Hyper Parameter search can be run with mlp/src/experiments/hyper_parameter_experiments.py
  - The finite difference gradient check can be run with mlp/src/experiments/gradient_check_experiment.py
- `convnet`
- `kaggle`
  - Setup by copying all the training examples into kaggle/data/trainset and all the test examples into kaggle/data/testset
  - Train by running kaggle/main.py. This will save a model called best_model.bak
  - Predict labels using the trained model by running kaggle/predict.py. This will use best_model.bak to predict. 