# MLFlow Experiment Tracking with DagsHub

This repository demonstrates the use of MLFlow to track experiments both locally and remotely on DagsHub. The focus of this project is on the prediction of water potability using a dataset available [here](https://github.com/Digvijay25/Datasets).

## Overview

In this project, I have utilized:
- **MLFlow**: For tracking experiments locally.
- **DagsHub**: For tracking experiments remotely.

### Key Features
- **Auto Logging**: Automatically logs metrics, parameters, and models during training.
- **Hyperparameter Tuning**: Optimization of hyperparameters to improve model performance.

## Dataset

The dataset used for this project can be found in my dataset repository. It includes various parameters related to water quality, which are used to predict water potability.

## Experiments

You can view the experiments and their results on DagsHub:
[View Experiments on DagsHub](https://dagshub.com/Digvijay25/mlflow_exp_dagshub.mlflow)

## Getting Started

### Prerequisites

- Python
- MLFlow
- DagsHub

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Digvijay25/mlflow_exp_dagshub.git
    cd mlflow_exp_dagshub
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Experiments

1. Set up MLFlow tracking:
    ```bash
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
    ```

2. Run the script to start tracking experiments:
    ```bash
    python run_experiment.py
    ```

## Conclusion

This project showcases the effective use of MLFlow and DagsHub for experiment tracking. The integration of these tools allows for efficient tracking and management of machine learning experiments, both locally and remotely.
