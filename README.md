# Dynamic DQN-based IoT Scheduler

## Overview

This repository contains the implementation of a **Dynamic DQN-based Scheduler** for **IoT device task allocation**. The scheduler aims to **optimize energy consumption, trustworthiness**, and **system throughput** for a network of **120 IoT devices**. Tasks are assigned dynamically based on computational intensity and energy demands, and the model is trained using **Deep Q-Networks (DQN)** with **federated learning** principles.

The system utilizes the **CIFAR-10 dataset** as a simulation for task demands, and trust scores for IoT devices are evaluated using the **LIME** interpretability algorithm based on historical performance.

### Key Features

- **Dynamic DQN-based Scheduler**: A reinforcement learning-based scheduler that allocates tasks to IoT devices.
- **Federated Learning Integration**: Devices learn collaboratively to improve the task allocation model while preserving privacy.
- **LIME-based Trust Assessment**: Trust scores are assigned to each IoT device based on explainable AI techniques.
- **Simulation Environment**: A custom IoT environment that models energy consumption, trust levels, and utilization of 120 devices.
- **Training over 1000 Iterations**: The model is trained for 1000 iterations to achieve stability and reliability in scheduling.

## Project Structure

- **`dqn_scheduler_training.py`**: Main script for training the DQN-based scheduler over 1000 iterations.
- **`q_network.py`**: Implementation of the Q-network used for action-value estimation.
- **`iot_environment.py`**: Simulation environment for the IoT network, including energy, trust, and utilization metrics for devices.
- **`lime_trust_assessment.py`**: Module for computing trust scores using LIME (Local Interpretable Model-Agnostic Explanations).
- **`README.md`**: This file.

## Requirements

- **Python 3.8+**
- **TensorFlow 2.x**
- **NumPy**
- **LIME**
- **Matplotlib** (Optional, for visualizing training results)

You can install the required packages using `pip`:

```sh
pip install -r requirements.txt

```


## Installation
### 1. Clone the Repository
```sh
git clone https://github.com/gaith7/Dynamic-Deep-Q-Network.git
cd Dynamic-Deep-Q-Network
```

### 2. Install Dependencies:
```sh
pip install -r requirements.txt
```

### 3. Data Setup:

###### The CIFAR-10 dataset is used for simulation. The dataset will be automatically downloaded by TensorFlow when you run the training script.


## How to Run the Project
###### To start training the DQN-based scheduler, use the command:
```sh
python dqn_scheduler_training.py
```








