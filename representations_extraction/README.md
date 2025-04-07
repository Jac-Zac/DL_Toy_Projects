# Representation Extraction & Visualization Exercise

## Quick Start
```bash
# Get all instructions
python src/main.py --help

# Train a new model
python src/main.py --model train

# Use existing model
python src/main.py --model load
```

## Folder Structure

Here’s a suggested folder structure for your project:

```bash
.
├── src
│   ├── utils
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── args_parser.py
│   │   ├── data.py
│   │   ├── environment.py
│   │   └── plot.py
│   ├── __init__.py
│   ├── activation_hook.py
│   ├── main.py
│   ├── model.py
│   └── train_test.py
├── README.md
└── requirements.txt
```

## Overview

This exercise focuses on extracting and visualizing neural network representations using the MNIST dataset. You'll train a small MLP and apply dimensionality reduction (UMAP/T-SNE) to visualize how representations evolve during training.

## Steps to Complete the Exercise

### 1. **Choose a Dataset**
   - Use the **MNIST dataset**, which is already available in the `data/MNIST` folder.

### 2. **Construct a Small MLP**
   - Build a multi-layer perceptron (MLP) with the following architecture:
     - 128 units in the first hidden layer
     - 64 units in the second hidden layer
     - 32 units in the third hidden layer
   - You are free to modify the number of units in each layer based on your exploration.

   > [!NOTE]
   > It's important to ensure that your MLP is not too large, as this is a small dataset. This will allow you to observe meaningful representations without overfitting.

### 3. **Train the Model**
   - Train your MLP using an optimizer of your choice (e.g., SGD, Adam, etc.).
   - Ensure that your model achieves reasonable performance, such as an accuracy of ~98%.

### 4. **Extract Representations**
   - After training, extract the representations from the test set.
   - Store these representations in matrices for further analysis.

### 5. **Project the Representations**
   - Use **UMAP** (recommended) or **T-SNE** to project the extracted representations into a 2D space.
   - **Color-code the projections** based on their class labels to visualize how well the model has learned to separate different classes.
   
   [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/basic_usage.html)
   
   > [!WARNING]
   > UMAP is preferred for its ability to preserve local and global structures better than T-SNE in many cases, especially for higher-dimensional datasets.

### 6. **Compare Projections**
   - Perform the same projections on both the **training** and **test** data.
   - Compare the two 2D maps to see how well the representations generalize from training to test data.

   > [!NOTE]
   >   You may observe that the training data projections form tighter clusters, while the test data projections may appear more spread out depending on model performance.

### 7. **Monitor Representations During Training**
   - Select one or more hidden layers to monitor during training.
   - After each epoch or at regular intervals, extract and project the representations of the chosen layers into 2D space.
   - This will help you visualize how representations evolve as the model learns.

   > [!TIP]
   > Start by monitoring the output of the first hidden layer to understand the early stages of learning, then move to deeper layers as you progress.

### 8. **Include Pre-training Representations**
   - Before training, extract and store representations from the selected layers.
   - Compare these pre-training representations with those obtained during training to observe how they evolve over time.

   > [!IMPORTANT]
   >  This step is crucial to understanding how the network gradually "learns" and refines its internal representations.
