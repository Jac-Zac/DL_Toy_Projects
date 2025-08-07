# Neural Representations and Backpropagation

> [!WARNING]
> This is in the works

This project is inspired by the seminal paper [_Learning Representations by Back-Propagating Errors_](https://jontallen.ece.illinois.edu/uploads/498-NS.S21/RumelhartHintonWilliams-BackPropError.86.pdf). You'll revisit core concepts from the course by reproducing Figure 1 from the paper using PyTorch, and analyzing the internal representations learned by a neural network.

---

## 📌 Objectives

- 📖 Read and reflect on the paper: _Rumelhart, Hinton & Williams (1986)_
- 🧠 Identify course concepts covered in the paper and in your implementation
- 🔁 Reproduce Figure 1 using PyTorch
- 🗣️ Comment on:
  - Whether your results match the original (it's okay if they don’t)
  - How to interpret the network’s learned representation

---

## 🚀 Quick Start

```bash
# View all options
python src/main.py --help

# Train a model from scratch
python src/main.py --model train

# Load a pre-trained model
python src/main.py --model load
```

## Folder Structure

Here’s a suggested folder structure for your project:

```bash
.
├── src
│   ├── utils
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── environment.py
│   │   └── plot.py
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── train_test.py
├── README.md
└── requirements.txt
```
