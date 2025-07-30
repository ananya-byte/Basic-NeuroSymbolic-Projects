# 🧠🔢 Project 1: Digit Recognition and Logical Reasoning (MNIST)

## Overview

This project demonstrates a simple **neurosymbolic AI pipeline** by combining a neural network for **digit recognition** with **symbolic logic reasoning**. The system first predicts a handwritten digit using the MNIST dataset, then applies classic mathematical rules to reason about the result (e.g., checking if the digit is even or prime).

---

## 🚀 Goal

- Train a neural network to recognize handwritten digits (0–9).
- Pass the predicted digit to symbolic logic functions to classify it as:
  - Even or Odd
  - Prime or Not Prime
- Illustrate the concept of **perception → reasoning**, a key paradigm in neurosymbolic AI.

---

## 🔧 How It Works

### 1. Neural Step (Perception)

- A small neural network is trained on the **MNIST dataset**.
- The network predicts digits from 0 to 9 given 28×28 pixel grayscale images.

### 2. Symbolic Step (Reasoning)

- Python logic functions analyze the predicted digit to determine:
  - Is it **even** or **odd**?
  - Is it **prime**?

### 3. Integration

- The neural network output is passed to the symbolic reasoning engine.
- The system outputs the digit along with symbolic interpretations.

# 🧠🖼️ Project 2: Image-to-Knowledge Reasoning (CIFAR-10 + Symbolic Logic)

## 📌 Goal

Build a hybrid AI system that combines a neural image classifier with symbolic reasoning. Using the CIFAR-10 dataset, the model predicts object categories (e.g., bird, car) and then reasons about their properties like:

- “Can it fly?”
- “Is it a pet?”

---

## ⚙️ How It Works

### 1. Neural Step: CNN-Based Classification

- A simple CNN is trained to classify CIFAR-10 images into one of 10 categories:
