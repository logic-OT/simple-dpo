# Simple DPO: Understanding Direct Preference Optimization with MNIST

> **Based on my article:** [What Happens When You Train a CNN to Think like an LLM](https://medium.com/@joaolages/direct-preference-optimization-dpo-622fc1f18707)

## 🧠 What is this?
This repository contains a minimalist implementation of **Direct Preference Optimization (DPO)**.

Usually, DPO is discussed in the context of aligning massive Large Language Models (LLMs) using Transformers. This project strips away all that complexity to demonstrate the core algorithm in its purest form: **Training a simple Convolutional Neural Network (CNN) to classify noisy handwritten digits.**

It answers the question: *Can we teach a model to "prefer" the correct answer over a specific mistake, rather than just maximizing the probability of the truth?*

## 📓 The Notebook
The core of this repo is `Mnist_DPO.ipynb`. It guides you through:
1.  **The Task**: Classifying MNIST images that have been corrupted with motion blur and noise.
2.  **The Reference Model**: Pre-training a standard CNN on clean data.
3.  **Generating Preferences**: Automatically creating "Chosen" vs "Rejected" pairs by finding where the Reference Model fails.
4.  **DPO Training**: Implementing the DPO loss function in raw PyTorch to align the model.
5.  **Evaluation**: Comparing the "aligned" Policy Model against the baseline.

## 🚀 Why does this matter?
*   **Intuition**: It uses analogies (like teaching a child) to explain DPO concepts.
*   **Simplicity**: No HuggingFace `TRL`, no tokenizers, no GPUs required (though supported). Just ~200 lines of PyTorch.
*   **Visuals**: You can literally *see* the noisy images and the model's preferences, making the abstract math concrete.

## 🛠️ Getting Started

### Prerequisites
You will need Python and the following libraries:
```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn tqdm tensorflow
```
*(Note: TensorFlow is only used for easy MNIST data loading)*

### Running the Code
1.  Clone this repository.
2.  Open `Mnist_DPO.ipynb` in Jupyter or VS Code.
3.  Run all cells! (The notebook automatically generates the necessary noisy data).

## 📚 Acknowledgements
This project is based on the article **"What Happens When You Train a CNN to Think like an LLM"**, which provides the conceptual framework for applying DPO to computer vision.
