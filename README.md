# CNN-RNN-DNN-Classification

# Deep Learning for Image and Sequence Classification

## Overview
This repository contains implementations of **Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Deep Neural Networks (DNNs)** for classification tasks using **PyTorch**. The models are trained on datasets such as **FashionMNIST** (for CNNs and DNNs) and **text classification datasets** (for RNNs).

---
## **Concepts Covered**

### **1. Convolutional Neural Networks (CNNs)**
- CNNs are used primarily for **image classification** by extracting spatial features.
- Uses **convolutional layers**, **pooling layers**, and **fully connected layers**.
- The model applies **ReLU activation**, batch normalization, and dropout.

### **2. Recurrent Neural Networks (RNNs)**
- RNNs are designed for **sequential data processing** such as text classification.
- Includes **vanilla RNNs, GRUs (Gated Recurrent Units), and LSTMs (Long Short-Term Memory Networks).**
- The network learns dependencies in sequences and can be used for NLP tasks.

### **3. Deep Neural Networks (DNNs)**
- Fully connected **feed-forward networks** used for classification.
- Employs **two hidden layers with ReLU activation**.
- Optimized using **Stochastic Gradient Descent (SGD) or Adam optimizer**.

---
## **Mathematical Foundations**

### **1. Gaussian Error Linear Unit (GELU) Activation**
The **GELU** activation function is widely used in modern deep learning architectures like **Vision Transformers (ViTs)** and **BERT**. Defined as:

\[ GELU(x) = x \Phi(x) \]

where \( \Phi(x) \) is the **cumulative distribution function (CDF)** of a Gaussian distribution.

### **2. Cross-Entropy Loss (CE Loss)**
Cross-Entropy Loss is used for **classification problems** and is defined as:

\[ CE(y, \hat{y}) = - \sum_{i} y_i \log(\hat{y_i}) \]

where:
- \( y \) is the true class label (one-hot encoded).
- \( \hat{y} \) is the predicted probability.

**Why is CE loss always non-negative?**
- Since \( log(p) \) is always negative for \( 0 < p \leq 1 \), the loss remains positive.
- The loss reaches **zero** when the model predicts the correct class with **100% confidence (p=1).**

### **3. Backpropagation and Gradient Descent**
- Backpropagation is used to compute the **gradient of the loss function** with respect to the model parameters.
- It updates weights using **Stochastic Gradient Descent (SGD):**

\[ W = W - \eta \frac{\partial L}{\partial W} \]

where:
- \( W \) is the weight matrix.
- \( \eta \) is the learning rate.
- \( \frac{\partial L}{\partial W} \) is the computed gradient.

---
## **Model Implementations**

### **1. CNN Model for Image Classification**
#### **Architecture:**
- **Conv2D → ReLU → MaxPooling → Conv2D → ReLU → MaxPooling → Fully Connected → Softmax**

#### **Dataset:**
- **FashionMNIST** (10 classes of clothing items)

#### **Training:**
- Loss Function: **Cross-Entropy Loss**
- Optimizer: **SGD or Adam**
- Batch Size: **64**
- Epochs: **50**

### **2. RNN Model for Text Classification**
#### **Architecture:**
- **Embedding → RNN (or GRU/LSTM) → Fully Connected → Softmax**

#### **Dataset:**
- **Text classification dataset (6 categories: ABBR, ENTY, DESC, HUM, LOC, NUM)**

#### **Training:**
- Loss Function: **Cross-Entropy Loss**
- Optimizer: **Adam**
- Sequence Length: **Variable**

---
## **Installation**
To run this project, install the required dependencies:
```bash
pip install torch torchvision numpy matplotlib scipy transformers datasets
```

---
## **How to Run**
### **1. Clone the Repository**
```bash
git clone https://github.com/mohibkhann/Deep-Learning-Models.git
cd Deep-Learning-Models
```

### **2. Run CNN Model**
```bash
python train_cnn.py
```

### **3. Run RNN Model**
```bash
python train_rnn.py
```

---
## **Conclusion**
This repository provides a detailed exploration of **CNNs, RNNs, and DNNs**, including **activation functions (GELU), loss functions (Cross-Entropy), and training methods (SGD, Adam).**

### **Future Improvements:**
- Implement **Attention Mechanisms** for RNN models.
- Add **Batch Normalization and Dropout** for DNNs.
- Explore **Transformer-based models** for text classification.

---
**Author:** Mohib Ali Khan

