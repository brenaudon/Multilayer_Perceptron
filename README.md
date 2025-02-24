# **Activation Functions**

## 🔹 **1. ReLU (Rectified Linear Unit)**
### **Definition**
ReLU is one of the most widely used activation functions in deep learning. It is defined as:

```math
f(Z) = \max(0, Z)
```

This means that for any negative input, the output is 0, and for any positive input, the output is the same as the input.

### **Derivative**
The derivative of ReLU is simple:

```math
f'(Z) =
\begin{cases}
1, & Z > 0 \\
0, & Z \leq 0
\end{cases}
```

### **Pros**
✅ Helps mitigate the **vanishing gradient problem**.  
✅ Computationally efficient (simple operations).  
✅ Works well in deep networks.

### **Cons**
❌ **Dying ReLU problem**: If a neuron only gets negative inputs, it stops learning (gradient = 0).  
❌ Does not handle negative values well.

### **Best Use Cases**
- Works well for deep networks.
- Recommended for **hidden layers** in most architectures.

---

## 🔹 **2. Leaky ReLU**
### **Definition**
Leaky ReLU solves the **dying ReLU problem** by allowing a small slope for negative values:

```math
f(Z) = \max(0.01Z, Z)
```

This means that instead of setting negative values to **0**, we allow a small negative slope (usually **0.01**, but it can be tuned).

### **Derivative**
```math
f'(Z) =
\begin{cases}
1, & Z > 0 \\
0.01, & Z \leq 0
\end{cases}
```

### **Pros**
✅ Avoids **dying ReLU problem** (neurons always have some gradient).  
✅ Works better than ReLU when dealing with negative values.

### **Cons**
❌ Slightly more computationally expensive than ReLU.

### **Best Use Cases**
- Use **instead of ReLU** when you notice many neurons **dying (stuck at 0)**.

---

## 🔹 **3. ELU (Exponential Linear Unit)**
### **Definition**
ELU is similar to Leaky ReLU but uses an **exponential curve** instead of a fixed slope for negative values:

```math
f(Z) =
\begin{cases}
Z, & Z > 0 \\
\alpha (e^Z - 1), & Z \leq 0
\end{cases}
```

where **α (alpha)** is usually set to **1**.

### **Derivative**
```math
f'(Z) =
\begin{cases}
1, & Z > 0 \\
f(Z) + \alpha, & Z \leq 0
\end{cases}
```

### **Pros**
✅ Avoids dying ReLU problem.  
✅ Allows small negative values, making it more robust than ReLU.  
✅ Can speed up learning.

### **Cons**
❌ Slightly more computationally expensive.  
❌ Requires tuning the **α** parameter.

### **Best Use Cases**
- Works well for deep networks where ReLU might struggle.

---

## 🔹 **4. SELU (Scaled Exponential Linear Unit)**
### **Definition**
SELU (Scaled Exponential Linear Unit) is an activation function designed to **self-normalize** neural networks. It was introduced in the paper:  
👉 *Klambauer et al., 2017 - "Self-Normalizing Neural Networks"*

The SELU activation function is defined as:

```math
SELU(x) =
\begin{cases}
\lambda x & \text{if } x > 0 \\
\lambda \alpha (e^x - 1) & \text{if } x \leq 0
\end{cases}
```

Where:

- $\( \lambda \approx 1.0507 \)$ (scaling factor)
- $\( \alpha \approx 1.67326 \)$ (negative slope factor)


This scaling ensures that activations **automatically normalize** their mean and variance, helping deep networks stabilize during training.


### **Derivative**
```math
SELU'(x) =
\begin{cases}
\lambda & \text{if } x > 0 \\
\lambda \alpha e^x & \text{if } x \leq 0
\end{cases}
```

### **Pros**
✅ **Self-normalizing** — reduces the need for Batch Normalization.  
✅ **Helps with vanishing/exploding gradients** due to its smooth non-linearity.  
✅ **Better for deep networks** — prevents the network from suffering from vanishing gradients or saturation issues.  
✅ **Works well with LeCun Normal initialization and Alpha Dropout**.  
✅ **Can help improve convergence speed and stability in deep networks**.

### **Cons**
❌ **Not effective in CNNs and RNNs** — works best for fully connected (dense) networks.  
❌ **Requires careful weight initialization (LeCun Normal)** and **Alpha Dropout**.  
❌ **Computationally more expensive** than simpler activation functions like ReLU, due to the exponential part for negative values.

### **Best Use Cases**
- Deep **fully connected networks** (especially for classification and regression tasks).
- When you want **self-normalization** and stable training without extra normalization layers like BatchNorm.
- Networks where careful **weight initialization** (LeCun Normal) and **Alpha Dropout** are used to keep activations properly normalized.
- Tasks like **image classification**, **speech recognition**, or **NLP** (where deep, stable networks are crucial).

---

## 🔹 **5. Swish (Self-Gated)**
### **Definition**
Swish is a **smooth, non-monotonic** function developed by Google that often outperforms ReLU:

```math
f(Z) = Z \cdot \sigma(Z) = \frac{Z}{1 + e^{-Z}}
```

where **σ(Z)** is the **sigmoid** function.

### **Derivative**
```math
f'(Z) = \sigma(Z) + Z \cdot \sigma(Z) \cdot (1 - \sigma(Z))
```

### **Pros**
✅ Helps prevent **dying neurons**.  
✅ Works well in **deep networks**.  
✅ Often outperforms ReLU.

### **Cons**
❌ More computationally expensive than ReLU.

### **Best Use Cases**
- Deep networks (especially for NLP and computer vision).
- If you want to **experiment** beyond ReLU.

---

## 🔹 **6. GELU (Gaussian Error Linear Unit)**
### **Definition**
GELU is used in **BERT, GPT, and transformers**. Instead of using a simple threshold (like ReLU), it smoothly adjusts activations based on a Gaussian curve:

```math
f(Z) = 0.5 Z \left(1 + \tanh \left( \sqrt{\frac{2}{\pi}} \left( Z + 0.044715 Z^3 \right) \right) \right)
```

### **Derivative**
The derivative is more complex, but it helps with **smooth gradient updates**:

```math
f'(Z) = 0.5 \left(1 + \tanh \left( \sqrt{\frac{2}{\pi}} \left( Z + 0.044715 Z^3 \right) \right) \right) + 0.5 Z \left(1 - \tanh^2 \left( \sqrt{\frac{2}{\pi}} \left( Z + 0.044715 Z^3 \right) \right) \right) \left( \sqrt{\frac{2}{\pi}} \left(1 + 3 \times 0.044715 Z^2 \right) \right)
```

### **Pros**
✅ Used in **state-of-the-art deep learning models**.  
✅ Helps with training deep architectures.

### **Cons**
❌ Very computationally expensive.

### **Best Use Cases**
- If you're working on **transformers** or large **deep networks**.
- If you want to **experiment with cutting-edge techniques**.

---

### 🔥 **Summary Table**

| Activation Function | Best For | Pros | Cons |
|--------------------|---------|------|------|
| **ReLU** | Most deep learning tasks | Fast, simple, avoids vanishing gradients | Can have dead neurons (dying ReLU problem) |
| **Leaky ReLU** | Tasks where ReLU fails | Prevents dying neurons | Slightly more expensive |
| **ELU** | Faster training, better convergence | No dead neurons, good for deep networks | Requires tuning α, slower than ReLU |
| **SELU** | Deep, fully connected networks | Self-normalizing, reduces need for BatchNorm, good for deep networks | Requires careful weight initialization and Alpha Dropout |
| **Swish** | Deep networks (Google-developed) | Can outperform ReLU | More complex to compute |
| **GELU** | Transformer models (BERT, GPT) | Used in state-of-the-art networks | Computationally expensive |

---

# **Initialization functions**

## 🚀 **Weight Initialization Techniques in Deep Learning**
Weight initialization is crucial in training deep neural networks because it influences how gradients propagate during backpropagation. Poor initialization can lead to issues like vanishing or exploding gradients.

Below is a **detailed list of weight initialization techniques**, their mathematical background, and the activation functions they work best with.

---

## 🔹 **1. Zero Initialization (For bias matrix)**
### **Definition:**
All weights are initialized to **zero**, and biases are typically also initialized to **zero**.

```math
W = 0, \quad b = 0
```

### **Problems:**
- If all weights are zero, neurons in the same layer will receive the same gradients.
- This symmetry makes neurons learn the same features, making the network ineffective.

### **Best Used With:** ❌ **Not recommended for deep networks.**
It can be used **only** for bias initialization.

---

## 🔹 **2. Random Initialization**
### **Definition:**
Weights are initialized randomly using a uniform or normal distribution:

- **Uniform Distribution**:  
  ```math
  W \sim U(-a, a)
    ```
  
- **Normal Distribution**:  
  ```math
  W \sim \mathcal{N}(0, \sigma^2)
  ```

where \( a \) and \( \sigma \) are chosen empirically.

### **Problems:**
- If values are too large → **Exploding gradients**.
- If values are too small → **Vanishing gradients**.

### **Best Used With:**
✅ Works for shallow networks, but not ideal for deep networks.

---

## 🔹 **3. Xavier (Glorot) Initialization**
### **Definition:**
Designed to keep the variance of activations constant across layers. Based on the idea that:

```math
\text{Var}(W \cdot X) = \text{Var}(X)
```

To achieve this, Xavier initialization sets:

- **For uniform distribution:**
  ```math
  W \sim U\left(-\frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}\right)
  ```

- **For normal distribution:**
  ```math
  W \sim \mathcal{N}\left(0, \frac{1}{n_{\text{in}} + n_{\text{out}}}\right)
  ```

where:
- $\( n_{\text{in}} \)$ = Number of inputs to the neuron
- $\( n_{\text{out}} \)$ = Number of outputs from the neuron

### **Advantages:**
✅ Helps maintain stable variance across layers.  
✅ Prevents vanishing/exploding gradients in deep networks.

### **Best Used With:**
✅ **Sigmoid, Tanh**  
⏳ **Not ideal for ReLU-based activations** (since ReLU tends to produce asymmetric activations).

---

## 🔹 **4. He Initialization (Kaiming Initialization)**
### **Definition:**
Optimized for **ReLU** and its variants. Unlike Xavier, He initialization only considers **input neurons** because ReLU deactivates half of the neurons.

- **For uniform distribution:**
  ```math
  W \sim U\left(-\frac{\sqrt{6}}{\sqrt{n_{\text{in}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}}}}\right)
  ```

- **For normal distribution:**
  ```math
  W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
  ```

### **Advantages:**
✅ Keeps activations from shrinking or exploding.  
✅ Works well with ReLU-based activations.

### **Best Used With:**
✅ **ReLU, Leaky ReLU, ELU**  
❌ **Not ideal for Sigmoid/Tanh** (because it leads to large activation outputs and saturation).

---

## 🔹 **5. LeCun Initialization**
### **Definition:**
Specialized for **Sigmoid and Tanh**, where weights are initialized to:

- **For uniform distribution:**
  ```math
  W \sim U\left(-\frac{\sqrt{3}}{\sqrt{n_{\text{in}}}}, \frac{\sqrt{3}}{\sqrt{n_{\text{in}}}}\right)
  ```

- **For normal distribution:**
  ```math
  W \sim \mathcal{N}\left(0, \frac{1}{n_{\text{in}}}\right)
  ```

### **Advantages:**
✅ Works well for networks using **Tanh and Sigmoid**.  
✅ Helps prevent saturation.

### **Best Used With:**
✅ **Tanh, Sigmoid**  
⏳ **Not ideal for ReLU** (He initialization is better).

---

## 🔹 **6. SELU (Scaled Exponential Linear Unit) Initialization**
### **Definition:**
Designed for **Self-Normalizing Networks** (SNNs). SELU has a unique property:  
✅ It **automatically normalizes activations** across layers.

Weights follow:

```math
W \sim \mathcal{N} \left(0, \frac{1}{n_{\text{in}}} \right)
```

and biases should be **zero**.

### **Advantages:**
✅ Allows deep networks to **self-normalize**.  
✅ Eliminates need for Batch Normalization.

### **Best Used With:**
✅ **SELU Activation**  
❌ **Not recommended for ReLU, Sigmoid, or Tanh**.

---

## 🔥 **Summary Table**
| **Initialization**  | **Best For**                 | **Formula**                                                             | **Best Activation Functions**             |
|---------------------|------------------------------|-------------------------------------------------------------------------|-------------------------------------------|
| **Zero Init**       | Bias Matrix                  | $\( W = 0 \)$                                                           | ✅ Bias Matrix                             |
| **Random Init**     | Small networks               | $\( W \sim U(-a, a) \) or \( W \sim \mathcal{N}(0, \sigma^2) \)$        | ✅ Any (but not optimal for deep networks) |
| **Xavier (Glorot)** | Avoiding vanishing gradients | $\( W \sim \mathcal{N}(0, \frac{1}{n_{\text{in}} + n_{\text{out}}}) \)$ | ✅ Sigmoid, Tanh                           |
| **He Init**         | ReLU-based activations       | $\( W \sim \mathcal{N}(0, \frac{2}{n_{\text{in}}}) \)$                  | ✅ ReLU, Leaky ReLU, ELU, Swish, GELU      |
| **LeCun Init**      | Self-normalizing nets        | $\( W \sim \mathcal{N}(0, \frac{1}{n_{\text{in}}}) \)$                  | ✅ Sigmoid, Tanh                           |
| **SELU Init**       | Self-Normalizing Nets        | $\( W \sim \mathcal{N}(0, \frac{1}{n_{\text{in}}}) \)$                  | ✅ SELU                                    |

## **Best Initialization for Each Activation**
| Activation     | Best Initialization                                                      |
|----------------|--------------------------------------------------------------------------|
| **Sigmoid**    | Xavier Glorot Normal, Xavier Glorot Uniform                              |
| **Tanh**       | Xavier Glorot Normal, Xavier Glorot Uniform, Lecun Normal, Lecun Uniform |
| **ReLU**       | He Normal, He Uniform                                                    |
| **Leaky ReLU** | He Normal, He Uniform                                                    |
| **ELU**        | He Normal, He Uniform                                                    |
| **SELU**       | Lecun Normal, Lecun Uniform, SELU Initialization                         |
| **Swish**      | He Normal                                                                |
| **GELU**       | He Normal                                                                |

---
