# Mathematical Foundations of NeuralForge Core

## 1. Reverse-Mode Automatic Differentiation (Autograd)

For a computation graph representing a scalar objective $L$, reverse-mode automatic differentiation computes the vector-Jacobian product (VJP) in reverse topological order:

$$\frac{\partial L}{\partial x} = \sum_{y \in \text{children}(x)} \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}$$

### 1.1 Matrix Multiplication
Given $Z = XW$ where $X \in \mathbb{R}^{N \times D_{in}}$ and $W \in \mathbb{R}^{D_{in} \times D_{out}}$, and upstream gradient $G = \frac{\partial L}{\partial Z} \in \mathbb{R}^{N \times D_{out}}$:

$$\frac{\partial L}{\partial X} = G W^T, \quad \frac{\partial L}{\partial W} = X^T G$$

### 1.2 Gradient Unbroadcasting
When a binary operation broadcasts input shapes (e.g. $Z = X + b$ where $X \in \mathbb{R}^{N \times C}$ and $b \in \mathbb{R}^{C}$):
The upstream gradient $G$ has shape $(N, C)$. The gradient with respect to $b$ must be summed across the broadcast axis:

$$\frac{\partial L}{\partial b} = \sum_{i=1}^N G_{i, :}$$

In general, `unbroadcast(G, target_shape)` sums out:
1. All leading dimensions where `G.ndim > len(target_shape)`.
2. All axes where `target_shape[i] == 1` and `G.shape[i] > 1`.

---

## 2. Numerically Stable Softmax & Cross-Entropy

### 2.1 The Log-Sum-Exp Trick
A naive softmax $\frac{e^{z_i}}{\sum_j e^{z_j}}$ overflows in floating-point representation for large $z_i > 709$ (in float64).
By identity, shifting all inputs by $m = \max_j(z_j)$ does not change the quotient:

$$\text{Softmax}(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}$$

The log partition function $\log \sum_j e^{z_j}$ is computed stably as:

$$\text{LogSumExp}(z) = m + \log \left( \sum_j e^{z_j - m} \right)$$

### 2.2 Cross-Entropy Gradient
For logits $Z$ and ground-truth one-hot targets $Y$:

$$L = - \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^C Y_{ij} \log(\text{softmax}(Z)_{ij})$$

The analytical gradient with respect to logits $Z$ is exact and elegant:

$$\frac{\partial L}{\partial Z} = \frac{1}{N} (\text{softmax}(Z) - Y)$$

This fused formulation prevents both overflow (from exponentiating large numbers) and underflow (from taking the logarithm of zero).

---

## 3. Optimizers

### 3.1 Stochastic Gradient Descent (SGD) with Momentum
For parameters $\theta$, gradient $g_t$, learning rate $\eta$, momentum $\gamma$, and weight decay $\lambda$:

$$g_t \leftarrow g_t + \lambda \theta_t$$
$$v_{t+1} = \gamma v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

### 3.2 Adaptive Moment Estimation (Adam)
Adam computes exponential moving averages of the gradient ($m_t$) and squared gradient ($v_t$) with decay hyperparameters $\beta_1 = 0.9, \beta_2 = 0.999$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

To counteract initialization bias toward zero during early time steps ($t = 1, 2, \dots$):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 4. Finite-Difference Numerical Verification

Every analytical gradient in NeuralForge Core is validated using two-sided central finite differences:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}, \quad \epsilon = 10^{-6}$$

The relative error metric used in automated testing is:

$$\text{RelError} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|_2}{\|\nabla_{\text{analytical}}\|_2 + \|\nabla_{\text{numerical}}\|_2 + 10^{-12}} < 10^{-4}$$
