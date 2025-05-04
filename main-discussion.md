# Gradient Descent

## Introduction: Why Gradient Descent?

In many machine learning and optimization problems, our goal is to find the parameter values that minimize a given cost function — a mathematical expression that quantifies how well a model performs.

This cost function, often denoted as $J(\boldsymbol{\beta})$, depends on a set of parameters:

$$
\boldsymbol{\beta} = [\beta_0, \beta_1, \dots, \beta_p]
$$

The optimization objective is to find:

$$
\boldsymbol{\beta}^* = \arg\min_{\boldsymbol{\beta}} J(\boldsymbol{\beta})
$$

This equation says:

> *“We want to find the value of $\boldsymbol{\beta}$ that makes the cost function $J(\boldsymbol{\beta})$ as small as possible.”*

Let’s break it down:

- $\boldsymbol{\beta}$ is a set of parameters (like weights in a model).

- $J(\boldsymbol{\beta})$ is the cost or error function — it tells us **how bad** our model is with a given set of parameters.

- $\arg\min$ means: “Give me the value of $\boldsymbol{\beta}$ that makes $J(\boldsymbol{\beta})$ the smallest.”

So,

- $\boldsymbol{\beta}^*$ is the **best** set of parameters — the one that **minimizes the cost**.

In simple cases, this can be achieved analytically. But in most real-world applications — especially those involving large datasets or complex models like neural networks — a closed-form solution is either **infeasible** or **non-existent**.

This is where **gradient-based optimization** techniques come in.

---

## What Is Gradient Descent?

**Gradient Descent** is a widely used iterative optimization algorithm that seeks to minimize the cost function by taking steps proportional to the **negative of the gradient** at the current point. In other words:

> Move in the direction that most rapidly decreases the cost.

<div align="center">
  <img src="https://i.sstatic.net/L0B4L.png" alt="gd" width="500" height="330"/>
</div>

At each iteration, the parameter vector is updated using the rule:

$$
\boxed{
\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \cdot \nabla J(\boldsymbol{\beta})
}
$$

$$
\boxed{
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \cdot \nabla J(\boldsymbol{\beta}^{(t)})
}
$$

Where:

- $\alpha$ is the **learning rate**(a small positive scalar),

- $J(\boldsymbol{\beta})$ is any differentiable cost function (e.g., MSE, cross-entropy, hinge loss, etc.),

- $\nabla J(\boldsymbol{\beta})$ is the **gradient vector**, containing partial derivatives of the cost function with respect to each parameter:

$$
\nabla J(\boldsymbol{\beta}) = 
\left[
\frac{\partial J}{\partial \beta_0}, \;
\frac{\partial J}{\partial \beta_1}, \;
\dots, \;
\frac{\partial J}{\partial \beta_p}
\right]
$$

Each component of the gradient tells us how sensitive the cost function is to changes in the corresponding parameter.

---

## Why Does It Work?

The gradient vector always points in the direction of **steepest ascent** of the cost function. By moving in the **opposite direction**, gradient descent ensures that we gradually move toward a **local minimum** of the cost.

---

## Gradient Descent in Practice

Although conceptually simple, gradient descent can behave in **unexpected and inefficient ways** depending on factors like:

- **Initialization**: Where you start matters — poor initial guesses can lead to slow convergence or local minima.

- **Learning rate**: Too small and convergence is slow; too large and the algorithm may **diverge** or oscillate.

- **Curvature of the cost surface**: Sharp valleys or flat plateaus can make progress inefficient.

- **Feature scaling** and **saddle points** may also introduce further challenges.

---

## Gradient Descent Code implementation

The original implementation of the gradient descent algorithm can be found in the folder `grad_descent_lib -> algo.py`. More detailed explanations and comments can be found directly in the original file. Below is the core implementation:

```python
def gradient_descent(derivative_func, initial_guess, learning_rate=0.02, precision=0.001, max_iter=300):
    new_x = initial_guess
    multiplier = learning_rate
    x_list = [new_x]
    slope_list = [derivative_func(new_x)]
    step_size = precision + 1
    iter_num = 1

    while step_size > precision:
        if iter_num > max_iter:
            break
        previous_x = new_x
        gradient = derivative_func(previous_x)
        new_x = previous_x - multiplier * gradient
        step_size = abs(new_x - previous_x)
        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))
        iter_num += 1

    local_min = new_x

    return local_min, x_list, slope_list, iter_num
```

In this section, we will not only explain how to implement gradient descent but also explore its behavior under different conditions, such as varying learning rates, precision, and iteration limits. Understanding how gradient descent works in practice will help you refine your approach and improve its effectiveness.

---

## Case Studies

### Case Study 1: How Starting Point Affects Gradient Descent Efficiency

Gradient Descent is an iterative optimization algorithm. In this case study, we demonstrate how the **initial starting point** directly affects the **number of iterations** required to reach the minimum.

We define a basic quadratic function as a simulated cost function:

$$f(x) = x^2 + x + 1$$

Its derivative (gradient):

$$f’(x) = 2x + 1$$

<div align="center">
  <img src="images/c1-func.png" alt="func-1" />
</div>

If we initialize gradient descent at either 3 or -1, we can observe the optimization paths on the original function (left) and its derivative (middle) in the images below. The image on the right illustrates how the total cost decreases over the iterations.

This experiment confirms that the **starting point of Gradient Descent significantly affects convergence efficiency**, even for a simple convex function like:

$$f(x) = x^2 + x + 1$$
- When starting at x = 3, it takes **32 steps** to reach the minimum.
- When starting at x = -1, it takes **23 steps**, despite being closer to the true minimum at x = -0.5.

| Starting Point | Steps to Converge | Distance from Minimum | Initial Gradient | Comments |
|----------------|-------------------|------------------------|------------------|----------|
| `x = 3`        | 32            | 3 units              | Large positive   | Overshoots initially, then slowly descends |
| `x = -1`       | 23                | 1 unit              | Small negative   | Starts closer, but smaller slope leads to slower early movement |

<div align="center">
  <img src="images/c1-path1.png" alt="func-1" />
</div>

<div align="center">
  <img src="images/c1-path2.png" alt="func-1" />
</div>

Now, if we plot both cost functions together, it’s clear that the “randomly” selected initial point significantly affects the efficiency and converging speed of gradient descent.

<div align="center">
  <img src="images/c1-costcomp.png" alt="func-1" />
</div>

<br>

> - **Gradient Descent is sensitive to the initial point.** Even on a simple quadratic, the step count can vary significantly based on the distance to the minimum and the steepness of the slope at the starting point.
> A **larger initial slope** can sometimes help move faster initially, but risks oscillations or overshooting if learning rate is not well-tuned.
> A **smaller slope** near the minimum might cause slower early updates, even though the point is closer to the optimum.
> Thus, **"closer ≠ always faster"** if the slope is too flat.

**In real-world high-dimensional problems:**

- Poor initialization can lead to **slow convergence** or **getting stuck in plateaus**.

- Good initialization strategies (e.g., Xavier, He) are **not just tricks**—they are **essential for performance**.

**Practical Implication:**

How to Choose a Good Initial Value in Gradient Descent? The choice of initial value affects **convergence speed**, **stability**, and whether the model can **escape poor local minima**. Here’s how to do it right:

1. If the Problem is Convex (e.g., Linear Regression)

- You **can choose any random value**, and gradient descent will eventually converge.
- But:
  - Starting **too far from the minimum** → slow convergence
  - Starting **in a flat region** → small gradients, slow updates

**Tip:**  
Use small random values near 0, like:

```python
x_init = np.random.uniform(-1, 1)
```

2. If the Problem is Non-Convex (e.g., Neural Networks)

Poor initialization can cause:
- Vanishing or exploding gradients
- Getting stuck in poor local minima
- Failed or slow convergence

Xavier Initialization (Glorot)

Best for sigmoid, tanh, etc.

$$
W \sim \mathcal{U}\left[-\frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}\right]
$$

He Initialization

Best for ReLU, LeakyReLU:

$$
W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{\text{in}}}})
$$

3. Data-aware Initialization
- Normalize inputs: mean = 0, std = 1
- Use domain knowledge if available
- For time-series/spatial problems, initialize from recent values or bounded centers

4. Multi-start Strategy (Random Restarts)
- Try several random initializations
- Pick the one that converges to the lowest cost

```python
best_result = min([gradient_descent(x0) for x0 in range(-3, 4)], key=lambda res: res.cost)
```

---

### Case Study 2: Divergence and Overflow in Gradient Descent

In this case study, we explore a **non-convex cost function** that exhibits **divergence** when the learning rate is too high, causing the algorithm to **overflow** instead of converge.

The cost function is defined as:

$$
t(x) = x^5 - 2x^4 + 2
$$

Its derivative (gradient) is:

$$
t'(x) = 5x^4 - 8x^3
$$

<div align="center">
  <img src="images/c2-func.png" alt="func-2" />
</div>

- **Divergence**: The algorithm starts at the initial guess (x = -0.4) and moves leftward with increasing steps, causing the cost function to increase. If we did not set the iteration limit, the gradient descent would continue to move away from the minimum.

- **Overflow**: As the gradient descent steps become excessively large due to the high learning rate, the computed values for the cost function and its gradient grow uncontrollably. This results in overflow, where the cost function value becomes extremely large or even reaches computational limits (inf or very large numbers). This overflow prevents meaningful updates, and the algorithm fails to converge.

> **Divergence Explanation**: In gradient descent, **divergence** refers to the phenomenon where, during the iterative process of updating parameters, the value of the loss function does not decrease but instead increases continuously, ultimately moving away from the optimal solution.

<div align="center">
  <img src="images/c2-path.png" alt="func-1" />
</div>

<br>

> - **Learning rate selection is crucial**:  A learning rate that is too large can cause **divergence** during the gradient descent process, where the loss function value increases rather than decreases. On the other hand, a learning rate that is too small results in **slow convergence**, wasting computational resources.
> 
> - **Causes of divergence**
>   
>   - **High learning rate**: Large update steps may overshoot the optimal solution, causing the results to move away from the target.
>   
>   - **Unnormalized features**: When input features have significantly different scales, the gradient descent updates will be inconsistent across directions, leading to divergence.
>   
>   - **Noise in the data**: Noise or outliers in the data can affect the stability of gradient descent, leading to divergence.
> 
> - **Prevent divergence by adjusting the learning rate**: A suitable learning rate helps gradient descent converge stably, while a high learning rate leads to divergence. Starting with a smaller learning rate and gradually adjusting it can effectively prevent this issue. 

**Practical Implication:**

1. **Adjust the learning rate**:  
   - **Start with a small learning rate** (e.g., \( \alpha = 0.01 \)), then gradually increase it and observe the effect on convergence speed.
   - **Use learning rate scheduling**: Techniques like learning rate decay or adaptive learning rate methods (such as Adam) can dynamically adjust the learning rate to improve training efficiency.

2. **Normalize input data**:  
   Standardize or normalize the data to ensure all features have similar scales. This can significantly enhance the stability of gradient descent and prevent divergence caused by extreme values in some features.

3. **Consider using adaptive optimizers**:  
   For cases where divergence or slow convergence is a problem, **adaptive learning rate methods** such as **Adam** or **RMSProp** can be used. These methods dynamically adjust the learning rate for each parameter, preventing divergence and speeding up convergence.

---

### Case Study 3: Impact of Learning Rate on Gradient Descent

In this case study, we explore how the **choice of learning rate** affects the convergence behavior of gradient descent, using the non-convex cost function:

$$
h(x) = x^4 - 4x^2 + 5
$$

Its derivative is:

$$
h'(x) = 4x^3 - 8x
$$

The learning rate $\alpha$(alpha) or $\gamma$(gamma) controls how large each update step is. It has a profound impact on the efficiency and stability of optimization.

<div align="center">
  <img src="images/c3-func.png" alt="func-1" />
</div>

A Proper Learning Rate Can Accelerate Convergence:

   - A larger learning rate means **larger steps** are taken along the negative gradient direction.
   
   - When the learning rate is **well-chosen** (e.g., `γ = 0.01`), the algorithm can reach the minimum **significantly faster** than a smaller learning rate (e.g., `γ = 0.001`).
   
   - The **cost vs. iteration** plot shows a **steeper decline**, reducing the number of iterations required to approach the optimal point.
   
   - This is especially helpful when dealing with large-scale datasets or high-dimensional models where iteration cost is expensive.
   
   - A moderately increased learning rate can improve convergence speed without sacrificing stability.
   
   - Faster convergence is desirable in large-scale or high-dimensional problems.

<div align="center">
  <img src="images/c3-path1.png" alt="func-1" />
</div>

<div align="center">
  <img src="images/c3-path2.png" alt="func-1" />
</div>

<div align="center">
  <img src="images/c3-costcomp.png" alt="func-1" />
</div>

A Too-High Learning Rate Can Lead to Oscillations or Divergence:

   - When the learning rate becomes **too large** (e.g., `γ = 0.2`), it can cause the algorithm to **overshoot the minimum**, jumping back and forth across it.
   
   - In our function, this manifests as **zigzag motion** in the path plot and **non-monotonic cost behavior** in the cost plot.
   
   - Instead of stabilizing at the minimum, the algorithm **oscillates** and may even **fail to converge** if the overshooting grows worse.
   
   - Higher learning rate may compromise convergence and lead to oscillatory or divergent behavior.

<div align="center">
  <img src="images/c3-path3.png" alt="func-1" />
</div>

<div align="center">
  <img src="images/c3-costcomp2.png" alt="func-1" />
</div>

| Learning Rate      | Behavior                         | Trade-off                            |
|--------------------|----------------------------------|--------------------------------------|
| Too Small  | Very slow convergence             | Stable but inefficient               |
| Good       | Fast and smooth convergence       | Optimal choice (requires tuning)     |
| Too Large    | Oscillation or divergence         | Fast updates but unstable            |

> - Learning rate must be **carefully balanced** to trade off between **speed** and **stability**.
> 
> - Selecting a good learning rate is one of the most critical decisions in gradient descent-based optimization.  
> 
> - When possible, visualize the **cost trajectory** and **parameter path** to diagnose potential issues like oscillation or stagnation.
>
> - In modern machine learning, consider **adaptive optimizers** to dynamically adjust learning rate and improve convergence robustness.

**Practical Implication:**

- **Adaptive learning rate methods** (often used in deep learning) aim to solve the problem of manual tuning:
  
  - `AdaGrad`: Adjusts the learning rate individually for each parameter based on the past gradients.
  
  - `RMSProp`: Improves AdaGrad by preventing the learning rate from shrinking too much.
  
  - `Adam`: Combines momentum and RMSProp techniques; widely adopted for its robustness and effectiveness.

These methods make learning more stable and often remove the need to manually choose a fixed global learning rate.

---

### Case Study 4: Multiple Minima in a Non-Convex Cost Function

We now analyze a new cost function, This is a **non-convex** function — it has more than one valley (local minimum), unlike a simple U-shaped parabola:

$$g(x) = x^4 - 4x^2 + 0.5x + 5$$

Its derivative (gradient):

$$g{\prime}(x) = 4x^3 - 8x + 0.5$$

We want to understand:

1. Can Gradient Descent always find the global minimum?

2. What happens when the function has more than one valley (local minimum)?

3. How do different initial guesses affect convergence?

<div align="center">
  <img src="images/c4-func.png" alt="func-1" />
</div>

This example reveals key limitations of gradient descent when applied to **non-convex** functions:

- Different starting points lead to different local minima

- Without prior knowledge of the cost surface, you cannot:
	
    1.	Guarantee convergence to the global minimum
	
    2.	Know how many minima exist

This behavior is typical in optimization problems involving **non-convex functions** (e.g., neural networks, energy landscapes, etc.)

<div align="center">
  <img src="images/c4-path1.png" alt="func-1" />
</div>

<div align="center">
  <img src="images/c4-path2.png" alt="func-1" />
</div>

<br>

> - Gradient Descent can get trapped in local minima when the cost function is non-convex.
>
> - The algorithm is not aware of the global structure of the cost surface.
>
> - Choosing different starting points will lead to different solutions, and the result is not guaranteed to be optimal.

**Practical Implication:**

To address these issues:
- Use **multiple random initializations** to increase the chance of finding a better solution.
- Try **[Stochastic Gradient Descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)** or **simulated annealing**, which introduce randomness and may help escape local minima.
- Consider advanced optimizers like **Adam**, **RMSProp**, or **momentum-based methods**.

More reading on [Stochastic Gradient Descent (SGD)](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/)