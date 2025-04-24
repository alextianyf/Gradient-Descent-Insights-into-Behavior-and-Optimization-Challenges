# Gradient Descent: Insights into Behavior and Optimization Challenges

This notebook demonstrates how different initialization strategies and learning rates affect the convergence of the gradient descent optimizer on various cost functions.

## Project Overview

this notebook investigates:

- How the **initial values** of coefficients affect convergence
- The impact of **learning rate selection** on optimization efficiency
- **Visualization of the MSE surface** to illustrate optimization paths
- Common pitfalls such as **slow convergence**, **divergence**, and **local minima**

The goal is to **diagnose and understand the practical limitations** of gradient-based optimization methods — a valuable perspective for both data science and machine learning engineering tasks.

## File Structure

```bash
├── grad_descent_lib/
│   ├── algo.py             # Core implementation of gradient descent algorithm
│   ├── plot_func.py        # Encapsulated plotting utilities for surface and trajectory visualization
│
├── discussion.ipynb        # Main notebook demonstrating 4 case studies
├── LICENSE                 # Open-source license information
├── .gitignore              # Files and folders to ignore in version control
└── README.md               # Project documentation (this file)
```

## Main Discussion Decomposition

This notebook contains:

- **Introduction**
- **Theoretical Background**
- **Mathematical Formulation**
- **Case Studies (4 total)**

Each case includes:

- A unique function setup
- Gradient descent behavior observation
- Insights, takeaways, and practical implications

### Overview of the 4 Case Studies

| Case | Title | Focus Description |
|------|-------|-------------------|
| **1** | Initialization and Convergence | Demonstrates how different starting points lead to different convergence paths and rates — some fast, some slow, some possibly stuck. |
| **2** | Divergence and Overflow | Shows how excessively large learning rates cause the optimizer to overshoot the minimum, explode in value, or diverge entirely. |
| **3** | Oscillation vs. Slow Descent | Compares moderate vs. small learning rates — large rates oscillate near the minimum, while small rates converge slowly but safely. |
| **4** | Local Minima in Non-Convex Landscapes | Highlights the challenges of optimizing non-convex functions — the optimizer may settle into a local minimum, depending on the initial value. |
