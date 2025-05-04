# Gradient Descent: Insights into Behavior and Optimization Challenges

This notebook demonstrates how different initialization strategies and learning rates affect the convergence of the gradient descent optimizer on various cost functions.

## Project Motivation

Gradient descent is fundamental to modern machine learning. Understanding its failure modes helps practitioners debug models and select better hyperparameters.

## Project Overview

This notebook investigates the following key aspects of gradient descent:

- How the **initial values** of coefficients affect convergence
- The impact of **learning rate selection** on optimization efficiency
- **Visualization of the MSE surface** to illustrate optimization paths
- Common pitfalls such as **slow convergence**, **divergence**, and **local minima**
- How to **manually implement gradient descent from scratch**, without using external libraries — which deepens understanding of the algorithm and may be beneficial for future debugging or job interviews

The goal is to **diagnose and understand the practical limitations** of gradient-based optimization methods — a valuable perspective for both data science and machine learning engineering tasks.

## File Structure

```bash
.
├── grad_descent_lib/
│   ├── algo.py             # Core implementation of gradient descent algorithm
│   ├── plot_func.py        # Encapsulated plotting utilities for surface and trajectory visualization
├── images/                 # Conceptual visual illustrations
├── notebooks/              # Jupyter notebooks used to generate supporting diagrams 
│
├── main-discussion.md      # Main demonstrating 4 case studies
├── requirements.txt        # Minimal dependencies needed to run the notebook
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

The following table summarizes the theme and learning objectives of each case study:

| Case | Title | Focus Description |
|------|-------|-------------------|
| **1** | Initialization and Convergence | Demonstrates how different starting points lead to different convergence paths and rates — some fast, some slow, some possibly stuck. |
| **2** | Divergence and Overflow | Shows how excessively large learning rates cause the optimizer to overshoot the minimum, explode in value, or diverge entirely. |
| **3** | Oscillation vs. Slow Descent | Compares moderate vs. small learning rates — large rates oscillate near the minimum, while small rates converge slowly but safely. |
| **4** | Local Minima in Non-Convex Landscapes | Highlights the challenges of optimizing non-convex functions — the optimizer may settle into a local minimum, depending on the initial value. |

## Development & Reproduction

To run, verify, or modify this project, the following environment is recommended:

- **Python version**: 3.12.4
- **IDE**: Visual Studio Code  
  Alternatively, you may explore the notebook interactively using **JupyterLab** or **Google Colab**.

Install the required packages using:

```bash
pip install -r requirements.txt
```

> The requirements.txt file was automatically generated using pipreqs:

```bash
pipreqs . --force
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.  
You are free to use and adapt the materials for **non-commercial** purposes, but **you must credit the author**.

See the full license text in the [LICENSE](./LICENSE) file.

## Authour

**Alex Tian**  
MASc | Data Scientist | Machine Learning Researcher  