import numpy as np
import matplotlib.pyplot as plt

def plot_cost_and_derivative(f, df, x_range=(-3, 3), num_points=100, figsize=(6, 9), xlim=None, ylim=None):
    """
    Plots the cost function and its derivative (slope) for the given function f and its derivative df.
    
    Parameters:
    f (function): The cost function to plot.
    df (function): The derivative of the cost function to plot.
    x_range (tuple): The range of x values to plot (default is (-3, 3)).
    num_points (int): The number of points to generate in the x range (default is 100).
    figsize (tuple): The size of the figure (default is (6, 9)).
    xlim (tuple): The limits for the x-axis (optional).
    ylim (tuple): The limits for the y-axis (optional).
    """
    # Generate x values
    x_values = np.linspace(start=x_range[0], stop=x_range[1], num=num_points)

    # Plot cost function and its slope
    plt.figure(figsize=figsize)

    # 1. Cost Function
    plt.subplot(2, 1, 1)
    plt.title('Cost Function')
    plt.plot(x_values, f(x_values), color='blue', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('f(X)')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid()

    # 2. Derivative
    plt.subplot(2, 1, 2)
    plt.title('Derivative(Slope) of the Cost Function')
    plt.plot(x_values, df(x_values), color='skyblue', linewidth=2)
    plt.xlabel('X')
    plt.ylabel("f'(X)")
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_gradient_descent_trajectories(f, df, x_list, slope_list, x_range, initial_guess, iter, local_min, learning_rate, figsize=(24, 7), x_lim=None, y_lim=None):
    plt.figure(figsize=figsize)

    # 1. Plotting the cost function and the trajectory
    plt.subplot(1, 3, 1)
    plt.plot(x_range, f(x_range), color='blue', linewidth=2)
    plt.scatter(x_list, f(np.array(x_list)), color='red', s=50, zorder=5)
    plt.plot(x_list, f(np.array(x_list)), color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('f(X)')
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(f'Function Path \nInitial: {initial_guess}, LR: {learning_rate}, Min: X={local_min:.4f}, Steps: {iter}', fontsize=14)
    plt.grid()

    # 2. Plotting the derivative (slope) and the trajectory
    plt.subplot(1, 3, 2)
    plt.plot(x_range, df(x_range), color='skyblue', linewidth=2)
    plt.scatter(x_list, slope_list, color='red', s=50, zorder=5)
    plt.plot(x_list, slope_list, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.xlabel('X')
    plt.ylabel("f'(X)")
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(f"Derivative Path \nInitial: {initial_guess}, LR: {learning_rate}, Min: X={local_min:.4f}, Steps: {iter}", fontsize=14)
    plt.grid()

    # 3. Plotting the cost value over iterations
    plt.subplot(1, 3, 3)
    cost_values = [f(x) for x in x_list]
    plt.plot(range(len(cost_values)), cost_values, color='green', linewidth=2, marker='o', markersize=5)
    plt.xlabel('Iteration')
    plt.ylabel('Cost f(X)')
    plt.title(f'Cost per Iteration \nInitial: {initial_guess}, LR: {learning_rate}', fontsize=14)
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_cost_comparison(f, x_lists, labels=None, figsize=(10, 6)):
    """
    Plot cost (f(x)) trajectories for multiple gradient descent runs.

    Parameters:
    - f: The cost function to evaluate
    - x_lists: A list of x_list from different GD runs
    - labels: Optional list of labels (e.g., learning rates or methods)
    - figsize: Size of the figure
    """
    plt.figure(figsize=figsize)
    
    for i, x_list in enumerate(x_lists):
        cost_values = [f(x) for x in x_list]
        label = labels[i] if labels is not None else f"Run {i+1}"
        plt.plot(range(len(cost_values)), cost_values, marker='o', linewidth=2, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Cost f(X)")
    plt.title("Comparison of Cost Reduction over Iterations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()