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


def plot_gradient_descent_trajectories(f, df, x_list, slope_list, x_range, initial_guess, iter, local_min, learning_rate, figsize=(20, 7), x_lim=None, y_lim=None):
    plt.figure(figsize=figsize)

    # Plotting the cost function and the trajectory
    plt.subplot(1, 2, 1)
    plt.plot(x_range, f(x_range), color='blue', linewidth=2)
    plt.scatter(x_list, f(np.array(x_list)), color='red', s=50, zorder=5)
    plt.plot(x_list, f(np.array(x_list)), color='red', linestyle='--', linewidth=2, alpha=0.7)  # Dashed line for path
    plt.xlabel('X')
    plt.ylabel('f(X)')

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(f'Gradient Path of Original Function \nInitial guess: {initial_guess}, learning rate: {learning_rate}, estimate local min: X={local_min:.4f}, steps: {iter}', fontsize=15)
    plt.grid()

    # Plotting the derivative (slope) and the trajectory
    plt.subplot(1, 2, 2)
    plt.plot(x_range, df(x_range), color='skyblue', linewidth=2)
    plt.scatter(x_list, slope_list, color='red', s=50, zorder=5)
    plt.plot(x_list, slope_list, color='red', linestyle='--', linewidth=2, alpha=0.7)  # Dashed line for path
    plt.xlabel('X')
    plt.ylabel("f'(X)")

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(f"Gradient Path of Derivative Function \nInitial guess: {initial_guess}, learning rate: {learning_rate}, estimate local min: X={local_min:.4f}, steps: {iter}", fontsize=15)
    plt.grid()

    plt.tight_layout()
    plt.show()