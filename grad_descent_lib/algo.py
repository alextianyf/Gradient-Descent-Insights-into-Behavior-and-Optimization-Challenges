def gradient_descent(derivative_func, initial_guess, learning_rate=0.02, precision=0.001, max_iter=300):
    """
    This function implements the Gradient Descent optimization algorithm.
    Gradient Descent is an iterative optimization algorithm used to minimize a function.
    The function adjusts parameters based on the derivative (gradient) of the function.

    Parameters:
    derivative_func (function): A function that computes the derivative (gradient) of the cost function.
    initial_guess (float): The initial value (starting point) for the parameter to be optimized.
    learning_rate (float): The step size (or learning rate) used to update the parameter. Default is 0.02.
    precision (float): The precision or tolerance. The algorithm stops when the step size is smaller than this value. Default is 0.001.
    max_iter (int): The maximum number of iterations the algorithm will run before stopping. Default is 300.

    Returns:
    best_x (float): The optimized value of the parameter after the algorithm converges.
    x_list (list): A list of all the x values computed during the iterations.
    slope_list (list): A list of all the gradient values computed during the iterations.
    iter_num (int): The total number of iterations the algorithm ran before stopping.
    """

    # Initialize the starting point and set up necessary variables
    new_x = initial_guess  # Starting guess for the value of x
    multiplier = learning_rate  # The learning rate (step size)
    x_list = [new_x]  # List to store x values over iterations
    slope_list = [derivative_func(new_x)]  # List to store slope (derivative) values over iterations
    step_size = precision + 1  # Initialize step_size to ensure the loop starts
    iter_num = 1  # Initialize iteration count
    
    # Iterate until convergence or maximum iterations are reached
    while step_size > precision:  # Continue while the step size is larger than the desired precision
        if iter_num > max_iter:  # Break the loop if max iterations are exceeded
            break
        previous_x = new_x  # Save the current value of x for the next iteration
        gradient = derivative_func(previous_x)  # Compute the gradient (derivative) at the current x
        new_x = previous_x - multiplier * gradient  # Update the value of x using the gradient descent formula
        step_size = abs(new_x - previous_x)  # Calculate the step size (difference between old and new x)
        x_list.append(new_x)  # Append the updated x to the list
        slope_list.append(derivative_func(new_x))  # Append the new slope to the list
        iter_num += 1  # Increment the iteration count
    
    best_x = new_x  # The final value of x after convergence

    return best_x, x_list, slope_list, iter_num  # Return the final value, the x and slope history, and the iteration count