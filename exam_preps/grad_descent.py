"""
grad_descent.py a simple implementation of gradient descent for a 1D quadratic function, with visualisation.
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """The function we're trying to minimise."""
    return (x - 3) ** 2 + 2

def grad_f(x):
    """The gradient of the function."""
    return 2 * (x - 3)

def gradient_descent(start_x, learning_rate, num_iterations):
    """Performs gradient descent on the function f."""
    x = start_x
    history = [x]
    for _ in range(num_iterations):
        grad = grad_f(x)
        x = x - learning_rate * grad
        history.append(x)
    return history

def gradient_descent_momentum(x0=3.0, lr=0.01, momentum=0.9, tol=1e-6, max_steps=1000):
    x = x0
    velocity = 0.0
    for step in range(max_steps):
        grad = grad_f(x)
        if abs(grad) < tol:
            break
        velocity = momentum * velocity - lr * grad
        x += velocity
    return x, f(x), step + 1

# Parameters for gradient descent
start_x = 0.0
learning_rate = 0.1
num_iterations = 40
# Run gradient descent
history = gradient_descent(start_x, learning_rate, num_iterations)

# Visualisation
x_values = np.linspace(-1, 7, 100)
y_values = f(x_values)
plt.plot(x_values, y_values, label='f(x)')

# Compute the corresponding f(x) values for the history of x values
y_history = [f(x) for x in history]

print("Gradient descent history (x values):", history)
print("Gradient descent history (f(x) values):", y_history)

plt.scatter(history, y_history, color='red', label='Gradient Descent Steps')
plt.title('Gradient Descent on f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()


x, loss, steps = gradient_descent_momentum(x0=3.0, lr=0.01, momentum=0.9)
print(f"minimum at x={x:.6f}, f(x)={loss:.6f}, steps={steps}")