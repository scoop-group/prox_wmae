"""
This script demonstrates how to use the implementation of the the proximal map
of the weighted mean absolute error function prox_wmae.py, see Algorithm 1 in 

  Baumgärtner, Herzog, Schmidt, Weiß
  The Proximal Map of the Weighted Mean Absolute Error

This function depends on numpy.

.. code: 

    pip3 install numpy

"""
 
import numpy as np
from prox_wmae import prox_wmae

# Demonstrate a call which evaluates a single proximal map at a single point.
# The number of weights and data points is N = 4.
x = 0.6
gamma = 0.8
w = np.array([0.1, 0.7, 0.0, 0.3]) 
d = np.array([-0.2, 0.4, 0.8, 1.5])
y = prox_wmae(x, w, d, gamma)

# Demonstrate a call which evaluates 3 proximal maps in parallel at a single point each.
# The number of weights and data points is N = 5.
x_vec = np.array([-1, 1, 0.5])
gamma_vec = np.array([0.1, 0.2, 0.3])
w_vec = np.array([[0.5, 0.7, 0.9, 1.5, 0.25], [0.3, 0.4, 0.5, 0.6, 0.7], [0.1, 0.11, 0.111, 0.2, 0.22]])
d_vec = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [-1, -0.3, 0.1, 1.5, 2], [0, 0.25, 0.4, 0.6, 1.5]])
y_vec = prox_wmae(x_vec, w_vec, d_vec, gamma_vec)
