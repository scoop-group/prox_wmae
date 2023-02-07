""" 
    This function implements the proximal map of the weighted mean absolute
    error function, see Algorithm 1 in 

    Baumgärtner, Herzog, Schmidt, Weiß: The Proximal Map of the Weighted Mean Absolute Error

    Equation numbers in the comments refer to this publication.

    This function depends on numpy.

    .. code: 

      pip3 install numpy

    See demo_prox_wmae.py and demo_cameraman.py for demonstrations.

"""

import numpy as np

def prox_wmae(x, weights, data, gamma):
    r""" evaluates the proximal map of the weighted mean absolute error function at the point x,

    .. math::
        \operatorname{prox}_{\gamma f}(x) = \argmin_{y\in\mathbb{R}} \gamma \sum_{i=1}^N w_i |y-d_i| + 1/2 (y-x)^2.

    Here, :math:`w \in \mathbb{R}^N` is a vector of non-negative :code:`weights`, and :math:`d \in \mathbb{R}^N` is a vector of :code:`data` points.
    The entries in :math:`d` must be sorted in non-decreasing order.

    This is an implementation of Algorithm 1 presented in 
    Baumgärtner, Herzog, Schmidt, Weiß: The Proximal Map of the Weighted Mean Absolute Error.

    The implementation can solve :math:`k` problems at once, which share the number :math:`N` of data points but may differ in :code:`weights`, :code:`data` and :code:`gamma`.

    Parameters
    ----------
    x : float or np.array of shape (k,)
        point of evaluation
    weights : np.array of shape (N,) or (k,N)
        weights of the mean absolute error function
    data : np.array of shape (N,) or (k,N)
        data points of the mean absolute error function
    gamma : float or np.array of shape (k,)
        value of the prox parameter

    Returns
    -------
    float or np.array of shape (k,)
        :math:`\\operatorname{prox}_{\\gamma f}(x)`
    """

    # Transform all shapes in case of a single problem.
    if len(weights.shape) == 1:
        weights = weights[np.newaxis,:]
    if len(data.shape) == 1:
        data = data[np.newaxis,:]
    if len(np.array(x).shape) == 0:
        x = np.array([x])[:,np.newaxis] 

    # Evaluate the forward and reverse cumulative weights.
    # Notice that nu[0] equals \nu_1 in the paper.
    mu = np.hstack((np.zeros((weights.shape[0],1)), np.cumsum(weights,axis=-1), np.sum(weights,axis = -1, keepdims = True)))
    nu = np.hstack((np.flip(np.cumsum(np.flip(weights, axis = -1), axis = -1), axis = -1), np.zeros((weights.shape[0],1)), np.zeros((weights.shape[0],1))))

    # Evaluate the limiting derivatives (2.7), (2.10) of the objective (2.5) at all data points.
    data_extended = np.hstack((-np.inf * np.ones((data.shape[0],1)), data, np.inf * np.ones((data.shape[0],1))))
    derivatives = ((gamma * (mu - nu).T).T + data_extended - x[:,np.newaxis]) 

    # Find the smallest index k where derivatives[k] is non-negative.
    k = np.apply_along_axis(np.searchsorted, axis = -1, arr = derivatives, side = 'left', v = 0)

    # Evaluate the proximal map according to (2.9).
    y = np.minimum(data_extended[np.arange(len(k)),k], x - gamma * (mu[np.arange(len(k)),k-1] - nu[np.arange(len(k)),k-1]))

    # Reshape the result in case of a single problem.
    if y.shape == (1,1):
        return y[0,0]
    else:
        return y
