""" 
This script demonstrates the use of the proximal map of the weighted mean
absolute error function in a total-variation image denoising problem. It
implements Algorithm 2 in

Baumgärtner, Herzog, Schmidt, Weiß
The Proximal Map of the Weighted Mean Absolute Error

Equation numbers in the comments refer to this publication.
The algorithm operates on cameraman_noisy.png and outputs cameraman_result.png. 

This function depends on several python packages:

.. code: 

    pip3 install Pillow 
    pip3 install numpy
    pip3 install osqp
    pip3 install scipy
"""

from PIL import Image, ImageOps
from prox_wmae import prox_wmae
from scipy.sparse import csc_matrix, eye
import numpy as np
import osqp

def perform_checkerboard_step(img_data, u, beta, group):
    """ performs one checkerboard step for one group of pixels in parallel

    Calls :py:meth:`prox_wmae` once. 

    Parameters
    ----------
    img_data : np.array of shape (width, height)
        array containing the image data 
    u : np.array of shape (width, height)
        current iterate with image values
    beta : float
        total-variation parameter
    group : 0 or 1 
        determines the group of pixels being operated on
    
    Returns
    -------
    np.array of shape (width, height)
        updated iterate with image values
    """

    # Evaluate the values of neighbors (north, south, west, east) of all pixels in the current iterate.
    # Missing values are replaced by center values but masked below through zero weights.
    u_n = np.pad(u, ((1,0), (0,0)), mode = 'edge')[:-1]
    u_s = np.pad(u, ((0,1), (0,0)), mode = 'edge')[1:]
    u_w = np.pad(u, ((0,0), (1,0)), mode = 'edge')[:,:-1]
    u_e = np.pad(u, ((0,0), (0,1)), mode = 'edge')[:,1:]
    neighbors_u = np.stack((u_n, u_s, u_w, u_e), axis = 2)
    
    # Create the array of weights.
    # Missing values are replaced by zeros.
    w_n = np.pad(np.ones(u.shape), ((1,0), (0,0)), mode = "constant", constant_values = 0)[:-1]
    w_s = np.pad(np.ones(u.shape), ((0,1), (0,0)), mode = "constant", constant_values = 0)[1:]
    w_w = np.pad(np.ones(u.shape), ((0,0), (1,0)), mode = "constant", constant_values = 0)[:,:-1]
    w_e = np.pad(np.ones(u.shape), ((0,0), (0,1)), mode = "constant", constant_values = 0)[:,1:]
    neighbors_w = np.stack((w_n, w_s, w_w, w_e), axis = 2)
    
    # Create index lists of pixel coordinates in the group, separately for even and odd rows.
    ind_even = (slice(0, u.shape[0], 2), slice(group,   u.shape[1], 2))
    ind_odd  = (slice(1, u.shape[0], 2), slice(1-group, u.shape[1], 2))
    
    # Sort the neighbors' values in ascending order for even rows.
    # Apply the same sorting to the corresponding weights.
    order_even = np.argsort(neighbors_u[ind_even].reshape(-1,4), axis = 1)
    neighbors_u_even_sorted = np.take_along_axis(neighbors_u[ind_even].reshape(-1,4), order_even, axis = 1)
    neighbors_w_even_sorted = np.take_along_axis(neighbors_w[ind_even].reshape(-1,4), order_even, axis = 1)

    # Sort the neighbors' values in ascending order for odd rows.
    # Apply the same sorting to the corresponding weights.
    order_odd = np.argsort(neighbors_u[ind_odd].reshape(-1,4), axis = 1)
    neighbors_u_odd_sorted = np.take_along_axis(neighbors_u[ind_odd].reshape(-1,4), order_odd, axis = 1)
    neighbors_w_odd_sorted = np.take_along_axis(neighbors_w[ind_odd].reshape(-1,4), order_odd, axis = 1)

    # Split the image data in the current group into even and odd rows.
    img_data_reshape_even = img_data[ind_even].reshape(-1)
    img_data_reshape_odd  = img_data[ind_odd].reshape(-1)
    
    # Concatenate relevant image data, weights and current iterate.
    img_data_reshape   = np.concatenate((img_data_reshape_even,   img_data_reshape_odd),   axis = 0)
    neighbors_w_sorted = np.concatenate((neighbors_w_even_sorted, neighbors_w_odd_sorted), axis = 0)
    neighbors_u_sorted = np.concatenate((neighbors_u_even_sorted, neighbors_u_odd_sorted), axis = 0)
  
    # Evaluate the proximal map of the weighted mean absolute error function.
    result = prox_wmae(img_data_reshape, neighbors_w_sorted, neighbors_u_sorted, beta)

    # Assemble and return the updated iterate.
    u_new = np.copy(u)
    u_new[ind_even] = result[:img_data_reshape_even.size].reshape(u_new[ind_even].shape)
    u_new[ind_odd]  = result[img_data_reshape_even.size:].reshape(u_new[ind_odd].shape)
    return u_new

def objective(u, f, beta, D):
    r""" computes the objective value (4.1) for the image denoising problem at the image u, image data :code:`f` and total-variation parameter :code:`beta`

    .. math:
        
        objective(u) = fidelity(u) + \beta * TV(u)

    with data fidelity measure

    .. math::

        fidelity(u) = \frac{1}{2} \sum_{i=1}^{D_1} \sum_{j=1}^{D_2} (u_{i,j} - f_{i,j})^2.

    and total-variation semi-norm

    .. math::

        TV(u) = \sum_{i=1}^{D_1} \sum_{j=1}^{D_2} |u_{i+1,j} - u_{i,j}| + |u_{i,j+1} - u_{i,j}|,

    where out-of-bounds differences are set to :math:`0`. Using the jump matrix :code:`D`, this amounts to

    .. math::

        TV(u) = \| D \operatorname{vec}(u) \|_1,

    where :math:`\operatorname{vec}(u)` represents the image reshaped as vector.

    Parameters
    ----------
    f : np.array of shape (width, height)
        array containing the image data 
    u : np.array of shape (width,height)
        current iterate with image values
    beta : float
        total-variation parameter
    D : sparse matrix
        jump matrix, precomputed by :py:meth:`allocate_jump_matrix`, to evaluate TV(u)
    """

    # Evaluate the fidelity term.
    fidelity_term = 0.5 * np.sum((u - f)**2, axis = (0,1))

    # Evaluate the total-variation term.
    TV_term = np.sum(np.abs(D.dot(u.flatten())))

    # Return the result.
    return fidelity_term + beta * TV_term

def subdiff_projection(u, f, beta, D, eps_jump = 0.5):
    r""" evaluates the orthogonal projection of the zero matrix onto the subdifferential of the objective (4.1) at u.

    This amounts to solving a quadratic programming (QP) problem with sparse linear
    equality constraints as well as bound constraints. The problem to be solved is

    .. math::
    
        \text{Minimize} \| \operatorname{vec}(u) - \operatorname{vec}(f) + beta D^T w \|_2^2

    with respect to the variable :math:`w`, subject to the constraint that :math:`w` belongs to the subdifferential of the 1-norm at :math:`D \operatorname{vec}(u)`.
    The latter condition amounts to the constraints :math:`w_j = 1` where :math:`[D \operatorname{vec}(u)]_j > 0`, :math:`w_j = -1` where :math:`[D \operatorname{vec}(u)]_j < 0`, and :math:`w_j \in [-1,1]` otherwise.
    In the implementation, :math:`0` is replaced by a positive number :code:`eps_jump`.

    Before solving, the QP is reduced by eliminating the components of :math:`w` which are known to be equal to :math:`\pm 1`.

    Parameters
    ----------
    u : np.array 
        current iterate with image values
    f : np.array of same shape as u
        array containing the image data 
    beta : float
        total-variation parameter
    D : sparse matrix
        jump matrix, precomputed by :py:meth:`allocate_jump_matrix`, to evaluate TV(u)
    eps_jump : float, optional
        tolerance for treating a jump as no jump (default 1e-4)

    Returns
    -------
    np.array of shape (width, height)
        projection of the zero matrix onto the subdifferential of the objective
    """ 

    # Evaluate all differences between neighboring pixel values.
    jumps = D.dot(u.flatten()) 

    # Find the indices where w is free in [-1,1].
    free_indices = np.where(np.abs(jumps) <= eps_jump)[0]

    # Initialize w to some element in the subdifferential.
    w = np.sign(jumps)

    # In case the subdifferential is trivial, return it
    if len(free_indices) == 0:
      return u - f + beta * (D.T.dot(w)).reshape(u.shape)

    # Initialize w to zero on the free indices.
    w[free_indices] = 0

    # Keep only the relevant rows of the jump matrix.
    Dreduced = D[free_indices]

    # Setup the Hessian of the reduced objective.
    # For numerical stability, add a small multiple of the identity matrix.
    P = beta**2 * Dreduced.dot(Dreduced.T) + 1e-16 * eye(len(free_indices)) 

    # Setup the gradient of the reduced objective.
    q = beta * Dreduced.dot(u.flatten() - f.flatten()) + beta**2 * Dreduced.dot(D.T.dot(w))

    # Setup the bound constraints as linear inequality constraints.
    A = eye(len(free_indices), format = "csc")
    lb = -np.ones(len(free_indices))
    ub = np.ones(len(free_indices))

    # Solve the QP using osqp.
    QP = osqp.OSQP()
    QP.setup(P, q, A, lb, ub, verbose = False, polish = 1, eps_abs = 1e-10, eps_rel = 1e-5)
    res = QP.solve()

    # Copy the solution into the free indices of w.
    # Make sure the solution lies within [-1,1].
    w[free_indices] = np.clip(res.x, -1, 1)

    # Return the solution.
    return u - f + beta * (D.T.dot(w)).reshape(u.shape) 

def allocate_jump_matrix(u):
    r""" allocates the jump matrix :math:`D`.
    
    :math:`D` is the matrix representing the differences between neighboring pixel values.
    The dimensions are taken from :code:`u`.

    Parameters
    ----------
    u : np.array 
        array with image values (only used for shape)

    Returns 
    -------
    D : scipy.sparse.csc_matrix
        sparse matrix representing the differences between neighboring pixel values
    """

    # Get the dimensions of the image.
    n = u.shape[0]
    m = u.shape[1]

    # Get flattened indices of neighbors involved in vertical differences.
    vdiff_plus  = [np.ravel_multi_index([i+1,j], u.shape) for i in range(n-1) for j in range(m)]
    vdiff_minus = [np.ravel_multi_index([i,  j], u.shape) for i in range(n-1) for j in range(m)]

    # Get flattened indices of neighbors involved in horizontal differences.
    hdiff_plus  = [np.ravel_multi_index([i,j+1], u.shape) for i in range(n) for j in range(m-1)]
    hdiff_minus = [np.ravel_multi_index([i,j],   u.shape) for i in range(n) for j in range(m-1)]

    # Get the number of vertical and horizontal differences.
    N_v = len(vdiff_plus)
    N_h = len(hdiff_plus)

    # Create the (values, rows, cols) information for the vertical differences.
    values = np.hstack([-np.ones(N_v), np.ones(N_v)])
    rows   = np.hstack([np.arange(N_v), np.arange(N_v)])
    cols   = np.hstack([vdiff_minus, vdiff_plus])

    # Append the (values, rows, cols) information for the horizontal differences.
    values = np.hstack([values, -np.ones(N_h), np.ones(N_h)])
    rows   = np.hstack([rows, np.arange(N_v, N_v + N_h), np.arange(N_v, N_v + N_h)])
    cols   = np.hstack([cols, hdiff_minus, hdiff_plus])

    # Allocate the matrix in CSC sparse format.
    return csc_matrix((values, (rows,cols)), shape = [N_h + N_v, u.size])

def main():
    # Load the noisy image and convert it to float.
    img_noisy = ImageOps.grayscale(Image.open("cameraman_noisy.png"))
    img_data = np.array(img_noisy).astype('float')

    # Evaluate the jump matrix.
    D = allocate_jump_matrix(img_data)
    
    # Set problem and algorithmic parameters.
    beta = 10 
    TOL_outer = 300
    TOL_inner = 1e-4

    # Initialize the image.
    u = np.copy(img_data) 

    # Enter checkerboard scheme (Algorithm 2).
    while True:
      norm_deltau = np.inf

      # Enter the inner loop performing alternating checkerboard steps.
      while norm_deltau > TOL_inner:

        # Perform one round of updates and evaluate its Frobenius norm.
        u_old = np.copy(u) 
        u = perform_checkerboard_step(img_data, u, beta, 0) 
        u = perform_checkerboard_step(img_data, u, beta, 1)
        norm_deltau = np.linalg.norm(u - u_old)

      # Evaluate the steepest descent direction.
      direction = - subdiff_projection(u, img_data, beta, D, eps_jump = 0.5)

      # Verify the stopping criterion for the outer loop.
      if np.linalg.norm(direction) < TOL_outer: 
        break 

      # Set initial trial step size and evaluate current objective.
      alpha = 0.5 
      current_obj = objective(u, img_data, beta, D)

      # Backtrack until objective decreases.
      while alpha > 1e-4:
        if objective(u + alpha * direction, img_data, beta, D) < current_obj:
          break
        alpha = 0.5 * alpha

      # Perform the update.
      u = u + alpha * direction

    # Crop u to data range [0,255] to prevent automatic rescaling by PIL.
    u = np.minimum(255, np.maximum(0,u)) 

    # Convert to PIL image and export to PNG.
    final_im = Image.fromarray(u).convert('RGB')
    final_im.save("cameraman_result.png")
    return

if __name__ == "__main__":
    main()
