""" 
This script realizes the numerical minimization of the deflection energy, as presented in Section 5 in the paper
  Baumgärtner, Herzog, Schmidt, Weiß: 
  The Proximal Map of the Weighted Mean Absolute Error
  arxiv: [2209.13545](https://arxiv.org/abs/2209.13545)

Equation numbers in the comments refer to this publication.
This script exports the optimal deflection y of the 2 domains :math:`\\Omega_1` and :math:`\\Omega_2` into the files "Omega_1.xdmf" and "Omega_2.xdmf".

In order to render the results, run 

.. code: 

    pvpython paraview_render_results.py

TODO Fix error in the max-to-abs-replacement. For the paper, this was fixed in commit 6060a387cc44443b522c1b14e9679c6546d37509

"""

import os
import numpy as np
from scipy.sparse import csr_matrix
import dolfin as df
import mshr
import prox_wmae

def ADMM(mesh,f,weights,d,alpha,rho,maxiter=50,tolerance=1e-20):
    """ performs the ADMM algorithm minimizing the discrete energy J as defined in (5.4)

    Parameters
    ----------
    mesh : df.Mesh
        underlying mesh :math:`\\Omega`
    f : df.Function(mesh,'CG',1)
        function :math:`f` representing the external forces
    weights : df.VectorFunction(mesh,'CG',1,dim=N)
        weights :math:`w` of the additional forces
    d : df.VectorFunction(mesh,'CG',1,dim=N)
        deflection points :math:`d`, where additional forces are activated 
    alpha : ufl.form
        form on the boundary representing the Robin boundary condition
    rho : Float
        augmentation parameter :math:`\\rho` for the augmented Lagrangian method 
    maxiter : Int, optional
        maximal number of outer ADMM iterations, by default 50
    tolerance : float, optional
        steplength tolerance until convergence, by default 1e-20
    """
    # precompute mass matrix M and stiffness matrix K
    V = df.FunctionSpace(mesh,'CG',1)
    K,M = compute_mass_matrices(V,alpha)
    # initialize all variables with 0
    y = df.Function(V)
    y.rename('y','y')
    z = df.Function(V)
    z.rename('z','z')
    mu = df.Function(V) 
    mu.rename('mu','mu')

    # create variables to store the previous iterates to compute the steplength after each outer step
    y_old = df.Function(V)
    z_old = df.Function(V)
    mu_old = df.Function(V) 

    # precompute sum of weights as needed in every z-update
    sum_of_weights = df.Function(V)
    N = weights.ufl_shape[0]
    sum_of_weights_np = np.sum(weights.vector().get_local().reshape(-1,N),axis=1)
    sum_of_weights.vector().set_local(sum_of_weights_np)
    
    # compute f_tilde modification following equation (5.4)
    f_tilde = df.project(f - 0.5 * sum_of_weights,V)


    # Perform ADMM updates until maxiter is reached or steplength is smaller than 1e-20
    for iter in range(maxiter):
        # save current iterates for steplength computation
        y_old.vector().set_local(np.copy(y.vector().get_local()))
        z_old.vector().set_local(np.copy(z.vector().get_local()))
        mu_old.vector().set_local(np.copy(mu.vector().get_local()))

        # perform ADMM update steps
        update_z(z,y,mu,f_tilde,rho,K,M,sum_of_weights)
        update_y(z,y,mu,d,weights,rho)
        update_dual(z,y,mu)

        if norm_M(z_old,z,M) < tolerance and norm_M(y_old,y,M)<tolerance and norm_M(mu_old,mu,M)<tolerance: 
            # did iter+1, since starting with 0 
            print("Reached required tolerance {0} after {1} iterations.".format(tolerance,iter+1))
            break
    return z,y,mu

def update_z(z,y,mu,f_tilde,rho,K,M,sum_of_weights): 
    r""" performs the update on the variable z (5.7a), storing the result into the df.Function :code:`z`

    The optimality condition for the :code:`z` minimization reads 

    .. math:

        Kz - Mf - M*1 + \rho M ( z - y + \mu ) = 0

    Here, :math`1` means the constant function 1. 
    We can easily rearrange this to 

    .. math:: 

        (K + \rho M ) z = M*1 + Mf + \rho M ( y - \mu ). 


    Parameters
    ----------
    z : df.Function
        current iterate :math:`z`
    y : df.Function
        current iterate :math:`y`
    mu : df.Function
        current dual variable :math:`\mu`
    f_tilde : df.Function
        external force f_tilde, after the replacement introudced after equation (5.4)
    rho : Float
        augmentation parameter of the ADMM 
    K : df.Matrix 
        stiffness matrix of the underlying problem, precomputed using :py:meth:`compute_mass_matrices`
    M : df.Matrix
        mass-lumped matrix, precomputed using :py:meth:`compute_mass_matrices`
    sum_of_weights : df.Function
        precomputed sum of the :code:`weights` for every dof
    """
    # define the required linear form A to solve A * z = b
    A = K + rho * M
    b =  M * (-sum_of_weights.vector() + f_tilde.vector() + rho * y.vector() - rho * mu.vector()) 
    # call df linear solver
    df.solve(A , z.vector()  , b) # result stored into z
    return 

def update_y(z,y,mu,d,weights,rho): 
    r""" performs the update on the variable y (5.8), storing the result into the df.Function :code:`y`

    This solves 
    
    .. math::

        Minimize_{y_i\in\R} \quad
            \frac{1}{2} (z_i-y_i+\mu_i)^2 
            + \frac{1}{2\rho} \sum_{i=1}^N w_i| d_i - y_i | 

    The problems decouple for each dof of the mesh. 
    This is solved using the vectorized implementation :py:meth:`prox_wmae.prox_wmae`. 

    Parameters
    ----------
    z : df.Function
        current iterate :math:`z`
    y : df.Function
        current iterate :math:`y`
    mu : df.Function
        current dual variable :math:`\mu`
    d : df.VectorFunction
        deflections where the mass energy starts
    weights : df.VectorFunction
        weight of the masses at :code:`d`
    rho : Float
        augmentation parameter of the ADMM 
    """
    # setup x as the proximal point, in the notation of prox_wmae.py
    x = z.vector().get_local() + mu.vector().get_local()

    N = d.ufl_shape[0]
    # Evaluate the proximal map of the weighted mean absolute error function
    y_new = prox_wmae.prox_wmae(x , weights.vector().get_local().reshape(-1,N) , d.vector().get_local().reshape(-1,N) , 1/(2*rho)*np.ones(x.shape))
    y.vector().set_local(y_new)
    return 

def update_dual(z,y,mu):
    r""" performs the update of the dual variable (5.7c), storing the result into the df.Function :code:`mu`

    Parameters
    ----------
    z : df.Function
        current iterate :math:`z`
    y : df.Function
        current iterate :math:`y`
    mu : df.Function
        current dual variable :math:`\mu`

    """
    mu_new_loc = mu.vector().get_local() + z.vector().get_local() - y.vector().get_local()
    mu.vector().set_local(mu_new_loc)
    return

def compute_mass_matrices(V,alpha): 
    """ computes the stiffness matrix :code:`K` and the lumped mass matrix :code:`M` 

    Parameters
    ----------
    V : df.FunctionSpace
        function space for the variables :code:`z,y`
    alpha : ufl.form
        form on the boundary representing the Robin boundary condition

    Returns
    ---------

    df.Matrix
        stiffness matrix :math:`K`
    df.Matrix
        lumped mass matrix :math:`M` of the problem 

    """
    # compute stiffness K by the definition
    # define the stiffness K as a bilinear operator on Test-/ and TrialFunctions
    K_form = df.inner(df.grad(df.TrialFunction(V)),df.grad(df.TestFunction(V))) * df.dx + alpha * df.TestFunction(V)*df.TrialFunction(V)* df.ds
    # assemble the bilinearform K_form into a df.Matrix
    K = df.assemble(K_form)

    # compute lumped mass matrix M following [](https://fenicsproject.org/qa/4284/mass-matrix-lumping/)
    # define normal mass matrix 
    mass_form = df.TrialFunction(V) * df.TestFunction(V) * df.dx
    M = df.assemble(mass_form)
    # compute mass-lumped matrix by summing the rows for each dof
    mass_action_form = df.action(mass_form,df.Constant(1)) # df.action(a,u) inserts the function u into the bilinear operator a
    # set mass matrix entries to 0 
    M.zero()
    # update the diagonal entries of the mass matrix
    M.set_diagonal(df.assemble(mass_action_form))
    return K,M

def norm_M(z,y,M):
    """ evaluates the M-norm of :code:`z-y`

    This is defined by 

    .. math::
        
        norm_M(z,y) = \\sqrt{(z-y)^\\top M (z-y) }

    Parameters
    ----------
    z : df.Function
        1st function
    y : df.Function
        2nd function 
    M : df.Matrix
        matrix that defines a norm on the functions z,y

    Returns
    --------
    float
        M-distance between the functions z and y 

    """

    diff = z.vector()-y.vector()
    return diff.inner(M * diff)

def main():
    Omega_1 = df.RectangleMesh(df.Point(0,0),df.Point(1,1),35,35,diagonal='crossed')
    # create df.Functions for the input values according to (5.8) in the paper
    V_1 = df.FunctionSpace(Omega_1,'CG',1)
    f_1 = df.project(df.Constant(0.5), V_1)
    f_1.rename('f','external force')
    alpha_1 = df.Constant(10)
    W_1 = df.VectorFunctionSpace(Omega_1,'CG',1,dim=4)
    d_1 = df.project(df.Constant([0.01,0.02,0.03,0.04]) , W_1)
    d_1.rename('d','deflection points')
    weights_1 = df.project(df.Constant([0.02, 0.02, 0.02, 0.02]) , W_1)
    weights_1.rename('weights','weights of the deflection points')
    # perform ADMM
    z_1,y_1,mu_1 = ADMM(Omega_1,f_1,weights_1,d_1,alpha_1,rho=100,maxiter=2000)

    # export final deflection z=y
    fil = df.XDMFFile(df.MPI.comm_world,'Omega_1.xdmf')
    fil.write_checkpoint(z_1,'z',0,append=False)
    fil.close()

    # define L-shaped mesh Omega_2
    rectangle_big = mshr.Rectangle(df.Point(0,0),df.Point(1.1,1.1))
    rectangle_small = mshr.Rectangle(df.Point(0.6,0.6),df.Point(1.1,1.1))
    L_domain = rectangle_big - rectangle_small
    Omega_2 = mshr.generate_mesh(L_domain,45)
    V_2 = df.FunctionSpace(Omega_2,'CG',1)
    f_2 = df.project(df.Constant(0.5), V_2)
    f_2.rename('f','external force')
    alpha_2 = df.Constant(10)
    W_2 = df.VectorFunctionSpace(Omega_2,'CG',1,dim=4)
    d_2 = df.project(df.Constant([0.01,0.02,0.03,0.04]) , W_2)
    d_2.rename('d','deflection points')
    weights_2 = df.project(df.Constant([0.02, 0.02, 0.02, 0.02]) , W_2)
    weights_2.rename('weights','weights of the deflection points')
    # perform ADMM
    z_2,y_2,mu_2 = ADMM(Omega_2,f_2,weights_2,d_2,alpha_2,rho=100,maxiter=2000)
    # export final deflections z=y
    fil = df.XDMFFile(df.MPI.comm_world,'Omega_2.xdmf')
    fil.write_checkpoint(z_2,'z',0,append=False)
    fil.close()

if __name__ == '__main__':
    main()
