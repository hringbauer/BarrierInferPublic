'''
This is a script that takes input-data:
nr_individuals x 2 Position-Matrix X
and 
nr_individuals x k Genotype Matrix
and runs a MLE estimate and Standard Error Estimates (From Fisher Information Matrix)
'''

import numpy as np
import tensorflow as tf
import pylab
import cPickle as pickle  # @UnusedImport

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from random import shuffle # For shuffling start position list

import time
import scipy.optimize as opt




#############################
# Some important constants:
mean_param = np.pi/2.0  # Set everything to true mean allele frequency 0.5 (in latent space)
hyper_paramss = np.array([0.02, 15, 0.5]) # The initial values for a & L & c !!!
data_set_nr = 1  # Which data-set to use
#############################



############################# # The data:
path1 = "data_lists.p"  # Load data from pickle file!
x_data_sets, y_data_sets  = pickle.load(open(path1, "rb"))  # Load data from pickle file!
print(len(x_data_sets))


X_data=x_data_sets[data_set_nr]
Y_data=y_data_sets[data_set_nr]


nr_inds=200  # Nr. of Individuals to Check

inds=range(len(X_data))
shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
inds=inds[:nr_inds]  # Only load first nr_inds

# Only load first nr_inds entries
X_data = X_data[inds,:]  # Here is where the magic happens!
Y_data = Y_data[inds,:]  # Here as well!

# Read out the Dimensions
N = X_data.shape[0]  # Load the dimensions
k = X_data.shape[1]
print("Dim. of X-data: %i x %i" % (N,k))
M1 = Y_data.shape[0]
nr_loci = Y_data.shape[1] 
assert(len(X_data) == len(Y_data))   # Asserts that X- and Y-data have the same length.
mean_paramss = np.array([mean_param for _ in range(nr_loci)])  # Set everything to true mean allele frequency 0.5 (in latent space)
###################################

############# The Full Kernel:
def full_kernel_function(coords, l, a, c):
    '''Return barrier Kernel - describing reduced correlation across barrier
    and increased correlation next to barrier. Coords is nx2 Numpy array.
    Barrier is assumed to be at x=0'''
    x = coords[:, 0]  # Extracts x-coords
    
    mask = tf.constant([-1.0, 1.0], dtype=tf.float64) # Mask used for reflecting coordinates
    
    coords_refl = coords * mask[None,:]  # Reflects the x-Coordinate
    
    g = tf.sign(x)  # Calculates Signum of x
    same_side = (g[:,None] * g + 1) / 2  # Whether the x-Values are on the same side
    #print("Same Side Vector: ")
    #print(same_side)
    
    r_sqrd = tf.reduce_sum(((coords[:, None] - coords[None, :]) ** 2), reduction_indices=[2])  # Calculates pairwise Distance
    r_refl_sqrd = tf.reduce_sum(((coords_refl[:, None] - coords[None, :]) ** 2), reduction_indices=[2])  # Calculates the reflected Distance 
    
    # Calculate the normal Kernel:
    cov_mat = a * tf.exp(-r_sqrd / (2. * l ** 2))  # Calculate the co-variance matrix. Added diagonal term
    cov_mat_refl = a * tf.exp(-r_refl_sqrd / (2. * l ** 2))  # Calculate the covariance matrix for reflected coordinates.
    K = same_side * (cov_mat + c * cov_mat_refl) + (1 - same_side) * (1 - c) * cov_mat
    return K
##############################

#############################################################
# From Alex: To calculate the log determinant
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
def py_logdet(x, name=None):
    
    with ops.op_scope([x], name, "Logdet") as name:
        logdet = py_func(lambda x: np.linalg.slogdet(x)[1],
                         [x],
                         [tf.float64],
                         name=name,
                         grad=_LogdetGrad)  # <-- here's the call to the gradient
    return logdet[0]

def _LogdetGrad(op, grad):
    x = op.inputs[0]
    return tf.matrix_inverse(x) * grad[:, None, None]

############################################################



# Sin^2(x/2) Link function (Modifications by Harald)
def link_f(f):   # Define Link function
    '''Sinus Squared Link function'''
    pi=0.999*np.pi*tf.ones((N, nr_loci), dtype=tf.float64)  # Matrix - for values bigger than Pi. 
    #Allow for values slightly smaller than Pi to infinte likelihood (to avoid infinities)
    z=0.001*tf.ones((N, nr_loci), dtype=tf.float64)       
    # Matrix - for values smaller than 0. Allow for small values slightly bigger than 1 (to avoid infinities)
    f=tf.where(f>np.pi, pi, f)       # Make values bigger than pi almost pi
    f=tf.where(f<0, z, f)            # Make values smaller than 0 almost 0
    
    y = tf.sin(0.5*f) ** 2        # Do the actual calculation
    return y

with tf.device('/gpu:0'):
    X = tf.Variable(dtype=tf.float64, initial_value = X_data, trainable=False)
    Y = tf.Variable(dtype=tf.float64, initial_value = Y_data, trainable=False)
    F = tf.Variable(dtype=tf.float64, initial_value = np.random.normal(0.0, 0.1, (N, nr_loci)).astype('float64'),
                    trainable=True)

    #hyper_params = tf.placeholder(shape = [3 + nr_loci], dtype=tf.float64)      # HyperParameters for the Kernel
    hyper_params = tf.placeholder(shape=[3], dtype=tf.float64)                # The Kernel Hyper Parameters
    hyper_params_means =tf.placeholder(shape=[nr_loci], dtype=tf.float64)      # The Mean Hyper Parameters
    # First two are for the kernel; nr_loci are for the means of the GRFs!
    #mean_params = tf.placeholder(shape=[nr_loci], dtype=tf.float64)  # HyperParameters for the mean Values
    
    a = tf.abs(hyper_params[0])     # The Parameter for the absolute correlation
    l = tf.abs(hyper_params[1])     # The Parameter for the length scale
    c = tf.abs(hyper_params[2])
    #mean_param = hyper_params[3:]   # The mean parameters
    mean_param = hyper_params_means
    
    eye = tf.eye(N, dtype=tf.float64)

    # The Kernel for the classical Covariance Matrix
    #K = a * tf.exp(-tf.reduce_sum(((X[:, None] - X[None, :]) ** 2) / (2 * l**2), reduction_indices=[2])) + 0.00001*eye\
    #0.00001 * tf.eye(N, dtype=tf.float64) # Calculate Matrix of Covariances
    
    # The Kernel for the full covariance Matrix:
    #K = rbf_kernel_function(X, l, a) + 0.00001 *eye
    K = full_kernel_function(X, l, a, c) + 0.00001 * eye # Add identity to make positive definite
 
    
    f_tot = F + mean_param[None, :]    # Adds mean term; None is at position of an individual
    #f_tot= F + f_mean[None,:]          # Adds empirical mean term
    #f_tot = F + f_mean                 # For the case that there is a single Value
    p = link_f(f_tot)    # Calculate novel p, assuming sin^2 Link Function
    Kinv_F = tf.matrix_solve(K, F)      # Calculate K^(-1)*F
    
    g0 = tf.greater(Y, 0.5)        # Values where Y is greater than 0; i.e. data is 1.
    g0 = tf.cast(g0, tf.float64)   # Transform so that can be multiplied

    y_f= g0*p + (1-g0)*(1-p)  # Probability of observing y given probabilities p
    
    data_fit = tf.reduce_sum(tf.log(y_f), reduction_indices=[0])   # Calculate Data Fit: Check  
    prior_fit = -0.5 * tf.reduce_sum(F * Kinv_F, reduction_indices=[0])          # Calculate 1st term of prior probability: Check
    
    logL = tf.reduce_sum(data_fit + prior_fit) # Modulo terms not depending on f: Check
    
    d1 = 1.0/tf.tan(0.5*f_tot)   # Gradient coming from f_tot (for y=1)
    d0 = -tf.tan(0.5*f_tot)      # Gradient comfing from t_tot (for y=0)
    
    grad1 = g0*d1 + (1-g0)*d0  # First part of the gradient
    
    g =  grad1 - Kinv_F       # Calculate Matrix for Gradients Check
    
    h1 = -0.5 / ((tf.sin(0.5 * f_tot))**2) # Minus second derivative of data. (for y=1)
    h0 = -1.0 / (1 + tf.cos(f_tot))        # Second derivative of data (for y=0)
    W = - g0*h1 - (1-g0)*h0       # Calculate first part of Hessian. It is diagonal (but here diagonal in every column)
    
    lhs = -K[:, :, None] * W[None, :, :] - tf.eye(N, dtype=tf.float64)[:, :, None] # Check (HEAD-ACHE)
    #lhs = -K[:, :, None] * W[None, :, :] - eye[:, :, None] 
    rhs = tf.matmul(K, g)
    
    update = tf.matrix_solve(tf.transpose(lhs, [2, 0, 1]), tf.transpose(rhs, [1, 0])[:, :, None])[:, :, 0]
    update = tf.transpose(update)
    
    opt_op = F.assign(F - update)
    
    #####
    
    B = (W[:, None, :] ** 0.5) * K[:, :, None] * (W[None, :, :] ** 0.5) + tf.eye(N, dtype=tf.float64)[:, :, None]  # Check
    #B = (W[:, None, :] ** 0.5) * K[:, :, None] * (W[None, :, :] ** 0.5) + eye[:, :, None] 
    
    logdet = tf.reduce_sum(py_logdet(tf.transpose(B, [2, 0, 1])))  # Factor of 2??
    
    margL = logL - 0.5 * logdet
    
    ##### Gradients and Hessian #######################3
    
    grad = tf.gradients(margL, hyper_params)
    hessian = tf.hessians(margL, hyper_params)  # Comment out to avoid computational overhead.
    
###########################################################################################

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.allow_soft_placement = True

############################################################################################ 



 

### Optimization Run with Method of Harald
# Uses Nelder-Mead to be faster
mps = mean_paramss

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    for j in range(5):
        _ = sess.run([opt_op], {hyper_params: hyper_paramss, hyper_params_means: mps}) 
    
        
    def f(x):
        print("\nParameters: ")
        print(x)
        
        for i in range(2):
            _, = sess.run([opt_op,], {hyper_params: x, hyper_params_means: mps})   # Optimize the F 2x (Very fast; quadratic convergence)
            
        f, = sess.run([margL,], {hyper_params: x, hyper_params_means: mps})    # Calculates the Likelihood
        print("Likelihood: ")
        print(f)
        return -f # Returns the negative function Value

    result = opt.fmin(f, hyper_paramss, disp=1, ftol=1.0)   # Do 1 iteration of bfgs_b
    hyper_paramss_est=np.array([result[0], result[1], result[2]])
    g, h = sess.run([grad, hessian], {hyper_params: hyper_paramss_est, hyper_params_means: mps})
    
    fisher_info = np.matrix(h[0])
    stds = np.sqrt(np.diag(-fisher_info.I)) # Calculates the Standard Deviations (Sqrt of Variance)
    
    res=[result[:3]]
    print(res)
    print(stds)
    #return (res, stds)


















