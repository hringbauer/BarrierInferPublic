'''
Created on March 2nd, 2017:
@Harald: Contains class for MLE estimaton. Is basically a wrapper
for stuff which is implemented in Tensor-Flow and Kernel Calculation
'''

from statsmodels.base.model import GenericLikelihoodModel
from kernels import fac_kernel
from time import time
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from random import shuffle 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cPickle as pickle
    
class MLE_estimator(GenericLikelihoodModel):
    '''
    Class for MLE estimation. Inherits from GenericLikelihoodModel.
    This there to automatically run Maximum Likelihood Estimation.
    coords (nx2) and genotypes (nxk) are saved in self.exog and self.endog.
    '''
    # Diverse variables:
    estimates = []  # Array for the fitted estimates
    start_params = []  # The starting Parameters for the Fit
    kernel = 0  # Class that can calculate the Kernel
    fixed_params = np.array([200, 0.001, 1.0, 0.04])  # Full array
    param_mask = np.array([0, 1, 3])  # Parameter Mask used to change specific Parameters
    nr_params = 0
    parameter_names = []
    mps = [] 
    
    def __init__(self, kernel_class, coords, genotypes, start_params = None,
                 param_mask=None, multi_processing=0, **kwds):
        '''Initializes the Class.'''
        self.kernel = fac_kernel(kernel_class)                  # Loads the kernel object. Use factory funciton to branch
        self.kernel.multi_processing = multi_processing          # Whether to do multi-processing: 1 yes / 0 no
        exog = coords  # The exogenous Variables are the coordinates
        endog = genotypes  # The endogenous Variables are the Genotypes
        
        self.mps = np.array([np.pi / 2.0 for _ in range(np.shape(genotypes)[1])])  # Set everything corresponding to p=0.5 (in f_space for ArcSin Model)
        
        super(MLE_estimator, self).__init__(endog, exog, **kwds)  # Run initializer of full MLE object.
        
        # Load Parameters and Parameter names
        self.nr_params = self.kernel.give_nr_parameters()
        self.parameter_names = self.kernel.give_parameter_names()
        if start_params != None:
            self.start_params = start_params 
        if param_mask != None:
            self.param_mask = param_mask

        
        
        # Some Output that everything loaded successfully:
        nr_inds, nr_loci = np.shape(genotypes)
        print("Successfully Loaded MLE-Object.")
        print("Nr inds: %i" % nr_inds)
        print("Nr loci: %i \n" % nr_loci)
        
        print("MultiProcessing for Kernel: %i" % self.kernel.multi_processing)
        
    def loglike(self, params):
        '''Return Log Likelihood of the Genotype Matrix given Coordinate Matrix.'''
        # First some out-put what the current Parameters are:
        params = self.expand_params(params)  # Expands Parameters to full array
        print("Calculating Likelihood:")
        for i in xrange(self.nr_params):
            print(self.parameter_names[i] + ":\t %.4f" % params[i])
        
        if np.min(params) <= 0:  # If any Parameters non-positive - return infinite negative ll
            ll = -np.inf
            print("Total log likelihood: %.4f \n" % ll)
            return ll  # Return the log likelihood
                   
        tic = time()     
        
        (config, update, opt_op, logL, margL, F, K, mean_param) = self.set_tf_model()  # Sets the tensorflow Model.
        
    
        coords = self.exog
        self.kernel.set_parameters(params)
        kernel_mat = self.kernel.calc_kernel_mat(coords) 
        
        toc = time()
        print("Runtime Calculation Kernel: %.4f" % (toc - tic))
        

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            # Optimize the F 4x (Very fast; quadratic convergence)
            for i in range(4):
                _, = sess.run([opt_op, ], {K: kernel_mat, mean_param: self.mps})  
            
            ll, = sess.run([margL, ], {K: kernel_mat, mean_param: self.mps})  # Now calculate the Marginal Likelihood
        
        toc = time()
        print("Total runtime: %.4f " % (toc - tic))
        print("Total log likelihood: %.4f \n" % ll)
        return ll  # Return the log likelihood
    
    def fit(self, start_params=None, maxiter=500, maxfun=1000, **kwds):  # maxiter was 5000; maxfun was 5000
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            start_params = self.start_params  # Set the starting parameters for the fit
        
        # Check whether the length of the start parameters is actually right:
        assert(len(start_params) == len(self.param_mask))  
        
        fit = super(MLE_estimator, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
        self.estimates = fit.params
        return fit    
    
    def expand_params(self, params):
        '''Method to expand subparameters as defined in self.param_mask to full parameters'''
        all_params = self.fixed_params
        all_params[self.param_mask] = params  # Set the subarray
        return all_params
        
    def likelihood_surface(self, range1, range2, wp1, wp2, fix_params, true_vals):
        '''Method for creating and visualizing likelihood surface.
        w p ...which parameters.
        fix_params: Fixed Parameters.
        Range1 and Range2 are vectors'''
        res = []  # Vector for the results.
        
        for val1 in range1:
            for val2 in range2:
                # Set the Parameters
                fix_params[wp1] = val1
                fix_params[wp2] = val2
                ll = self.loglike(params)
                res.append(ll)
                
        pickle.dump(res, open("temp_save.p", "wb"))  # Pickle
        self.plot_loglike_surface(range1, range2, true_vals, res)  # Plots the Data
    
    def plot_loglike_surface(self, range1, range2, true_vals, res):
        '''Method to plot the loglikelihood surface'''
        surface = np.array(res).reshape((len(range1), len(range2)))
        
        plt.figure()
        plt.pcolormesh(range2, range1, surface)  # L and nbh
        # pylab.pcolormesh(, a_list, surface)
        plt.xscale('log')
        plt.yscale('log')
        # pylab.xlabel('L')
        # pylab.ylabel('Nbh Size')
        plt.colorbar()
        # pylab.plot(25, 0.1, 'ko', linewidth=5)
        plt.plot(true_vals[1], true_vals[0], 'ko', linewidth=5)
        plt.show()
        
        # Now one Plot were the 
        plt.figure()
        levels = np.arange(max(res) - 30, max(res) + 1, 2)  # Every two likelihood units
        # ax=pylab.contourf(l_list, a_list, surface, alpha=0.9, levels=levels)
        ax = plt.contourf(range2, range1, surface, alpha=0.9, levels=levels)
        
        cb = plt.colorbar(ax, format="%i")
        cb.ax.tick_params(labelsize=16)
        plt.title("Log Likelihood Surface", fontsize=20)
        plt.xlabel("L", fontsize=20)  # l
        plt.ylabel("NBH", fontsize=20)  # a
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(0.001, 62.8 * 4, 'ko', linewidth=5, label="True Value")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
    ################## Methods from Tensorflow-Analysis:
    def set_tf_model(self):
        '''This is the tensorflow Implementation of the Likelihood Model.'''
        coords, genotypes = self.exog, self.endog 
        nr_inds, nr_loci = np.shape(genotypes)
        # print("\nNr inds: %i" % nr_inds)
        # print("Nr loci: %i" % nr_loci)
        
        with tf.device('/cpu:0'):  # GPU or CPU.
            X = tf.Variable(dtype=tf.float64, initial_value=coords, trainable=False)
            Y = tf.Variable(dtype=tf.float64, initial_value=genotypes, trainable=False)
            F = tf.Variable(dtype=tf.float64, initial_value=np.random.normal(0.0, 0.1, (nr_inds, nr_loci)).astype('float64'),
                            trainable=True)  # Initial Value for the F
        
            K = tf.placeholder(shape=[nr_inds, nr_inds], dtype=tf.float64)  # Placeholder for the Kernel Matrix
            mean_param = tf.placeholder(shape=[nr_loci], dtype=tf.float64)  # The Mean Hyper Parameters
          
            f_tot = F + mean_param[None, :]  # Adds mean term; None is at position of an individual
            g0 = tf.greater(Y, 0.5)  # Values where Y is greater than 0; i.e. data is 1.
            g0 = tf.cast(g0, tf.float64)  # Transform so that can be multiplied
            
            p = link_f(f_tot)  # Calculate novel p, assuming sin^2 Link Function.
            grad1 = grad_f(f_tot, g0)  # Calculate the Gradient.
            W = hessian_f(f_tot, g0)  # Calculates the Hessian of the second derivative.
            
            Kinv_F = tf.matrix_solve(K, F)  # Calculate K^(-1)*F
            
            # Calculate Probabilities of observing y given p
            y_f = g0 * p + (1 - g0) * (1 - p)  # Probability of observing y given probabilities p
            
            data_fit = tf.reduce_sum(tf.log(y_f), reduction_indices=[0])  # Calculate Data Fit.  
            prior_fit = -0.5 * tf.reduce_sum(F * Kinv_F, reduction_indices=[0])  # Calculate 1st term of prior probability.
            
            logL = tf.reduce_sum(data_fit + prior_fit)  # Modulo terms not depending on f: Check
            
            g = grad1 - Kinv_F  # Calculate Matrix for Gradients Check
            lhs = -K[:, :, None] * W[None, :, :] - tf.eye(nr_inds, dtype=tf.float64)[:, :, None]  # Check (HEAD-ACHE)
            rhs = tf.matmul(K, g)
            
            update = tf.matrix_solve(tf.transpose(lhs, [2, 0, 1]), tf.transpose(rhs, [1, 0])[:, :, None])[:, :, 0]
            update = tf.transpose(update)
            opt_op = F.assign(F - update)
            
            ##################
            B = (W[:, None, :] ** 0.5) * K[:, :, None] * (W[None, :, :] ** 0.5) + tf.eye(nr_inds, dtype=tf.float64)[:, :, None]  # Check 
            
            det = tf.reduce_sum(tf.log(tf.matrix_diag_part(tf.cholesky(tf.transpose(B, [2, 0, 1])))))  # Factor of 2??
            # logdet = tf.reduce_sum(py_logdet(tf.transpose(B, [2, 0, 1])))  # Factor of 2?? Alex Log 
            margL = logL - det
            # margL = logL - 0.5 * logdet
            
            # Set up configuration
            config = tf.ConfigProto()  
            config.gpu_options.per_process_gpu_memory_fraction = 0.01
            
            return((config, update, opt_op, logL, margL, F, K, mean_param))
        
        
    ###################################################
       
     
# Sin^2(x/2) Link function which is used to transform from f to p space so that
# variance is right. Function, 1st and 2nd derivative:
def link_f(f):  # Define Link function
    '''Sinus Squared Link function'''
    pi = 0.999 * np.pi * tf.ones(tf.shape(f), dtype=tf.float64)  # Matrix - for values bigger than Pi. 
    # Allow for values slightly smaller than Pi to infinte likelihood (to avoid infinities)
    z = 0.001 * tf.ones(tf.shape(f), dtype=tf.float64)       
    # Matrix - for values smaller than 0. Allow for small values slightly bigger than 1 (to avoid infinities)
    f = tf.where(f > np.pi, pi, f)  # Make values bigger than pi almost pi
    f = tf.where(f < 0, z, f)  # Make values smaller than 0 almost 0
    
    y = tf.sin(0.5 * f) ** 2  # Do the actual calculation
    return y

def grad_f(f_tot, g0):
    '''Return Gradient of the Link function'''
    d1 = 1.0 / tf.tan(0.5 * f_tot)  # Gradient coming from f_tot (for y=1)
    d0 = -tf.tan(0.5 * f_tot)  # Gradient comfing from t_tot (for y=0)    
    grad1 = g0 * d1 + (1 - g0) * d0  # First part of the gradient
    return grad1

def hessian_f(f_tot, g0):
    '''Returns second derivative of Link Funciton'''
    h1 = -0.5 / ((tf.sin(0.5 * f_tot)) ** 2)  # Minus second derivative of data. (for y=1)
    h0 = -1.0 / (1 + tf.cos(f_tot))  # Second derivative of data (for y=0)
    W = -g0 * h1 - (1 - g0) * h0  # Calculate first part of Hessian. It is diagonal (but here diagonal in every column)
    return W

# From Alex: Some custom Functions to calculate the Log-Determinant.
# In there for historic reasons - used to be that this was needed for the Gradient.
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

   
######################### Some lines to test the code and make plots
if __name__ == "__main__":
    X_data = np.loadtxt('./Data/coordinates00.csv', delimiter='$').astype('float64')  # Load the complete X-Data
    Y_data = np.loadtxt('./Data/data_genotypes00.csv', delimiter='$').astype('float64')  # Load the complete Y-Data
    # Extract some information regarding the mean allele frequency:
    p_means = np.mean(Y_data, axis=0)
    print("Standard Deviation of mean allele frequencies: %.4f" % np.std(p_means))
    
    nr_inds_analysis = 1000
    inds = range(len(X_data))
    shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
    inds = inds[:nr_inds_analysis]  # Only load first nr_inds
    

    position_list = X_data[inds, :]
    genotype_mat = Y_data[inds, :]
    start_params = [75.0, 0.02, 0.04]  # nbh, mu, t0, ss
    MLE_obj = MLE_estimator("DiffusionK0", position_list, genotype_mat, start_params, multi_processing=1) 
    # MLE_obj.loglike([200, 0.001, 1, 0.04])  # Test Run for a Likelihood
    
    # Run a likelihood surface
    nbh_list = np.logspace(1.8, 2.7, 10)  # Neighborhood List
    L_list = np.logspace(-3.4, -2, 10)  # Lengthscale List
    params = [200, 0.001, 1.0, 0.04]
    true_vals = [251.33, 0.001]
    
    # res = pickle.load(open("temp_save.p", "rb")) # Load the Pickle Data
    # MLE_obj.likelihood_surface(nbh_list, L_list, 0, 1, params, true_vals) # create the likelihood surface
    # MLE_obj.plot_loglike_surface(nbh_list, L_list, true_vals, res)  # Plots the Data
    
    
    # Do the actual Fitting: 
    fit = MLE_obj.fit(start_params=[75, 0.02, 0.035])  # Could alter method here.
    pickle.dump(fit, open("fit.p", "wb"))  # Pickle
    print(fit.summary())
