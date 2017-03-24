'''
Created on March 2nd, 2017:
@Harald: Contains class for MLE estimaton. Based on
a simple model for pairwise coalescence.
It sums up all pairwise likelihoods. This makes it a
composite Maximum Likelihood scheme.
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
    
class MLE_pairwise(GenericLikelihoodModel):
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
        
        ll = self.likelihood_function(kernel_mat)  # Calculates the log likelihood
        

    
        toc = time()
        print("Total runtime: %.4f " % (toc - tic))
        print("Total log likelihood: %.4f \n" % ll)
        return ll  # Return the log likelihood
    
    def likelihood_function(self):
        '''Function to calculate pairwise likelihood directly from simple model'''
        coords, genotypes = self.exog, self.endog 
        nr_inds, nr_loci = np.shape(genotypes)
        
        # Calculate Mean and Variance
        p=np.mean(genotypes, axis=0)
        p=np.var(genotypes, axis=0)
        
        genotype_mat= np.abs(genotypes[:,None] - genotypes[None,:])  # Calculates Matrix of all Genotype pairs. Do this via 
        print("TO IMPLEMENT")
        print("TO IMPLEMENT")
        
        return ll # Returns the Log Likelihood
        
    
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
    
        
    ###################################################

   
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
