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
from scipy.stats import sem
from analysis import Fit_class
from scipy.optimize.minpack import curve_fit
from analysis import group_inds

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cPickle as pickle
import gc
    
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
    fixed_params = np.array([200, 0.001, 1.0, 0.])  # Full array nbh, L , t0 , ss
    param_mask = np.array([0, 1, 3])  # Parameter Mask used to change specific Parameters
    nr_params = 0
    parameter_names = []
    mps = [] 
    
    def __init__(self, kernel_class, coords, genotypes, start_params=None,
                 param_mask=None, multi_processing=0, **kwds):
        '''Initializes the Class.'''
        self.kernel = fac_kernel(kernel_class)  # Loads the kernel object. Use factory funciton to branch
        self.kernel.multi_processing = multi_processing  # Whether to do multi-processing: 1 yes / 0 no
        exog = coords  # The exogenous Variables are the coordinates
        endog = genotypes  # The endogenous Variables are the Genotypes
        
        self.mps = np.array([np.pi / 2.0 for _ in range(np.shape(genotypes)[1])])  # Set everything corresponding to p=0.5 (in f_space for ArcSin Model)
        
        super(MLE_pairwise, self).__init__(endog, exog, **kwds)  # Run initializer of full MLE object.
        
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
        '''Return Log Likelihood of the Genotype-Matrix given Coordinate-Matrix.'''
        # First some out-put what the current Parameters are:
        params = self.expand_params(params)  # Expands Parameters to full array
        print("Calculating Likelihood:")
        for i in xrange(self.nr_params):
            print(self.parameter_names[i] + ":\t %.4f" % params[i])
        
        if np.min(params) < 0:  # If any Parameters non-positive - return infinite negative ll
            ll = -np.inf
            print("Total log likelihood: %.4f \n" % ll)
            return ll  # Return the log likelihood
                   
        tic = time()     
        
        # Calculate Kernel matrix
        coords = self.exog
        print("Maximum Memory usage before Kernel calculation: %.4f MB" % memory_usage_resource())
        self.kernel.set_parameters(params)
        kernel_mat = self.kernel.calc_kernel_mat(coords) 
        
        var = params[-1]  # Assumes that the last parameter of the param-vector gives the all. freq. Variance.
        # Calculate Log Likelihood
        
        
        ll = self.likelihood_function(kernel_mat, var)
        

    
        toc = time()
        print("Maximum Memory usage: %.4f MB" % memory_usage_resource())
        print("Total runtime: %.4f " % (toc - tic))
        print("Total log likelihood: %.4f \n" % ll)
        return ll  # Return the log likelihood
    
    def likelihood_function(self, kernel_mat, var):
        '''Function to calculate pairwise likelihood directly from simple model'''
        genotypes = self.endog 
        nr_inds, nr_loci = np.shape(genotypes)
        mean_ps = np.mean(genotypes, axis=0)
        
        # Calculate Mean and Variance. Do it empirically
        # p_mean = np.mean(mean_ps)
        # p_var = np.var(mean_ps)
        
        # Estimate the variance:
        p_mean = 0.5
        p_var = var
        
        print("Mean Allele Freq: %.4f" % p_mean)
        print("Variance in Allele Freq: %.4f" % p_var)
        
        # Set Mean and Variance of the opposing genotypes.
        q_mean, q_var = 1 - p_mean, p_var
        
        inds = np.triu_indices(nr_inds, 1)  # Only take everything above diagonal.
        
        def ll_per_locus(genotypes, inds):
            '''Calculates the likelihood per locus - Written to save Memory.
            Genotype: Single array of genotypes.  But also works for multiple rows - Warning: Memory 
            Inds: Which positions to use in difference Matrix.'''
            # Calculates Matrix of mismatches of Genotype pairs. 
            genotype_mat = np.abs(genotypes[:, None] - genotypes[None, :]) 
            genotypes11 = genotypes[:, None] * genotypes[None, :]  # Where both genotypes are 1.
            genotypes00 = (1 - genotypes[:, None]) * (1 - genotypes[None, :])  # Where both genotypes are 0.
             
            # Extract upper triangular values into vectors to avoid double calculation:
            genotype_vec = genotype_mat[inds]  # Gives list of lists of differences.
            genotype11_vec = genotypes11[inds]  # Gives list where both are 1.
            genotype00_vec = genotypes00[inds]  # Gives list where both are 0.
            kernel_vec = kernel_mat[inds]  # Gives list
            
            # Do the composite likelihood Calculations:
            ll_same0 = genotype00_vec * (kernel_vec * q_mean + (1 - kernel_vec) * (q_var + q_mean ** 2))
            ll_same1 = genotype11_vec * (kernel_vec * p_mean + (1 - kernel_vec) * (p_var + p_mean ** 2))
            ll_different = genotype_vec * (1 - kernel_vec) * (p_mean - p_var - p_mean ** 2)  # That works. check
            
            ll = np.sum(np.log((ll_same0 + ll_same1 + ll_different)))  # Calulate the sum of all log-likelihoods. There should be no more 0.
            return ll  # Return the Log Likelihood
        
        ll_vec = [ll_per_locus(genotype_row, inds) for genotype_row in genotypes.T]  # calculates ll per locus. Iterate over columns of matrix
        total_ll = np.sum(ll_vec)  # Sum all log likelihoods
        return total_ll  # Return the total Likelihood.
        
    
    def fit(self, start_params=None, maxiter=500, maxfun=1000, **kwds):  # maxiter was 5000; maxfun was 5000
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            start_params = self.start_params  # Set the starting parameters for the fit
        
        # Check whether the length of the start parameters is actually right:
        assert(len(start_params) == len(self.param_mask))  
        
        fit = super(MLE_pairwise, self).fit(start_params=start_params,
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
        
################################################################################################################################        
################################################################################################################################

class MLE_f_emp(GenericLikelihoodModel):
    '''
    Class for MLE estimation. Inherits from GenericLikelihoodModel.
    This there to automatically run Maximum Likelihood Estimation.
    coords (nx2) and genotypes (nxk) are saved in self.exog and self.endog.
    One does a curve Fit via the Scipy model to fit all pairwise empirical estimates of F.
    The error is then estimated by taking the residuals (and curve fit assumes them equally distributed)
    '''
    # Diverse variables:
    estimates = []  # Array for the fitted estimates
    fixed_params = []  # Vector of all params that are fixed
    start_params = []  # The starting Parameters for the Fit
    kernel = 0  # Class that can calculate the Kernel
    nr_params = 0
    parameter_names = []
    nr_ind_demes = []  # Nr of Individuals per Deme  
    fit_params = []  # Which parameters to fit - the rest is fixed to start parameters
    min_distance = 0  # The minimum pairwise Distance that is analyzed
    inds = []  # Which indices to use based on min pw. distance
    fit_t0 = 0  # Whether to fit t0 as well
    
    def __init__(self, kernel_class, coords, genotypes, multi_processing=0,
                fit_t0=0, min_dist=0, max_dist=0, nr_inds=[],
                fit_params=[], fixed_params=[], start_params=[], **kwds):
        '''Initializes the Class and loads everything'''
        self.kernel = fac_kernel(kernel_class)  # Loads the kernel object. Use factory funciton to branch
        self.kernel.multi_processing = multi_processing  # Whether to do multi-processing: 1 yes / 0 no
        self.min_distance = min_dist  # What is the minimum Distance for pairs
        self.max_distance = max_dist  # What is the maximum Distance for pairs
        
        if max_dist == 0:
            self.max_distance = np.inf  # Set maximum Distance to Infinity in case not given.
        self.fit_t0 = fit_t0
        exog = coords  # The exogenous Variables are the coordinates
        endog = genotypes  # The endogenous Variables are the Genotypes
        
        if len(nr_inds) == 0:  # In case no Nr. of Inds per Deme supplied; 
            self.nr_ind_demes = np.ones(len(coords))  # Set everything to 1.
        else: self.nr_ind_demes = nr_inds
        
        
        super(MLE_f_emp, self).__init__(endog, exog, **kwds)  # Run initializer of full MLE object.
        
        # Load Parameters and Parameter names
        self.nr_params = self.kernel.give_nr_parameters()
        self.parameter_names = self.kernel.give_parameter_names()

        
        # Sets the fixed Parameters; i.e. the base for fitting:
        if len(fixed_params) == 0:
            raise RuntimeError("Supply Fixed Parameters!!")
        else: self.fixed_params = np.array(fixed_params)
        
        assert(len(self.fixed_params)==self.kernel.give_nr_parameters()) # Sanity Check.
        
        # Load which parameters to fit:
        if len(fit_params) == 0:
            self.fit_params = range(self.kernel.give_nr_parameters())  # All parameters are fit!
        else:
            self.fit_params = fit_params
            
        # Check for old version of code:
        if start_params == None:
            raise ValueError("Supply Starting Parameters!")
        if len(self.start_params) == 0:
            self.start_params = self.fixed_params[self.fit_params]
        
        # Sanity Check
        assert(len(self.fit_params) == len(self.start_params))

        
        # Some Output that everything loaded successfully:
        nr_inds, nr_loci = np.shape(genotypes)
        print("Successfully Loaded MLE-Object.")
        print("Nr inds: %i" % nr_inds)
        print("Nr loci: %i \n" % nr_loci)
        
        print("MultiProcessing for Kernel: %i" % self.kernel.multi_processing)
            

    def fit_function(self, coords, *args):
        '''Function that calculates the expected ratio of homozygotes based on Parameters
        *args for Kernel Function
        Return the Vector of fitted Values'''
        print(args)  # Prints arguments so that one knows where one is
        args = np.array(args)  # Make Arguments Numpy array so that it everything is fluent
        
        params = np.array(self.fixed_params)  # Set the base value
        params[self.fit_params] = args  # Overwrite the values to fit
        
        var = params[-1] # Extract Variance
        params[-1] = 0  # Set ss=0
        
        print("Parameters Sent to Kernel:")
        print(params)
        # Sets the variance Parameter 0; so that one can calculate the Kernel fluently
        assert(self.kernel.give_nr_parameters() == len(params))  # Checks whether Nr. of Parameters is right.
        self.kernel.set_parameters(params)  # Sets the kernel parameters
        
        tic = time()   
        kernel_mat = self.kernel.calc_kernel_mat(coords)  # Calculates the full kernel matrix
        kernel_vec = kernel_mat[self.inds]  # Extracts the Kernel as Vector for the right indices
        toc = time()
        print("Runtime Kernel: %.4f" % (toc - tic))
        
        predictor = kernel_vec + (1 - kernel_vec) * var
        
        return predictor

        
    def extract_right_indices(self, coords):
        '''Given Coords, calculates pw. Distance Matrix and then extracts
        indices where bigger than self.min_distance'''
        nr_inds = len(coords)  # How many individual data points
        inds = np.triu_indices(nr_inds, 1)  # Only take everything above diagonal.
        inds0, inds1 = inds
        
        pw_dist_mat = np.sqrt(np.sum((coords[:, None] - coords[None, :]) ** 2, axis=2))  # Calculates Pw. Distances.
        pw_dist_list = pw_dist_mat[inds]
        
        # Extract indices where greater than min. Distance and smaller than max. Distance:
        inds_md = np.where((pw_dist_list > self.min_distance) & (pw_dist_list < self.max_distance))[0]  
        inds = (inds0[inds_md], inds1[inds_md])  # Extracts right Matrix indices
        self.inds = inds  # Remembers so that class
        return inds
           
    def calc_mean_indentical(self, genotypes):
        '''Function to calculate matrix with counts how many Genotypes
        are identical'''
        genotypes11 = genotypes[:, None] * genotypes[None, :]  # Where both genotypes are 1.
        genotypes00 = (1 - genotypes[:, None]) * (1 - genotypes[None, :])  # Where both genotypes are 0.
        
        # Whats the right fract
        frac_genotypes_id = np.mean(genotypes11 + genotypes00, axis=2)  # Calculate Fraction shared
        frac_genotypes_sem = sem(genotypes11 + genotypes00, axis=2)  # Calculates SEMs
        
        return frac_genotypes_id, frac_genotypes_sem
    
        
    def fit(self, start_params=None, maxiter=500, maxfun=1000, **kwds):  # maxiter was 5000; maxfun was 5000
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            start_params = self.start_params  # Set the starting parameters for the fit
        
        # First extract and calculate pairwise distances;
        coords = self.exog
        nr_inds = len(coords)
        
        # Calculate Matrix with fraction of identical genotypes per pair
        genotypes = self.endog
        frac_genotypes_id, sems = self.calc_mean_indentical(genotypes)
        nr_pair_comps = self.nr_ind_demes[:, None] * self.nr_ind_demes[None, :]  # Calculate Number of pairwise comparisons.
        
        # Extract Pairs with right minimum pairwise Distance
        print("Nr of all pairwise comparisons: %i" % (nr_inds * (nr_inds - 1) / 2))
        inds = self.extract_right_indices(coords)
        y_values = frac_genotypes_id[inds]  # Makes a vector out of identical Genotypes
        nr_pair_comps = nr_pair_comps[inds]
        y_errors = sems[inds]  # Makes vector out of standard errors.        
        print("Extracted pairwise comparisons: %i" % len(y_values))
        print("Doing the Fitting...")
        
        lower_bounds = 0.0  # Lower Bound for the fit.
        upper_bounds = [np.inf for _ in start_params]  # Sets all upper bounds to infinity
        if len(upper_bounds) == 5:  # If Barrier is fitted as well set upper bound to 1.0 (no barrier)
            upper_bounds[2] = 1.0 
            
        sigma = 1.0 / np.sqrt(nr_pair_comps)  # The error in the pairwise values is proportion to 1/sqrt(nr_pair_comps)
        
        # Do the Fitting:
        parameters, cov_matrix = curve_fit(self.fit_function, coords, y_values,  # sigma=y_errors, absolute_sigma=True
                    p0=start_params, bounds=(lower_bounds, upper_bounds), sigma=sigma)  # @UnusedVariable p0=(C / 10.0, -r)
        
        std_params = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
        
        print("Parameters:")
        print(parameters)
        print("Unc. Estimates: ")
        print(std_params)
        
        # Create and fill up Fit object
        fit = Fit_class(parameters, std_params)
        return fit

######################################################### 
def memory_usage_resource():
    '''Returns Maximum Memory usage'''
    import resource
    rusage_denom = 1024.
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem

   
######################### Some lines to test the code and make plots
def analyze_barrier(position_list, genotype_mat, ind_deme_nr,
                    position_barrier=2, nr_inds=200, fit_t0=0, start_params=[136.07, 0.0122, 0.375, 0.53]):
    '''Method that analyzes a barrier. Use Method 2.'''
    # inds = range(len(position_list))
    # shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
    # inds = inds[:nr_inds]  # Only load first nr_inds
    # position_list = position_list[inds, :]
    # genotype_mat = genotype_mat[inds, :]
    
    MLE_obj = MLE_f_emp("DiffusionBarrierK0", position_list, genotype_mat, nr_ind_demes=ind_deme_nr,
                start_params=start_params, multi_processing=1, fit_t0=fit_t0)
    MLE_obj.kernel.position_barrier = position_barrier  # Sets the Barrier
    tic = time()
    fit = MLE_obj.fit(start_params=start_params)
    pickle.dump(fit, open("fitbarrier.p", "wb"))
    toc = time()
    print("Total Running Time of Fitting: %.4f" % (toc - tic))
    
def analyze_normal(position_list, genotype_mat, nr_inds=1000, fixed_params=[62, 0.006, 1 ,0.5], fit_params=[0,1,3], limit_inds=0,
                   nr_x_bins=0, nr_y_bins=0, min_ind_nr=1):
    '''Method that analyzes data without a barrier. Use Method 2.'''
    # Load only certain Number of Individuals
    if limit_inds == 1:
        inds = range(len(position_list))
        shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
        inds = inds[:nr_inds]  # Only load first nr_inds


    # Group Inds:
    if nr_x_bins > 0 or nr_y_bins > 0:
        position_list, genotype_mat, nr_inds = group_inds(position_list, genotype_mat,
                demes_x=nr_x_bins, demes_y=nr_y_bins, min_ind_nr=min_ind_nr)  
        
        
    # position_list = position_list[inds, :]
    # genotype_mat = genotype_mat[inds, :]
    # MLE_obj = MLE_pairwise("DiffusionK0", position_list, genotype_mat, start_params=[75, 0.02, 0.01], multi_processing=1) 
    MLE_obj = MLE_f_emp("DiffusionK0", position_list, genotype_mat, fixed_params=fixed_params, 
                        fit_params=fit_params, multi_processing=1)
    
    # MLE_obj.loglike([200, 0.001, 1, 0.04])  # Test Run for a Likelihood
    
    # Run a likelihood surface
    nbh_list = np.logspace(0.5, 2.5, 10)  # Neighborhood List
    L_list = np.logspace(-3.5, -1.5, 10)  # Length-Scale List
    # params = [4*np.pi*5, 0.001, 1.0, 0]
    # true_vals = [4*np.pi*5, 0.001]
    
    # res = pickle.load(open("temp_save.p", "rb")) # Load the Pickle Data
    # MLE_obj.likelihood_surface(nbh_list, L_list, 0, 1, params, true_vals) # create the likelihood surface
    # MLE_obj.plot_loglike_surface(nbh_list, L_list, true_vals, res)  # Plots the Data
    
    
    # Do the actual Fitting: 
    fit = MLE_obj.fit()  # Could alter method here. nbh, mu
    pickle.dump(fit, open("fit.p", "wb"))  # Pickle
        


if __name__ == "__main__":
    # position_list = np.loadtxt('./nbh_folder_gauss/nbh_file_coords200.csv', delimiter='$').astype('float64')  # Load the complete X-Data
    # genotype_mat = np.loadtxt('./nbh_folder_gauss/nbh_file_genotypes200.csv', delimiter='$').astype('float64')  # Load the complete Y-Data
    position_list = np.loadtxt('./multi_barrier_hz/mb_posHZ_coords00.csv', delimiter='$').astype('float64')  # Load the complete X-Data
    genotype_mat = np.loadtxt('./multi_barrier_hz/mb_posHZ_genotypes00.csv', delimiter='$').astype('float64')  # Load the complete Y-Data
    #ind_deme_nr = np.loadtxt('./Data/inds_per_deme_HZall2.csv', delimiter='$')  
    # ind_deme_nr = np.ones(len(position_list))  # Load the Nr of Individuals per Deme
    #analyze_barrier(position_list, genotype_mat, ind_deme_nr)  # Do not forget to set position of barrier
    
    analyze_normal(position_list, genotype_mat, nr_x_bins=50, nr_y_bins=10)
    
#########################################


