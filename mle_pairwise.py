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
        self.kernel.set_parameters(params)
        kernel_mat = self.kernel.calc_kernel_mat(coords) 
        
        var = params[-1]  # Assumes that the last parameter of the param-vector gives the all. freq. Variance.
        # Calculate Log Likelihood
        ll = self.likelihood_function(kernel_mat, var)
        

    
        toc = time()
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
        
        # Calculates Matrix of mismatches of Genotype pairs. 
        genotype_mat = np.abs(genotypes[:, None] - genotypes[None, :]) 
        genotypes11 = genotypes[:, None] * genotypes[None, :]  # Where both genotypes are 1.
        genotypes00 = (1 - genotypes[:, None]) * (1 - genotypes[None, :])  # Where both genotypes are 0.
         
        # Extract upper triangular values into vectors to avoid double calculation:
        inds = np.triu_indices(nr_inds, 1)  # Only take everything above diagonal.
        genotype_vec = genotype_mat[inds]  # Gives list of lists of differences.
        genotype11_vec = genotypes11[inds]  # Gives list where both are 1.
        genotype00_vec = genotypes00[inds]  # Gives list where both are 0.
        kernel_vec = kernel_mat[inds]  # Gives list
        
        # Do the composite likelihood Calculations:
        ll_same0 = genotype00_vec * (kernel_vec[:, None] * q_mean + (1 - kernel_vec[:, None]) * (q_var + q_mean ** 2))
        ll_same1 = genotype11_vec * (kernel_vec[:, None] * p_mean + (1 - kernel_vec[:, None]) * (p_var + p_mean ** 2))
        ll_different = genotype_vec * (1 - kernel_vec[:, None]) * (p_mean - p_var - p_mean ** 2)  # That works. check
        
        ll = np.sum(np.log((ll_same0 + ll_same1 + ll_different)))  # Calulate the sum of all log-likelihoods. There should be no more 0.
        
        return ll  # Return the Log Likelihood
        
    
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
    The error is then estimated via taking the residuals (and curve fit assumes them equally distributed)
    '''
    # Diverse variables:
    estimates = []  # Array for the fitted estimates
    start_params = []  # The starting Parameters for the Fit
    kernel = 0  # Class that can calculate the Kernel
    fixed_params = np.array([200, 0.001, 1.0, 0.])  # Full array nbh, L , t0 , ss
    param_mask = np.array([0, 1, 3])  # Parameter Mask used to change specific Parameters
    nr_params = 0
    parameter_names = []
    min_distance = 0  # The minimum pairwise Distance that is analyzed
    inds = []  # Which indices to use based on min pw. distance
    
    def __init__(self, kernel_class, coords, genotypes, start_params=None,
                 param_mask=None, multi_processing=0, **kwds):
        '''Initializes the Class.'''
        self.kernel = fac_kernel(kernel_class)  # Loads the kernel object. Use factory funciton to branch
        self.kernel.multi_processing = multi_processing  # Whether to do multi-processing: 1 yes / 0 no
        exog = coords  # The exogenous Variables are the coordinates
        endog = genotypes  # The endogenous Variables are the Genotypes
        
        
        super(MLE_f_emp, self).__init__(endog, exog, **kwds)  # Run initializer of full MLE object.
        
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
            

    def fit_function(self, coords, *args):
        '''Function that calculates the expected ratio of homozygotes based on Parameters
        *args for Kernel Function
        Return the Vector of fitted Values'''
        print(args)  # Prints arguments so that one knows where one is
        args = np.array(args)  # Make Arguments Numpy array so that it everything is fluent
        var = args[-1]  # Gets the variance Parameter
        args[-1] = 1.0  # Sets t0 
        args = np.append(args, 0)  # appends ss=0
        # Sets the variance Parameter 0; so that one can calculate the Kernel fluently
        assert(self.kernel.give_nr_parameters() == len(args))  # Checks whether Nr. of Parameters is right.
        self.kernel.set_parameters(args)  # Sets the kernel parameters
        
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
        inds_md = np.where(pw_dist_list > self.min_distance)[0]  # Extract indices where greater than min. Distance
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
        inds = self.extract_right_indices(coords)
        
        # Calculate Matrix with fraction of identical genotypes per pair
        genotypes = self.endog
        frac_genotypes_id, sems = self.calc_mean_indentical(genotypes)
        y_values = frac_genotypes_id[inds]  # Makes a vector out of identical Genotypes
        y_errors = sems[inds]  # Makes vector out of standard errors.
        
        print("Doing the Fitting...")
        
        parameters, cov_matrix = curve_fit(self.fit_function, coords, y_values,  # sigma=y_errors, absolute_sigma=True
                    p0=start_params, bounds=(0, np.inf))  # @UnusedVariable p0=(C / 10.0, -r)
        
        std_params = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
        
        print("Parameters:")
        print(parameters)
        print("Unc. Estimates: ")
        print(std_params)
        
        # Create and fill up Fit object
        fit = Fit_class(parameters, std_params)
        return fit
    
    def expand_params(self, params):
        '''Method to expand subparameters as defined in self.param_mask to full parameters'''
        all_params = self.fixed_params
        all_params[self.param_mask] = params  # Set the subarray
        return all_params
        


   
######################### Some lines to test the code and make plots
def analyze_barrier(position_list, genotype_mat, position_barrier=500.5):
    '''Test Method that analyzes a barrier'''
    nr_inds_analysis = 200
    inds = range(len(position_list))
    shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
    inds = inds[:nr_inds_analysis]  # Only load first nr_inds
    # position_list = position_list[inds, :]
    # genotype_mat = genotype_mat[inds, :]
    
    MLE_obj = MLE_f_emp("DiffusionBarrierK0", position_list, genotype_mat, start_params=[65, 0.006, 0.5, 0.5], multi_processing=1)
    MLE_obj.kernel.position_barrier = position_barrier  # Sets the Barrier
    tic = time()
    fit = MLE_obj.fit(start_params=[65, 0.006, 0.5, 0.5])
    pickle.dump(fit, open("fitbarrier.p", "wb"))
    toc = time()
    print("Total Running Time of Fitting: %.4f" % (toc - tic))
    
def analyze_normal(position_list, genotype_mat):
    '''Method that analyzes data without a barrier.'''

    
    # Load only certain Number of Individuals
    nr_inds_analysis = 1000
    inds = range(len(position_list))
    shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
    inds = inds[:nr_inds_analysis]  # Only load first nr_inds
    

    # position_list = position_list[inds, :]
    # genotype_mat = genotype_mat[inds, :]
    # MLE_obj = MLE_pairwise("DiffusionK0", position_list, genotype_mat, start_params=[75, 0.02, 0.01], multi_processing=1) 
    MLE_obj = MLE_f_emp("DiffusionK0", position_list, genotype_mat, start_params=[75, 0.02, 0.5], multi_processing=1)
    
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
    fit = MLE_obj.fit(start_params=[65, 0.005, 0.5])  # Could alter method here. nbh, mu
    pickle.dump(fit, open("fit.p", "wb"))  # Pickle
    
if __name__ == "__main__":
    # position_list = np.loadtxt('./nbh_folder/nbh_file_coords200.csv', delimiter='$').astype('float64')  # Load the complete X-Data
    # genotype_mat = np.loadtxt('./nbh_folder/nbh_file_genotypes200.csv', delimiter='$').astype('float64')  # Load the complete Y-Data
    position_list = np.loadtxt('./Data/coordinates00b.csv', delimiter='$').astype('float64')  # Load the complete X-Data
    genotype_mat = np.loadtxt('./Data/data_genotypes00b.csv', delimiter='$').astype('float64')  # Load the complete Y-Data
    # analyze_barrier(position_list, genotype_mat) # Do not forget to set position of barrier
    analyze_normal(position_list, genotype_mat)
    
#########################################
