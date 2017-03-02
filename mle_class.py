'''
Created on March 2nd, 2017:
@Harald: Contains class for MLE estimaton. Is basically a wrapper
for stuff which is implemented in Tensor-Flow and Kernel Calculation
'''

from statsmodels.base.model import GenericLikelihoodModel
import matplotlib.pyplot as plt
import numpy as np
    
class MLE_estim_error(GenericLikelihoodModel):
    '''
    Class for MLE estimation. Inherits from GenericLikelihoodModel.
    There to automatically run Maximum Likelihood Estimation
    '''
    estimates = []     # Array for the fitted estimates
    start_params = []  # The starting Parameters for the Fit
    # Parameters for MLE analysis

    
    def __init__(self, kernel_func, coords, genotypes, **kwds):
        '''Takes the Coordinates as Input. Plugs them into Kernel.'''
        exog = coords        # The exogenous Variables are the coordinates
        endog = genotypes    # The endogenous Variables are the Genotypes
        super(MLE_estim_error, self).__init__(endog, exog, **kwds)  # Create the full object.
        self.create_bins()  # Create the Mid Bin vector
        self.fp_rate = fp_rate(self.mid_bins) * self.bin_width  # Calculate the false positives per bin
        self.density_fun = bl_dens_fun  # Set the block density function 
        self.start_params = start_params 
        self.error_model = error_model  # Whether to use error model
        if self.error_model == True:  # In case required:  
            self.calculate_trans_mat()  # Calculate the Transformation matrix
        
    def loglike(self, params):
        '''Return Log Likelihood'''
        print("To Implement - Probably Copy and Paste from Tensor Flow")

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            start_params = self.start_params  # Set the starting parameters for the fit
        fit = super(MLE_estim_error, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
        self.estimates = fit.params
        return fit
    
    def pairwise_ll(self, l, exog, params):
        '''Log likelihood function for every raw of data (sharing between countries).
        Return log likelihood.'''
        r, pw_nr = exog[0], exog[1]  # Distance between populations
        l = np.array(l)  # Make l an Numpy vector for better handling
        
        bins = self.mid_bins[self.min_ind:self.max_ind + 1] - 0.5 * self.bin_width  # Rel. bin edges
        l = l[(l >= bins[0]) * (l <= bins[-1])]  # Cut out only blocks of interest
        
        self.calculate_thr_shr(r, params)  # Calculate theoretical sharing PER PAIR
        self.calculate_full_bin_prob()  # Calculate total sharing PER PAIR /TM Matrix and FP-rate dont need update
        shr_pr = self.full_shr_pr[self.min_ind:self.max_ind]
        
        log_pr_no_shr = -np.sum(shr_pr) * pw_nr  # The negative sum of all total sharing probabilities
        if len(l) > 0:
            indices = np.array([(bisect_left(bins, x) - 1) for x in l])  # Get indices of all shared blocks
            l1 = np.sum(np.log(shr_pr[indices]))
        else: l1 = 0
        ll = l1 + log_pr_no_shr
        return(ll)    
    
        
    def block_shr_density(self, l, r, start_params):
        '''Returns block sharing density per cM; if l vector return vector
        Uses self.density_fun as function'''
        return self.density_fun(l, r, start_params)

############# Functions the class uses.      





   
######################### Some lines to test the code and make some plots
if __name__ == "__main__":
    print("To Implement")

    
##############################
