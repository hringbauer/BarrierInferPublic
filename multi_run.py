'''
Created on March 5th, 2017:
@Harald: Contains methods for producing as well as
analyzing multiple data-sets. There is one Master Class, and all other classes inherit from it:
Every MultiRun subclass will require implementation of all methods required for analysis.
Every class saves single result runs to numbered files in a folder - and loads them from there for analysis.
Analysis is then conducted and results/uncertainty estimates (confidence intervalls) are pickled.
For visualization the pickled file is loaded then.
'''

from statsmodels.base.model import GenericLikelihoodModel
from kernels import fac_kernel
from time import time
from grid import Grid, Secondary_Grid
from mle_class import MLE_estimator
from mle_class import calculate_ss
from mle_pairwise import MLE_pairwise
from mle_pairwise import MLE_f_emp
from random import shuffle 
from analysis import Analysis, group_inds, bootstrap_genotypes, clean_hz_data, flip_gtps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import os
import pickle as pickle

class MultiRun(object):
    '''
    Class that can produce as well as analyze multiple barrier datasets.
    '''
    data_folder = ""  # Folder
    name = "run"
    param_estimates = 0  # Will be matrix for parameters
    uncert_estimates = 0  # Will be matrix for parameter uncertainties
    nr_data_sets = 0  # Number of the datasets
    multi_processing = 0  # Whether to actually use multi-processing

    
    def __init__(self, data_folder, nr_data_sets, nr_params, multi_processing=0):
        '''
        Constructor. Sets the Grid class to produce and the MLE class to analyze the Data
        '''
        self.data_folder = data_folder
        self.nr_data_sets = nr_data_sets
        self.param_estimates = np.zeros((nr_data_sets, nr_params))  # Creates array for Parameter Estimates
        self.uncert_estimates = np.zeros((nr_data_sets, nr_params * 2))  # Creates array for Uncertainty Estimates
        self.multi_processing = multi_processing 
        
    def set_grid_params(self, grid, run_nr):
        '''Sets the Parameters for the grid depending on the run-nr'''
        raise NotImplementedError("Implement this!")
        
    def set_mle_object(self):
        '''Analyzes Data Set'''
        raise NotImplementedError("Implement this!")
    
    def set_data_path(self, position_path, gtp_path):
        '''Sets the path to the position and genotype files.'''
        self.position_path = position_path
        self.gtp_path = gtp_path
    
    def create_data_set(self, data_set_nr, position_path=None, genotype_path=None):
        '''Method to create data_set nr data_set_nr.'''
        # If method is called without path:
        if (position_path == None) or (genotype_path == None):
            raise NotImplementedError("Implement creation of Data-Set!")
        
        else:    
            position_list = np.loadtxt(position_path, delimiter='$').astype('float64')
            genotype_matrix = np.loadtxt(genotype_path, delimiter='$').astype('float64')
            self.save_data_set(position_list, genotype_matrix, data_set_nr) 
        print("Dataset successfully created!")
    
    def analyze_data_set(self, data_set_nr):
        '''Method that analyze data-set i.
        Return Parameter estimates and 95% Uncertainties'''
        raise NotImplementedError("Implement this!")
    
    def visualize_results(self):
        '''Method to visualize the results and uncertainty estimates'''
        raise NotImplementedError("Implement this!")
    
    def create_all_data_sets(self):
        '''Method that creates all data sets.
        Could be parallelized'''
        for i in xrange(self.nr_data_sets):
            self.create_data_set(i)
        
    def analyze_all_data_sets(self):
        '''Method that analyzes all data sets.
        Could be parallelized'''
        for i in xrange(self.nr_data_sets):
            self.analyze_data_set(i)
    
    def save_data_set(self, coords, genotype_matrix, data_set_nr):
        '''Method to save a Data set'''
        data_set_name_coords = self.data_folder + self.name + "_coords" + str(data_set_nr).zfill(2) + ".csv"
        data_set_name_genotypes = self.data_folder + self.name + "_genotypes" + str(data_set_nr).zfill(2) + ".csv"
        # Check whether Directory exists and creates it if necessary
        print("Saving...")
        directory = os.path.dirname(data_set_name_coords)  # Extract Directory
        print("Directory: " + directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        np.savetxt(data_set_name_coords, coords, delimiter="$")  # Save the coordinates
        np.savetxt(data_set_name_genotypes, genotype_matrix, delimiter="$")  # Save the data 
        
    def load_data_set(self, data_set_nr):
        '''Loads the data set'''
        data_set_name_coords = self.data_folder + self.name + "_coords" + str(data_set_nr).zfill(2) + ".csv"
        data_set_name_genotypes = self.data_folder + self.name + "_genotypes" + str(data_set_nr).zfill(2) + ".csv"
        

        position_list = np.loadtxt(data_set_name_coords, delimiter='$').astype('float64')
        genotype_matrix = np.loadtxt(data_set_name_genotypes, delimiter='$').astype('float64')
        return position_list, genotype_matrix
    
    def pickle_parameters(self, parameters_names, parameter_values, other_info):
        '''Pickles all parameters that have been used into the folder.
        Should be called in create_data_sets
        For later information'''
        path = self.data_folder + "parameters.p"
        pickle.dump((parameters_names, parameter_values, other_info), open(path, "wb"))  # Pickle the Info
        
    def save_analysis(self):
        '''Pickles the Outcome of the Analysis. For later information. '''
        path = self.data_folder + "analysis.p"
        pickle.dump((self.param_estimates, self.uncert_estimates), open(path, "wb"))
        
    def load_analysis(self):
        '''Loads Results.'''
        path = self.data_folder + "analysis.p"
        (self.param_estimates, self.uncert_estimates) = pickle.load(open(path, "rb"))
        return self.param_estimates, self.uncert_estimates
    
    def fit_IBD(self, start_params, data_set_nr, method=2,
                    res_folder=None, position_list=[], genotype_mat=[], random_ind_nr=None,
                    fit_params=[0, 1, 3], fixed_params=[50, 0.005, 1, 0]):
        '''Method to fit no-Barrier-Model'''
        
        # If not Position_List Data or no Genotype Data; put it in:
        if len(position_list) == 0 or len(genotype_mat) == 0:  
            position_list, genotype_mat = self.load_data_set(data_set_nr)  # Loads the Data 
            
        # Pick Random_ind_nr many Individuals in case needed:
        if random_ind_nr:
            inds = range(len(position_list))
            shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
            inds = inds[:random_ind_nr]  # Only load first nr_inds
    
            position_list = position_list[inds, :]
            genotype_mat = genotype_mat[inds, :]
                
        if method == 0:
            MLE_obj = MLE_estimator("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing) 
            start_params = start_params + [0.004, ]
        elif method == 1:
            MLE_obj = MLE_pairwise("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing)
            start_params = start_params + [0.01, ]
        elif method == 2:
            MLE_obj = MLE_f_emp("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing,
                                fixed_params=fixed_params, fit_params=fit_params)
            start_params = start_params + [0.5, ]
        elif method == 3:  # Do the fitting based on binned data
            MLE_obj = Analysis(position_list, genotype_mat) 
        else: raise ValueError("Wrong Input for Method!!")
        
        fit = MLE_obj.fit(start_params=start_params)

        params = fit.params
        try:
            conf_ind = fit.conf_int()
        except:  # In case Covariances failed (i.e. Inverting Hessian failed...)
            conf_ind = np.array([[param, param] for param in params])  # Set CI to 0
        
        # Pickle Parameter Estimates:
        if res_folder == None:  # In case SubFolder was passed on:
            subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
            
        else: subfolder_meth = res_folder
            
        path = self.data_folder + subfolder_meth + "result" + str(data_set_nr).zfill(2) + ".p"
        
        directory = os.path.dirname(path)  # Extract Directory
        if not os.path.exists(directory):  # Creates Folder if not already existing
            os.makedirs(directory)
            
        pickle.dump((params, conf_ind), open(path, "wb"))  # Pickle the Info
        
        print("Successfully saved to: ")
        print(path)
        
        
    def fit_barrier(self, position_barrier, start_params, data_set_nr=0, method=2,
                    res_folder=None, position_list=[], genotype_mat=[], nr_inds=[], random_ind_nr=None,
                    deme_x_nr=0, deme_y_nr=0, min_dist=0, max_dist=0,
                    fit_params=[0, 1, 2, 4], fixed_params=[50, 0.005, 0.5, 1, 0], loci=[]):
        '''Method to fit a Barrier; and save the estimates and Unc.
        position_barrier: Where to fit the Barrier
        data_set_nr: Which data_set to load; or to save to.
        genotype_mat, position_list: Use this as input data if given.
        res_folder: Where to save to
        nr_inds: How many individuals at least per deme
        min_dist: Minimum Distance
        max_dist: Maximum Distance
        fixed_params: Base for fitting
        fit_params: Which Parameters to fit; i.e. overwrite from fixed params
        start_params: Where to start from
        Loci: Vector. Describes which Loci to actually fit.'''
        # Load Data Sets in case not given:
        if len(position_list) == 0 or len(genotype_mat) == 0:  
            position_list, genotype_mat = self.load_data_set(data_set_nr)  # Loads the Data 
          
        # Group Individuals in case Deme_Nr is given:  
        if deme_x_nr > 0 and deme_y_nr > 0:
            position_list, genotype_mat, nr_inds = group_inds(position_list, genotype_mat,
                                                demes_x=deme_x_nr, demes_y=deme_y_nr, min_ind_nr=1)  
            print(nr_inds)
        
        # Pick Loci if given:
        if len(loci) > 0:
            genotype_mat = genotype_mat[:, loci]
            print("Reduced to %i loci." % len(genotype_mat[0, :]))
            
        # Pick Random_ind_nr many Individuals in case needed:
        if random_ind_nr:
            inds = range(len(position_list))
            shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
            inds = inds[:random_ind_nr]  # Only load first nr_inds
    
            position_list = position_list[inds, :]
            genotype_mat = genotype_mat[inds, :]
            
            
        # First Choose the right MLE-Object and set the right starting Parameters:
        if method == 0:
            MLE_obj = MLE_estimator("DiffusionBarrierK0", position_list, genotype_mat, multi_processing=self.multi_processing) 
            start_params = start_params + [0.004, ]
            fixed_params[-1] = 0.004
        elif method == 1:
            MLE_obj = MLE_pairwise("DiffusionBarrierK0", position_list, genotype_mat, multi_processing=self.multi_processing)
            start_params = start_params + [0.01, ]
            fixec_params[-1] = 0.01
        elif method == 2:
            MLE_obj = MLE_f_emp("DiffusionBarrierK0", position_list, genotype_mat, nr_inds=nr_inds,
                                min_dist=min_dist, max_dist=max_dist, multi_processing=self.multi_processing,
                                fit_params=fit_params, fixed_params=fixed_params)
            start_params = start_params + [0.5, ]
            fixed_params[-1] = 0.5
        elif method == 3:  # Do the fitting based on binned data
            MLE_obj = Analysis(position_list, genotype_mat) 
        else: raise ValueError("Wrong Input for Method!!")
        
        MLE_obj.kernel.position_barrier = position_barrier  # Sets the Barrier Position
        print("Assumed Barrier Position: %g" % MLE_obj.kernel.position_barrier)
        fit = MLE_obj.fit(start_params=start_params)

        params = fit.params
        try:
            conf_ind = fit.conf_int()
        except:  # In case Covariances failed (i.e. Inverting Hessian failed...)
            conf_ind = np.array([param, param] for param in params)  # Set CI to 0
            
        
        # Pickle Parameter Estimates:
        if res_folder == None:  # In case SubFolder was passed on:
            subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
            
        else: subfolder_meth = res_folder
            
        path = self.data_folder + subfolder_meth + "result" + str(data_set_nr).zfill(2) + ".p"
        
        directory = os.path.dirname(path)  # Extract Directory
        if not os.path.exists(directory):  # Creates Folder if not already existing
            os.makedirs(directory)
            
        pickle.dump((params, conf_ind), open(path, "wb"))  # Pickle the Info
        
        print("Successfully saved to: ")
        print(path)

    def save_mat(self, file, filename="add_info.csv", method=2, res_folder=None):
        '''Method to save additional Information via np.savetxt'''
        if res_folder == None:  # In case SubFolder was passed:
            subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
        else: subfolder_meth = res_folder
        
        path = self.data_folder + subfolder_meth + filename
        
        directory = os.path.dirname(path)  # Extract Directory
        if not os.path.exists(directory):  # Creates Folder if not already existing!
            os.makedirs(directory)
            
        np.savetxt(path, file, delimiter="$")  # Save the additional Info
        
        
###############################################################################################################################

class MultiNbh(MultiRun):
    '''The Full Goal is to find out at which Neighborhood Size the Method to estimate IBD works best.
    Can run different Methods and can compare them
    Everything set so that 100 Data-Sets are run. With 4x25 Parameters'''
    def __init__(self, folder, nr_data_sets=100, nr_params=4, **kwds):
        super(MultiNbh, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
        self.name = "nbh_file"
        # self.data_folder = folder
        
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method.'''
        print("Creating Dataset: %i" % data_set_nr)
        # First set all the Parameter Values:
        ips_list = 25 * [2.0] + 25 * [10.0] + 25 * [18.0] + 25 * [26.0]
        ips = ips_list[data_set_nr]  # Number of haploid Individuals per Node (For D_e divide by 2)  Loads the right Neighborhood Size
        
        
        position_list = np.array([(500 + i, 500 + j) for i in range(-19, 21, 2) for j in range(-49, 51, 2)])  # 1000 Individuals; spaced 2 sigma apart.
        nr_loci = 200
        t = 5000
        gridsize_x, gridsize_y = 1000, 1000
        sigma = 0.965  # 0.965 # 1.98
        mu = 0.003  # Mutation/Long Distance Migration Rate # Idea is that at mu=0.01 there is quick decay which stabilizes at around sd_p
        sd_p = 0.1
        p_delta = np.random.normal(scale=sd_p, size=nr_loci)  # Draw some random Delta p from a normal distribution
        p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
        p_mean = p_mean + p_delta
        
        # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
        # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
        
        genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
        
        for i in range(nr_loci):
            grid = Grid()  # Creates new Grid. Maybe later on use factory Method
            grid.set_parameters(gridsize_x, gridsize_y, sigma, ips, mu)
            print("Doing data set: %i, Simulation: %i " % (data_set_nr, i))
            grid.set_samples(position_list)
            grid.update_grid_t(t, p=p_mean[i])  # Uses p_mean[i] as mean allele Frequency.
            genotype_matrix[:, i] = grid.genotypes
            
        
        self.save_data_set(position_list, genotype_matrix, data_set_nr)
        
            
        # Now Pickle Some additional Information:
        p_names = ["Nr Loci", "t", "p_mean", "sigma", "mu", "ips", "sd_p", "Position List"]
        ps = [nr_loci, t, p_mean, sigma, mu, ips, sd_p, position_list]
        additional_info = ("1 Test Run for Grid object with high neighborhood size")
        self.pickle_parameters(p_names, ps, additional_info)
            
    def analyze_data_set(self, data_set_nr, random_ind_nr=4000, method=2, fit_t0=0):
        '''Create Data Set. Override Method. fit_t0: Whether to fit t0. (at the moment only for method 2!)
        method 0: GRF; method 1: Pairwise LL method 2: Individual Curve Fit. method 3: Binned Curve fit.'''
        position_list, genotype_mat = self.load_data_set(data_set_nr)  # Loads the Data 
        
        # Creates the "right" starting parameters:
        ips_list = 25 * [2.0] + 25 * [10.0] + 25 * [18.0] + 25 * [26.0]
        ips_list = np.array(ips_list)
        nbh_sizes = ips_list / 2.0 * 4 * np.pi  # 4 pi sigma**2 D = 4 * pi * 1 * ips/2.0
        
        start_params_vec = [[nbh_size, 0.006] for nbh_size in nbh_sizes]  # General Vector for Start-Lists
        if fit_t0 == 1:  # If t0 is to be fitted as well
            start_params_vec = [[nbh_size, 0.006, 1.0] for nbh_size in nbh_sizes]  # General Vector for Start-Lists
        
        # Pick the right DataSet
        start_params = start_params_vec[data_set_nr]

        self.fit_IBD(start_params, data_set_nr, method=method,
                    res_folder=None, position_list=[], genotype_mat=[], random_ind_nr=None)
        
#         if method == 0:
#             MLE_obj = MLE_estimator("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing) 
#         elif method == 1:
#             MLE_obj = MLE_pairwise("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing)
#             start_list = [[nbh_size, 0.006, 0.01] for nbh_size in nbh_sizes]  # Update Vector of Start Lists
#         elif method == 2:
#             MLE_obj = MLE_f_emp("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing, fit_t0=fit_t0)
#             start_list = [[nbh_size, 0.006, 0.5] for nbh_size in nbh_sizes]  # Update Vector of Start Lists
#             if fit_t0 == 1:
#                 start_list = [[nbh_size, 0.006, 1.0, 0.5] for nbh_size in nbh_sizes]  # Update Vector of Start Lists
#         elif method == 3:  # Do the fitting based on binned data
#             MLE_obj = Analysis(position_list, genotype_mat) 
#         else: raise ValueError("Wrong Input for Method!!")
#         
#         fit = MLE_obj.fit(start_params=start_list[data_set_nr])
# 
#         params = fit.params
#         conf_ind = fit.conf_int()
#         
#         # Pickle Parameter Estimates:
#         subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
#         path = self.data_folder + subfolder_meth + "result" + str(data_set_nr).zfill(2) + ".p"
#         
#         directory = os.path.dirname(path)  # Extract Directory
#         if not os.path.exists(directory):  # Creates Folder if not already existing
#             os.makedirs(directory)
#             
#         pickle.dump((params, conf_ind), open(path, "wb"))  # Pickle the Info

        
    

###############################################################################################################################

class MultiNbhModel(MultiNbh):
    '''Class that generates the data UNDER the Gaussian model, not from a population genetics perspective.
    Simply overwerite the data creation method of MultiNbh, the rest (data analysis/visualization) is the same.'''
    
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method of MultiNbh.'''
        print("Craeting Dataset: %i" % data_set_nr)
        # First set all the Parameter Values:
        ips_list = 25 * [2.0] + 25 * [10.0] + 25 * [18.0] + 25 * [26.0] 
        ips = ips_list[data_set_nr]  # Number of haploid Individuals per Node (For D_e divide by 2)  Loads the right Neighborhood Size for specific run.
        
        
        position_list = np.array([(500 + i, 500 + j) for i in range(-19, 21, 2) for j in range(-49, 51, 2)])  # 1000 Individuals; spaced 2 sigma apart.
        nr_loci = 200
        # t = 5000
        # gridsize_x, gridsize_y = 1000, 1000
        sigma = 0.965  # 0.965 # 1.98
        mu = 0.003  # Mutation/Long Distance Migration Rate # Idea is that at mu=0.01 there is quick decay which stabilizes at around sd_p.
        ss = 0.04  # The std of fluctuations in f-space.
        t0 = 1.0  # When do start the integration.
        p_mean = 0.5  # Sets the mean allele frequency.
        
        # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
        # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
        
        genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
        
        nbh = 4.0 * np.pi * ips / 2.0 * 1.0  # Calculate Neighborhood size
        L = 2 * mu  # Calculate the effective mutation rate
        KC = fac_kernel("DiffusionK0")  # Create the object that will calculate the Kernel.
        KC.set_parameters([nbh, L, t0, ss])  # Sets parameters: Nbh, L, t0, ss
        
        print("Calculating Kernel...")
        
        tic = time()
        kernel_mat = KC.calc_kernel_mat(position_list)  # Calculate the co-variance Matrix
        toc = time()
        print("Calculation Kernel runtime: %.6f:" % (toc - tic))
        
        grid = Grid()  # Load the grid (contains method to created correlated data.
        
        for i in range(nr_loci):  # Iterate over all loci
            print("Doing Locus: %i" % i)
            genotypes = grid.draw_corr_genotypes_kernel(kernel_mat=kernel_mat, p_mean=p_mean)
            genotype_matrix[:, i] = genotypes
            
        self.save_data_set(position_list, genotype_matrix, data_set_nr)  # Save the Data.
        
            
        # Now Pickle Some additional Information:
        p_names = ["Nr Loci", "t0", "p_mean", "sigma", "ss", "mu", "ips", "Position List"]
        ps = [nr_loci, t0, p_mean, sigma, ss, mu, ips, position_list]
        additional_info = ("Data generated under a gaussian Model")
        self.pickle_parameters(p_names, ps, additional_info)
        
        
###############################################################################################################################

class MultiBarrier(MultiRun):
    '''
    Tests 100 Runs for different Barrier strengths.
    Everything set so that 100 Data-Sets are run; with 4x25 Parameters.'''
    name = "barrier_file"
    
    def __init__(self, folder, nr_data_sets=100, nr_params=5, **kwds):
        super(MultiBarrier, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
        
        # self.data_folder = folder
        
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method.'''
        print("Creating Dataset: %i" % data_set_nr)
        # First set all the Parameter Values:
        # barrier_strength_list = 25 * [0.0] + 25 * [0.05] + 25 * [0.1] + 25 * [0.15]
        # barrier_strength = barrier_strength_list[data_set_nr]
        barrier_strength = 0.05
        
        ips = 10  # Number of haploid Individuals per Node (For D_e divide by 2)
        
        
        # position_list = np.array([(500 + i, 500 + j) for i in range(-19, 21, 2) for j in range(-49, 51, 2)])  # 1000 Individuals; spaced 2 sigma apart. Original data_set
        position_list = np.array([(500 + i, 500 + j) for i in range(-29, 31, 1) for j in range(-19, 21, 1)])  # 1000 Individuals; space 2 sigma apart.
        # position_list = np.array([(500 + i, 500 + j) for i in range(-9, 11, 1) for j in range(-9, 11, 1)])  # Updated position list.
        nr_loci = 200
        t = 5000
        gridsize_x, gridsize_y = 1000, 1000
        barrier_pos = 500.5
        sigma = 0.965  # 0.965 # 1.98
        mu = 0.003  # Mutation/Long Distance Migration Rate # Idea is that at mu=0.01 there is quick decay which stabilizes at around sd_p
        sd_p = 0.1  # Standard Deviation Allele Frequency
        p_delta = np.random.normal(scale=sd_p, size=nr_loci)  # Draw some random Delta p from a normal distribution
        p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
        p_mean = p_mean + p_delta
        
        # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
        # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
        
        genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
        
        for i in range(nr_loci):
            grid = Grid()  # Creates new Grid. Maybe later on use factory Method
            grid.set_parameters(gridsize_x, gridsize_y, sigma, ips, mu)
            print("Doing data set: %i, Simulation: %i " % (data_set_nr, i))
            grid.set_samples(position_list)
            grid.set_barrier_parameters(barrier_pos, barrier_strength)  # Where to set the Barrier and its strength
            grid.update_grid_t(t, p=p_mean[i], barrier=1)  # Uses p_mean[i] as mean allele Frequency.
            genotype_matrix[:, i] = grid.genotypes
        position_list = position_list.astype("float")  # So it works when one subtracts a float.
        # position_list_update = position_list[:, 0] - grid.barrier 
        self.save_data_set(position_list, genotype_matrix, data_set_nr)
        
            
        # Now Pickle Some additional Information:
        p_names = ["Nr Loci", "t", "p_mean", "sigma", "mu", "ips", "sd_p", "Position List"]
        ps = [nr_loci, t, p_mean, sigma, mu, ips, sd_p, position_list]
        additional_info = ("1 Test Run for Grid object with high neighborhood size")
        self.pickle_parameters(p_names, ps, additional_info)
            
    def analyze_data_set(self, data_set_nr, random_ind_nr=1000, res_folder=None,
                         position_barrier=500.5, deme_x_nr=0, deme_y_nr=0, method=0):
        '''Create Data Set. Override Method. mle_pw: Whether to use Pairwise Likelihood.
        If no positon list or genotype data is given; load it for the according Data Set.
        method 0: GRF; method 1: Pairwise LL method 2: Individual Curve Fit. method 3: Binned Curve fit.'''
        
        # Creates the "right" starting parameters:
        # barrier_strength_list = 25 * [0.01] + 25 * [0.2] + 25 * [0.5] + 25 * [1.0]
        barrier_strength_list = 100 * [0.5]
        l = 0.006
        nbh_size = 4 * np.pi * 5  # 4 pi sigma**2 D = 4 * pi * 1 * ips/2.0
        start_list = [[nbh_size, l, bs] for bs in barrier_strength_list]  # General Vector for Start-Lists
        
        # Then pick the Starting Parameters:
        start_params = start_list[data_set_nr]        
    
        self.fit_barrier(position_barrier, start_params, data_set_nr, method=method,
                         deme_x_nr=deme_x_nr, deme_y_nr=deme_y_nr)
        
        
    def barrier_ll(self, data_set_nr, nbh, L, t0, random_ind_nr=1000, position_barrier=500.5, barrier_strengths=21):
        '''Calculate LL for different strengths of the barrier in GRF Framework'''
        print("Running Dataset: %i" % data_set_nr)
        k_vec = np.linspace(0.00001, 1, barrier_strengths)  # Creates the Grid for k
        
        position_list, genotype_mat = self.load_data_set(data_set_nr)  # Loads the Dataset
        ss = calculate_ss(genotype_mat)  # Calculates ss from empirical Data
        
        MLE_obj = MLE_estimator("DiffusionBarrierK0", position_list, genotype_mat, multi_processing=self.multi_processing) 
        MLE_obj.kernel.position_barrier = position_barrier  # Sets the Barrier Position
        
        params = [[nbh, L, k, t0, ss] for k in k_vec]  # Prepares the parameter vector
        print(params[0])
        ll_vec = np.array([MLE_obj.loglike(p) for p in params])  # Calculates Vector of marginal Likelihoods
        
        
        # Pickle Parameter Estimates:
        subfolder_meth = "method_k" + "/"  # Sets subfolder on which Method to use.
        path = self.data_folder + subfolder_meth + "result" + str(data_set_nr).zfill(2) + ".p"
        
        directory = os.path.dirname(path)  # Extract Directory
        if not os.path.exists(directory):  # Creates Folder if not already existing
            os.makedirs(directory)
            
        pickle.dump(ll_vec, open(path, "wb"))  # Pickle the Info

        
##############################################################################################################################
class MultiCluster(MultiBarrier):
    '''Tests 100 runs for different degrees of clustering
    4x25; 25 Datasets are the same.'''
    
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method.'''
        print("Creating Dataset: %i" % data_set_nr)
        # First set all the Parameter Values:
        barrier_strength_list = 25 * [0.1]
        barrier_strength = barrier_strength_list[data_set_nr]
        
        if data_set_nr > 25:
            raise ValueError("Only 25 Datasets available")
        
        ips = 10  # Number of haploid Individuals per Node (For D_e divide by 2)
        
        
        position_list = np.array([(500 + i, 500 + j) for i in range(-11, 12, 1) for j in range(-17, 18, 1)])  # Spaced 1 sigma apart
        # 24*36
        nr_loci = 200
        t = 5000
        gridsize_x, gridsize_y = 1000, 1000
        barrier_pos = 500.5
        sigma = 0.965  # 0.965 # 1.98
        mu = 0.003  # Mutation/Long Distance Migration Rate # Idea is that at mu=0.01 there is quick decay which stabilizes at around sd_p
        sd_p = 0.1
        p_delta = np.random.normal(scale=sd_p, size=nr_loci)  # Draw some random Delta p from a normal distribution
        p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
        p_mean = p_mean + p_delta
        
        # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
        # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
        
        genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
        
        for i in range(nr_loci):
            grid = Grid()  # Creates new Grid. Maybe later on use factory Method
            grid.set_parameters(gridsize_x, gridsize_y, sigma, ips, mu)
            print("Doing data set: %i, Simulation: %i " % (data_set_nr, i))
            grid.set_samples(position_list)
            grid.set_barrier_parameters(barrier_pos, barrier_strength)  # Where to set the Barrier and its strength
            grid.update_grid_t(t, p=p_mean[i], barrier=1)  # Uses p_mean[i] as mean allele Frequency.
            genotype_matrix[:, i] = grid.genotypes
        position_list = position_list.astype("float")  # So it works when one subtracts a float.
        # position_list_update = position_list[:, 0] - grid.barrier 
        self.save_data_set(position_list, genotype_matrix, data_set_nr)
        
            
        # Now Pickle Some additional Information:
        p_names = ["Nr Loci", "t", "p_mean", "sigma", "mu", "ips", "sd_p", "Position List"]
        ps = [nr_loci, t, p_mean, sigma, mu, ips, sd_p, position_list]
        additional_info = ("1 Test Run for Grid object with high neighborhood size")
        self.pickle_parameters(p_names, ps, additional_info)
            
    def analyze_data_set(self, data_set_nr, random_ind_nr=None, position_barrier=500.5, method=2):
        '''Create Data Set. Override Method. mle_pw: Whether to use Pairwise Likelihood
        method 0: GRF; method 1: Pairwise LL method 2: Individual Curve Fit. method 3: Binned Curve fit.'''
        
        if data_set_nr >= 100:
            raise ValueError("DataSet does not exist.")
            
        data_set_eff = data_set_nr % 25  # Which data-set to use
        batch_nr = np.floor(data_set_nr / 25)  # Which batch to use
        assert(batch_nr * 25 + data_set_eff == data_set_nr)  # Make sure everything works.
        
        position_list, genotype_mat = self.load_data_set(data_set_eff)  # Loads the Data     
        
        # Then group the Individuals:
        if batch_nr == 0:
            position_list, genotype_mat = position_list, genotype_mat  # No grouping whatsoever: 24x36
            
        elif batch_nr == 1:
            position_list, genotype_mat, _ = group_inds(position_list, genotype_mat, demes_x=12, demes_y=18)  # 2x2 clustering
            
        elif batch_nr == 2:
            position_list, genotype_mat, _ = group_inds(position_list, genotype_mat, demes_x=8, demes_y=12)  # 3x3 clustering
            
        elif batch_nr == 3:
            position_list, genotype_mat, _ = group_inds(position_list, genotype_mat, demes_x=6, demes_y=9)  # 4x4 clustering
        
        
        # Creates the "right" starting parameters:
        # barrier_strength_list = 25 * [0.01] + 25 * [0.2] + 25 * [0.5] + 25 * [1.0]
        barrier_strength_list = 100 * [0.1]
        l = 0.006

        nbh_size = 4 * np.pi * 5  # 4 pi sigma**2 D = 4 * pi * 1 * ips/2.0
        
        start_list = [[nbh_size, l, bs] for bs in barrier_strength_list]  # General Vector for Start-Lists
        start_params = start_list[data_set_nr]
        
        self.fit_barrier(position_barrier, start_params, data_set_nr, method=method,
                    position_list=position_list, genotype_mat=genotype_mat, random_ind_nr=random_ind_nr)
        
        

##############################################################################################################################

class MultiBootsTrap(MultiBarrier):
    '''
    Class that bootstrap over Datasets.
    Generates Bootstraps when creating data-set
    
    When initializing it; one can give it the path to the 
    data set over which one has to bootstrap.
    
    Inherits from MultiBarrier. Uses its Analyze Data-Set Method.
    '''
    
    # Paths to the File over which to bootstrap
    position_path = "./cluster_folder/barrier_file_coords00.csv"  # Path to the data set over which one has to bootstrap
    gtp_path = "./cluster_folder/barrier_file_genotypes00.csv"
    
    def __init__(self, folder, nr_data_sets=100, nr_params=4, **kwds):
        '''Initializes the BootsTrap class. Path_data_set is the path to the 
        data set over which to bootstrap.'''
        super(MultiBootsTrap, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
        self.name = "bt_file"
    
    def create_data_set(self, data_set_nr):
        '''In this case one actually does 
        the bootstrapping'''
        
        print("Creating Data Set: %i" % data_set_nr)
        
        # Load the data.
        position_list = np.loadtxt(self.position_path, delimiter='$').astype('float64')
        genotype_matrix = np.loadtxt(self.gtp_path, delimiter='$').astype('float64')
        
        position_list, genotype_matrix, _ = group_inds(position_list, genotype_mat, demes_x=8, demes_y=12)
        # position_list, genotype_matrix, _ = group_inds(position_list, genotype_matrix, demes_x=8, demes_y=12)  # 3x3 clustering
    
        
        nr_inds, nr_genotypes = np.shape(genotype_matrix)  # Could in principle also bootstrap over Individuals
        
        
        r_ind = np.random.randint(nr_genotypes, size=nr_genotypes)  # Get Indices of random resampling
        gtps_sample = genotype_matrix[:, r_ind]  # Do the actual Bootstrap; pick the columns
        
        
        self.save_data_set(position_list, gtps_sample, data_set_nr)  # Save the Data Set
        
        # self.pickle_parameters(p_names, ps, additional_info)      Dont't pickle additional Info; as it is not clear what it was
        
##############################################################################################################################
class MultiBT_HZ(MultiBarrier):
    '''
    Class that bootstrap over Datasets.
    Generates Bootstraps when creating data-set
    
    The first data_set is the original Data-Set;
    the rest are 99 Bootstraps.
    
    Inherits from MultiBarrier. Uses its Analyze Data-Set Method.
    '''
    
    # Paths to the File over which to bootstrap
    position_path = "./Data/coordinatesHZall1.csv"  # Path to the data set over which one has to bootstrap
    gtp_path = "./Data/genotypesHZall1.csv"
    
    def __init__(self, folder, nr_data_sets=100, nr_params=4, **kwds):
        '''Initializes the BootsTrap class. Path_data_set is the path to the 
        data set over which to bootstrap.'''
        super(MultiBT_HZ, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
        self.name = "hz_file"
    
    def create_data_set(self, data_set_nr):
        '''In this case one actually does 
        the bootstrapping'''
        
        if not 0 <= data_set_nr < self.nr_data_sets:  # Check whether everything alright
            raise ValueError("Data Set out of Range!")
        
        
        print("Creating Data Set: %i" % data_set_nr)
        
        # Load the data.
        position_list = np.loadtxt(self.position_path, delimiter='$').astype('float64')
        genotype_matrix = np.loadtxt(self.gtp_path, delimiter='$').astype('float64')
        
        # The first data set is the original one
        if data_set_nr == 0: 
            self.save_data_set(position_list, genotype_matrix, data_set_nr)
            return 
        
        # Do the Bootstrap
        gtps_sample = bootstrap_genotypes(genotype_mat)
        
        self.save_data_set(position_list, gtps_sample, data_set_nr)  # Save the Data Set
        

##############################################################################################################################
class MultiSecondaryContact(MultiBarrier):
    '''
    Tests 100 Runs for different Barrier strengths.
    Everything set so that 100 Data-Sets are run; with 4x25 Parameters.
    '''
    def create_data_set(self, data_set_nr, barrier_strength=None):
        '''Create a Data_Set for secondary contact.
        Use Secondary Contact Subclass of Grid for Simulations'''
        
        # Set overall Parameters:  
        std_pl, std_pr = 0.1, 0.1  # Standard Deviation of Allele Frequency to the left and to the right.
        t_contact = 100  # Time until secondary Contact.
        ips = 10  # Number of haploid Individuals per Node (For D_e divide by 2)  Loads the right Neighborhood Size
        position_list = np.array([(500 + i, 500 + j) for i in range(-19, 21, 2) for j in range(-49, 51, 2)])  # 1000 Individuals; spaced 2 sigma apart.
        nr_loci = 200
        gridsize_x, gridsize_y = 1000, 1000
        sigma = 0.965  # 0.965 # 1.98
        mu = 0.0001  # Mutation/Long Distance Migration Rate # Idea is that at mu=0.01 there is quick decay which stabilizes at around sd_p
        sd_p = 0.1
        barrier = 500.5  # Where to set the Barrier
                
        # Draw allele frequencies to the left and to the right:
        p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
        p_delta_l = np.random.normal(scale=std_pl, size=nr_loci)  # Draw some random Delta p from a normal distribution
        p_delta_r = np.random.normal(scale=std_pr, size=nr_loci)  # Draw some random Delta p from a normal distribution
        
        p_mean_l = p_mean + p_delta_l
        p_mean_r = p_mean + p_delta_r
        
        # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
        # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
        
        genotype_matrix = -np.ones((len(position_list), nr_loci))  # Set Genotype Matrix to -1 to detect Errors.
        
        for i in range(nr_loci):
            grid = Secondary_Grid()
            grid.set_parameters(gridsize_x, gridsize_y, sigma, ips, mu)
            grid.barrier = barrier
            print("Doing data set: %i, Simulation: %i " % (data_set_nr, i))
            grid.set_samples(position_list)
            if barrier_strength:  # If Barrier:
                grid.barrier_strength = barrier_strength  # Set the strength of the Barrier.
                grid.update_grid_t(t_contact, barrier=1, p1=p_mean_l[i], p2=p_mean_r[i])
            
            else:  # Produce Grid in the normal way: 
                grid.update_grid_t(t_contact, barrier=0, p1=p_mean_l[i], p2=p_mean_r[i])  # Updates for t Generations with 
            
            genotype_matrix[:, i] = grid.genotypes
            
        self.save_data_set(position_list, genotype_matrix, data_set_nr)
        
            
        # Now Pickle Some additional Information:
        p_names = ["Nr Loci", "p_mean_l", "p_mean_r", "sigma", "mu", "ips", "Barrier", "Position List", "t_contact"]
        ps = [nr_loci, p_mean_l, p_mean_r, sigma, mu, ips, barrier, position_list, t_contact]
        additional_info = ("Secondary Contact Simulations; with different allele frequencies left and right")
        self.pickle_parameters(p_names, ps, additional_info)
        
    def analyze_data_set_cleaning(self, data_set_nr, method=2, res_folder=None):
        '''Analyze Data Set data_set_nr. Analyses the Data-Set with various levels of "cleaning"
        I.e. it tries to remove loci with differences in allele frequencies. Then passes Data-Sets to 
        Standard-Analysis Method'''
        if data_set_nr >= 100 or data_set_nr < 0:
            raise ValueError("DataSet does not exist.")
        max_r2_vec = np.array([1.0, 0.02, 0.01, 0.005])
        
        # Load the right Data-Set Number (Modulo 25!!)
        data_set_eff = data_set_nr % 25  # Which data-set to use
        batch_nr = int(np.floor(data_set_nr / 25))  # Which batch to use
        assert(batch_nr * 25 + data_set_eff == data_set_nr)  # Make sure everything works.
        
        print(batch_nr)
        max_r2 = max_r2_vec[batch_nr]  # Load the right Cut-Off
        
        position_list, genotype_mat = self.load_data_set(data_set_eff)  # Loads the Data 
        
        # Extract Indices on which side of the Barrier to look at
        # barrier = 500.5
        # inds_l = np.where(position_list[:, 0] <= barrier)[0]
        # inds_r = np.where(position_list[:, 0] > barrier)[0]
        
        # genotypes_left = genotype_mat[inds_l,:]
        # genotypes_right = genotype_mat[inds_r,:]
        
        # assert(len(genotypes_left) + len(genotypes_right) == len(genotype_mat))   # Assert that splitting worked.
        
        # Calculate mean Allele Frequencies:
        # p_l = np.mean(genotypes_left, axis=0)
        # p_r = np.mean(genotypes_right, axis=0)
        
        # print("Standard Deviation left: %.2f" % np.std(p_l))
        # print("Standard Deviation right: %.2f" % np.std(p_r))
        # print("Standard Deviation between: %.2f" % np.std(p_r-p_l))
        
        # Plot to compare allel-frequs
        
        # p_diff = 0.05
        # close_inds=np.where(np.abs(p_l-p_r)< p_diff)[0]
        
        # ## Do the Correction via geographic R**2:
        x_coords = position_list[:, 0] 
        y_coords = position_list[:, 1]
        nr_gtps = np.shape(genotype_mat)[1]
        assert(nr_gtps == 200)  # Quick Hack to check whether everything okay. Remove if not!!
        
        x_corr = np.array([np.corrcoef(x_coords, genotype_mat[:, i])[0, 1] ** 2 for i in xrange(nr_gtps)])
        y_corr = np.array([np.corrcoef(y_coords, genotype_mat[:, i])[0, 1] ** 2 for i in xrange(nr_gtps)])
        tot_corr = np.maximum(x_corr, y_corr)
        
        x = range(nr_gtps)
        
        print("Nr. of Loci before cleaning: %i" % nr_gtps)
        good_lc_inds = np.where(tot_corr < max_r2)[0]
        print("Nr. of Loci with difference < %.4f: %i" % (max_r2, len(good_lc_inds)))
        gen_mat_clean = genotype_mat[:, good_lc_inds]
        
        # Set the Position of the Barrier:
        position_barrier = 500.5  # Where to set the Barrier
        
        # Set the starting Parameters:
        bs = 0.1  # Barrier Strength
        l = 0.006
        nbh_size = 4 * np.pi * 5
        start_params = [nbh_size, l, bs]  # General Vector for Start-Lists
        
        # Call the original Method of MultiBarrier
        self.fit_barrier(position_barrier, start_params, data_set_nr, method=method,
                    res_folder=res_folder, position_list=position_list, genotype_mat=gen_mat_clean)
        
        # self.analyze_data_set(data_set_nr, method=method,
        #                      position_list=position_list, genotype_mat=gen_mat_clean)
        
        # Also Save the number of all and extracted_loci: (Need not check the existence of Path - it was already created
        subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
        path = self.data_folder + subfolder_meth + "nr_good_loci" + str(data_set_nr).zfill(2) + ".csv"
        
        data = np.array([nr_gtps, len(good_lc_inds)])  # Saves the number of all Genotypes; and the Number of filtered Loci
        np.savetxt(path, data, delimiter="$")  # Save the coordinates
        print("Nr. of filtered Loci successfully saved!!")
        
class MultiBarrierPosition(MultiRun):
    '''Analyzes Multiple Barrier-Positions throughout the Population.
    For every of them; do Bootstraps'''
    name = "mb_pos"
    
    def __init__(self, folder, nr_data_sets=200, nr_params=5, **kwds):
        super(MultiBarrierPosition, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
    
    def analyze_data_set(self, data_set_nr, method=2, nr_x_bins=50, nr_y_bins=10, nr_bts=20,
                         res_folder=None, min_ind_nr=5, barrier_pos=[], use_ind_nr=1,
                         min_dist=0, max_dist=0):
        '''Analyzes the data-set. First bins the Data; then do nr_bts many estimates.
        For synthetic data set: nr_x_bins=30; nr_y_bins=20. For HZ: nr_x_bins=50; nr_y_bins=10'''
        
        # First load and bin the data:
        position_list, genotype_mat = self.load_data_set(0)  # Load the Data
        nr_inds, nr_genotypes = np.shape(genotype_mat)
        print("Nr of total Inds: %i" % nr_inds)
        print("Nr of total Genotypes: %i" % nr_genotypes)
        
        # Group Inds:
        position_list, genotype_mat, nr_inds = group_inds(position_list, genotype_mat,
                                                    demes_x=nr_x_bins, demes_y=nr_y_bins, min_ind_nr=min_ind_nr)  
        # If not wished to use individual number:
        if use_ind_nr == 0:
            nr_inds = np.ones(len(position_list))
            
        print("Demes: ")
        print(nr_inds)
        
        nr_individuals, nr_genotypes = np.shape(genotype_mat)
        print("Nr of analyzed Inds: %i" % nr_individuals)
        print("Nr of analyzed Genotypes: %i" % nr_genotypes)
        
        
        if len(barrier_pos) == 0:
            x_coords = np.unique(position_list[:, 0])  # Get the unique, sorted x-Coordinates
            x_barriers = (x_coords[1:] + x_coords[:-1]) / 2.0  # Calculate the barrier positions
            x_barriers = x_barriers[0::2]  # Only take every second Barrier Step. 1 Start for uneven
            print(x_barriers)
        
        else: x_barriers = barrier_pos
        
        effective_data_set_nr = data_set_nr / nr_bts  # Get the effetive number; i.e. what position of the Barreier
        assert(0 <= effective_data_set_nr <= len(x_barriers))  # Sanity Check
        
        position_barrier = x_barriers[effective_data_set_nr]  # Load the Barrier Position
        
        # For the non-bootstrap data-sets:
        if (data_set_nr % nr_bts) != 0:  
            genotype_mat = bootstrap_genotypes(genotype_mat) 
        
        # Create start parameters:
        bs = 0.5
        l = 0.006
        nbh_size = 4 * np.pi * 5  # 4 pi sigma**2 D = 4 * pi * 1 * ips/2.0
        start_params = [nbh_size, l, bs]
        
        # Fit the Barrier:
        self.fit_barrier(position_barrier, start_params, data_set_nr, method=method,
                         res_folder=res_folder, position_list=position_list, genotype_mat=genotype_mat, nr_inds=nr_inds,
                         min_dist=min_dist, max_dist=max_dist)
        
        # Save the Barrier Positions which have been used:
        self.save_mat(x_barriers, filename="barrier_pos.csv", method=2, res_folder=res_folder)
        
    
    def analyze_data_set_k_only(self, data_set_nr, nbh=50, l=0.003, method=2, nr_x_bins=30, nr_y_bins=20, nr_bts=20,
                         res_folder=None, min_ind_nr=1, barrier_pos=[], use_ind_nr=0,
                         min_dist=0, max_dist=0, loci=[]):
        '''Analyzes the data-set. Only does the estimation of k and ss. First bins the Data; then do nr_bts many estimates.
        For synthetic data set: nr_x_bins=30; nr_y_bins=20. For HZ: nr_x_bins=50; nr_y_bins=10
        Loci: Which Loci to use for the Analysis'''
        
        # First load and bin the data:
        position_list, genotype_mat = self.load_data_set(0)  # Load the Data
        nr_inds, nr_genotypes = np.shape(genotype_mat)
        print("Nr of total Inds: %i" % nr_inds)
        print("Nr of total Genotypes: %i" % nr_genotypes)
        
        # Group Inds:
        position_list, genotype_mat, nr_inds = group_inds(position_list, genotype_mat,
                                                    demes_x=nr_x_bins, demes_y=nr_y_bins, min_ind_nr=min_ind_nr)  
        # If not wished to use individual number:
        if use_ind_nr == 0:
            nr_inds = np.ones(len(position_list))
            
        print("First 10 Demes: ")
        print(nr_inds[:10])
        
        nr_individuals, nr_genotypes = np.shape(genotype_mat)
        print("Nr of analyzed Inds: %i" % nr_individuals)
        print("Nr of analyzed Genotypes: %i" % nr_genotypes)
        
        
        if len(barrier_pos) == 0:
            x_coords = np.unique(position_list[:, 0])  # Get the unique, sorted x-Coordinates
            x_barriers = (x_coords[1:] + x_coords[:-1]) / 2.0  # Calculate the barrier positions
            x_barriers = x_barriers[0::2]  # Only take every second Barrier Step. 1 Start for uneven
            print(x_barriers)
        
        else: x_barriers = barrier_pos
        
        effective_data_set_nr = data_set_nr / nr_bts  # Get the effetive number; i.e. what position of the Barreier
        assert(0 <= effective_data_set_nr <= len(x_barriers))  # Sanity Check
        
        position_barrier = x_barriers[effective_data_set_nr]  # Load the Barrier Position
        
        # For the non-bootstrap data-sets:
        if (data_set_nr % nr_bts) != 0:  
            genotype_mat = bootstrap_genotypes(genotype_mat) 
            print("Bootstrapping...")
        
        # Create start parameters; fixed and fit Params:
        bs = 0.5
        ss = 0
        
        fixed_params = [nbh, l, bs, 1, 0]
        fit_params = [2, 4]  # Fit k and ss
        start_params = [bs, ]  # Ss will get added above
        
        # Fit the Barrier:
        self.fit_barrier(position_barrier, start_params, data_set_nr, method=method,
                         res_folder=res_folder, position_list=position_list, genotype_mat=genotype_mat, nr_inds=nr_inds,
                         min_dist=min_dist, max_dist=max_dist, fit_params=fit_params, fixed_params=fixed_params, loci=loci)
        
        # Save the Barrier Positions which have been used:
        self.save_mat(x_barriers, filename="barrier_pos.csv", method=2, res_folder=res_folder)
        # Save the Parameters for nbh and l which have been used:
        params_used = np.array([nbh, l])
        self.save_mat(params_used, filename="fix_params.csv", method=2, res_folder=res_folder)
        
class MultiHZPosition(MultiBarrierPosition):
    '''Analyzes Multiple Barrier-Positions throughout the HZ. 
    Inherits from the MultiBarrierPosition Class to do the analysis;
    and it has some some create part that picks the data based on the full HZ-data
    with custom cutoffs'''
    
    name = "mb_posHZ"
    
    def __init__(self, folder, nr_data_sets=1000, nr_params=5, **kwds):
        super(MultiHZPosition, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
        
    def create_data_set(self, data_set_nr=0, position_path="", genotype_path="", loci_path="",
            scale_factor=50, chromosome=0, geo_r2=0.015, p_HW=0.00001,
            ld_r2=0.03, min_p=0.15):
        '''Creates the HZ Data-Set. Takes raw Data and pre-processes it:
        Filtering out number of Loci; based on chromosome
        Position_Path: Where to find the Positions of one Individual
        Genotype_Path: Where to find the Genotypes of one Individual
        loci_path: Where to find the loci information'''
        # Load the row data
        # Raw Data from Hybrid Zone
        position_list = np.loadtxt(position_path, delimiter='$').astype('float64')
        position_list = position_list / scale_factor
        print("Position List successfully loaded and scaled. ")
        genotypes = np.loadtxt(genotype_path, delimiter='$').astype('float64')
        print("Genotype position sucessfully loaded. ")
        assert(len(genotypes) == len(position_list))  # Sanity Check
        print("Nr. of Individuals: %i" % np.shape(genotypes)[0])
        print("Nr. of Genotypes: %i" % np.shape(genotypes)[1])
        
        # Raw Data from Loci:
        loci_info = pd.read_csv(loci_path)
        assert(len(loci_info) == np.shape(genotypes)[1])  # Make sure that Data match!
        
        genotypes = clean_hz_data(genotypes, loci_info, geo_r2=geo_r2,
                                  p_HW=p_HW, ld_r2=ld_r2, min_p=min_p, chromosome=chromosome, plot=True)
        
        
        # Randomly flip some Genotypes:
        genotypes = flip_gtps(genotypes)
        print("Randomly flipping Genotypes: Done")
        
        print("Nr. of cleaned Genotypes: %i " % np.shape(genotypes)[1])
        
        self.save_data_set(position_list, genotypes, data_set_nr)
        

###############################################################################################################################

class MultiIndNr(MultiNbhModel):
    '''Generate Data-Sets under the Model. A big one at 0 with 4000
    individuals; and then 99 smaller ones with decreasing number 
    of individuals (randomly subchosen)
    Inherits from MultiNbhModel to analyze the Data'''
    
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method of MultiNbh.'''
        print("Creating Dataset: %i" % data_set_nr)
        
        # Create Vector of Numbers of Individuals:
        ind_nr_vec = range(436, 4001, 36)  # Length 100: from 404 to 4000

        if data_set_nr == 0:  # Create the first data set
            # Set all the Parameter Values:
            # 4000 Individuals; each spaced 1 Sigma apart
            ips = 10  # Number of haploid Individuals per Node (For D_e divide by 2
            position_list = np.array([(500 + i, 500 + j) for i in range(-19, 21, 1) for j in range(-49, 51, 1)])  
            nr_loci = 200  # 1 for testing reasons.
            # t = 5000
            # gridsize_x, gridsize_y = 1000, 1000
            sigma = 0.965  # 0.965 # 1.98
            mu = 0.003  # Mutation/Long Distance Migration Rate # Idea is that at mu=0.01 there is quick decay which stabilizes at around sd_p.
            ss = 0.04  # The std of fluctuations in f-space.
            t0 = 1.0  # When do start the integration.
            p_mean = 0.5  # Sets the mean allele frequency.
            
            # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
            # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
            
            genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
            
            nbh = 4.0 * np.pi * ips / 2.0 * 1.0  # Calculate Neighborhood size
            L = 2 * mu / 1.0  # Calculate the effective mutation rate
            KC = fac_kernel("DiffusionK0")  # Create the object that will calculate the Kernel.
            KC.set_parameters([nbh, L, t0, ss])  # Sets parameters: Nbh, L, t0, ss
            
            print("Calculating Kernel...")
            
            tic = time()
            kernel_mat = KC.calc_kernel_mat(position_list)  # Calculate the co-variance Matrix
            toc = time()
            print("Calculation Kernel runtime: %.6f:" % (toc - tic))
            
            grid = Grid()  # Load the grid (contains method to created correlated data.
            
            for i in range(nr_loci):  # Iterate over all loci
                print("Doing Locus: %i" % i)
                genotypes = grid.draw_corr_genotypes_kernel(kernel_mat=kernel_mat, p_mean=p_mean)
                genotype_matrix[:, i] = genotypes
                
            self.save_data_set(position_list, genotype_matrix, 0)  # Save the Data.
            
                
            # Now Pickle Some additional Information:
            p_names = ["Nr Loci", "t0", "p_mean", "sigma", "ss", "mu", "ips", "Position List"]
            ps = [nr_loci, t0, p_mean, sigma, ss, mu, ips, position_list]
            additional_info = ("Data generated under a gaussian Model")
            self.pickle_parameters(p_names, ps, additional_info)
            
        elif 0 < data_set_nr < 100:  # In case of valid Data-Set Nr.
            ind_nr = ind_nr_vec[data_set_nr]  # Number of Individuals to Load
            position_list, genotype_mat = self.load_data_set(0)  # Loads the Data
            
            # Do the random Choice of individuals
            inds = range(len(position_list))
            shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
            inds = inds[:ind_nr]  # Only load first nr_inds
            self.save_data_set(position_list[inds, :], genotype_mat[inds, :], data_set_nr)  # Save the Data.
            
            
        else: 
            raise ValueError("Invalid Data-Set Nr.!")
        
###############################################################################################################################

class MultiLociNr(MultiNbh):
    '''Generate Data-Sets under the PopgenModel:
    From 50-350 Loci in steps of three
    Inherits from MultiNbh to analyze the Data'''
    
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method of MultiNbh.'''
        print("Creating Dataset: %i" % data_set_nr)
        
        # Create Vector of Numbers of Individuals:
        nr_loci_vec = range(50, 350, 3)  # From 50 to 350 in steps of 3.
        assert(len(nr_loci_vec) == 100)  # Check whether the right length

        if 0 <= data_set_nr < 100:  # In case of valid Data-Set Nr.
            '''Create a Data_Set. Override method of MultiNbh.'''
            print("Creating Dataset: %i" % data_set_nr)
            # First set all the Parameter Values:
            ips = 10
            
            position_list = np.array([(500 + i, 500 + j) for i in range(-19, 21, 2) for j in range(-49, 51, 2)])  # 1000 Individuals; spaced 2 sigma apart.
            nr_loci = nr_loci_vec[data_set_nr]
            t = 5000
            gridsize_x, gridsize_y = 1000, 1000
            sigma = 0.965  # 0.965 # 1.98
            mu = 0.003  # Mutation/Long Distance Migration Rate # Idea is that at mu=0.01 there is quick decay which stabilizes at around sd_p
            sd_p = 0.1
            p_delta = np.random.normal(scale=sd_p, size=nr_loci)  # Draw some random Delta p from a normal distribution
            p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
            p_mean = p_mean + p_delta
            
            # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
            # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
            
            genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
            
            for i in range(nr_loci):
                grid = Grid()  # Creates new Grid. Maybe later on use factory Method
                grid.set_parameters(gridsize_x, gridsize_y, sigma, ips, mu)
                print("Doing data set: %i, Simulation: %i " % (data_set_nr, i))
                grid.set_samples(position_list)
                grid.update_grid_t(t, p=p_mean[i])  # Uses p_mean[i] as mean allele Frequency.
                genotype_matrix[:, i] = grid.genotypes
                
            self.save_data_set(position_list, genotype_matrix, data_set_nr)
                
            # Now Pickle Some additional Information:
            p_names = ["Nr Loci", "t0", "p_mean", "sigma", "ss", "mu", "ips", "Position List", "Loci Nr Vec"]
            ps = [nr_loci, t0, p_mean, sigma, ss, mu, ips, position_list, nr_loci_vec]
            additional_info = ("Data generated under a Gaussian Model")
            self.pickle_parameters(p_names, ps, additional_info)  
            
        else: 
            raise ValueError("Invalid Data-Set Nr.!")
        
               
###############################################################################################################################
        

def fac_method(method, folder, multi_processing=0):
    '''Factory Method to give the right Class which creates and analyzes the data-set'''
    if method == "testrun":
        return TestRun(folder, multi_processing=multi_processing)
    
    elif method == "multi_nbh":
        return MultiNbh(folder, multi_processing=multi_processing)
    
    elif method == "multi_barrier":
        return MultiBarrier(folder, multi_processing=multi_processing)
    
    elif method == "multi_nbh_gaussian":
        return MultiNbhModel(folder, multi_processing=multi_processing)
    
    elif method == "multi_cluster":
        return MultiCluster(folder, multi_processing=multi_processing)
    
    elif method == "multi_bts":
        return MultiBootsTrap(folder, multi_processing=multi_processing)  # IMPORTANT: Set the path to the bootstrap up there.
    
    elif method == "multi_HZ":
        return MultiBT_HZ(folder, multi_processing=multi_processing)  # IMPORTANT: Set the path to the bootstrap up there.
    
    elif method == "multi_inds":
        return MultiIndNr(folder, multi_processing=multi_processing)
    
    elif method == "multi_loci":
        return MultiLociNr(folder, multi_processing=multi_processing)
        
    elif method == "multi_2nd_cont":
        return MultiSecondaryContact(folder, multi_processing=multi_processing)
    
    elif method == "multi_barrier_pos":
        return MultiBarrierPosition(folder, multi_processing=multi_processing)
    
    elif method == "multi_hz_pos":
        return MultiHZPosition(folder, multi_processing=multi_processing)
    
    
    else: raise ValueError("Wrong method entered!")

#########################################################################################
#########################################################################################       
#######################
# Some methods to test & and run this class:
######################### Some lines to test the code and make plots

def run_mult_nbh(folder):
    '''Method that can be run to simulate multiple Neighborhood Sizes'''
    MultiRun = fac_method("multi_nbh", folder)
    MultiRun.create_all_data_sets()  # Create all Datasets
    print("Creation of all Data Sets finished...")
    
    
def an_mult_nbh(folder):
    '''Analyze multiple Neighborhood Sizes'''
    MultiRun = fac_method("multi_nbh", folder)
    MultiRun.analyze_all_data_sets()
    MultiRun.save_analysis()
    print("Analysis finished and saved...")
    
def vis_mult_nbh(folder, method):
    '''Visualize the analysis of Multiple Neighborhood Sizes.'''
    MultiRun = fac_method("multi_nbh_gaussian", folder)
    MultiRun.temp_visualize(method)
    MultiRun.visualize_all_methods()
    
##########################################################################################
# Run all data-sets
    
if __name__ == "__main__":
    # Test = fac_method("multi_nbh", "./testfolder/")
    # Test.create_all_data_sets()  # Create all Datasets and saves them
    # print("Finished!")
    # Test.analyze_all_data_sets()  # Analyze all Datasets
    # Test.save_analysis()  # Saves the Analysis
    # Test.load_analysis()
    # print(Test.param_estimates)
    # print(Test.uncert_estimates)
    # MultiRun.create_data_set(78)
    # MultiRun.analyze_data_set(78, method=2)
    
    ####Method to Run Multiple Neighborhood Sizes:
    # run_mult_nbh("./nbh_folder/")
    
    ####Method to Analyze Multiple Neighborhood Sizes:
    # an_mult_nbh("./nbh_folder/")
    
    ####Method to Visualize Multiple Neighborhood Sizes:
    # vis_mult_nbh("./nbh_folder_gauss/", method=2)
    
    ####################################################
    ####Create Multi Barrier Data Set
    # MultiRun = fac_method("multi_barrier", "./Data/", multi_processing=1)
    # MultiRun = fac_method("multi_nbh", "./nbh_folder/", multi_processing=1)
    # MultiRun = fac_method("multi_nbh_gaussian", "./nbh_folder_gauss/", multi_processing=1)
    # MultiRun.create_data_set(1)
    # MultiRun.analyze_data_set(1, method=2, deme_x_nr=30, deme_y_nr=20)
    
    
    # MultiRun.temp_visualize(method=2)
    # MultiRun.visualize_barrier_strengths(res_numbers=range(0, 100))
    
    ####################################################
    ####Create Multi Cluster Data Set:
    # MultiRun = fac_method("multi_cluster", "./cluster_folder/", multi_processing=1)
    # MultiRun.create_data_set(51)
    # MultiRun.analyze_data_set(99, method=2)
    
    
    ####################################################
    ####Create Bootrap Data Set
    # MultiRun = fac_method("multi_bts", "./bts_folder_test/", multi_processing=1)
    # MultiRun.analyze_data_set(0, method=2)
    # for i in xrange(100):
    # MultiRun.create_data_set(i)
        
    ####################################################
    # MultiRun = fac_method("multi_inds", "./multi_ind_nr/", multi_processing=1)
    # MultiRun.create_data_set(0)
    # MultiRun.create_data_set(25)
    # MultiRun.analyze_data_set(1, method=2)
    
    ######################################################
    # MultiRun = fac_method("multi_loci", "./multi_loci/", multi_processing=1)
    # MultiRun.create_data_set(5)
    # MultiRun.analyze_data_set(5, method=2)
    
    
    ####################################################
    # MultiRun = fac_method("multi_2nd_cont", "./multi_2nd_b/", multi_processing=1)
    # multi_barrier_pos
    # MultiRun.analyze_data_set(10, method=2)
    # MultiRun.analyze_data_set_cleaning(50, method=2, res_folder="TestYOLO/")
    
    ####################################################
    # Multi Barrier Position Data Set:
    # MultiRun = fac_method("multi_barrier_pos", "./multi_barrier_synth/", multi_processing=1)
    # MultiRun.create_data_set(0, position_path= "./Data/barrier_file_coords01.csv", 
    #                       genotype_path="./Data/barrier_file_genotypes01.csv")
    # MultiRun.analyze_data_set(1, method=2)
    # MultiRun.analyze_data_set_k_only(0, nbh=60.2095, l=0.007882, method=2, nr_x_bins=30, nr_y_bins=20, nr_bts=20,
    #                     res_folder="test_k/", min_ind_nr=1, loci=range(10))
    
    ####################################################
    # Multi Position Hybrid Zone Data Set:
    MultiRun = fac_method("multi_hz_pos", "./multi_barrier_hz/chr4/", multi_processing=1)
    MultiRun.create_data_set(0, position_path="./Data/coordinatesHZall.csv",
                        genotype_path="./Data/genotypesHZall.csv", loci_path="./Data/loci_info.csv",
                         chromosome=4)
    # MultiRun.analyze_data_set(45, method=2, res_folder="ind_info/", barrier_pos=[2.0,], use_ind_nr=0,
    #                          min_dist=1.0, max_dist=42, nr_bts=100, nr_x_bins=100, nr_y_bins=20, min_ind_nr=3,
    #                          chromosome=5)
    #####################################################
    
    
    
    

    
    
    
    
    
