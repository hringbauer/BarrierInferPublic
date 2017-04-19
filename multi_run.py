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
from grid import Grid
from mle_class import MLE_estimator
from mle_class import calculate_ss
from mle_pairwise import MLE_pairwise
from mle_pairwise import MLE_f_emp
from random import shuffle 
from analysis import Analysis
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pickle
    

class MultiRun(object):
    '''
    Class that can produce as well as analyze multiple barrier datasets.
    '''
    data_folder = ""  # Folder
    file_name = "run"  # Which name to use for the files. 
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
    
    def create_data_set(self, data_set_nr):
        '''Method to create data_set nr data_set_nr.'''
        raise NotImplementedError("Implement this!")
    
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
    
    def plot_estimates(self):
        '''Method to plot estimates and uncertainties'''
        raise NotImplementedError("Implement this!")
    
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
        print("Craeting Dataset: %i" % data_set_nr)
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
            
    def analyze_data_set(self, data_set_nr, random_ind_nr=1000, method=0, fit_t0=0):
        '''Create Data Set. Override Method. fit_t0: Whether to fit t0. (at the moment only for method 2!)
        method 0: GRF; method 1: Pairwise LL method 2: Individual Curve Fit. method 3: Binned Curve fit.'''
        position_list, genotype_mat = self.load_data_set(data_set_nr)  # Loads the Data 
        
        # Creates the "right" starting parameters:
        ips_list = 25 * [2.0] + 25 * [10.0] + 25 * [18.0] + 25 * [26.0]
        ips_list = np.array(ips_list)
        nbh_sizes = ips_list / 2.0 * 4 * np.pi  # 4 pi sigma**2 D = 4 * pi * 1 * ips/2.0
        start_list = [[nbh_size, 0.005, 0.04] for nbh_size in nbh_sizes]  # General Vector for Start-Lists
        
        if fit_t0==1:  # If t0 is to be fitted as well
            start_list = [[nbh_size, 0.005, 1.0 ,0.04] for nbh_size in nbh_sizes]  # General Vector for Start-Lists
        
        # Pick Random_ind_nr many Individuals:
        inds = range(len(position_list))
        shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
        inds = inds[:random_ind_nr]  # Only load first nr_inds

        position_list = position_list[inds, :]
        genotype_mat = genotype_mat[inds, :]
        
        if method == 0:
            MLE_obj = MLE_estimator("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing) 
        elif method == 1:
            MLE_obj = MLE_pairwise("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing)
            start_list = [[nbh_size, 0.006, 0.01] for nbh_size in nbh_sizes]  # Update Vector of Start Lists
        elif method == 2:
            MLE_obj = MLE_f_emp("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing, fit_t0=fit_t0)
            start_list = [[nbh_size, 0.006, 0.5] for nbh_size in nbh_sizes]  # Update Vector of Start Lists
            if fit_t0==1:
                start_list = [[nbh_size, 0.006, 1.0, 0.5] for nbh_size in nbh_sizes]  # Update Vector of Start Lists
        elif method == 3:  # Do the fitting based on binned data
            MLE_obj = Analysis(position_list, genotype_mat) 
        else: raise ValueError("Wrong Input for Method!!")
        
        fit = MLE_obj.fit(start_params=start_list[data_set_nr])

        params = fit.params
        conf_ind = fit.conf_int()
        
        # Pickle Parameter Estimates:
        subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
        path = self.data_folder + subfolder_meth + "result" + str(data_set_nr).zfill(2) + ".p"
        
        directory = os.path.dirname(path)  # Extract Directory
        if not os.path.exists(directory):  # Creates Folder if not already existing
            os.makedirs(directory)
            
        pickle.dump((params, conf_ind), open(path, "wb"))  # Pickle the Info

        
    def visualize_results(self):
        '''Load and visualize the Results'''
        param_estimates, uncert_estimates = self.load_analysis()  # Loads and saves Parameter Estimates and Uncertainty estimates
        
        plt.figure()
        plt.errorbar()  # Fully Implement this plotting.
        plt.show()
        
    def temp_visualize(self, method=0):
        '''Temporary Function to plot the Estimates
        that were run on cluster.'''
        
        
        # First quick function to unpickle the data:
        def load_pickle_data(i, arg_nr, method=2):
            '''Function To load pickled Data.
            Also visualizes it.'''
            data_folder = self.data_folder
            # path = data_folder + "result" + str(i).zfill(2) + ".p"  # Path to Alex Estimates
            
            # subfolder_meth = "estimate" + str(2) + "/"  # Path to binned Estimates
            # path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            # Coordinates for more :
            subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
            path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            
            res = pickle.load(open(path, "rb"))  # Loads the Data
            return res[arg_nr]
        
        res_numbers = range(0, 100)
        # res_numbers = range(2, 7) + range(8, 86)
        # res_numbers = range(0, 2) + range(10, 13) + range(30, 33) + range(60, 61)  # + range(80, 83)
        # res_numbers = range(30,31)# To analyze Dataset 30
        
        res_vec = np.array([load_pickle_data(i, 0, method) for i in res_numbers])
        unc_vec = np.array([load_pickle_data(i, 1, method) for i in res_numbers])
        
        for l in range(len(res_numbers)):
            i = res_numbers[l]
            print("\nRun: %i" % i)
            for j in range(3):
                print("Parameter: %i" % j)
                print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
                
        
        
        # plt.figure()
        f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True)
        
        ax1.hlines(4 * np.pi, 0, 25, linewidth=2, color="r")
        ax1.hlines(4 * np.pi * 5, 25, 50, linewidth=2, color="r")
        ax1.hlines(4 * np.pi * 9, 50, 75, color="r")
        ax1.hlines(4 * np.pi * 13, 75, 100, color="r")
        ax1.errorbar(res_numbers, res_vec[:, 0], yerr=res_vec[:, 0] - unc_vec[:, 0, 0], fmt="bo", label="Nbh")
        ax1.set_ylim([0, 200])
        ax1.set_ylabel("Nbh", fontsize=18)
        # ax1.legend()
        
        ax2.errorbar(res_numbers, res_vec[:, 1], yerr=res_vec[:, 1] - unc_vec[:, 1, 0], fmt="go", label="L")
        ax2.hlines(0.006, 0, 100, linewidth=2)
        ax2.set_ylabel("L", fontsize=18)
        # ax2.legend()
        
        ax3.errorbar(res_numbers, res_vec[:, 2], yerr=res_vec[:, 2] - unc_vec[:, 2, 0], fmt="ko", label="ss")
        ax3.hlines(0.52, 0, 100, linewidth=2)
        ax3.set_ylabel("SS", fontsize=18)
        # ax3.legend()
        plt.xlabel("Dataset")
        plt.show()
        
    def visualize_all_methods(self):
        '''Visualizes the estimates of all three Methods
        Method0: GRF Method1: ML, Method2: Curve Fit'''
                # First quick function to unpickle the data:
        def load_pickle_data(i, arg_nr, method=2):
            '''Function To load pickled Data.
            Also visualizes it.'''
            data_folder = self.data_folder
            # path = data_folder + "result" + str(i).zfill(2) + ".p"  # Path to Alex Estimates
            
            # subfolder_meth = "estimate" + str(2) + "/"  # Path to binned Estimates
            # path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            # Coordinates for more :
            subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
            path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            
            res = pickle.load(open(path, "rb"))  # Loads the Data
            return res[arg_nr]
        
        res_numbers = range(2, 7) + range(8, 68)
        # res_numbers = range(0,86)
        # res_numbers = range(10, 13) + range(30, 33) + range(60, 61)  # + range(80, 83)
        # res_numbers = range(30,31)# To analyze Dataset 30
        
        # Load the Data for all three methods
        res_vec0 = np.array([load_pickle_data(i, 0, 0) for i in res_numbers])
        unc_vec0 = np.array([load_pickle_data(i, 1, 0) for i in res_numbers])
        
        res_vec1 = np.array([load_pickle_data(i, 0, 1) for i in res_numbers])
        unc_vec1 = np.array([load_pickle_data(i, 1, 1) for i in res_numbers])
        
        res_vec2 = np.array([load_pickle_data(i, 0, 2) for i in res_numbers])
        unc_vec2 = np.array([load_pickle_data(i, 1, 2) for i in res_numbers])
        
        
        f, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3, 3, sharex=True)
        
        ax1.hlines(4 * np.pi, 0, 25, linewidth=2, color="r")
        ax1.hlines(4 * np.pi * 5, 25, 50, linewidth=2, color="r")
        ax1.hlines(4 * np.pi * 9, 50, 75, color="r")
        ax1.hlines(4 * np.pi * 13, 75, 100, color="r")
        ax1.errorbar(res_numbers, res_vec0[:, 0], yerr=res_vec0[:, 0] - unc_vec0[:, 0, 0], fmt="bo", label="Nbh")
        ax1.set_ylim((0, 180))
        ax1.set_ylabel("Nbh", fontsize=18)

        ax2.errorbar(res_numbers, res_vec0[:, 1], yerr=res_vec0[:, 1] - unc_vec0[:, 1, 0], fmt="go", label="L")
        ax2.hlines(0.006, 0, 100, linewidth=2)
        ax2.set_ylabel("L", fontsize=18)
        ax2.set_ylim((0, 0.02))
        
        ax3.errorbar(res_numbers, res_vec0[:, 2], yerr=res_vec0[:, 2] - unc_vec0[:, 2, 0], fmt="ko", label="ss")
        ax3.hlines(0.04, 0, 100, linewidth=2)
        ax3.set_ylabel("SS", fontsize=18)
        
        ax4.hlines(4 * np.pi, 0, 25, linewidth=2, color="r")
        ax4.hlines(4 * np.pi * 5, 25, 50, linewidth=2, color="r")
        ax4.hlines(4 * np.pi * 9, 50, 75, color="r")
        ax4.hlines(4 * np.pi * 13, 75, 100, color="r")
        ax4.errorbar(res_numbers, res_vec1[:, 0], yerr=res_vec1[:, 0] - unc_vec1[:, 0, 0], fmt="bo", label="Nbh")
        ax4.set_ylim((0, 180))
        ax4.set_yticks([])
        
        ax5.errorbar(res_numbers, res_vec1[:, 1], yerr=res_vec1[:, 1] - unc_vec1[:, 1, 0], fmt="go", label="L")
        ax5.hlines(0.006, 0, 100, linewidth=2)
        ax5.set_ylim((0, 0.02))
        ax5.set_yticks([])
        
        ax6.errorbar(res_numbers, res_vec1[:, 2], yerr=res_vec1[:, 2] - unc_vec1[:, 2, 0], fmt="ko", label="ss")
        ax6.hlines(0.01, 0, 100, linewidth=2)
        ax6.set_yticks([])
        
        ax7.hlines(4 * np.pi, 0, 25, linewidth=2, color="r")
        ax7.hlines(4 * np.pi * 5, 25, 50, linewidth=2, color="r")
        ax7.hlines(4 * np.pi * 9, 50, 75, color="r")
        ax7.hlines(4 * np.pi * 13, 75, 100, color="r")
        ax7.errorbar(res_numbers, res_vec2[:, 0], yerr=res_vec2[:, 0] - unc_vec2[:, 0, 0], fmt="bo", label="Nbh")
        ax7.set_ylim((0, 180))
        ax7.set_yticks([])
        # ax1.legend()
        
        ax8.errorbar(res_numbers, res_vec2[:, 1], yerr=res_vec2[:, 1] - unc_vec2[:, 1, 0], fmt="go", label="L")
        ax8.hlines(0.006, 0, 100, linewidth=2)
        ax8.set_ylim((0, 0.02))
        ax8.set_yticks([])
        
        ax9.errorbar(res_numbers, res_vec2[:, 2], yerr=res_vec2[:, 2] - unc_vec2[:, 2, 0], fmt="ko", label="ss")
        ax9.hlines(0.52, 0, 100, linewidth=2)
        ax9.set_yticks([])
        
        # ax3.legend()
        plt.show()

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
        p_mean = 0.5 # Sets the mean allele frequency.
        
        # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
        # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
        
        genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
        
        nbh = 4.0 * np.pi * ips / 2.0 * 1.0  # Calculate Neighborhood size
        L = 2 * mu  # Calculate the effective mutation rate
        KC = fac_kernel("DiffusionK0")  # Create the object that will calculate the Kernel.
        KC.set_parameters([nbh, L, t0, ss])  # Sets parameters: Nbh, L, t0, ss
        
        print("Calculating Kernel...")
        
        tic=time()
        kernel_mat = KC.calc_kernel_mat(position_list)  # Calculate the co-variance Matrix
        toc=time()
        print("Calculation Kernel runtime: %.6f:" % (toc-tic))
        
        grid = Grid() # Load the grid (contains method to created correlated data.
        
        for i in range(nr_loci):  # Iterate over all loci
            print("Doing Locus: %i" % i)
            genotypes = grid.draw_corr_genotypes_kernel(kernel_mat=kernel_mat, p_mean=p_mean)
            genotype_matrix[:, i] = genotypes
            
        self.save_data_set(position_list, genotype_matrix, data_set_nr) # Save the Data.
        
            
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
    def __init__(self, folder, nr_data_sets=100, nr_params=5, **kwds):
        super(MultiBarrier, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
        self.name = "barrier_file"
        # self.data_folder = folder
        
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method.'''
        print("Creating Dataset: %i" % data_set_nr)
        # First set all the Parameter Values:
        barrier_strength_list = 25 * [0.0] + 25 * [0.05] + 25 * [0.1] + 25 * [0.15]
        barrier_strength = barrier_strength_list[data_set_nr]
        
        ips = 10  # Number of haploid Individuals per Node (For D_e divide by 2)
        
        
        # position_list = np.array([(500 + i, 500 + j) for i in range(-19, 21, 2) for j in range(-49, 51, 2)])  # 1000 Individuals; spaced 2 sigma apart. Original data_set
        position_list = np.array([(500 + i, 500 + j) for i in range(-9, 11, 1) for j in range(-9, 11, 1)])  # Updated position list.
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
            
    def analyze_data_set(self, data_set_nr, random_ind_nr=1000, position_barrier=500.5, method=0):
        '''Create Data Set. Override Method. mle_pw: Whether to use Pairwise Likelihood
        method 0: GRF; method 1: Pairwise LL method 2: Individual Curve Fit. method 3: Binned Curve fit.'''
        position_list, genotype_mat = self.load_data_set(data_set_nr)  # Loads the Data 
        
        # Creates the "right" starting parameters:
        # barrier_strength_list = 25 * [0.01] + 25 * [0.2] + 25 * [0.5] + 25 * [1.0]
        barrier_strength_list = 100 * [0.5]
        l = 0.006

        nbh_size = 4 * np.pi * 5  # 4 pi sigma**2 D = 4 * pi * 1 * ips/2.0
        start_list = [[nbh_size, l, bs, 0.004] for bs in barrier_strength_list]  # General Vector for Start-Lists
        
        # Pick Random_ind_nr many Individuals:
        # inds = range(len(position_list))
        # shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
        # inds = inds[:random_ind_nr]  # Only load first nr_inds

        # position_list = position_list[inds, :]
        # genotype_mat = genotype_mat[inds, :]
        
        if method == 0:
            MLE_obj = MLE_estimator("DiffusionBarrierK0", position_list, genotype_mat, multi_processing=self.multi_processing) 
        elif method == 1:
            MLE_obj = MLE_pairwise("DiffusionBarrierK0", position_list, genotype_mat, multi_processing=self.multi_processing)
            start_list = [[nbh_size, l, bs, 0.01] for bs in barrier_strength_list]  # Update Vector of Start Lists
        elif method == 2:
            MLE_obj = MLE_f_emp("DiffusionBarrierK0", position_list, genotype_mat, multi_processing=self.multi_processing)
            start_list = [[nbh_size, l, bs, 0.5] for bs in barrier_strength_list]  # Update Vector of Start Lists
        elif method == 3:  # Do the fitting based on binned data
            MLE_obj = Analysis(position_list, genotype_mat) 
        else: raise ValueError("Wrong Input for Method!!")
        
        MLE_obj.kernel.position_barrier = position_barrier  # Sets the Barrier Position
        
        fit = MLE_obj.fit(start_params=start_list[data_set_nr])

        params = fit.params
        conf_ind = fit.conf_int()
        
        # Pickle Parameter Estimates:
        subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
        path = self.data_folder + subfolder_meth + "result" + str(data_set_nr).zfill(2) + ".p"
        
        directory = os.path.dirname(path)  # Extract Directory
        if not os.path.exists(directory):  # Creates Folder if not already existing
            os.makedirs(directory)
            
        pickle.dump((params, conf_ind), open(path, "wb"))  # Pickle the Info
        
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
        
    
        
    def visualize_barrier_strengths(self, res_numbers=range(0, 100)):
        '''Method to visualize the strengths of the Barrier'''
        
        def load_pickle_data(i):
            '''Function To load pickled Data.
            Also visualizes it.'''
            data_folder = self.data_folder
            # path = data_folder + "result" + str(i).zfill(2) + ".p"  # Path to Alex Estimates
            
            # subfolder_meth = "estimate" + str(2) + "/"  # Path to binned Estimates
            # path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            # Coordinates for more :
            subfolder_meth = "method_k0" + "/"  # Sets subfolder to which Method to use.
            path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            res = pickle.load(open(path, "rb"))  # Loads the Data
            return res
        
        ll_vecs = np.array([load_pickle_data(i) for i in res_numbers])
        ll_vecs_max = np.max(ll_vecs, axis=1)
        
        ll_rel_vecs = ll_vecs - ll_vecs_max[:, None]
        
        # How many Barrier Strengths:
        k_len = len(ll_vecs[0])
        k_vec = np.linspace(0.0, 1, k_len)  # Creates the Grid for k
        
        plt.figure()
        ax = plt.gca()
        # ax.set_aspect('equal')
        im = ax.imshow(ll_rel_vecs.T, cmap="seismic", vmin=-6)
        plt.ylabel("Reduced Migration")
        plt.xlabel("Data Set")
        plt.title("Marginal Likelihood Barrier")
        plt.yticks(range(len(k_vec)), k_vec[::-1])
        plt.hlines(0 * (k_len - 1), -0.5, 24.5, linewidth=1, color="g")
        plt.hlines(0.25 * (k_len - 1), 24.5, 49.5, linewidth=1, color="g")
        plt.hlines(0.5 * (k_len - 1), 49.5, 74.5, linewidth=1, color="g")
        plt.hlines(1.0 * (k_len - 1), 74.5, 99.5, linewidth=1, color="g")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        
    def temp_visualize(self, method=0):
        '''Temporary Function to plot the Estimates
        that were run on cluster.'''
        # First quick function to unpickle the data:
        def load_pickle_data(i, arg_nr, method=2):
            '''Function To load pickled Data.
            Also visualizes it.'''
            data_folder = self.data_folder
            # path = data_folder + "result" + str(i).zfill(2) + ".p"  # Path to Alex Estimates
            
            # subfolder_meth = "estimate" + str(2) + "/"  # Path to binned Estimates
            # path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            # Coordinates for more :
            subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
            path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            
            res = pickle.load(open(path, "rb"))  # Loads the Data
            return res[arg_nr]
        
        
        res_numbers = range(1, 100)  # + range(8, 100)
        # res_numbers = range(10, 13) + range(30, 33) + range(60, 61)  # + range(80, 83)
        print(res_numbers)
        # res_numbers = range(30,31)# To analyze Dataset 30
        
        res_vec = np.array([load_pickle_data(i, 0, method) for i in res_numbers])
        unc_vec = np.array([load_pickle_data(i, 1, method) for i in res_numbers])
        
        for l in range(len(res_numbers)):
            i = res_numbers[l]
            print("\nRun: %i" % i)
            for j in range(3):
                print("Parameter: %i" % j)
                print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
                
        
        
        # plt.figure()
        f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True)
        
        ax1.hlines(4 * np.pi * 5, 0, 100, linewidth=2, color="r")
        ax1.errorbar(res_numbers, res_vec[:, 0], yerr=res_vec[:, 0] - unc_vec[:, 0, 0], fmt="bo", label="Nbh")
        # ax1.set_ylim([0, 200])
        ax1.set_ylabel("Nbh", fontsize=18)
        # ax1.legend()
        
        ax2.errorbar(res_numbers, res_vec[:, 1], yerr=res_vec[:, 1] - unc_vec[:, 1, 0], fmt="go", label="L")
        ax2.hlines(0.006, 0, 100, linewidth=2)
        ax2.set_ylabel("L", fontsize=18)
        # ax2.legend()
        
        ax3.errorbar(res_numbers, res_vec[:, 2], yerr=res_vec[:, 2] - unc_vec[:, 2, 0], fmt="ko", label="ss")
        # ax3.hlines(0.04, 0, 100, linewidth=2)
        ax3.set_ylabel("k", fontsize=18)
        
        ax3.hlines(0.0, 0, 25, linewidth=2, color="r")
        ax3.hlines(0.05, 25, 50, linewidth=2, color="r")
        ax3.hlines(0.1, 50, 75, linewidth=2, color="r")
        ax3.hlines(0.15, 75, 100, linewidth=2, color="r")
        ax3.set_ylim([0, 0.5])
        # ax3.set_ylim([0,0.01])
        # ax3.legend()
        plt.xlabel("Dataset")
        plt.show()
        
    
        
###############################################################################################################################
        
class TestRun(MultiRun):
    '''First simple class to test whether everything works'''
    def __init__(self, folder, nr_data_sets=1, nr_params=4, **kwds):
        super(TestRun, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
        self.name = "test_file"
        # self.data_folder = folder
        
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method.'''
        print("Craeting Dataset: %i" % data_set_nr)
        # First set all the Parameter Values:
        position_list = np.array([(500 + i, 500 + j) for i in range(-19, 20, 2) for j in range(-25, 25, 2)])  # Original position list
        nr_loci = 100
        t = 5000
        gridsize_x, gridsize_y = 1000, 1000
        sigma = 0.965  # 0.965 # 1.98
        ips = 12  # Number of haploid Individuals per Node (For D_e divide by 2)
        mu = 0.005  # Mutation Rate
        ss = 0.1
        p_delta = np.random.normal(scale=ss, size=nr_loci)  # Draw some random Delta p from a normal distribution
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
        p_names = ["Nr Loci", "t", "p_mean", "sigma", "mu", "ips", "ss", "Position List"]
        ps = [nr_loci, t, p_mean, sigma, mu, ips, ss, position_list]
        additional_info = ("1 Test Run for Grid object with high neighborhood size")
        self.pickle_parameters(p_names, ps, additional_info)
            
    def analyze_data_set(self, data_set_nr, random_ind_nr=500):
        '''Create Data Set. Override Method.'''
        position_list, genotype_mat = self.load_data_set(data_set_nr)  # Loads the Data 
        
        # Pick Random_ind_nr many Individuals:
        inds = range(len(position_list))
        shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
        inds = inds[:random_ind_nr]  # Only load first nr_inds

        position_list = position_list[inds, :]
        genotype_mat = genotype_mat[inds, :]
        
        start_params = [100.0, 0.005, 0.038]  # Sets the Start Parameters
        MLE_obj = MLE_estimator("DiffusionK0", position_list, genotype_mat) 
        fit = MLE_obj.fit(start_params=[100.0, 0.01, 0.04])
        
        
        print(fit.summary())
    
        # results0 = ml_estimator.fit(method="BFGS")  # Do the actual fit. method="BFGS" possible
        params = fit.params
        conf_ind = fit.conf_int()
        
        print(np.shape(conf_int))
        print(params)  # Print Parameter Estimates
        print(conf_ind)  # Print Confidence Intervals
        
        
        print("Trying to save. uhuhuh.")
        # Saves the results of Analysis to local object
        self.param_estimates[data_set_nr, :] = fit.params
        self.uncertain_estimates[data_set_nr, :] = fit.conf_int
        
        # Pickles the parameters and confidence intervalls
        path = self.data_folder + "result" + str(date_set_nr).zfill(2) + ".p"
        pickle.dump((params, conf_int), open(path, "wb"))  # Pickle the Info
        
        # self.param_estimates=fit.
        # self.uncertain_estimates=fit.
    
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
    MultiRun = fac_method("multi_nbh", folder)
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
    # vis_mult_nbh("./nbh_folder/", method=2)
    
    #######################################################
    ####Create Multi Barrier Data Set
    #MultiRun = fac_method("multi_barrier", "./barrier_folder1/", multi_processing=1)
    MultiRun = fac_method("multi_nbh_gaussian", "./nbh_gaussian_folder/", multi_processing=1)
    #MultiRun.create_data_set(30)
    MultiRun.analyze_data_set(30, method=2, fit_t0=0)
    
    
    # MultiRun.temp_visualize(method=2)
    #MultiRun.visualize_barrier_strengths(res_numbers=range(0, 100))
    




