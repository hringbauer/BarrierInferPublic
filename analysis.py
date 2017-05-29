'''
Created on Oct 13, 2016
Class that analyses the data produced by the grid object.
@author: hringbauer
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from scipy.stats import binned_statistic
from scipy.stats import sem
from kernels import fac_kernel  # Factory Method which yields Kernel Object
from random import shuffle
from scipy.optimize.minpack import curve_fit
from time import time


parameters_fit = [84.96, 0.0814, 1.0, 0.0596]  # Parameters used in manual Kernel Plot. Old: 0.04194
# 7.38282829e+01   9.44133108e-04   5.15210543e-01

class Fit_class(object):
    '''Simple class that contains the results of a fit'''
    params = []  # Vector of estimated Parameters
    std = []  # Vector of estimated Standard Deviations
    
    def __init__(self, params, std):
        self.params = params
        self.std = std
        
    def conf_int(self):
        '''Method that gives back Confidence Intervals.
        Calculates it as best estimates plus/minus 3 standard deviations'''
        conf_int = [[self.params[i] - 2 * self.std[i], self.params[i] + 2 * self.std[i]] for i in xrange(len(self.params))]  # 95 percent of data within 2 STD of Mean.
        return conf_int
            

class Analysis(object):
    '''
    General class that analyzes given Genotype matrix and position_list:
    '''
    position_list = []  # List of all positions (n individuals)
    genotypes = []  # List of all genotypes (nxk matrix. n: Individuals k: Genotypes)
    inds_per_deme = []  # Vector storing the number of Individuals per Deme
    barrier = 0  # Where one can find the barrier
    loci_info = 0  # Will be Pandas Data Frame

    def __init__(self, position_list, genotype_list, loci_path=None, barrier=0):
        '''
        Constructor
        '''
        self.position_list = position_list  # Sets the position_list
        self.genotypes = genotype_list  # Sets the geno-type Matrix (n inds x k loci)
        self.barrier = barrier  # Where the barrier sits

        print("Nr. of loci: %i" % len(genotype_list[0, :]))
        print("Nr. of individuals: % i" % len(genotype_list[:, 0]))
        print("Barrier assumed at: %.2f" % self.barrier)
        
        # Calculate standard deviations among means:
        means = np.mean(self.genotypes, axis=0)  # 
        print("\nMean of Allele Frequencies: %.6f: " % np.mean(means))
        print("Standard Deviations of Allele Frequencies: %.6f" % np.std(means))
        
        if loci_path:
            df = pd.read_csv(loci_path)
            self.loci_info = df
            assert(len(self.loci_info) == np.shape(self.genotypes)[1])  # Make sure that Data match!
            print("Loci Information successfully loaded!")
           
    def clean_hz_data(self, geo_r2=0.015, p_HW=0.00001, ld_r2=0.03, min_p=0.15, chromosome=0, plot=False):
        '''Method to clean HZ Data.
        Extracts loci with min. Geographic Correlation; min. p-Value for HW
        min. ld_score and minimal allele Frequency.'''
        loci_info = self.loci_info
        genotypes = self.genotypes
        
        # Call the cleaning Method:
        genotypes = clean_hz_data(genotypes, loci_info,
                                  geo_r2=geo_r2, p_HW=p_HW, ld_r2=ld_r2, 
                                  min_p=min_p, plot=plot, chromosome=chromosome)
        
        return genotypes, self.position_list
    
        
        
    def kinship_coeff(self, p1, p2, p):
        '''Takes two allele frequencies as input and calculates their correlation.'''
        f = kinship_coeff(p1, p2, p)
        # f = np.mean((p1 - p) * (p2 - p) / (p * (1 - p)))
        return f
    
    def mean_kinship_coeff(self, genotype_mat, p_mean=0.5):
        '''Calculate the mean Kinship coefficient for a Genotype_matrix; given some mean Vector 
        p_mean'''
        p_mean_emp = np.mean(genoytpe_mat, axis=0)  # Calculate the mean allele frequencies
        f_vec = (p_mean_emp - p_mean) * (p_mean_emp - p_mean) / (p_mean * (1 - p_mean))  # Calculate the mean f per locus
        f = np.mean(f_vec)  # Calculate the overall mean f
        return f
    
    def fit(self, p=0.5, nr_inds=10000, bins=50, start_params=[50, 0.005, 0.04]):
        '''Fits pairwise co-variance matrices'''
        # Some Code to draw random samples
        inds = range(len(self.position_list[:, 0]))  
        shuffle(inds)
        inds = inds[:nr_inds]
        
        positions = self.position_list[inds, :]
        genotypes = self.genotypes[inds, :]
        
        # First Calculate all individuals correlations
        distance = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Empty container
        correlation = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Container for correlation
        entry = 0
        
        print("Calculating pairwise Correlation...")
        for (i, j) in itertools.combinations(range(len(genotypes[:, 0])), r=2):
            distance[entry] = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))  # Calculate the pairwise distance
            correlation[entry] = self.kinship_coeff(genotypes[i, :], genotypes[j, :], p)  # Kinship coeff per pair, averaged over loci  
            entry += 1     
        
        # Bin the data
        bin_corr, bin_edges, _ = binned_statistic(distance, correlation, bins=bins, statistic='mean')  # Calculate Bin Values
        stand_errors, _, _ = binned_statistic(distance, correlation, bins=bins, statistic=sem)
        bin_dist = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Calculate the mean of the bins
        
        # Do the fitting.
        params, cov_matrix = fit_diffusion_kernel(bin_corr[:bins / 2], bin_dist[:bins / 2], stand_errors[:bins / 2], guess=start_params)
        std_params = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
        Fit = Fit_class(params, std_params)
        return Fit
        
    def ind_correlation(self, p=0.5, nr_inds=10000, bins=50):
        '''Analyze individual correlations.'''
        inds = range(len(self.position_list[:, 0]))  # Some Code to draw random samples
        shuffle(inds)
        inds = inds[:nr_inds]
        positions = self.position_list[inds, :]
        genotypes = self.genotypes[inds, :]
        
        distance = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Empty container
        correlation = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Container for correlation
        entry = 0
        
        for (i, j) in itertools.combinations(range(len(genotypes[:, 0])), r=2):
            distance[entry] = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))  # Calculate the pairwise distance
            correlation[entry] = self.kinship_coeff(genotypes[i, :], genotypes[j, :], p)  # Kinship coeff. per pair, averaged over loci  
            entry += 1     
        self.vis_correlation(distance, correlation, bins=bins)  # Visualize the correlation
    
    def vis_correlation(self, distance, correlation, bins=50, cut_off_frac=0.75):
        '''Take pairwise correlation and distances as inputs and visualizes them.
        Cut_Off_Frac: At which fraction to do the cut-off'''
        bin_corr, bin_edges, _ = binned_statistic(distance, correlation, bins=bins, statistic='mean')  # Calculate Bin Values
        stand_errors, _, _ = binned_statistic(distance, correlation, bins=bins, statistic=sem)
        
        bin_dist = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Calculate the mean of the bins
        
        # Fit the data:
        C, k, std_k = fit_log_linear(bin_dist[:bins / 3], bin_corr[:bins / 3])  # Fit first half of the distances bins
        Nb_est = 1 / (-k)
        Nb_std = (-std_k / k) * Nb_est
        
        
        # Fit Diffusion/RBF Kernel; Comment out depending on what is need:
        params, cov_matrix = fit_diffusion_kernel(bin_corr[:bins / 2], bin_dist[:bins / 2], stand_errors[:bins / 2])
        # params, cov_matrix = fit_rbf_kernel(bin_corr[:bins/2], bin_dist[:bins/2], stand_errors[:bins/2])
        
        std_params = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
        
        # print("Fitted Parameters + Errors from least square fit: ")
        print(params)
        print(std_params)
        
        print("Log Fit: ")
        print(Nb_est)
        print(Nb_std)
        
        
        x_plot = np.linspace(min(bin_dist), max(bin_dist) * cut_off_frac, 100)
        y_fit = diffusion_kernel(x_plot, *params)  # Calculate the best fits (diffusion Kernel is vector)
        # y_fit = rbf_kernel(x_plot, *params)  # Calculate the best fits (RBF Kernel is vector)
        
        KC = fac_kernel("DiffusionK0")
        
        
        KC.set_parameters(parameters_fit)  # Nbh Sz, Mu0, t0, ss. Sets known Parameters #[4*np.pi*6, 0.02, 1.0, 0.04]
        # KC.set_parameters([4 * np.pi * 5, 0.006, 1.0, 0.04])
        
        coords = [[0, 0], ] + [[0, i] for i in x_plot]  # Coordsvector
        print(coords[:5])
        kernel = KC.calc_kernel_mat(coords)
        
        plt.errorbar(bin_dist[:int(bins * cut_off_frac)], bin_corr[:int(bins * cut_off_frac)], stand_errors[:int(bins * cut_off_frac)], fmt='ro', label="Binwise estimated Correlation")
        plt.plot(x_plot, C + k * np.log(x_plot), 'g', label="Fitted Log Decay")
        
        plt.plot(x_plot, y_fit, 'yo', label="Least square fit.")
        plt.plot(x_plot, kernel[0, 1:], 'bo', label="From Kernel; known parameters")
        plt.axhline(np.mean(bin_corr), label="Mean Value", color='k', linewidth=2)
        plt.annotate(r'$\bar{N_b}=%.4G \pm %.2G$' % (Nb_est, Nb_std) , xy=(0.6, 0.7), xycoords='axes fraction', fontsize=15)
        plt.legend()
        plt.ylabel("F / Correlation")
        plt.xlabel("Distance")
        # plt.ylim([0,0.05])
        # plt.xscale("log")
        plt.show()
       
    def plot_positions(self, row=1, inds=[]):
        '''Method to plot position of Samples.
        If inds given; plot them'''
        plt.figure()
        plt.title("Sample Distribution", fontsize=30)
        color = self.genotypes[:, row].astype("float")
        
        if len(inds) > 0:
            color = inds
            
        plt.scatter(self.position_list[:, 0], self.position_list[:, 1], label="Samples", c=color)
        # pylab.vlines(0, min(X_data[:,1]), max(X_data[:,1]), linewidth=2, color="red", label="Barrier")
        plt.xlabel("X-Coordinate", fontsize=30)
        plt.ylabel("Y-Coordinate", fontsize=30)
        plt.show()
        
    def plot_all_freqs(self):
        '''Plot Figure of allele Frequency distribution'''
        p_mean = np.mean(self.genotypes, axis=0)  # Empirical average for every loci
        print(len(p_mean))
        print("Mean allele frequency: %.6f" % np.mean(p_mean))
        print("Empirical Standard Deviation of Allele Frequency: %.6f" % np.std(p_mean))
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.bar(range(len(p_mean)), p_mean, width=1.0)
        ax1.set_ylabel("Mean")
        ax1.axhline(y=0.5, c="r", linewidth=2)
        
        ax2 = fig.add_subplot(122)
        ax2.hist(p_mean)
        ax2.axvline(x=0.5, c="r")
        ax2.set_xlim([0, 1])
        plt.title("Distribution of mean all. freqs")
        plt.show()
        
    def geo_comparison(self, mean_all_freq=0.5, barrier=None):
        '''Compares kinship coefficients based on geography'''
        if barrier == None:
            barrier = self.barrier
        y_comps = []  # Comparisons among yellow
        m_comps = []  # Comparisons among magenta
        h_comps = []  # Comparisons among hybrids
        ym_comps = []  # Comparisons in-between yellow/magenta
        
        genotypes, positions = self.genotypes, self.position_list
        p = mean_all_freq
        
        geo_tag = np.zeros(len(positions))
        # close_tag =np.zeros(len(individual_list))
        
        i = 0  # Place-holder variable for iterating
        for ind in positions:  # Assign geo_tag based on x-coordinate
            if ind[0] < barrier:
                geo_tag[i] = 0
                
            elif ind[0] > barrier:
                geo_tag[i] = 2
                
            else: geo_tag[i] = 1
            
            i += 1
        
        # p_y = 0.5 * np.array([np.mean([data[j, i] for j in range(0, len(color)) if color[j] == 0 and data[j, i] > -8]) for i in range(0, len(data[0, :]))])
        # p_m = 0.5 * np.array([np.mean([data[j, i] for j in range(0, len(color)) if color[j] == 2 and data[j, i] > -8]) for i in range(0, len(data[0, :]))])
        # p_h = 0.5 * np.array([np.mean([data[j, i] for j in range(0, len(color)) if color[j] == 1 and data[j, i] > -8]) for i in range(0, len(data[0, :]))])
               
        for (i, j) in itertools.combinations(range(len(positions)), r=2):
            ind1 = positions[i]
            ind2 = positions[j]
            delta = np.linalg.norm(np.array(ind1) - np.array(ind2))  # Calculate the pairwise distance
                
            estimator = self.kinship_coeff(genotypes[i, :], genotypes[j, :], p)  # Kinship coeff per pair, averaged over loci
            if geo_tag[i] == 2 and geo_tag[j] == 2:  # Case: Two magentas
                m_comps.append([estimator, delta])  # Append Kinship coefff and pairwise distance         
                
            elif geo_tag[i] == 0 and geo_tag[j] == 0:  # Case: Two yellows
                y_comps.append([estimator, delta])
            
            elif geo_tag[i] == 1 and geo_tag[j] == 1:  # Case: Two hybrids
                h_comps.append([estimator, delta])
                    
            elif (geo_tag[i] == 0 and geo_tag[j] == 2) or (geo_tag[i] == 2 and geo_tag[j] == 0):  # Case: One yellow, one magenta
                ym_comps.append([estimator, delta])
                    
        y_comps = np.array(y_comps)  # Make everything a numpy array
        print(y_comps)
        m_comps = np.array(m_comps)
        print(m_comps)
        ym_comps = np.array(ym_comps)
        h_comps = np.array(h_comps)

        print("Y-mean Kinship: %.6f" % np.mean(y_comps[:, 0]))
        print("M-mean Kinship: %.6f" % np.mean(m_comps[:, 0]))
        print("YM-mean Kinship: %.6f" % np.mean(ym_comps[:, 0]))
            
        distance_mean_y, bin_edges, _ = binned_statistic(y_comps[:, 1], y_comps[:, 0], bins=20, statistic='mean', range=[0, 51])  # Calculate Bin Values
        distance_mean_m, bin_edges_m, _ = binned_statistic(m_comps[:, 1], m_comps[:, 0], bins=20, statistic='mean', range=[0, 51])  # Calculate Bin Values
        distance_mean_ym, bin_edges_ym, _ = binned_statistic(ym_comps[:, 1], ym_comps[:, 0], bins=10, statistic='mean', range=[0, 26])  # Calculate Bin Values
        # distance_mean_h, bin_edges_h, _ = binned_statistic(h_comps[:, 1], h_comps[:, 0], bins=5, statistic='mean', range=[0, 5])  # Calculate Bin Values
        
        bin_mean = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Calculate the mean of the bins
        bin_mean_m = (bin_edges_m[:-1] + bin_edges_m[1:]) / 2.0
        bin_mean_ym = (bin_edges_ym[:-1] + bin_edges_ym[1:]) / 2.0
        # bin_mean_h = (bin_edges_h[:-1] + bin_edges_h[1:]) / 2.0
        
        plt.figure()
        plt.scatter(y_comps[:, 1], y_comps[:, 0], alpha=0.5, c='y', label="Yellow-Yellow")
        plt.scatter(m_comps[:, 1], m_comps[:, 0], alpha=0.5, c='m', label="Magenta-Magenta")
        plt.scatter(ym_comps[:, 1], ym_comps[:, 0], alpha=0.5, c='g', label="Yellow-Magenta")
        plt.xlabel("Pairwise Euclidean distance")
        plt.ylabel("Kinship coefficient ")
        plt.xlim([0, max(bin_edges)])
        # plt.axhline(0.25, label="Expected kinship PO/Full Sibs", color='y')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(bin_mean, distance_mean_y, 'yo', markersize=10, label="Yellow-Yellow")
        plt.plot(bin_mean_m, distance_mean_m, 'mo', markersize=10, label="Magenta-Magenta")
        plt.plot(bin_mean_ym, distance_mean_ym, 'go', markersize=10, label="Magenta-Yellow")
        # plt.plot(bin_mean_h, distance_mean_h, 'wo', label="Hybrid-Hybrid")
        plt.xlabel(r"PW Distance [$\sigma$]", fontsize=25)
        plt.ylabel("Mean f", fontsize=25)
        plt.legend(prop={'size':20})
        plt.show() 
     
    def flip_gtps(self, genotypes): 
        '''Randomly flip Genotypes
        There for the case when SNPs are somehow ascertained'''
        genotypes_new = flip_gtps(genotypes)
        self.genotypes = genotypes_new
        return genotypes_new
        
       
    def group_inds(self, position_list, genotypes, demes_x=10, demes_y=10, min_ind_nr=0):
        '''Function that groups indviduals into demes and gives back mean deme position
        and mean deme genotype'''
        position_list_new, genotypes_new, inds_per_deme = group_inds(position_list, genotypes, demes_x=demes_x, demes_y=demes_y, min_ind_nr=min_ind_nr)
        self.position_list, self.genotypes, self.inds_per_deme = position_list_new, genotypes_new, inds_per_deme
        return position_list_new, genotypes_new, inds_per_deme
                
#####################################################################################################################
# Some Helper Functions for other classes to import from:
        
def fit_log_linear(t, y):
    '''Fitting log decay and returns parameters: y = A + B * ln(t) as (A,B)'''
    t = np.log(t)
    param, V = np.polyfit(t, y, 1, cov=True)  # Param has highest degree first
    return param[1], param[0], np.sqrt(V[0, 0])  # Returns parameters and STD
        
def diffusion_kernel(r, nbh, L, ss, t0=1): 
    '''Function which is used to fit diffusion kernel'''
    # print([nbh, L, 1, ss])  # Print were the fit is
    K0 = fac_kernel("DiffusionK0")  # Load the Diffusion Kernel
    K0.set_parameters([nbh, L, t0, ss])  # Set its parameters: diffusion, t0, mu, density
    print(K0.give_parameters())
    y = [K0.num_integral(i) for i in r]  # Calculates vector of outputs
    return y + ss  # As ss is not yet included in num_integral

def rbf_kernel(r, l, a):
    '''Function which is used to fit RBF kernel'''
    K1 = fac_kernel("RBFBarrierK")
    K1.set_parameters([l, a, 0, 0])  # l, a, c, s,
    y = [K1.calc_r(i) for i in r]
    return y
    
def fit_diffusion_kernel(f, r, error, guess=[200, 0.002, 0.05]):  # Uncomment the t0=1 if not to be fitted
    '''Fits vectors f,r and error to numerical Integration of
    Diffusion Kernel - Using non-linear, weighted least square.'''    
    parameters, cov_matrix = curve_fit(diffusion_kernel, r, f,
                            sigma=error, absolute_sigma=True, p0=guess, bounds=(0, np.inf))  # @UnusedVariable p0=(C / 10.0, -r)
    return parameters, cov_matrix

def fit_rbf_kernel(f, r, error, guess=[25, 0.2]):
    '''Fits vectors f,r and error to numerical Integration of
    Diffusion Kernel - Using non-linear, weighted least square.'''    
    parameters, cov_matrix = curve_fit(rbf_kernel, r, f,
                            sigma=error, absolute_sigma=True, p0=guess, bounds=(0, np.inf))  # @UnusedVariable p0=(C / 10.0, -r)
    return parameters, cov_matrix

def group_inds(position_list, genotypes, demes_x=10, demes_y=10, min_ind_nr=0):
    '''Function that groups indviduals into demes and gives back mean deme position
    and mean deme genotype'''
    nr_inds, nr_markers = np.shape(genotypes)
    x_coords, y_coords = position_list[:, 0], position_list[:, 1]
    
    x_bins = np.linspace(min(x_coords), max(x_coords) + 0.00001, num=demes_x + 1)
    # print(x_bins)
    y_bins = np.linspace(min(y_coords), max(y_coords) + 0.00001, num=demes_y + 1)
    
    x_inds = np.digitize(x_coords, x_bins)
    y_inds = np.digitize(y_coords, y_bins)
    
    nr_demes = demes_x * demes_y
    position_list_new = np.zeros((nr_demes, 2)) - 1.0  # Defaults everything to -1.
    genotypes_new = np.zeros((nr_demes, nr_markers)) - 1.0  # Same.
    inds_per_deme = np.zeros(nr_demes)
    
    # Iterate over every deme
    
    row = 0
    for i in xrange(1, demes_x + 1):
        for j in range(1, demes_y + 1):
            inds = np.where((x_inds == i) * (y_inds == j))[0]  # Ectract all individuals where match
            
            # row = (i - 1) * demes_y + (j - 1)  # Which row to set the data 
            
            if len(inds) <= min_ind_nr:  # In case no Individual fits in the grid space
                continue
                  
            position_list_new[row, :] = [(x_bins[i - 1] + x_bins[i]) / 2.0, (y_bins[j - 1] + y_bins[j]) / 2.0]
            inds_per_deme[row] = len(inds)
            
            matching_genotypes = genotypes[inds, :]
            genotypes_new[row, :] = np.mean(matching_genotypes, axis=0)  # Sets the new genotypes
            row += 1
    
    # Extract only the set individuals:
    position_list_new = position_list_new[:row, :]
    genotypes_new = genotypes_new[:row, :]
    inds_per_deme = inds_per_deme[:row]
    return position_list_new, genotypes_new, inds_per_deme

def bootstrap_genotypes(genotype_mat):
    '''Short helper Function to Bootstrap over Genotypes'''
    nr_inds, nr_genotypes = np.shape(genotype_mat)  # Get the shape of the Genotype Matrix
    sample_inds = np.random.randint(nr_genotypes, size=nr_genotypes)  # Get Indices of random resampling
    
    gtps_sample = genotype_mat[:, sample_inds]  # Do the actual Bootstrap; pick the columns
    return gtps_sample


def clean_hz_data(genotypes, loci_info, geo_r2=0.015, 
                  p_HW=0.00001, ld_r2=0.03, min_p=0.15, chromosome=0, plot=False):
    '''Method to clean HZ Data.
    Extracts loci with min. Geographic Correlation; min. p-Value for HW
    min. ld_score and minimal allele Frequency.
    Genotypes: Genotype-Matrix
    Loci_Info: Where to find information about Loci - as Pandas Dataframe'''
    df = loci_info
    
    inds_okay = (df['Geo_Score'] < geo_r2) & (df['HW p-Value'] > p_HW) & (df['LD_Score'] < ld_r2) & (df['Min All. Freq'] > min_p)
    
    # In case that also chromosomes should be filtered:
    if chromosome>0:
        inds_okay = inds_okay & (df["LG"] == chromosome)
        
    inds = np.where(inds_okay)[0]  # Extract Numpy Array Indices
    
    print("Filtered from %i to %i Loci." % (len(df), len(inds)))
    print("Reducing to SNPs:")
    print(df["Name"][inds_okay])
    
    genotypes = genotypes[:, inds_okay]
    nr_loci = len(df)
    
    if plot == True:
        f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True)
        ax1.plot(df['Geo_Score'], 'bo', label="Geo_Score")
        ax1.hlines(geo_r2, 0, nr_loci, linewidth=2, color="r")
        ax1.legend(loc="upper right")
        
        ax2.plot(df['LD_Score'], 'bo', label="LD R2")
        ax2.hlines(ld_r2, 0, nr_loci, linewidth=2, color="r")
        ax2.legend(loc="upper right")
        
        ax3.plot(df['Min All. Freq'], 'bo', label="MAF")
        ax3.hlines(min_p, 0, nr_loci, linewidth=2, color="r")
        ax3.legend(loc="upper right")
        
        ax4.plot(df['HW p-Value'], 'bo', label="HW p-Value")
        ax4.hlines(p_HW, 0, nr_loci, linewidth=2, color="r")
        ax4.legend(loc="upper right")
        
        plt.xticks(np.arange(nr_loci + 0.5), df["Name"], rotation='vertical')
        
        colors = np.array(["red", ] * nr_loci)
        colors[inds_okay] = "b"  # Sets the good SNPs to Blue!
        
        ax = plt.gca()
        [t.set_color(i) for (i, t) in zip(colors, ax.xaxis.get_ticklabels())]
        plt.show()

    return genotypes


def flip_gtps(genotypes): 
    '''Randomly flip Genotypes
    There for the case when SNPs are somehow ascertained'''
    nr_genotypes = np.shape(genotypes)[1]

    flip = np.random.random(size=nr_genotypes) < 0.5  # Whether to flip or not
    genotypes_new = (1 - flip[None, :]) * genotypes + flip[None, :] * (1 - genotypes)  # Does the flipping
    genotypes = genotypes_new
    return genotypes_new

def kinship_coeff(p1, p2, p):
    '''Takes two allele frequencies as input and calculates their correlation.'''
    f = np.mean((p1 - p) * (p2 - p) / (p * (1 - p)))
    return f

def calc_f_mat(genotypes, p=0.5):
    '''Calculates whole pairwise F-matrix for all Genotypes'''
    nr_inds=np.shape(genotypes)[0]
    f_mat=np.zeros((nr_inds, nr_inds))
    for i in xrange(nr_inds):
        f_mat[i,:] = np.mean((genotypes[i,:]-0.5)*(genotypes-0.5)/(p*(1-p)), axis=1)
    return f_mat




         
