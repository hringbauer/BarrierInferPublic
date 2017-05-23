'''
Created on April.4th.2017
Contains various methods to produce figures for the paper.
Calls these methods in the end. Aim is to have the figures
produced in high Quality for the final paper.
@author: Harald Ringbauer
'''
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from analysis import Analysis
from scipy.stats import binned_statistic
from scipy.stats import sem
from kernels import fac_kernel  # Factory Method which yields Kernel Object


multi_nbh_folder = "./nbh_folder/"
multi_nbh_gauss_folder = "./nbh_folder_gauss/"
multi_barrier_folder = "./barrier_folder1/"
cluster_folder = "./cluster_folder/"
hz_folder = "./hz_folder/"
multi_ind_folder = "./multi_ind_nr/"
multi_loci_folder = "./multi_loci_nr/"
secondary_contact_folder = "./multi_2nd/"
secondary_contact_folder_b = "./multi_2nd_b/"  # With a 0.05 Barrier
multi_pos_syn_folder = "./multi_barrier_synth/"

met2_folder = "method2/"
######################################################################################
#### First some helper functions

def mean_kinship_coeff(genotype_mat, p_mean=0.5):
        '''Calculate the mean Kinship coefficient for a 
        Genotype_matrix; given some mean Vector p_mean'''
        p_mean_emp = np.mean(genotype_mat, axis=0)  # Calculate the mean allele frequencies
        f_vec = (p_mean_emp - p_mean) * (p_mean_emp - p_mean) / (p_mean * (1 - p_mean))  # Calculate the mean f per locus
        f = np.mean(f_vec)  # Calculate the overall mean f
        return f

def load_pickle_data(folder, i, arg_nr, method=2, subfolder=None):
            '''Function To load pickled Data.
            Also visualizes it.'''
            
            # Coordinates for more :
            if subfolder == None:
                subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
            else: subfolder_meth = subfolder
            path = folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            
            res = pickle.load(open(path, "rb"))  # Loads the Data
            return res[arg_nr]

def calc_bin_correlations(positions, genotypes, bins=25, p=0.5, correction=True):
    '''Helper function Calculates Correlations and Bin them'''
    # Load the data:
    # position_list = position_list / 50.0  # Normalize; for position_list and genotype Matrix of HZ data!
    
    analysis = Analysis(positions, genotypes)
    distance = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Empty container
    correlation = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Container for correlation
    entry = 0
    
    for (i, j) in itertools.combinations(range(len(genotypes[:, 0])), r=2):
        distance[entry] = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))  # Calculate the pairwise distance
        correlation[entry] = analysis.kinship_coeff(genotypes[i, :], genotypes[j, :], p)  # Kinship coeff. per pair, averaged over loci  
        entry += 1 
    bin_dist, bin_corr, stand_errors, corr_factor = bin_correlations(distance, correlation, bins=bins, correction=correction)
    return bin_dist, bin_corr, stand_errors, corr_factor 
            
def bin_correlations(distance, correlation, bins=50, statistic='mean', correction=False):
    '''Bin the Correlations.'''
    bin_corr, bin_edges, _ = binned_statistic(distance, correlation, bins=bins, statistic='mean')  # Calculate Bin Values
    stand_errors, _, _ = binned_statistic(distance, correlation, bins=bins, statistic=sem)
    
    bin_dist = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Calculate the mean of the bins
    
    corr_factor = 0
    if correction == True:
        # Substract IBD of distant individuals
        corr_factor = np.mean(bin_corr[int(bins * 1 / 3):int(bins * 2 / 3)]) 
        bin_corr = bin_corr - corr_factor
    return bin_dist, bin_corr, stand_errors, corr_factor 


######################################################################################
######################################################################################
######################################################################################
# Do the actual Plots:


def multi_nbh_single(folder, method):
    '''Print several Neighborhood Sizes simulated under the model - using one method'''
    # First quick function to unpickle the data:
    res_numbers = range(0, 100)
    # res_numbers = [2, 3, 8, 11, 12, 13, 21, 22, 27, 29, 33, 35, 37, 38, 40, 75]  # 2
    # res_numbers = [1, 7, 8, 9, 14, 17, 18, 19, 20]
    
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
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
    ax1.title.set_text("Method: %s" % str(method))
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
    
def multi_barrier_single(folder, method, barrier_strengths=[0, 0.05, 0.1, 0.15]):
    '''Print Estimates from several Barrier strength from Folder.'''
    # Define the Result Numbers:
    res_numbers0 = range(0, 25)
    res_numbers1 = range(25, 50)
    res_numbers2 = range(50, 75)
    res_numbers3 = range(75, 100)
    res_numbers = res_numbers0 + res_numbers1 + res_numbers2 + res_numbers3
    
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    
    
    # plt.figure()
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True)
    ax1.hlines(4 * np.pi * 5, res_numbers[0], res_numbers[-1], linewidth=2, color="r")
    ax1.errorbar(res_numbers, res_vec[:, 0], yerr=res_vec[:, 0] - unc_vec[:, 0, 0], fmt="bo", label="Nbh")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.title.set_text("Method: %s" % str(method))
    # ax1.legend()
    
    ax2.errorbar(res_numbers, res_vec[:, 1], yerr=res_vec[:, 1] - unc_vec[:, 1, 0], fmt="go", label="L")
    ax2.hlines(0.006, 0, 100, linewidth=2)
    ax2.set_ylabel("L", fontsize=18)
    # ax2.legend()
    
    ax3.errorbar(res_numbers, res_vec[res_numbers, 2], yerr=res_vec[res_numbers, 2] - unc_vec[res_numbers, 2, 0], fmt="yo")
    ax3.hlines(barrier_strengths[0], res_numbers0[0], res_numbers0[-1], linewidth=2, color="r")
    ax3.hlines(barrier_strengths[1], res_numbers1[0], res_numbers1[-1], linewidth=2, color="r")
    ax3.hlines(barrier_strengths[2], res_numbers2[0], res_numbers2[-1], linewidth=2, color="r")
    ax3.hlines(barrier_strengths[3], res_numbers3[0], res_numbers3[-1], linewidth=2, color="r")
    ax3.set_ylabel("Barrier", fontsize=18)

    
    ax4.errorbar(res_numbers, res_vec[:, 3], yerr=res_vec[:, 3] - unc_vec[:, 3, 0], fmt="ko", label="ss")
    ax4.hlines(0.52, 0, 100, linewidth=2)
    ax4.set_ylabel("SS", fontsize=18)
    # ax3.legend()
    plt.xlabel("Dataset")
    plt.show()
    
    
def multi_ind_single(folder, method, res_numbers=range(0, 100)):
    '''Print several Neighborhood Sizes simulated under the model - using one method'''
    # First quick function to unpickle the data:
    res_numbers = range(0, 100)
    # res_numbers = range(3, 4)
    # res_numbers = [2, 3, 8, 11, 12, 13, 21, 22, 27, 29, 33, 35, 37, 38, 40, 75]  # 2
    # res_numbers = [1, 2, 7, 8, 9, 14, 17, 18, 19, 20]
    
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(3):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    
    x_vec = range(436, 4001, 36)  # Length 100: from 404 to 4000
    # plt.figure()
    f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True)
    
    ax1.hlines(4 * np.pi * 5, 500, 4000, linewidth=2, color="r")
    ax1.errorbar(x_vec, res_vec[:, 0], yerr=res_vec[:, 0] - unc_vec[:, 0, 0], fmt="bo", label="Nbh")
    ax1.set_ylim([50, 150])
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.title.set_text("Method: %s" % str(method))
    # ax1.legend()
    
    ax2.errorbar(x_vec, res_vec[:, 1], yerr=res_vec[:, 1] - unc_vec[:, 1, 0], fmt="go", label="L")
    ax2.hlines(0.006, 500, 4000, linewidth=2)
    ax2.set_ylabel("L", fontsize=18)
    # ax2.legend()
    
    ax3.errorbar(x_vec, res_vec[:, 2], yerr=res_vec[:, 2] - unc_vec[:, 2, 0], fmt="ko", label="ss")
    ax3.hlines(0.52, 500, 4000, linewidth=2)
    ax3.set_ylabel("SS", fontsize=18)
    # ax3.legend()
    plt.xlabel("Nr. of Individuals")
    plt.show()
    
    
def multi_loci_single(folder, method):
    '''Print several Neighborhood Sizes simulated under the model - using one method'''
    # First quick function to unpickle the data:
    res_numbers = range(0, 100)
    # res_numbers = [2, 3, 8, 11, 12, 13, 21, 22, 27, 29, 33, 35, 37, 38, 40, 75]  # 2
    # res_numbers = [1, 2, 7, 8, 9, 14, 17, 18, 19, 20]
    
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(3):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    
    x_vec = range(50, 350, 3)  # Length 100: from 50 to 350
    # plt.figure()
    f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True)
    
    ax1.hlines(4 * np.pi * 5, 50, 350, linewidth=2, color="r")
    ax1.errorbar(x_vec, res_vec[:, 0], yerr=res_vec[:, 0] - unc_vec[:, 0, 0], fmt="bo", label="Nbh")
    ax1.set_ylim([30, 150])
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.title.set_text("Method: %s" % str(method))
    # ax1.legend()
    
    ax2.errorbar(x_vec, res_vec[:, 1], yerr=res_vec[:, 1] - unc_vec[:, 1, 0], fmt="go", label="L")
    ax2.hlines(0.006, 50, 350, linewidth=2)
    ax2.set_ylabel("L", fontsize=18)
    # ax2.legend()
    
    ax3.errorbar(x_vec, res_vec[:, 2], yerr=res_vec[:, 2] - unc_vec[:, 2, 0], fmt="ko", label="ss")
    ax3.hlines(0.52, 50, 350, linewidth=2)
    ax3.set_ylabel("SS", fontsize=18)
    # ax3.legend()
    plt.xlabel("Nr. of Loci")
    plt.show()


def multi_secondary_contact_single(folder, method):
    '''Print several Neighborhood Sizes simulated under the model - using one method'''
    # First quick function to unpickle the data:
    res_numbers0 = range(0, 25)
    res_numbers1 = range(25, 50)
    res_numbers2 = range(50, 75)
    res_numbers3 = range(75, 100)
    res_numbers = res_numbers0 + res_numbers1 + res_numbers2 + res_numbers3
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    res_vec[res_numbers, 2] = np.where(res_vec[res_numbers, 2] > 1, 1, res_vec[res_numbers, 2])
    subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
    
    loci_nr_vec = [np.loadtxt(folder + subfolder_meth + "nr_good_loci" + str(i).zfill(2) + ".csv")[1] 
                   for i in res_numbers]
    # plt.figure()
    f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True)
    
    ax1.hlines(4 * np.pi * 5, 0, 100, linewidth=2, color="k")
    
    ax1.errorbar(res_numbers0, res_vec[res_numbers0, 0], yerr=res_vec[res_numbers0, 0] - unc_vec[res_numbers0, 0, 0], fmt="yo", label="R2=1")
    ax1.errorbar(res_numbers1, res_vec[res_numbers1, 0], yerr=res_vec[res_numbers1, 0] - unc_vec[res_numbers1, 0, 0], fmt="ro", label="R2=0.02")
    ax1.errorbar(res_numbers2, res_vec[res_numbers2, 0], yerr=res_vec[res_numbers2, 0] - unc_vec[res_numbers2, 0, 0], fmt="go", label="R2=0.01")
    ax1.errorbar(res_numbers3, res_vec[res_numbers3, 0], yerr=res_vec[res_numbers3, 0] - unc_vec[res_numbers3, 0, 0], fmt="bo", label="R2=0.005")
    ax1.set_ylim([0, 150])
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.title.set_text("Varying Quality Cutoff")
    ax1.legend(loc="upper right")
    
    # ax2.errorbar(res_numbers0, res_vec[res_numbers0, 1], yerr=res_vec[res_numbers0, 1] - unc_vec[res_numbers0, 1, 0], fmt="yo")
    # ax2.errorbar(res_numbers1, res_vec[res_numbers1, 1], yerr=res_vec[res_numbers1, 1] - unc_vec[res_numbers1, 1, 0], fmt="ro")
    # ax2.errorbar(res_numbers2, res_vec[res_numbers2, 1], yerr=res_vec[res_numbers2, 1] - unc_vec[res_numbers2, 1, 0], fmt="go")
    # ax2.errorbar(res_numbers3, res_vec[res_numbers3, 1], yerr=res_vec[res_numbers3, 1] - unc_vec[res_numbers3, 1, 0], fmt="bo")
    # ax2.hlines(0.006, 0, 100, linewidth=2)
    # ax2.set_ylabel("L", fontsize=18)
    # ax2.legend()
    
    ax2.errorbar(res_numbers0, res_vec[res_numbers0, 2], yerr=res_vec[res_numbers0, 2] - unc_vec[res_numbers0, 2, 0], fmt="yo")
    ax2.errorbar(res_numbers1, res_vec[res_numbers1, 2], yerr=res_vec[res_numbers1, 2] - unc_vec[res_numbers1, 2, 0], fmt="ro")
    ax2.errorbar(res_numbers2, res_vec[res_numbers2, 2], yerr=res_vec[res_numbers2, 2] - unc_vec[res_numbers2, 2, 0], fmt="go")
    ax2.errorbar(res_numbers3, res_vec[res_numbers3, 2], yerr=res_vec[res_numbers3, 2] - unc_vec[res_numbers3, 2, 0], fmt="bo")
    ax2.hlines(1, 0, 100, linewidth=2)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Barrier", fontsize=18)
    
    ax3.plot(res_numbers, loci_nr_vec, 'ro')
    ax3.hlines(200, 0, 100, linewidth=2)
    ax3.set_ylabel("Nr. of Loci", fontsize=18)
    
    plt.xlabel("Dataset")
    plt.show()
    
def multi_secondary_contact_all(folder, folder_b, method=2):
    '''Prints results of Multi Secondary Contact Simulations -
    left: wo Barrier; right Barrier'''
    res_numbers0 = range(0, 25)
    res_numbers1 = range(25, 50)
    res_numbers2 = range(50, 75)
    res_numbers3 = range(75, 100)
    res_numbers = res_numbers0 + res_numbers1 + res_numbers2 + res_numbers3
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    res_vec_b = np.array([load_pickle_data(folder_b, i, 0, method) for i in res_numbers])
    unc_vec_b = np.array([load_pickle_data(folder_b, i, 1, method) for i in res_numbers])
    
    res_vec[res_numbers, 2] = np.where(res_vec[res_numbers, 2] > 1, 1, res_vec[res_numbers, 2])
    res_vec_b[res_numbers, 2] = np.where(res_vec_b[res_numbers, 2] > 1, 1, res_vec_b[res_numbers, 2])
    
    subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
    
    loci_nr_vec = [np.loadtxt(folder + subfolder_meth + "nr_good_loci" + str(i).zfill(2) + ".csv")[1] 
                   for i in res_numbers]
    loci_nr_vec_b = [np.loadtxt(folder_b + subfolder_meth + "nr_good_loci" + str(i).zfill(2) + ".csv")[1] 
                   for i in res_numbers]
    
    
    # plt.figure()
    f, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, sharex=True)
    
    ax1.hlines(4 * np.pi * 5, 0, 100, linewidth=2, color="k")
    
    ax1.errorbar(res_numbers0, res_vec[res_numbers0, 0], yerr=res_vec[res_numbers0, 0] - unc_vec[res_numbers0, 0, 0], fmt="yo", label="R2=1")
    ax1.errorbar(res_numbers1, res_vec[res_numbers1, 0], yerr=res_vec[res_numbers1, 0] - unc_vec[res_numbers1, 0, 0], fmt="ro", label="R2=0.02")
    ax1.errorbar(res_numbers2, res_vec[res_numbers2, 0], yerr=res_vec[res_numbers2, 0] - unc_vec[res_numbers2, 0, 0], fmt="go", label="R2=0.01")
    ax1.errorbar(res_numbers3, res_vec[res_numbers3, 0], yerr=res_vec[res_numbers3, 0] - unc_vec[res_numbers3, 0, 0], fmt="bo", label="R2=0.005")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.title.set_text("No Barrier")
    ax1.legend(loc="upper right")
    
    ax4.errorbar(res_numbers0, res_vec_b[res_numbers0, 0], yerr=res_vec_b[res_numbers0, 0] - unc_vec_b[res_numbers0, 0, 0], fmt="yo", label="R2=1")
    ax4.errorbar(res_numbers1, res_vec_b[res_numbers1, 0], yerr=res_vec_b[res_numbers1, 0] - unc_vec_b[res_numbers1, 0, 0], fmt="ro", label="R2=0.02")
    ax4.errorbar(res_numbers2, res_vec_b[res_numbers2, 0], yerr=res_vec_b[res_numbers2, 0] - unc_vec_b[res_numbers2, 0, 0], fmt="go", label="R2=0.01")
    ax4.errorbar(res_numbers3, res_vec_b[res_numbers3, 0], yerr=res_vec_b[res_numbers3, 0] - unc_vec_b[res_numbers3, 0, 0], fmt="bo", label="R2=0.005")
    ax4.set_ylim([0, 200])
    ax4.hlines(4 * np.pi * 5, 0, 100, linewidth=2, color="k")
    ax4.title.set_text("Barrier: 0.05")
    ax4.yaxis.tick_right()
    # ax4.legend(loc="upper right")
    

    
    ax2.errorbar(res_numbers0, res_vec[res_numbers0, 2], yerr=res_vec[res_numbers0, 2] - unc_vec[res_numbers0, 2, 0], fmt="yo")
    ax2.errorbar(res_numbers1, res_vec[res_numbers1, 2], yerr=res_vec[res_numbers1, 2] - unc_vec[res_numbers1, 2, 0], fmt="ro")
    ax2.errorbar(res_numbers2, res_vec[res_numbers2, 2], yerr=res_vec[res_numbers2, 2] - unc_vec[res_numbers2, 2, 0], fmt="go")
    ax2.errorbar(res_numbers3, res_vec[res_numbers3, 2], yerr=res_vec[res_numbers3, 2] - unc_vec[res_numbers3, 2, 0], fmt="bo")
    ax2.hlines(1, 0, 100, linewidth=2)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Barrier", fontsize=18)
    
    ax5.errorbar(res_numbers0, res_vec_b[res_numbers0, 2], yerr=res_vec_b[res_numbers0, 2] - unc_vec_b[res_numbers0, 2, 0], fmt="yo")
    ax5.errorbar(res_numbers1, res_vec_b[res_numbers1, 2], yerr=res_vec_b[res_numbers1, 2] - unc_vec_b[res_numbers1, 2, 0], fmt="ro")
    ax5.errorbar(res_numbers2, res_vec_b[res_numbers2, 2], yerr=res_vec_b[res_numbers2, 2] - unc_vec_b[res_numbers2, 2, 0], fmt="go")
    ax5.errorbar(res_numbers3, res_vec_b[res_numbers3, 2], yerr=res_vec_b[res_numbers3, 2] - unc_vec_b[res_numbers3, 2, 0], fmt="bo")
    ax5.hlines(0.05, 0, 100, linewidth=2)
    ax5.set_ylim([0, 0.1])
    ax5.yaxis.tick_right()
    
    ax3.plot(res_numbers, loci_nr_vec, 'ro')
    ax3.hlines(200, 0, 100, linewidth=2)
    ax3.set_ylabel("Nr. of Loci", fontsize=18)
    
    
    ax6.plot(res_numbers, loci_nr_vec_b, 'ro')
    ax6.hlines(200, 0, 100, linewidth=2)
    ax6.yaxis.tick_right()
    
    plt.xlabel("Dataset")
    plt.show()
    
def multi_barrier(folder):
    '''Prints Inference of multiple Barrier strenghts'''
    print("To Implement")
    
def cluster_plot(folder, method=2):
    '''Plots multiple results of various degrees of clustering.'''
    res_numbers0 = range(0, 25)
    res_numbers1 = range(25, 50)
    res_numbers2 = range(50, 75)
    res_numbers3 = range(75, 100)
    res_numbers = res_numbers0 + res_numbers1 + res_numbers2 + res_numbers3
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    
    
    # plt.figure()
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True)
    
    ax1.hlines(4 * np.pi * 5, 0, 100, linewidth=2, color="k")
    
    ax1.errorbar(res_numbers0, res_vec[res_numbers0, 0], yerr=res_vec[res_numbers0, 0] - unc_vec[res_numbers0, 0, 0], fmt="yo", label="1x1")
    ax1.errorbar(res_numbers1, res_vec[res_numbers1, 0], yerr=res_vec[res_numbers1, 0] - unc_vec[res_numbers1, 0, 0], fmt="ro", label="2x2")
    ax1.errorbar(res_numbers2, res_vec[res_numbers2, 0], yerr=res_vec[res_numbers2, 0] - unc_vec[res_numbers2, 0, 0], fmt="go", label="3x3")
    ax1.errorbar(res_numbers3, res_vec[res_numbers3, 0], yerr=res_vec[res_numbers3, 0] - unc_vec[res_numbers3, 0, 0], fmt="bo", label="4x4")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh Size", fontsize=18)
    ax1.title.set_text("Various Clustering")
    ax1.legend()
    
    ax2.errorbar(res_numbers0, res_vec[res_numbers0, 1], yerr=res_vec[res_numbers0, 1] - unc_vec[res_numbers0, 1, 0], fmt="yo", label="1x1")
    ax2.errorbar(res_numbers1, res_vec[res_numbers1, 1], yerr=res_vec[res_numbers1, 1] - unc_vec[res_numbers1, 1, 0], fmt="ro", label="2x2")
    ax2.errorbar(res_numbers2, res_vec[res_numbers2, 1], yerr=res_vec[res_numbers2, 1] - unc_vec[res_numbers2, 1, 0], fmt="go", label="3x3")
    ax2.errorbar(res_numbers3, res_vec[res_numbers3, 1], yerr=res_vec[res_numbers3, 1] - unc_vec[res_numbers3, 1, 0], fmt="bo", label="4x4")
    ax2.hlines(0.006, 0, 100, linewidth=2)
    ax2.set_ylabel("L", fontsize=18)
    # ax2.legend()
    
    ax3.errorbar(res_numbers0, res_vec[res_numbers0, 2], yerr=res_vec[res_numbers0, 2] - unc_vec[res_numbers0, 2, 0], fmt="yo", label="1x1")
    ax3.errorbar(res_numbers1, res_vec[res_numbers1, 2], yerr=res_vec[res_numbers1, 2] - unc_vec[res_numbers1, 2, 0], fmt="ro", label="2x2")
    ax3.errorbar(res_numbers2, res_vec[res_numbers2, 2], yerr=res_vec[res_numbers2, 2] - unc_vec[res_numbers2, 2, 0], fmt="go", label="3x3")
    ax3.errorbar(res_numbers3, res_vec[res_numbers3, 2], yerr=res_vec[res_numbers3, 2] - unc_vec[res_numbers3, 2, 0], fmt="bo", label="4x4")
    ax3.hlines(0.1, 0, 100, linewidth=2)
    ax3.set_ylabel("Barrier", fontsize=18)
    
    ax4.errorbar(res_numbers0, res_vec[res_numbers0, 3], yerr=res_vec[res_numbers0, 3] - unc_vec[res_numbers0, 3, 0], fmt="yo", label="1x1")
    ax4.errorbar(res_numbers1, res_vec[res_numbers1, 3], yerr=res_vec[res_numbers1, 3] - unc_vec[res_numbers1, 3, 0], fmt="ro", label="2x2")
    ax4.errorbar(res_numbers2, res_vec[res_numbers2, 3], yerr=res_vec[res_numbers2, 3] - unc_vec[res_numbers2, 3, 0], fmt="go", label="3x3")
    ax4.errorbar(res_numbers3, res_vec[res_numbers3, 3], yerr=res_vec[res_numbers3, 3] - unc_vec[res_numbers3, 3, 0], fmt="bo", label="4x4")
    ax4.hlines(0.52, 0, 100, linewidth=2)
    ax4.set_ylim([0.5, 0.53])
    ax4.set_ylabel("SS", fontsize=18)
    # plt.xticks([10,35,60,85], ['1x1', '2x2', '3x3','4x4'])
    
    plt.xlabel("Dataset")
    plt.show()
    
def ll_barrier(folder):
    '''Visualize Likelihoods of Barrier'''
    
    def load_pickle_data(folder, i):
        '''Special Function to load pickled Data'''
        # path = data_folder + "result" + str(i).zfill(2) + ".p"  # Path to Alex Estimates
        
        # subfolder_meth = "estimate" + str(2) + "/"  # Path to binned Estimates
        # path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
        
        # Coordinates for more :
        subfolder_meth = "method_k0" + "/"  # Sets subfolder to which Method to use.
        path = folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
        
        res = pickle.load(open(path, "rb"))  # Loads the Data
        return res
    
    res_numbers = range(0, 100)
    ll_vecs = np.array([load_pickle_data(folder, i) for i in res_numbers])
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


def boots_trap(folder, method=2):
    '''Plots Bootstrap Estimates'''
    res_numbers = range(0, 100)
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    
    
    # plt.figure()
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True)
    
    ax1.hlines(4 * np.pi * 5, 0, 100, linewidth=2, color="k")
    
    
    inds = np.argsort(res_vec[res_numbers, 0])
    ax1.errorbar(res_numbers, res_vec[inds, 0], yerr=res_vec[inds, 0] - unc_vec[inds, 0, 0], fmt="ro")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh Size", fontsize=18)
    ax1.title.set_text("BootsTrap over Test Data Set")
    
    inds = np.argsort(res_vec[res_numbers, 1])
    ax2.errorbar(res_numbers, res_vec[inds, 1], yerr=res_vec[inds, 1] - unc_vec[inds, 1, 0], fmt="ro")
    ax2.hlines(0.006, 0, 100, linewidth=2)
    ax2.set_ylabel("L", fontsize=18)
    # ax2.legend()
    
    inds = np.argsort(res_vec[res_numbers, 2])
    ax3.errorbar(res_numbers, res_vec[inds, 2], yerr=res_vec[inds, 2] - unc_vec[inds, 2, 0], fmt="ro")
    ax3.hlines(0.1, 0, 100, linewidth=2)
    ax3.set_ylabel("Barrier", fontsize=18)
    
    inds = np.argsort(res_vec[res_numbers, 3])
    ax4.errorbar(res_numbers, res_vec[inds, 3], yerr=res_vec[inds, 3] - unc_vec[inds, 3, 0], fmt="ro")
    ax4.hlines(0.52, 0, 100, linewidth=2)
    ax4.set_ylim([0.5, 0.53])
    ax4.set_ylabel("SS", fontsize=18)
    # plt.xticks([10,35,60,85], ['1x1', '2x2', '3x3','4x4'])
    
    plt.xlabel("Dataset")
    plt.show()
    
def hz_barrier_bts(folder, subfolder, method=2):
    '''Plot BootsTrap Estimates for HZ.
    Same as normal one but with a few modificatins.'''
    
    res_numbers = range(0, 100)
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method, subfolder=subfolder) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method, subfolder=subfolder) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    
    
    # plt.figure()
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True)
    
    # ax1.hlines(4 * np.pi * 5, 0, 100, linewidth=2, color="k")
    
    
    inds = np.argsort(res_vec[res_numbers, 0])
    ax1.errorbar(res_numbers, res_vec[inds, 0], yerr=res_vec[inds, 0] - unc_vec[inds, 0, 0], fmt="ro")
    inds0 = np.where(inds == 0)[0][0]  # Get the index of the first individual
    ax1.errorbar(res_numbers[inds0], res_vec[0, 0],
                 yerr=res_vec[0, 0] - unc_vec[0, 0, 0], fmt="bo")  # Plot first Data Set
    ax1.set_ylim([5, 400])
    ax1.set_ylabel("Nbh Size", fontsize=18)
    ax1.hlines(res_vec[0, 0], 0, 100, linewidth=2, color="b")
    ax1.title.set_text("Bootstrap over Hybrid Zone Data")
    
    inds = np.argsort(res_vec[res_numbers, 1])
    ax2.errorbar(res_numbers, res_vec[inds, 1], yerr=res_vec[inds, 1] - unc_vec[inds, 1, 0], fmt="ro")
    ax2.hlines(res_vec[0, 1], 0, 100, linewidth=2, color="blue")
    ax2.set_ylim([0, 0.1])
    ax2.set_ylabel("L", fontsize=18)
    inds0 = np.where(inds == 0)[0][0]  # Get the index of the first individual
    ax2.errorbar(res_numbers[inds0], res_vec[0, 1],
                 yerr=res_vec[0, 1] - unc_vec[0, 1, 0], fmt="bo")  # Plot first Data Set
    
    # ax2.legend()
    
    inds = np.argsort(res_vec[res_numbers, 2])
    ax3.errorbar(res_numbers, res_vec[inds, 2], yerr=res_vec[inds, 2] - unc_vec[inds, 2, 0], fmt="ro")
    ax3.hlines(res_vec[0, 2], 0, 100, linewidth=2)
    ax3.set_ylim([0, 5])
    ax3.set_ylabel("Barrier", fontsize=18)
    inds0 = np.where(inds == 0)[0][0]  # Get the index of the first individual
    ax3.errorbar(res_numbers[inds0], res_vec[0, 2],
                 yerr=res_vec[0, 2] - unc_vec[0, 2, 0], fmt="bo")  # Plot first Data Set
    
    
    
    inds = np.argsort(res_vec[res_numbers, 3])
    ax4.errorbar(res_numbers, res_vec[inds, 3], yerr=res_vec[inds, 3] - unc_vec[inds, 3, 0], fmt="ro")
    ax4.hlines(res_vec[0, 3], 0, 100, linewidth=2)
    ax4.set_ylim([0.52, 0.58])
    ax4.set_ylabel("SS", fontsize=18)
    inds0 = np.where(inds == 0)[0][0]  # Get the index of the first individual
    ax4.errorbar(res_numbers[inds0], res_vec[0, 3],
                 yerr=res_vec[0, 3] - unc_vec[0, 3, 0], fmt="bo")  # Plot first Data Set
    
    # plt.xticks([10,35,60,85], ['1x1', '2x2', '3x3','4x4'])
    
    plt.xlabel("Dataset")
    plt.show()
    
def barrier_var_pos(folder, subfolder0, subfolder1, subfolder2, method=2):
    '''Plots barriers at various positions'''
    res_numbers = range(0, 100)
    
    # Load the results:
    res_vec = -1.0 * np.ones((len(res_numbers), 3))  # Set empty array to -1.
    
    res_vec0 = np.array([load_pickle_data(folder, i, 0, method, subfolder=subfolder0) for i in res_numbers])[:, 2]
    res_vec1 = np.array([load_pickle_data(folder, i, 0, method, subfolder=subfolder1) for i in res_numbers])[:, 2]
    res_vec2 = np.array([load_pickle_data(folder, i, 0, method, subfolder=subfolder2) for i in res_numbers])[:, 2]
    
    res_vec[:, 0] = np.where(res_vec0 > 1, 1, res_vec0)
    res_vec[:, 1] = np.where(res_vec1 > 1, 1, res_vec1)
    res_vec[:, 2] = np.where(res_vec2 > 1, 1, res_vec2)
    
    
    # unc_vec0 = np.array([load_pickle_data(folder, i, 1, method, subfolder=subfolder0) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(3):
            print("Parameter: %i" % j)
            print("Value: %f " % res_vec[l, j])
            
    
    
    # plt.figure()
    f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True)
    
    # ax1.hlines(4 * np.pi * 5, 0, 100, linewidth=2, color="k")
    
    
    inds = np.argsort(res_vec[res_numbers, 0])
    ax1.plot(res_numbers, res_vec[inds, 0], "ro")
    inds0 = np.where(inds == 0)[0][0]  # Get the index of the first individual
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel("-1000m", fontsize=18)
    # ax1.hlines(res_vec[0, 0], 0, 100, linewidth=2, color="b")
    ax1.title.set_text("BootsTrap over Hybrid Zone Data")
    ax1.plot(res_numbers[inds0], res_vec[0, 0], "bo")  # Plot first Data Set
    
    inds = np.argsort(res_vec[res_numbers, 1])
    ax2.plot(res_numbers, res_vec[inds, 1], "ro")
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel("Center", fontsize=18)
    inds0 = np.where(inds == 0)[0][0]  # Get the index of the first individual
    ax2.plot(res_numbers[inds0], res_vec[0, 1], "bo")  # Plot first Data Set
    
    # ax2.legend()
    
    inds = np.argsort(res_vec[res_numbers, 2])
    ax3.plot(res_numbers, res_vec[inds, 2], "ro")
    # ax3.hlines(res_vec[0, 2], 0, 100, linewidth=2)
    ax3.set_ylim([0, 1.1])
    ax3.set_ylabel("1000m", fontsize=18)
    inds0 = np.where(inds == 0)[0][0]  # Get the index of the first individual
    ax3.plot(res_numbers[inds0], res_vec[0, 2], "bo")  # Plot first Data Set

    # plt.xticks([10,35,60,85], ['1x1', '2x2', '3x3','4x4'])

    plt.xlabel("Dataset")
    plt.show()
    
def plot_multi_barrier_pos(position_path, result_folder, subfolder):
    '''Loads and plots the multiple Bootstrap per Position Estimates.'''
    print("ToDo")  
    
def plot_IBD_bootstrap(position_path, genotype_path, result_folder, subfolder,
                       bins=50, p=0.5, nr_bootstraps=10):
    '''Plot IBD of real data and bootstrapped real data (over loci).
    Load from Result Folder and plot binned IBD from HZ'''  
    
    # res_numbers = range(0, 100)
    positions = np.loadtxt(position_path, delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
    genotypes = np.loadtxt(genotype_path, delimiter='$').astype('float64')
    
    
    # Calculate best Estimates
    bin_dist, bin_corr0, stand_errors, corr_factor_e = calc_bin_correlations(positions, genotypes, bins=bins)
    
    # Calculates Estimates for left and right half of HZ
    inds_l = np.where(positions[:, 0] < 0)[0]
    inds_r = np.where(positions[:, 0] > 0)[0]
    
    bin_dist_l, bin_corr_l, stds_l, _ = calc_bin_correlations(positions[inds_l, :],
                                    genotypes[inds_l, :], bins=bins / 3.0, correction=False)
    bin_dist_r, bin_corr_r, stds_r, _ = calc_bin_correlations(positions[inds_r, :],
                                    genotypes[inds_r, :], bins=bins / 3.0, correction=False)
    
    # Take the same correction as Overall
    bin_corr_l = bin_corr_l - corr_factor_e 
    bin_corr_r = bin_corr_r - corr_factor_e

    
    # Load the Best-Fit Estimates:
    res_vec = load_pickle_data(result_folder, 0, 0, method=2, subfolder=subfolder)
    print("Best Fit Results:")
    print(res_vec)
    
    
    
    res_vec[ -1] = 0.0  # 605  # Set the mean to mean
    # else: res_vec[:,-1] = 0.0605
    # print(res_vec)
    
    # plt.plot()
    # plt.hist(res_vec[:, 0])
    # plt.xlabel("Histogram of NBH Estimates")
    # plt.show()
    
    
    # Do the Bootstrapping
    nr_genotypes = np.shape(genotypes)[1]
    f_estimates = np.zeros((nr_bootstraps, len(bin_dist)))  # Empty Vector for the F-Estimates
    
    for i in range(nr_bootstraps):
        print("Doing BootsTrap Nr. %i" % i)
        r_ind = np.random.randint(nr_genotypes, size=nr_genotypes)  # Get Indices for random resampling
        gtps_sample = genotypes[:, r_ind]  # Do the actual Bootstrap; pick the columns
        _, bin_corr, _, _ = calc_bin_correlations(positions, gtps_sample, bins=bins)
        f_estimates[i, :] = bin_corr
    
    upper = np.percentile(f_estimates, 97.5, axis=0)  # Calculate Upper Bound
    lower = np.percentile(f_estimates, 2.5, axis=0)  # Calculate Lower Bound
    
    
    x_plot = np.linspace(min(bin_dist), max(bin_dist) / 2.0, 100)
    coords = [[0, 0], ] + [[0, i] for i in x_plot]  # Coordsvector
    
    # Calculate the best fit
    res_true = res_vec
    KC = fac_kernel("DiffusionK0")
    KC.set_parameters([res_true[0], res_true[1], 1.0, res_true[3]])  # Nbh Sz, Mu0, t0, ss. Sets known Parameters #[4*np.pi*6, 0.02, 1.0, 0.04]
    kernel = KC.calc_kernel_mat(coords)
    corr_fit = kernel[0, 1:] - np.mean(kernel[0, 40:60])  # Substract the middle Value
    
    
    # scale_factor=50   # Scaling factor for x-Distance
    scale_factor = 50
    
    plt.figure()
    plt.errorbar(bin_dist[:bins / 2] * scale_factor, bin_corr0[:bins / 2], stand_errors[:bins / 2], fmt='ro', label="Binwise Correlation")
    # plt.plot(x_plot, C + k * np.log(x_plot), 'g', label="Fitted Log Decay")
    
    # plt.plot(x_plot, y_fit, 'yo', label="Least square fit.")
    plt.plot(x_plot * scale_factor, kernel[0, 1:], 'b-', linewidth=2, label="Best Fit Estimates")
    plt.plot(bin_dist[:bins / 2] * scale_factor, lower[:bins / 2], 'k-', label="Bootstrap 2.5 %")
    plt.plot(bin_dist[:bins / 2] * scale_factor, upper[:bins / 2], 'k-', label="Bootstrap 97.5 %")
    
    # Plot IBD of Left and Right half of HZ:
    nr_bins_l = 20
    nr_bins_r = 24
    plt.plot(bin_dist_l[:nr_bins_l] * scale_factor, bin_corr_l[:nr_bins_l], 'y-', label="Left Half HZ")  # stds_l[:nr_bins_l]
    plt.plot(bin_dist_r[:nr_bins_r] * scale_factor, bin_corr_r[:nr_bins_r], 'm-', label="Right Half HZ")  # stds_r[:nr_bins_r]
    
    # plt.axhline(np.mean(bin_corr), label="Mean Value", color='k', linewidth=2)
    # plt.annotate(r'$\bar{N_b}=%.4G \pm %.2G$' % (Nb_est, Nb_std) , xy=(0.6, 0.7), xycoords='axes fraction', fontsize=15)
    plt.legend()
    plt.ylabel("F / Correlation")
    plt.xlabel("Distance [m]")
    plt.title("IBD in HZ")
    # plt.ylim([0,0.05])
    # plt.xscale("log")
    plt.show()
    
def plot_IBD_across_Zone(position_path, genotype_path, bins=30, max_dist=4.0, nr_bootstraps=100):
    '''Plot IBD in bins across hybrid zones.
    Includes BT estimates; so that one knows about the order of the error.'''
    
    # ## First Produce the Data:
    # Load the raw data:
    position_list = np.loadtxt(position_path, delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
    nr_inds = len(position_list)
    
    genotypes = np.loadtxt(genotype_path, delimiter='$').astype('float64')
    nr_genotypes = np.shape(genotypes)[1]
    x_coords = position_list[:, 0]  # Get the x_coordinates
    x_min = np.min(x_coords) - 0.000001  # So that everything falls into something
    x_max = np.max(x_coords) + 0.000001
    
    
        
    # Get the x_Bins:
    x_bins = np.linspace(x_min, x_max + 0.001, num=bins + 1)
    x_coords_mean = (x_bins[1:] + x_bins[:-1]) / 2.0  # Calculate the mean x-Values
    delta_x = x_bins[1] - x_bins[0]
    
    print("Calculating Distance Matrix...")
    tic = time()
    dist_mat = np.linalg.norm(position_list[:, None] - position_list, axis=2)
    close_inds = dist_mat < max_dist  # Calculate Boolean Matrix of all nearby Individuals
    toc = time()
    print("Runtime Dist_Mat: %.4f" % (toc - tic))
    nr_nearby_inds = [np.sum(nearby_inds) for nearby_inds in close_inds]  # Calculate the total Number of nearby Individuals
    power = np.array(nr_nearby_inds) ** 2  # Number of Pairwise comparisons grows quadratically; so thus the weight factor
    
    
    def get_f_vec(position_list, genotype_mat, x_bins, bin_size=5):
        '''Given Nr of Bins and Bin Size;
        calculate mean F within bins across HZ'''
        nr_inds = len(position_list)
        f_nb = -100 * np.ones(nr_inds)
        
        # Calculate the deviation from mean of all nearby individuals
        f_mean_tot = np.array([mean_kinship_coeff(genotype_mat[nearby_inds, :], p_mean=0.5) for nearby_inds in close_inds])  # Get F-Mean over total HZ
        assert(len(power) == nr_inds)  # Sanity Check
        
        # Do the binning:
        f_vec = -100 * np.ones(bins)
        for i in range(bins):
            # Where to cut off:
            bin_minx = x_bins[i]
            bin_maxx = x_bins[i + 1]
            
            # All inds that fall into the x-bin:
            inds = np.where((x_coords >= bin_minx) & (x_coords <= bin_maxx))[0]
            # Calculate F-Mean withing this bin
            f_vec[i] = np.sum(f_mean_tot[inds] * power[inds]) / np.sum(power[inds])
            
        f_tot = mean_kinship_coeff(genotype_mat, p_mean=0.5)  # Calculate the total Mean
        return f_vec - f_tot
    
    
    
    f_res = -np.ones((bins, nr_bootstraps))  # Set everything to minus 1; so that errors are realized
    
    f_res[:, 0] = get_f_vec(position_list, genotypes, x_bins)
    
    for i in xrange(1, nr_bootstraps):
        print("Calculating BootsTrap Nr. %i" % i)
        r_gtps_ind = np.random.randint(nr_genotypes, size=nr_genotypes)  # Get Indices for random resampling
        gtps_sample = genotypes[:, r_gtps_ind]  # Do the actual Bootstrap; pick the columns
        
        f_res[:, i] = get_f_vec(position_list, gtps_sample, x_bins)
        

    # Now do the Plot:    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    ax1.set_title("Excess F within %i m" % (max_dist * 50))
    ax1.plot(x_coords_mean, f_res[:, 1:], 'go', alpha=0.2)
    ax1.plot(x_coords_mean, f_res[:, 0], 'ro', label="Empricial Values")
    ax1.hlines(0, min(position_list[:, 0]), max(position_list[:, 0]), alpha=0.8)
    ax1.legend(loc="upper right")
    
    f_mean_tot = np.array([mean_kinship_coeff(genotypes[nearby_inds, :], p_mean=0.5) for nearby_inds in close_inds])  # Get F-Mean over total HZ
    ax2.scatter(position_list[:, 0], position_list[:, 1], c=f_mean_tot)  # c = nr_nearby_inds
    for x in x_bins:
        plt.vlines(x, min(position_list[:, 1]), max(position_list[:, 1]), alpha=0.6)
    plt.show()
    
    
def multi_pos_plot(folder, method_folder, res_numbers=range(0, 300), nr_bts=20, barrier_pos=500.5):
    '''Plots multiple Barrier positions throughout the area.
    Upper Plot: For every Barrier-Position plot the most likely estimate -
    as well as Bootstrap Estimates around it! Lower Plot: Plot the Positions of the
    Demes/Individuals'''
    
    # Load the Results
    res_vec = np.array([load_pickle_data(folder, i, 0, subfolder=method_folder) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, subfolder=method_folder) for i in res_numbers])
    
    # Put the Barrier Estimates >1 to 1:
    res_vec[res_numbers, 2] = np.where(res_vec[res_numbers, 2] > 1, 1, res_vec[res_numbers, 2])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(3):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    # Mean Estimates:
    mean_inds = range(0, np.max(res_numbers), nr_bts)
    res_mean = res_vec[mean_inds, :]  
    print(res_mean)   
            
    # Load the Barrier Positions:
    barrier_fn = "barrier_pos.csv"
    path = folder + method_folder + barrier_fn
    barrier_pos = np.loadtxt(path, delimiter='$').astype('float64')
    
    # Add upper bound here if lower number of results are available
    barrier_pos = barrier_pos[:]  
    
    print("Barrier Positions loaded: ")
    print(barrier_pos)
    barr_pos_plot = [val for val in barrier_pos for _ in xrange(nr_bts)]
    # print(barr_pos_plot)
    
    # Load the Position File:
    path_pos = folder + "mb_pos_coords00.csv"
    position_list = np.loadtxt(path_pos, delimiter='$').astype('float64')
    print("Position List loaded: %i Entries" % len(position_list))
    
    
    # Do the plotting:
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # Plot the Nbh Estiamtes:
    ax1.plot(barr_pos_plot, res_vec[res_numbers, 0], 'ko', label="Bootstrap Estimate", alpha=0.5)
    ax1.plot(barrier_pos, res_mean[:, 0], 'go', label="Mean Estimate")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.hlines(4 * np.pi * 5, min(position_list[:, 0]), max(position_list[:, 0]), linewidth=2, color="y")
    # ax1.title.set_text("No Barrier")
    ax1.legend(loc="upper right")
    
    ax2.plot(barr_pos_plot, res_vec[res_numbers, 2], 'ko', label="Bootstrap Estimate", alpha=0.5)
    ax2.plot(barrier_pos, res_mean[:, 2], 'go', label="Mean Estimate")
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Barrier", fontsize=18)
    ax2.hlines(0.05, min(position_list[:, 0]), max(position_list[:, 0]), linewidth=2, color="y")
    
    # ax1.title.set_text("No Barrier")
    # ax2.legend(loc="upper right")
    
    # Plot the Positions:
    ax3.scatter(position_list[:, 0], position_list[:, 1])  # c = nr_nearby_inds
    ax3.set_xlabel("x-Position", fontsize=18)
    ax3.set_ylabel("y-Position", fontsize=18)
    for x in barrier_pos:
        ax3.vlines(x, min(position_list[:, 1]), max(position_list[:, 1]), alpha=0.8, linewidth=3)
    ax3.vlines(500.5, min(position_list[:, 1]), max(position_list[:, 1]), color="red", linewidth=6)
    plt.show()
    
    
######################################################
if __name__ == "__main__":
    '''Here one chooses which Plot to do:'''
    # multi_nbh_single(multi_nbh_folder, method=2)
    # multi_nbh_single(multi_nbh_gauss_folder, method=2)
    # multi_ind_single(multi_ind_folder, method=2)
    # multi_loci_single(multi_loci_folder, method=2)
    multi_barrier_single(multi_barrier_folder, method=2)  # Mingle with the above for different Barrier Strengths.
    # multi_secondary_contact_single(secondary_contact_folder_b, method=2)
    # multi_secondary_contact_all(secondary_contact_folder, secondary_contact_folder_b, method=2)
    
    # cluster_plot(cluster_folder, method=2)
    # boots_trap("./bts_folder_test/", method=2)   # Bootstrap over Test Data Set: Dataset 00 from cluster data-set; clustered 3x3
    # ll_barrier("./barrier_folder1/")
    # multi_pos_plot(multi_pos_syn_folder, met2_folder)
    
    # Plots for Hybrid Zone Data
    # hz_barrier_bts(hz_folder, "barrier2/")  # Bootstrap over all Parameters for Barrier Data
    # barrier_var_pos(hz_folder, "barrier18p/", "barrier2/", "barrier20m/", method=2) # Bootstrap over 3 Barrier pos
    # plot_IBD_bootstrap("./Data/coordinatesHZall2.csv", "./Data/genotypesHZall2.csv", hz_folder, "barrier2/")    # Bootstrap in HZ to produce IBD fig
    # plot_IBD_bootstrap("./nbh_folder/nbh_file_coords30.csv", "./nbh_folder/nbh_file_genotypes30.csv", hz_folder, "barrier2/")  # Bootstrap Random Data Set
    # plot_IBD_across_Zone("./Data/coordinatesHZall0.csv", "./Data/genotypesHZall0.csv", bins=20, max_dist=4, nr_bootstraps=200)  # Usually the dist. factor is 50
    
    
    
