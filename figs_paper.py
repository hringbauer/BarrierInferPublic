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
from matplotlib import colors
from matplotlib import cm
import itertools
import os
from scipy.special import kv as kv  # Import Bessel Function
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from analysis import kinship_coeff, calc_f_mat, calc_h_mat, group_inds  # Import functions from Analysis
from scipy.stats import binned_statistic
from scipy.stats import sem
from kernels import fac_kernel  # Factory Method which yields Kernel Object
from grid import Grid
from scipy.special import erfc


multi_nbh_folder = "./nbh_folder/"
multi_nbh_gauss_folder = "./nbh_folder_gauss/"
multi_barrier_folder = "./barrier_folder2/"
cluster_folder = "./cluster_folder/"
hz_folder = "./hz_folder/"
multi_ind_folder = "./multi_ind_nr/"
multi_ind_folder1 = "./multi_ind_nr1/"
multi_loci_folder = "./multi_loci_nr/"
secondary_contact_folder = "./multi_2nd/"
secondary_contact_folder_b = "./multi_2nd_b/"  # With a 0.05 Barrier
multi_pos_syn_folder = "./multi_barrier_synth/"
multi_pos_hz_folder = "./multi_barrier_hz/"

met2_folder = "method2/"

# Some Default Colors:
c_dots = "crimson"
c_lines = "g"
c0, c1, c2, c3 = "yellow", "orange", "crimson", "purple"

######################################################################################
#### First some helper functions

def mean_kinship_coeff(genotype_mat, p_mean=0.5):
        '''Calculate the mean Kinship coefficient for a 
        Genotype_matrix; given some mean Vector p_mean'''
        p_mean_emp = np.mean(genotype_mat, axis=0)  # Calculate the mean allele frequencies
        f_vec = (p_mean_emp - p_mean) * (p_mean_emp - p_mean) / (p_mean * (1 - p_mean))  # Calculate the mean f per locus
        f = np.mean(f_vec)  # Calculate the overall mean f
        return f

def check_if_data(folder, i, method=2, subfolder=None):
    '''Check if Data-Set is there. Return 1 if yes. Return 0 if no.'''
    if subfolder == None:
            subfolder_meth = "method" + str(method) + "/"  # Sets subfolder on which Method to use.
    else: subfolder_meth = subfolder
    
    path = folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
    
    exist = os.path.exists(path)
    return exist

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
        
def give_result_stats(folder, res_vec=range(100), method=2, subfolder=None, bootstraps=20):
    '''Helper Function that gives stats of results'''
    res_vec = np.array([load_pickle_data(folder, i, 0, method, subfolder) for i in res_vec])
    _, nr_params = np.shape(res_vec)
    
    means = np.mean(res_vec, axis=0)
    median = np.median(res_vec, axis=0)
    std = np.std(res_vec, axis=0)
    upper = np.percentile(res_vec, 97.5, axis=0)  # Calculate Upper Bound
    lower = np.percentile(res_vec, 2.5, axis=0)  # Calculate Lower Bound
    
    for i in xrange(nr_params):
        print("\nParameter: %i" % i)
        print("Mean:\t\t\t %g" % means[i])
        print("Median:\t\t\t %g" % median[i])
        print("SD:\t\t\t %g" % std[i])
        print("2.5 Percent Quantile:\t %g" % lower[i])
        print("97.5 Percent Quantile:\t %g" % upper[i])
        
    plt.figure()
    plt.boxplot(res_vec[:, 0])
    plt.title("Nbh Size")
    plt.show()
    
    nr_bootstraps = len(res_vec) / bootstraps
    for i in xrange(nr_bootstraps):
        print("Estimate %i:" % i)
        print(res_vec[i * bootstraps])
    
def print_run_params(folder, info_file_name="parameters.p"):
    '''Load the Parameters of the scenario, that were saved with pickle'''
    path = folder + info_file_name
    
    param_names, params, additional_info = pickle.load(open(path, "rb"))
    print(param_names)
    print(params)
    
    print("Parameters Values of this Scenario:")
    for i in xrange(len(param_names)):
        try:
            print("Parameters %s: %.4g" % (param_names[i], params[i]))
        except:
            continue
        
        
    print("Additional Info:")
    print(additional_info)
    
def calc_bin_homos(positions, genotypes, bins=25, max_dist=0, print_res=False):
    '''Calculate the average Number of Homozygotes in Distance Bins'''
    distance = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Empty container
    correlation = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Container for correlation
    entry = 0
    
    h_mat = calc_h_mat(genotypes)  # Does the whole calculation a bit quicker
    dist_mat = np.linalg.norm(positions[:, None] - positions, axis=2)  # Calculate the whole Distance Matrix
    
    # print("positions: ")
    # print(positions)
    # print("Max Dist.: %.4f" % max_dist)
    # Extract the upper triangular Matrix of the Estimates:
    nr_inds = np.shape(genotypes)[0]
    xi, yi = np.tril_indices(nr_inds, -1)  # Gives the corresponding indices to array in matrix of thr. sharing
    
    
    pw_h = h_mat[xi, yi]
    pw_dist = dist_mat[xi, yi]
    
    
    if max_dist > 0:
        inds_dist = np.where(pw_dist < max_dist)[0]
        assert(len(inds_dist) > 0)
        pw_h = pw_h[inds_dist]
        pw_dist = pw_dist[inds_dist]
        
    bin_dist, bin_corr, stand_errors, _ = bin_correlations(pw_dist, pw_h, bins=bins)
    if print_res == True:
        print(bin_dist)
        print(bin_corr)
    return bin_dist, bin_corr, stand_errors
    
def calc_bin_correlations(positions, genotypes, bins=25, p=0.5, correction=True):
    '''Helper function Calculates Correlations and bins them'''
    # Load the data:
    # position_list = position_list / 50.0  # Normalize; for position_list and genotype Matrix of HZ data!
    
    distance = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Empty container
    correlation = np.zeros(len(positions) * (len(positions) - 1) / 2)  # Container for correlation
    entry = 0
    
    f_mat = calc_f_mat(genotypes, p)  # Does the whole calculation a bit quicker
    dist_mat = np.linalg.norm(positions[:, None] - positions, axis=2)  # Calculate the whole Distance Matrix
    
    # Calculate everything:
    for (i, j) in itertools.combinations(range(len(genotypes[:, 0])), r=2):
        # distance[entry] = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))  # Calculate the pairwise distance
        distance[entry] = dist_mat[i, j]
        # correlation[entry] = kinship_coeff(genotypes[i, :], genotypes[j, :], p)  # Kinship coeff. per pair, averaged over loci  
        correlation[entry] = f_mat[i, j]
        entry += 1 
          
    # Now bin them:
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
        ind_min = int(bins * 1 / 3)
        ind_max = int(bins * 1 / 2)
        print("Min Dist for Correction: %.4f" % bin_edges[ind_min])
        print("Max Dist for correction: %.4f" % bin_edges[ind_max])
        corr_factor = np.mean(bin_corr[ind_min:ind_max]) 
        bin_corr = bin_corr - corr_factor
    return bin_dist, bin_corr, stand_errors, corr_factor 

def argsort_bts(x, nr_bts):
    '''Arg Sorts a Vector within Bootstraps.
    Return: 
    Sorted indices
    Indices of true value'''
    assert(len(x) % nr_bts == 0)  # Check whether Bootstraps diveds nr of given array
    
    inds_sorted = np.zeros(len(x)).astype("int")
    true_inds = []
    
    # Iterate over batches
    for i in range(0, len(x), nr_bts):
        inds = range(i, i + nr_bts)
        
        inds_batch = np.argsort(x[inds])
        inds_sorted[inds] = inds_batch + i  # Return the sorted indices shifted.
        true_ind = np.where(inds_batch == 0)[0][0]  # Get the index of the true value.
        
        true_inds.append(true_ind + i)
        
    true_inds = np.array(true_inds)
    return (inds_sorted, true_inds)
    
######################################################################################
######################################################################################
######################################################################################
# Do the actual Plots:


def multi_nbh_single(folder, method, res_numbers=range(100)):
    '''Print several Neighborhood Sizes simulated under the model - using one method'''
    
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
    
    ax1.hlines(4 * np.pi, 0, 25, linewidth=3, color=c_lines)
    ax1.hlines(4 * np.pi * 5, 25, 50, linewidth=3, color=c_lines)
    ax1.hlines(4 * np.pi * 9, 50, 75, linewidth=3, color=c_lines)
    ax1.hlines(4 * np.pi * 13, 75, 100, linewidth=3, color=c_lines)
    ax1.errorbar(res_numbers, res_vec[:, 0], yerr=res_vec[:, 0] - unc_vec[:, 0, 0], fmt="o", label="Nbh", color=c_dots)
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.title.set_text("Method: %s" % str(method))
    # ax1.legend()
    
    ax2.errorbar(res_numbers, res_vec[:, 1], yerr=res_vec[:, 1] - unc_vec[:, 1, 0], fmt="o", label="L", color=c_dots, zorder=0)
    ax2.hlines(0.006, 0, 100, linewidth=2, color=c_lines, zorder=1)
    ax2.set_ylim([0, 0.03])
    ax2.set_ylabel("L", fontsize=18)
    # ax2.legend()
    
    ax3.errorbar(res_numbers, res_vec[:, 2], yerr=res_vec[:, 2] - unc_vec[:, 2, 0], fmt="o", label="ss", color=c_dots, zorder=0)
    ax3.hlines(0.52, 0, 100, linewidth=2, color=c_lines, zorder=1)
    ax3.set_ylabel("SS", fontsize=18)
    # ax3.legend()
    plt.xlabel("Dataset")
    plt.show()
    
    
def multi_nbh_all(folder, res_numbers=range(100)):
        '''Visualizes the estimates of all three Methods
        Method0: GRF Method1: ML, Method2: Curve Fit. NEEEDS LOVE AND CHECKIN'''
    
        # res_numbers = range(0,86)
        # res_numbers = range(10, 13) + range(30, 33) + range(60, 61)  # + range(80, 83)
        # res_numbers = range(30,31)# To analyze Dataset 30
        lw = 3
        ul_nb = 350  # Upper Limit Neighborhood Size
        # Load the Data for all three methods
        res_vec0 = np.array([load_pickle_data(folder, i, 0, method=0) for i in res_numbers])
        unc_vec0 = np.array([load_pickle_data(folder, i, 1, method=0) for i in res_numbers])
        
        res_vec1 = np.array([load_pickle_data(folder, i, 0, method=1) for i in res_numbers])
        unc_vec1 = np.array([load_pickle_data(folder, i, 1, method=1) for i in res_numbers])
        
        res_vec2 = np.array([load_pickle_data(folder, i, 0, method=2) for i in res_numbers])
        unc_vec2 = np.array([load_pickle_data(folder, i, 1, method=2) for i in res_numbers])
        
        f, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3, 3, sharex=True, figsize=(11, 11))
        
        ax1.hlines(4 * np.pi, 0, 25, linewidth=lw, color=c_lines)
        ax1.hlines(4 * np.pi * 5, 25, 50, linewidth=lw, color=c_lines)
        ax1.hlines(4 * np.pi * 9, 50, 75, linewidth=lw, color=c_lines)
        ax1.hlines(4 * np.pi * 13, 75, 100, linewidth=lw, color=c_lines)
        ax1.plot(res_numbers, res_vec0[:, 0], "ko", label="Nbh", color=c_dots, zorder=0)
        ax1.set_ylim((0, ul_nb))
        ax1.set_ylabel("Nbh", fontsize=18)
        ax1.set_title("Method 1", fontsize=18)

        ax2.plot(res_numbers, res_vec0[:, 1], "ko", label="L", color=c_dots, zorder=0)
        ax2.hlines(0.006, 0, 100, linewidth=lw, color=c_lines)
        ax2.set_ylabel("L", fontsize=18)
        ax2.set_ylim((0, 0.03))
        
        ax3.plot(res_numbers, res_vec0[:, 2], "ko", label="ss", color=c_dots, zorder=0)
        ax3.hlines(0.04, 0, 100, linewidth=lw, color=c_lines)
        ax3.set_ylim((0.01, 0.08))
        ax3.set_ylabel("Var. Par.", fontsize=18)
        
        ax4.hlines(4 * np.pi, 0, 25, linewidth=lw, color=c_lines)
        ax4.hlines(4 * np.pi * 5, 25, 50, linewidth=lw, color=c_lines)
        ax4.hlines(4 * np.pi * 9, 50, 75, linewidth=lw, color=c_lines)
        ax4.hlines(4 * np.pi * 13, 75, 100, linewidth=lw, color=c_lines)
        ax4.plot(res_numbers, res_vec1[:, 0], "ko", label="Nbh", color=c_dots, zorder=0)
        ax4.set_ylim((0, ul_nb))
        ax4.set_title("Method 2", fontsize=18)
        ax4.set_yticks([])
        
        ax5.plot(res_numbers, res_vec1[:, 1], "ko", label="L", color=c_dots, zorder=0)
        ax5.hlines(0.006, 0, 100, linewidth=lw, color=c_lines, zorder=1)
        ax5.set_ylim((0, 0.03))
        ax5.set_yticks([])
        
        ax6.plot(res_numbers, res_vec1[:, 2], "ko", label="Variance Parameter", color=c_dots, zorder=0)
        ax6.hlines(0.01, 0, 100, linewidth=lw, color=c_lines, zorder=1)
        ax6.set_ylim((0.001, 0.015))
        # ax6.tick_params(direction='out', pad=-30)
        ax6.get_yaxis().set_tick_params(which='both', direction='out', pad=-35)
        # ax6.yaxis.set_ticklabels(longlabs,position=(0.06,0))
        # ax6.set_yticks([])
        
        ax7.hlines(4 * np.pi, 0, 25, linewidth=lw, color=c_lines)
        ax7.hlines(4 * np.pi * 5, 25, 50, linewidth=lw, color=c_lines)
        ax7.hlines(4 * np.pi * 9, 50, 75, linewidth=lw, color=c_lines)
        ax7.hlines(4 * np.pi * 13, 75, 100, linewidth=lw, color=c_lines)
        ax7.plot(res_numbers, res_vec2[:, 0], "ko", label="Nbh", color=c_dots, zorder=0)
        ax7.set_ylim((0, ul_nb))
        ax7.set_title("Method 3", fontsize=18)
        ax7.set_yticks([])
        # ax1.legend()
        
        ax8.plot(res_numbers, res_vec2[:, 1], "ko", label="L", color=c_dots, zorder=0)
        ax8.hlines(0.006, 0, 100, linewidth=lw, color=c_lines, zorder=1)
        ax8.set_ylim((0, 0.03))
        ax8.set_yticks([])
        
        ax9.plot(res_numbers, res_vec2[:, 2], "ko", label="Variance Parameter", color=c_dots, zorder=0)
        ax9.hlines(0.52, 0, 100, linewidth=lw, color=c_lines, zorder=1)
        ax9.set_ylim((0.45, 0.55))
        ax9.yaxis.tick_right()
        # ax9.set_yticks([])
        plt.gcf().text(0.5, 0.04, "Data Set Nr.", ha="center", fontsize=18)  # Set the x-Label
        plt.show()
        # ax3.legend()
    
def multi_barrier_single(folder, method, barrier_strengths=[0, 0.05, 0.1, 0.15]):
    '''Print Estimates from several Barrier strength from Folder.'''
    # Define the Result Numbers:
    res_numbers0 = range(0, 25)
    res_numbers1 = range(25, 50)
    res_numbers2 = range(50, 75)
    res_numbers3 = range(75, 100)
    res_numbers = res_numbers0 + res_numbers1 + res_numbers2 + res_numbers3
    
    lw = 3
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    
    
    # plt.figure()
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True, figsize=(6, 6))
    ax1.hlines(4 * np.pi * 5, res_numbers[0], res_numbers[-1], linewidth=lw, color=c_lines)
    ax1.plot(res_numbers, res_vec[:, 0], "bo", color=c_dots, label="Nbh")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh", fontsize=18)
    # ax1.title.set_text("Method: %s" % str(method))
    # ax1.legend()
    
    ax2.plot(res_numbers, res_vec[:, 1], "go", color=c_dots, label="m")
    ax2.hlines(0.006, 0, 100, linewidth=lw, color=c_lines)
    ax2.set_ylabel("m", fontsize=18)
    # ax2.legend()
    
    ax3.plot(res_numbers, res_vec[res_numbers, 2], "yo", color=c_dots)
    ax3.hlines(barrier_strengths[0], res_numbers0[0], res_numbers0[-1], linewidth=lw, color=c_lines)
    ax3.hlines(barrier_strengths[1], res_numbers1[0], res_numbers1[-1], linewidth=lw, color=c_lines)
    ax3.hlines(barrier_strengths[2], res_numbers2[0], res_numbers2[-1], linewidth=lw, color=c_lines)
    ax3.hlines(barrier_strengths[3], res_numbers3[0], res_numbers3[-1], linewidth=lw, color=c_lines)
    ax3.set_ylabel(r"$\gamma$", fontsize=18)

    
    ax4.plot(res_numbers, res_vec[:, 3], "ko", label="ss", color=c_dots)
    ax4.hlines(0.52, 0, 100, linewidth=lw, color=c_lines)
    ax4.set_ylabel("s", fontsize=18)
    # ax3.legend()
    plt.xlabel("Dataset", fontsize=18)
    plt.show()
    
def multi_barrier10(folder, method=2, res_numbers=range(200), res_folder=None):
    '''Print the Estimates from 10x20 replicates of Barrier Strenght Estimation
    with Method 2. Max Plot: Up to what to plot'''
    bs = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0]
    nr_bts = 20
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method, subfolder=res_folder) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method, subfolder=res_folder) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    # Put the Barrier Estimates >1 to 1:
    res_vec[res_numbers, 2] = np.where(res_vec[res_numbers, 2] > 1, 1, res_vec[res_numbers, 2])
    
    f, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(5, 1, sharex=True)
    ax32 = ax3.twinx()  # Copy for different y-Scaling
    
    # Nbh Size Plot:
    ax1.hlines(4 * np.pi * 5, res_numbers[0], res_numbers[-1], linewidth=3, color="g")
    ax1.plot(res_numbers, res_vec[:, 0], "ko", label="Nbh", alpha=0.8,
                 zorder=0, color="grey")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh", fontsize=18, rotation=0, labelpad=18)
    # ax1.title.set_text("Method: %s" % str(method))

    # L Plot:
    ax2.plot(res_numbers, res_vec[:, 1], "o", color="grey", label="L", alpha=0.8,
                 zorder=0)
    ax2.hlines(0.006, res_numbers[0], res_numbers[-1], linewidth=3, color="g")
    ax2.set_ylim([0, 0.03])
    ax2.set_ylabel("m", fontsize=18, rotation=0, labelpad=10)
    # ax2.legend()
    
    # Barrier Plot Left Half:
    for i in xrange(5):  # Iterate over different parts of the Graph
        x0, x1 = i * nr_bts, (i + 1) * nr_bts
        y = bs[i]
        res_numbers_t = range(x0, x1)  # Where to Plot
        # Alternate the color:
        if i % 2 == 0:
            c = "cyan"
        else:
            c = "b"
        ax3.plot(res_numbers_t, res_vec[res_numbers_t, 2],
                    marker="o", linestyle="", alpha=0.8, color=c, zorder=0)
        ax3.hlines(y, x0, x1, linewidth=3, color="g", zorder=1)
    # Barrier Plot Right Half:
    for i in xrange(5, 10):
        x0, x1 = i * nr_bts, (i + 1) * nr_bts
        y = bs[i]
        res_numbers_t = range(x0, x1)  # Where to Plot
        # Alternate the color:
        if i % 2 == 0:
            c = "crimson"
        else:
            c = "coral"
        ax32.plot(res_numbers_t, res_vec[res_numbers_t, 2],
                    marker="o", linestyle="", alpha=0.8, color=c, zorder=0)
        ax32.hlines(y, x0, x1, linewidth=3, color="g", zorder=1)
        
        
    ax3.set_ylabel("$\gamma$", fontsize=18, rotation=0, labelpad=10)
    ax3.set_ylim([0, 0.2])
    ax3.tick_params('y', colors='blue')
    ax32.set_ylim([0, 1.1])
    ax32.tick_params('y', colors='crimson')

    # SS Plot:
    ax4.plot(res_numbers, res_vec[:, 3], "o", color="grey", label="ss", alpha=0.8,
                 zorder=0)
    ax4.hlines(0.52, res_numbers[0], res_numbers[-1], linewidth=3, color="g")
    ax4.set_ylabel("s", fontsize=18, rotation=0, labelpad=10)
    ax4.set_xlabel("Dataset", fontsize=18)
    
#     for i in xrange(len(bs)):  # Iterate over different parts of the Graph
#         x0, x1 = i * nr_bts, (i + 1) * nr_bts
#         y = bs[i]
#         res_numbers_t = range(x0, x1)  # Where to Plot
#         # Alternate the color:
#         if i % 2 == 0:
#             c = "cyan"
#         else:
#             c = "b"
#         ax5.plot(res_numbers_t, res_vec[res_numbers_t, 2],
#                     marker="o", linestyle="", alpha=0.8, color=c, zorder=0)
#         ax5.hlines(y, x0, x1, linewidth=3, color="g", zorder=1)
#     #ax5.yscale("log")
#     ax5.set_ylim([0.01,1.1])
#     ax5.set_yscale("log")
    plt.show()
    
    
    
def multi_bts_barrier(folder, method=2, nr_bts=25, k_vec=[0.002, 0.1, 0.5, 0.999], nr_reps=5, figsize=(12, 12)):
    '''Plots the Bootstraps over Barriers.'''
    nr_data_sets = len(k_vec) * nr_reps  # Nr of Independent DS
    nr_all_data = nr_bts * len(k_vec) * nr_reps  # Nr of all DS
    res_numbers = range(nr_all_data)
    
    # Load the data:
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    # Put the Barrier Estimates >1 to 1:
    res_vec[res_numbers, 2] = np.where(res_vec[res_numbers, 2] > 1, 1, res_vec[res_numbers, 2])
    
    k_vec_full = np.repeat(k_vec, nr_reps)  # Gets the Default Values for the k-vecs
    # k_vec_full = [j for j in k_vec for _ in range(nr_reps)]  # Gets the Default Values for the k-vecs
    x_vec_full = np.repeat(range(nr_data_sets), nr_bts)
    # x_vec_full = [j for j in range(nr_data_sets) for _ in range(nr_reps)] # Gets the x-Values for the Plots
    
    # Do some jitter for the x-Values:
    x_jitter = np.linspace(0, 1, nr_bts + 1)  # Construct the offset
    x_jitter = np.tile(x_jitter[:nr_bts], nr_data_sets)  
    x_vec_full1 = x_vec_full + x_jitter  # Constructs the x-Values
    
    # x-Values of true Estimates:
    x_true = range(0, nr_all_data, nr_bts)
    
    # Construct the color Values:
    colors = ["coral", "cyan"]
    color_vec = np.repeat(colors, nr_bts)
    color_vec = np.tile(color_vec, nr_data_sets)[:nr_all_data]  # Gets the color vector (double and extract what needed)

    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True)
    # ax32 = ax3.twinx()  # Copy for different y-Scaling
    
    # Nbh Size Plot:
    ax1.hlines(4 * np.pi * 5, 0, nr_data_sets, linewidth=2, color="g")
    inds_sorted, true_inds = argsort_bts(res_vec[:, 0], nr_bts)
    
    ax1.plot(x_vec_full1[true_inds], res_vec[x_true, 0], "ko", markersize=6, label="Estimate")
    ax1.scatter(x_vec_full1, res_vec[inds_sorted, 0], c=color_vec, label="Bootstrap Estimates")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh", fontsize=18, rotation=0, labelpad=16)
    # ax1.legend(loc="upper right")
    # ax1.title.set_text("Method: %s" % str(method))

    # L Plot:
    inds_sorted, true_inds = argsort_bts(res_vec[:, 1], nr_bts)
    ax2.scatter(x_vec_full1, res_vec[inds_sorted, 1], c=color_vec)
    ax2.plot(x_vec_full1[true_inds], res_vec[x_true, 1], "ko", markersize=6, label="Estimate")
    ax2.hlines(0.006, 0, nr_data_sets, linewidth=2, color="g")
    ax2.set_ylim([0, 0.03])
    ax2.set_ylabel("m", fontsize=18, rotation=0, labelpad=10)
    # ax2.legend()
    
    # Plot Barrier:   
    inds_sorted, true_inds = argsort_bts(res_vec[:, 2], nr_bts)
    ax3.scatter(x_vec_full1, res_vec[inds_sorted, 2], c=color_vec, alpha=0.6, zorder=0, label="Bootstrap Estimates")
    ax3.plot(x_vec_full1[true_inds], res_vec[x_true, 2], "ko", markersize=6, label="Estimate", alpha=0.6, zorder=2)
    for i in range(nr_data_sets):
        ax3.hlines(k_vec_full[i] , i, i + 1, linewidth=2, color="g", zorder=1)
    ax3.set_ylim([0, 1])
    ax3.legend(loc="upper left")
    ax3.set_ylabel("$\gamma$", fontsize=18, rotation=0, labelpad=10)
    ax3.set_ylim(0.01, 1.2)
    # ax3.set_yscale("log")
 
#     # SS Plot:
    inds_sorted, true_inds = argsort_bts(res_vec[:, 3], nr_bts)
    ax4.scatter(x_vec_full1, res_vec[inds_sorted, 3], c=color_vec)
    ax4.plot(x_vec_full1[true_inds], res_vec[x_true, 3], "ko", markersize=6, label="Estimate")
    ax4.hlines(0.52, 0, nr_data_sets, linewidth=2, color="g")
    ax4.set_ylabel("s", fontsize=18, rotation=0)
    plt.xlabel("Dataset", fontsize=18)
    plt.xlim([0, 20])
    plt.xticks(np.linspace(0, 20, (nr_all_data + 1) / 100), map(int, np.linspace(0, nr_all_data, (nr_all_data + 1) / 100)), fontsize=10)
    plt.show()
    
def multi_barrier_loci(folder, method=2, k_only_folder="k_only/", loci_nr=range(5, 101, 5), nr_reps=25, k_true=0.05):
    '''Plots the Estimates from multiple Loci and replicates.
    Plots both all estimates; and k_only estimates'''
    
    nr_data_sets = len(loci_nr) * nr_reps  # Nr of Independent DS
    res_numbers = range(nr_data_sets)
    
    # Load the data:
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])  # Full Data Set
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    res_vec_k = np.array([load_pickle_data(folder, i, 0, method, subfolder=k_only_folder) for i in res_numbers])  # K Only Datasets
    
    print(res_vec[:4])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
    
    
    
    
    # First Get all the colors
    colors = ["coral", "crimson"]
    color_vec = np.repeat(colors, nr_reps)
    color_vec = np.tile(color_vec, len(loci_nr))[:nr_data_sets]  # Gets the color vector (double and extract what needed)
    
    colors = ["cyan", "blue"]
    color_vec_k = np.repeat(colors, nr_reps)
    color_vec_k = np.tile(color_vec_k, len(loci_nr))[:nr_data_sets]
    
    x_vec_full1 = res_numbers
    x_ticks = nr_reps / 2.0 + np.arange(0, nr_data_sets - 1, nr_reps)
    # ax32 = ax3.twinx()  # Copy for different y-Scaling
    
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True)
    # Nbh Size Plot:
    ax1.hlines(4 * np.pi * 5, 0, nr_data_sets, linewidth=2, color=c_lines)
    
    ax1.scatter(x_vec_full1, res_vec[:, 0], c=color_vec, label="Full Estimates")
    # ax1.scatter(x_vev_full1, res_vec_k, c=color_vec, label="K-Only Estimates")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.legend(loc="upper right")
    ax1.title.set_text("Method: %s" % str(method))

    # L Plot:
    # inds_sorted, true_inds = argsort_bts(res_vec[:, 1], nr_bts)
    ax2.scatter(x_vec_full1, res_vec[:, 1], label="True Values", c=color_vec)
    ax2.hlines(0.006, 0, nr_data_sets, linewidth=2, color=c_lines)
    ax2.set_ylim([0, 0.03])
    ax2.set_ylabel("L", fontsize=18)
#     
#     # Plot Barrier:   
    ax3.scatter(x_vec_full1, res_vec[:, 2], c=color_vec, zorder=0)
    # ax3.scatter(x_vec_full1, res_vec_k[:,0], c= color_vec_k, zorder=0.5)
    ax3.hlines(0.05, 0, nr_data_sets, linewidth=2, color=c_lines, zorder=1)
    ax3.set_ylim([0, 1])
    ax3.set_ylabel("k", fontsize=18)
#  
# #     # SS Plot:
    ax4.scatter(x_vec_full1, res_vec[:, 3], c=color_vec)
    # ax4.plot(x_vec_full1[true_inds], res_vec[x_true, 3], "ko", markersize=6, label="True Values")
    ax4.hlines(0.52, 0, nr_data_sets, linewidth=2, color=c_lines)
    ax4.set_ylabel("SS", fontsize=18)
    
    plt.xlabel("Loci Nr")
    plt.xticks(x_ticks, loci_nr)
    plt.xlim([x_vec_full1[0], x_vec_full1[-1]])
    plt.show()
    
    f, ((ax1, ax2)) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
    # Plot Comparison k and k_only:
    ax1.scatter(x_vec_full1, res_vec[:, 2], c=color_vec, zorder=0)
    ax1.hlines(0.05, 0, nr_data_sets, linewidth=2, color=c_lines, zorder=1)
    ax1.set_ylim([0, 1])
    ax1.set_ylabel(r"$\gamma$", fontsize=18)
    
    ax2.scatter(x_vec_full1, res_vec_k[:, 0], c=color_vec_k, zorder=0.5)
    ax2.hlines(0.05, 0, nr_data_sets, linewidth=2, color=c_lines, zorder=1)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel(r"$\gamma$ only", fontsize=18)
    plt.xlabel("Loci Nr", fontsize=18)
    plt.xticks(x_ticks, loci_nr)
    plt.xlim([x_vec_full1[0], x_vec_full1[-1]])
    plt.show()
    
    print("Correlation k and k only: %.4g" % np.corrcoef(res_vec[250:, 2], res_vec_k[250:, 0])[0, 1])
    
    

def multi_ind_single(folder, method, res_numbers=range(0, 100)):
    '''Print several Neighborhood Sizes simulated under the model - using one method'''
    # First quick function to unpickle the data:
    # res_numbers = range(0, 100)
    # res_numbers = range(3, 4)
    # res_numbers = [2, 3, 8, 11, 12, 13, 21, 22, 27, 29, 33, 35, 37, 38, 40, 75]  # 2
    # res_numbers = [1, 2, 7, 8, 9, 14, 17, 18, 19, 20]
    
    # Extract Res-Numbers that actually exist
    res_numbers = np.array(res_numbers)
    data_there = np.array([check_if_data(folder, i, method) for i in res_numbers])
    res_numbers = res_numbers[data_there]
    
    
    res_vec = np.array([load_pickle_data(folder, i, 0, method) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(3):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    
    x_vec = np.array(range(200, 2200, 20))[res_numbers]  # Length 100: from 404 to 4000
    # plt.figure()
    f, ((ax1, ax2)) = plt.subplots(2, 1, sharex=True)
    
    ax1.hlines(4 * np.pi * 5, min(x_vec), max(x_vec), linewidth=3, color="g")
    ax1.plot(x_vec, res_vec[:, 0], "o", label="Nbh", color="crimson")
    ax1.set_ylim([0, 200])
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.title.set_text("Method: %s" % str(method))
    # ax1.legend()
    
    ax2.hlines(0.006, min(x_vec), max(x_vec), linewidth=3, color="g")
    ax2.plot(x_vec, res_vec[:, 1], "o", label="L", color="crimson")
    ax2.set_ylabel("L", fontsize=18)
    ax2.set_ylim([0, 0.02])
    # ax2.legend()
    
    # ax3.errorbar(x_vec, res_vec[:, 2], yerr=res_vec[:, 2] - unc_vec[:, 2, 0], fmt="ko", label="ss")
    # ax3.hlines(0.52, 500, 4000, linewidth=2)
    # ax3.set_ylabel("SS", fontsize=18)
    # ax3.legend()
    plt.xlabel("Nr. of Individuals")
    plt.show()
    
def multi_ind_all(folder, res_numbers=range(10100)):
    '''Plots the Estimates from multiple Individuals'''
    lw = 3
    ul_nb = 200  # Upper Limit Neighborhood Size
    # Load the Data for all three methods
    
    x_vec = np.array(range(200, 2200, 20))  # Length 100: From 200 to 2200 Individuals
    # x_vec = np.array(range(50, 350, 3))  # Length 100: from 50 to 350 Loci
    x_min = np.min(x_vec)
    x_max = np.max(x_vec)
     
    # Extract Res-Numbers that actually exist for Data-Set 0
    res_numbers = np.array(res_numbers)
    
    # Check for all Data whether it is there:
    data_there0 = np.array([check_if_data(folder, i, 0) for i in res_numbers])
    data_there1 = np.array([check_if_data(folder, i, 1) for i in res_numbers])
    data_there2 = np.array([check_if_data(folder, i, 2) for i in res_numbers])
    
    res_numbers0 = res_numbers[data_there0]
    res_numbers1 = res_numbers[data_there1]
    res_numbers2 = res_numbers[data_there2]
    
    x_vec0 = x_vec[res_numbers0]
    x_vec1 = x_vec[res_numbers1]
    x_vec2 = x_vec[res_numbers2]
    
    print("Loading (Method)\n 0: %i \n 1: %i \n 2: %i" 
          % (len(res_numbers0), len(res_numbers1), len(res_numbers2)))
    
    res_vec0 = np.array([load_pickle_data(folder, i, 0, method=0) for i in res_numbers0])
    unc_vec0 = np.array([load_pickle_data(folder, i, 1, method=0) for i in res_numbers0])
    
    res_vec1 = np.array([load_pickle_data(folder, i, 0, method=1) for i in res_numbers1])
    unc_vec1 = np.array([load_pickle_data(folder, i, 1, method=1) for i in res_numbers1])
    
    res_vec2 = np.array([load_pickle_data(folder, i, 0, method=2) for i in res_numbers2])
    unc_vec2 = np.array([load_pickle_data(folder, i, 1, method=2) for i in res_numbers2])
    
    f, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3, 3, sharex=True, figsize=(6, 6))
    
    ax1.hlines(4 * np.pi * 5, x_min, x_max, linewidth=lw, color=c_lines)
    ax1.plot(x_vec0, res_vec0[:, 0], "ko", label="Nbh", color=c_dots, zorder=0)
    ax1.set_ylim((0, ul_nb))
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.set_title("Method 1", fontsize=18)

    ax2.plot(x_vec0, res_vec0[:, 1], "ko", label="L", color=c_dots, zorder=0)
    ax2.hlines(0.006, x_min, x_max, linewidth=lw, color=c_lines)
    ax2.set_ylabel("m", fontsize=18)
    ax2.set_ylim((0, 0.03))
    
    ax3.plot(x_vec0, res_vec0[:, 2], "ko", label="ss", color=c_dots, zorder=0)
    ax3.hlines(0.04, x_min, x_max, linewidth=lw, color=c_lines)
    ax3.set_ylabel("Var. Par.", fontsize=18)
    
    ax4.hlines(4 * np.pi * 5, x_min, x_max, linewidth=lw, color=c_lines)
    ax4.plot(x_vec1, res_vec1[:, 0], "ko", label="Nbh", color=c_dots, zorder=0)
    ax4.set_ylim((0, ul_nb))
    ax4.set_title("Method 2", fontsize=18)
    ax4.set_yticks([])
    
    ax5.plot(x_vec1, res_vec1[:, 1], "ko", label="L", color=c_dots, zorder=0)
    ax5.hlines(0.006, x_min, x_max, linewidth=lw, color=c_lines, zorder=1)
    ax5.set_ylim((0, 0.03))
    ax5.set_yticks([])
    
    ax6.plot(x_vec1, res_vec1[:, 2], "ko", label="ss", color=c_dots, zorder=0)
    ax6.hlines(0.01, x_min, x_max, linewidth=lw, color=c_lines)
    ax6.get_yaxis().set_tick_params(which='both', direction='out', pad=-35)
    ax6.yaxis.tick_right()
    
    ax7.hlines(4 * np.pi * 5, x_min, x_max, linewidth=lw, color=c_lines)
    ax7.plot(x_vec2, res_vec2[:, 0], "ko", label="Nbh", color=c_dots, zorder=0)
    ax7.set_ylim((0, ul_nb))
    ax7.set_title("Method 3", fontsize=18)
    ax7.set_yticks([])
    # ax1.legend()
    
    ax8.plot(x_vec2, res_vec2[:, 1], "ko", label="L", color=c_dots, zorder=0)
    ax8.hlines(0.006, x_min, x_max, linewidth=lw, color=c_lines, zorder=1)
    ax8.set_ylim((0, 0.03))
    ax8.set_yticks([])
    
    ax9.plot(x_vec2, res_vec2[:, 2], "ko", label="ss", color=c_dots, zorder=0)
    ax9.hlines(0.52, x_min, x_max, linewidth=lw, color=c_lines)
    ax9.yaxis.tick_right()
   # ax9.set_yticks([])
    
    plt.gcf().text(0.5, 0.04, "Number of Individuals", ha="center", fontsize=18)  # Set the x-Label
    plt.show()
    
    
def multi_loci_single(folder, method, res_numbers=range(0, 100)):
    '''Print several Neighborhood Sizes simulated under the model - using one method'''
    # First quick function to unpickle the data:
    # res_numbers = range(0, 100)
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
            
    
    x_vec = np.array(range(50, 350, 3))  # Length 100: from 50 to 350
    x_vec = x_vec[res_numbers]
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
    
def multi_loci_all(folder, res_numbers=range(100)):
    '''Plot estimates for Data Sets with varying number of Loci.
    Test all three Methods.'''
    lw = 3
    ul_nb = 200  # Upper Limit Neighborhood Size
    # Load the Data for all three methods
    
    x_vec = np.array(range(50, 350, 3))  # Length 100: from 50 to 350 Loci
    x_min = np.min(x_vec)
    x_max = np.max(x_vec)
     
    # Extract Res-Numbers that actually exist for Data-Set 0
    res_numbers = np.array(res_numbers)
    
    # Check for all Data whether it is there:
    data_there0 = np.array([check_if_data(folder, i, 0) for i in res_numbers])
    data_there1 = np.array([check_if_data(folder, i, 1) for i in res_numbers])
    data_there2 = np.array([check_if_data(folder, i, 2) for i in res_numbers])
    
    res_numbers0 = res_numbers[data_there0]
    res_numbers1 = res_numbers[data_there1]
    res_numbers2 = res_numbers[data_there2]
    
    x_vec0 = x_vec[res_numbers0]
    x_vec1 = x_vec[res_numbers1]
    x_vec2 = x_vec[res_numbers2]
    
    print("Loading (Method)\n 0: %i \n 1: %i \n 2: %i" 
          % (len(res_numbers0), len(res_numbers1), len(res_numbers2)))
    
    res_vec0 = np.array([load_pickle_data(folder, i, 0, method=0) for i in res_numbers0])
    unc_vec0 = np.array([load_pickle_data(folder, i, 1, method=0) for i in res_numbers0])
    
    res_vec1 = np.array([load_pickle_data(folder, i, 0, method=1) for i in res_numbers1])
    unc_vec1 = np.array([load_pickle_data(folder, i, 1, method=1) for i in res_numbers1])
    
    res_vec2 = np.array([load_pickle_data(folder, i, 0, method=2) for i in res_numbers2])
    unc_vec2 = np.array([load_pickle_data(folder, i, 1, method=2) for i in res_numbers2])
    
    f, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3, 3, sharex=True, figsize=(7, 7))
    
    ax1.hlines(4 * np.pi * 5, x_min, x_max, linewidth=lw, color=c_lines)
    ax1.plot(x_vec0, res_vec0[:, 0], "ko", label="Nbh", color=c_dots, zorder=0)
    ax1.set_ylim((0, ul_nb))
    ax1.set_ylabel("Nbh", fontsize=18)
    ax1.set_title("Method 1", fontsize=18)

    ax2.plot(x_vec0, res_vec0[:, 1], "ko", label="L", color=c_dots, zorder=0)
    ax2.hlines(0.006, x_min, x_max, linewidth=lw, color=c_lines)
    ax2.set_ylabel("m", fontsize=18)
    ax2.set_ylim((0, 0.03))
    
    ax3.plot(x_vec0, res_vec0[:, 2], "ko", label="ss", color=c_dots, zorder=0)
    ax3.hlines(0.04, x_min, x_max, linewidth=lw, color=c_lines)
    ax3.set_ylabel("Var. Par.", fontsize=18)
    
    ax4.hlines(4 * np.pi * 5, x_min, x_max, linewidth=lw, color=c_lines)
    ax4.plot(x_vec1, res_vec1[:, 0], "ko", label="Nbh", color=c_dots, zorder=0)
    ax4.set_ylim((0, ul_nb))
    ax4.set_title("Method 2", fontsize=18)
    ax4.set_yticks([])
    
    ax5.plot(x_vec1, res_vec1[:, 1], "ko", label="L", color=c_dots, zorder=0)
    ax5.hlines(0.006, x_min, x_max, linewidth=lw, color=c_lines, zorder=1)
    ax5.set_ylim((0, 0.03))
    ax5.set_yticks([])
    
    ax6.plot(x_vec1, res_vec1[:, 2], "ko", label="ss", color=c_dots, zorder=0)
    ax6.hlines(0.01, x_min, x_max, linewidth=lw, color=c_lines)
    ax6.get_yaxis().set_tick_params(which='both', direction='out', pad=-35)
    ax6.yaxis.tick_right()
    
    ax7.hlines(4 * np.pi * 5, x_min, x_max, linewidth=lw, color=c_lines)
    ax7.plot(x_vec2, res_vec2[:, 0], "ko", label="Nbh", color=c_dots, zorder=0)
    ax7.set_ylim((0, ul_nb))
    ax7.set_title("Method 3", fontsize=18)
    ax7.set_yticks([])
    # ax1.legend()
    
    ax8.plot(x_vec2, res_vec2[:, 1], "ko", label="L", color=c_dots, zorder=0)
    ax8.hlines(0.006, x_min, x_max, linewidth=lw, color=c_lines, zorder=1)
    ax8.set_ylim((0, 0.03))
    ax8.set_yticks([])
    
    ax9.plot(x_vec2, res_vec2[:, 2], "ko", label="ss", color=c_dots, zorder=0)
    ax9.hlines(0.52, x_min, x_max, linewidth=lw, color=c_lines)
    ax9.yaxis.tick_right()
   # ax9.set_yticks([])
    
    plt.gcf().text(0.5, 0.04, "Loci Number", ha="center", fontsize=18)  # Set the x-Label
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
    
    loci_nr_vec = np.array([np.loadtxt(folder + subfolder_meth + "nr_good_loci" + str(i).zfill(2) + ".csv")[1] 
                   for i in res_numbers])
    loci_nr_vec_b = np.array([np.loadtxt(folder_b + subfolder_meth + "nr_good_loci" + str(i).zfill(2) + ".csv")[1] 
                   for i in res_numbers])
    
    
    # plt.figure()
    c0, c1, c2, c3 = "crimson", "coral", "sandybrown", "chocolate"
    f, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, sharex=True)
    
    ax1.hlines(4 * np.pi * 5, 0, 100, linewidth=3, color=c_lines)
    
    ax1.scatter(res_numbers0, res_vec[res_numbers0, 0], color=c0, label=r"$R^2=1$")
    ax1.scatter(res_numbers1, res_vec[res_numbers1, 0], color=c1, label=r"$R^2=0.02$")
    ax1.scatter(res_numbers2, res_vec[res_numbers2, 0], color=c2, label=r"$R^2=0.01$")
    ax1.scatter(res_numbers3, res_vec[res_numbers3, 0], color=c3, label=r"$R^2=0.005$")
    ax1.set_ylim([0, 300])
    ax1.set_ylabel("Nbh", fontsize=18, rotation=0, labelpad=18)
    ax1.set_title(r"No Barrier", fontsize=18)
    ax1.legend(loc="upper right", fontsize=8, labelspacing=0)
    
    ax4.scatter(res_numbers0, res_vec_b[res_numbers0, 0], color=c0, label=r"$R^2=1$")
    ax4.scatter(res_numbers1, res_vec_b[res_numbers1, 0], color=c1, label=r"$R^2=0.02$")
    ax4.scatter(res_numbers2, res_vec_b[res_numbers2, 0], color=c2, label=r"$R^2=0.01$")
    ax4.scatter(res_numbers3, res_vec_b[res_numbers3, 0], color=c3, label=r"$R^2=0.005$")
    ax4.set_ylim([0, 300])
    ax4.hlines(4 * np.pi * 5, 0, 100, linewidth=3, color=c_lines)
    ax4.set_title(r"Barrier: $\gamma=0.05$", fontsize=18)
    ax4.yaxis.tick_right()
    # ax4.legend(loc="upper right")
    

    
    ax2.scatter(res_numbers0, res_vec[res_numbers0, 2], color=c0)
    ax2.scatter(res_numbers1, res_vec[res_numbers1, 2], color=c1)
    ax2.scatter(res_numbers2, res_vec[res_numbers2, 2], color=c2)
    ax2.scatter(res_numbers3, res_vec[res_numbers3, 2], color=c3)
    ax2.hlines(0.99, 0, 100, linewidth=3, color=c_lines)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel(r"$\gamma$", fontsize=18, rotation=0, labelpad=18)
    
    ax5.scatter(res_numbers0, res_vec_b[res_numbers0, 2], color=c0)
    ax5.scatter(res_numbers1, res_vec_b[res_numbers1, 2], color=c1)
    ax5.scatter(res_numbers2, res_vec_b[res_numbers2, 2], color=c2)
    ax5.scatter(res_numbers3, res_vec_b[res_numbers3, 2], color=c3)
    ax5.hlines(0.05, 0, 100, linewidth=3, color=c_lines)
    ax5.set_ylim([0, 1.0])
    ax5.yaxis.tick_right()
    
    ax3.scatter(res_numbers0, loci_nr_vec[res_numbers0], color=c0)
    ax3.scatter(res_numbers1, loci_nr_vec[res_numbers1], color=c1)
    ax3.scatter(res_numbers2, loci_nr_vec[res_numbers2], color=c2)
    ax3.scatter(res_numbers3, loci_nr_vec[res_numbers3], color=c3)
    ax3.hlines(200, 0, 100, linewidth=3, color=c_lines)
    ax3.set_ylabel("Nr. of Loci", fontsize=18)
    
    ax6.scatter(res_numbers0, loci_nr_vec_b[res_numbers0], color=c0)
    ax6.scatter(res_numbers1, loci_nr_vec_b[res_numbers1], color=c1)
    ax6.scatter(res_numbers2, loci_nr_vec_b[res_numbers2], color=c2)
    ax6.scatter(res_numbers3, loci_nr_vec_b[res_numbers3], color=c3)
    ax6.hlines(200, 0, 100, linewidth=3, color=c_lines)
    ax6.yaxis.tick_right()
    f.text(0.5, 0.025, "Dataset", ha='center', fontsize=18)
    plt.show()
    
def plot_theory_f():
    '''Method to test the Kernel'''
    kc = fac_kernel("DiffusionBarrierK0")
    # Density: 5, mu=0.003, t0=1, Diff=1, k=0.5
    # kc.set_parameters([0, 1.0, 1.0, 0.001, 5.0])  # k, Diff, t0, mu, dens; In old ordering
    k = 0.01  # Barrier Strength
    kc.set_parameters([4 * np.pi * 5, 0.006, k, 1.0, 0.0])  # Nbh, L, k, t0, ss
    
    k0 = fac_kernel("DiffusionK0")
    k0.set_parameters([4 * np.pi * 5, 0.006, 1.0, 0.0])
    
    print("Parameters Barrier: ")
    print(kc.give_parameter_names())
    print(kc.give_parameters())
    
    print("Parameters No Barrier: ")
    print(k0.give_parameter_names())
    print(k0.give_parameters())
    mu = 0.003  # Set Mutation Rate
    # dens = k0.give_parameters
    
    # x_vec = np.logspace(-2, 2.0, 100) + 2.0
    x_vec = np.linspace(2.1, 30, 100) + 0.0001
    y_vec = [kc.num_integral_barrier(0, -1, -1 + x1) for x1 in x_vec]  # 0 Difference along the y-Axis ; 
    y_vec2 = [kc.num_integral_barrier(0, 1, 1 + x1) for x1 in x_vec]  # 0 Difference along the y-Axis ; 
    y_vec3 = [kc.num_integral_barrier(0, 5, 5 + x1) for x1 in x_vec]  # 0 Difference along the y-Axis ; 
    
     
    y_vec01 = np.array([k0.num_integral(r) for r in x_vec])  # Numerical Integral no barrier
    # y_vec1=np.array([num_integral(x, t0=1, sigma=sigma, mu=mu) for x in x_vec])
    # y_vec20=np.array([num_integral(x, t0=2, sigma=sigma, mu=mu) for x in x_vec])
    y_bessel = 1 / (4 * np.pi * 5) * kv(0, np.sqrt(2 * mu) * x_vec)  # The numerical Result from Mathematica
    
    c0, c1, c2, c3 = "black", "blue", "crimson", "yellow"
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios':[2, 2.5]})
    # Plot the IBD
    ax1.plot(x_vec, y_bessel, alpha=0.8, color=c0, linestyle='--', label="Analytical Solution", linewidth=2, zorder=1)
    ax1.plot(x_vec, y_vec01, alpha=0.8, label="No Barrier", color="Lime", linewidth=6, zorder=0)
    ax1.plot(x_vec, y_vec, alpha=0.9, color=c1, label="Different Side of Barrier", linewidth=5)
    ax1.plot(x_vec, y_vec2, alpha=0.9, color=c2, label="Same Side (Barrier near)", linewidth=5)
    ax1.plot(x_vec, y_vec3, alpha=0.9, color=c3, label="Same Side (Barrier distant)", linewidth=5)
    ax1.set_xlim([0, max(x_vec)])
    ax1.set_ylim([0, 0.05])
    ax1.set_xlabel("Pairwise Distance", fontsize=18)
    ax1.set_ylabel("Pairwise F", fontsize=18)
    
    # plt.xscale("log")
    ax1.legend(fontsize=12, labelspacing=0.3)
    
    #####################################
    ps = 1200  # Pointsize
    # Plot the positions
    pos = np.array([[i, j] for i in range(6) for j in range(5)])  # The Grid Points
    pos1 = np.array([[i, 2] for i in range(4)])
    pos2 = np.array([[i, 1] for i in range(5)])
    pos3 = np.array([[i, 3] for i in range(2)])
    
    ax2.scatter(pos[:, 0], pos[:, 1], marker='o', s=ps, color='grey', alpha=0.5)
    ax2.scatter(pos1[:, 0], pos1[:, 1], marker='o', s=ps, color=c2, label="Same Side Close")
    ax2.scatter(pos2[:, 0], pos2[:, 1], marker='o', s=ps, color=c1, label="Different Side")
    ax2.scatter(pos3[:, 0], pos3[:, 1], marker='o', s=ps, color=c3, label="Same Side Dist")
    ax2.scatter(3, 2, marker='o', s=100, color="white")  # Right End Point
    ax2.scatter(4, 1, marker='o', s=100, color="white")  # Right End Point
    ax2.scatter(1, 3, marker='o', s=100, color="white")  # Right End Point
    # ax2.set_xlabel("x-Axis", fontsize=18)
    # ax2.set_ylabel("y-Axis", fontsize=18)
    ax2.set_xlim([min(pos[:, 0]) - 0.5, max(pos[:, 0]) + 0.5])
    ax2.set_ylim([min(pos[:, 1]) - 0.5, max(pos[:, 1]) + 0.5])
    # plt.text(3.9, 2.9,"R",color="white",fontsize=18)
    # plt.text(4.9, 1.9,"R",color="white",fontsize=18)
    ax2.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax2.vlines(3.5, -0.5, 5.5, linewidth=6)
    # plt.legend(loc="upper right", borderpad=2)
    
    # Plot the arrows:
    ax2.arrow(-0.4, -0.4, 0, 1, head_width=0.1, head_length=0.2, fc='k', ec='k')
    ax2.arrow(-0.4, -0.4, 1, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')
    ax2.annotate("x", (0.85, -0.4))
    ax2.annotate("y", (-0.45, 0.85))
    
    plt.show()
    
def plot_theory_f_local_barrier():
    '''Plot the effects of a local barriers for David's grant'''
    ss = [0, 0.1, 0.5]
    k = len(ss)
    
    
    rr = np.linspace(-5.1, 5.1, 521)
    res = np.zeros((k, len(rr)))  # The Container for the Results
    
    # The Container for the 
    kc = fac_kernel("DiffusionBarrierK0")
    
    for i in xrange(k):
        for j in xrange(len(rr)):
            # calculate the Barrier Strength
            s = ss[i]
            r = np.abs(rr[j])/100.0
            print("s: %.2f, r: %.2f" % (s, r))
            gamma = r / (r + s)
            kc.set_parameters([4 * np.pi * 5, 0.006, gamma, 1.0, 0.0])  # Nbh, L, k, t0, ss

            f = kc.num_integral_barrier(0, -1, 1)
            res[i, j] = f

    
    
    c0, c1, c2 = "yellow", "orange", "crimson"
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios':[2, 2.5]})
    f = plt.figure(figsize=(8, 5))
    
    
    
    # Plot the IBD
    lw = 4
    a=0.95
    labels = [r"s=%.2f" % s for s in ss]
    plt.plot(rr, res[0, :], alpha=a, color=c0, label=labels[0], linewidth=lw)
    plt.plot(rr, res[1, :], alpha=a, color=c1, label=labels[1], linewidth=lw)
    plt.plot(rr, res[2, :], alpha=a, color=c2, label=labels[2], linewidth=lw)

    # ax1.set_xlim([0, max(x_vec)])
    # ax1.set_ylim([0, 0.05])
    plt.xlabel("Genetic Distance [cM]", fontsize=18)
    plt.ylabel("Expected F", fontsize=18)
    
    # plt.xscale("log")
    plt.legend(fontsize=12, labelspacing=0.3)
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
            
    c0, c1, c2, c3 = "yellow", "orange", "crimson", "purple"
    
    # plt.figure()
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True, figsize=(6, 6))
    
    ax1.hlines(4 * np.pi * 5, 0, 100, linewidth=2, color=c_lines)
    
    ax1.plot(res_numbers0, res_vec[res_numbers0, 0], "ko", color=c0, label="1x1", zorder=0)
    ax1.plot(res_numbers1, res_vec[res_numbers1, 0], "ko", color=c1, label="2x2", zorder=0)
    ax1.plot(res_numbers2, res_vec[res_numbers2, 0], "ko", color=c2, label="3x3", zorder=0)
    ax1.plot(res_numbers3, res_vec[res_numbers3, 0], "ko", color=c3, label="4x4", zorder=0)
    ax1.set_ylim([0, 200])
    # ax1.set_xlim([0,120])
    ax1.set_ylabel("Nbh", fontsize=18)
    # ax1.legend(loc="upper left")
    # ax1.title.set_text("Various Binning")
    
    
    ax2.plot(res_numbers0, res_vec[res_numbers0, 1], "ko", color=c0, label="1x1")
    ax2.plot(res_numbers1, res_vec[res_numbers1, 1], "ko", color=c1, label="2x2")
    ax2.plot(res_numbers2, res_vec[res_numbers2, 1], "ko", color=c2, label="3x3")
    ax2.plot(res_numbers3, res_vec[res_numbers3, 1], "ko", color=c3, label="4x4")
    ax2.hlines(0.006, 0, 100, linewidth=2, color=c_lines, zorder=1)
    ax2.set_ylabel("m", fontsize=18)
    # ax2.legend()
    
    ax3.plot(res_numbers0, res_vec[res_numbers0, 2], "ko", color=c0, label="1x1")
    ax3.plot(res_numbers1, res_vec[res_numbers1, 2], "ko", color=c1, label="2x2")
    ax3.plot(res_numbers2, res_vec[res_numbers2, 2], "ko", color=c2, label="3x3")
    ax3.plot(res_numbers3, res_vec[res_numbers3, 2], "ko", color=c3, label="4x4")
    ax3.hlines(0.1, 0, 100, linewidth=2, color=c_lines, zorder=1)
    ax3.set_ylabel(r"$\gamma$", fontsize=18)
    
    ax4.plot(res_numbers0, res_vec[res_numbers0, 3], "ko", color=c0, label="1x1")
    ax4.plot(res_numbers1, res_vec[res_numbers1, 3], "ko", color=c1, label="2x2")
    ax4.plot(res_numbers2, res_vec[res_numbers2, 3], "ko", color=c2, label="3x3")
    ax4.plot(res_numbers3, res_vec[res_numbers3, 3], "ko", color=c3, label="4x4")
    ax4.hlines(0.52, 0, 100, linewidth=2, color=c_lines, zorder=1)
    ax4.set_ylim([0.5, 0.53])
    ax4.set_ylabel("s", fontsize=18)
    
    # plt.xticks([10,35,60,85], ['1x1', '2x2', '3x3','4x4'])
    
    plt.xlabel("Dataset Nr", fontsize=18)
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
    
def plot_IBD_bootstrap(position_path, genotype_path, result_folder, subfolder,
                       bins=50, p=0.5, nr_bootstraps=10, res_number=1, scale_factor=50,
                       max_frac=0.5, plot_bootstraps=True):
    '''Plot IBD of real data and bootstrapped real data (over loci).
    Load from Result Folder and plot binned IBD from HZ
    res_number: How many results to plot.
    max_frac: Fraction of x-Values until which to plot.'''  
    
    # res_numbers = range(0, 100)
    positions = np.loadtxt(position_path, delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
    genotypes = np.loadtxt(genotype_path, delimiter='$').astype('float64')
    
    
    # Calculate best Estimates
    bin_dist, bin_corr0, stand_errors, corr_factor_e = calc_bin_correlations(positions, genotypes, bins=bins)
    
    print("Bin Distances:")
    print(bin_dist)
    
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
    res_vec = np.zeros((res_number, 4))  # Estimates are with Barrier
    for i in xrange(res_number):
        res_vec[i, :] = load_pickle_data(result_folder, i, 0, method=2, subfolder=subfolder)
        print("Results %i: " % i)
        print(res_vec[i, :])
    print("Best Fit Results:")
    print(res_vec[0, :])
    res_vec[:, -1] = 0.0  # 605  # Set the mean to mean
    
    # res_vec = res_vec[0,:]
    
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
    
    
    x_plot = np.linspace(min(bin_dist), max(bin_dist) * max_frac, 100)
    coords = [[0, 0], ] + [[0, i] for i in x_plot]  # Coordsvector
    
    corr_fit = np.zeros((res_number, len(x_plot)))
    
    # Calculate the best fits
    for i in xrange(res_number):
        res_true = res_vec[i, :]
        KC = fac_kernel("DiffusionK0")
        KC.set_parameters([res_true[0], res_true[1], 1.0, res_true[3]])  # Nbh Sz, Mu0, t0, ss. Sets known Parameters #[4*np.pi*6, 0.02, 1.0, 0.04]
        f_vec = KC.calc_f_vec(x_plot)
        corr_fit[i, :] = f_vec - np.mean(f_vec[40:60])  # Substract the middle Value
        
    
    
    plt.figure()
    # Plot Parameter Fits
    for i in xrange(1, res_number):
        plt.plot(x_plot * scale_factor, corr_fit[i, :], 'b-', linewidth=2, alpha=0.5)
        
    # ## Plot real Data and all Bootstraps
    # First Bootstrap:
    plt.plot(bin_dist[:int(bins * max_frac)] * scale_factor, f_estimates[0, :int(bins * max_frac)], 'r-', alpha=0.5, label="Bootstrap over GTPs")
    # All Bootstraps over Genotypes:
    if plot_bootstraps == True:
        for i in xrange(1, nr_bootstraps):
            plt.plot(bin_dist[:int(bins * max_frac)] * scale_factor, f_estimates[i, :int(bins * max_frac)], 'r-', alpha=0.5)
    plt.errorbar(bin_dist[:int(bins * max_frac)] * scale_factor, bin_corr0[:int(bins * max_frac)], stand_errors[:int(bins * max_frac)], fmt='go', label="Mean Correlation")
        
        
    plt.plot(bin_dist[:int(bins * max_frac)] * scale_factor, lower[:int(bins * max_frac)], 'k-', label="Bootstrap 2.5 %")
    plt.plot(bin_dist[:int(bins * max_frac)] * scale_factor, upper[:int(bins * max_frac)], 'k-', label="Bootstrap 97.5 %")
    
    # Plot IBD of Left and Right half of HZ:
    nr_bins_l = 10
    nr_bins_r = 12
    plt.plot(bin_dist_l[:nr_bins_l] * scale_factor, bin_corr_l[:nr_bins_l], 'y-', label="Left Half HZ")  # stds_l[:nr_bins_l]
    plt.plot(bin_dist_r[:nr_bins_r] * scale_factor, bin_corr_r[:nr_bins_r], 'm-', label="Right Half HZ")  # stds_r[:nr_bins_r]
    
    plt.plot(x_plot * scale_factor, corr_fit[0, :], 'k-', linewidth=2, alpha=1, label="Best Fit")
    
    plt.legend(loc="upper right")
    plt.ylabel("F / Correlation")
    plt.xlabel("Distance [m]")
    plt.title("IBD in HZ")
    plt.show()
    
def plot_homos(position_path, genotype_path, bins=50, max_dist=0, best_fit_params=[200, 0.02, 0.52],
                         bootstrap=False, nr_bootstraps=50, scale_factor=50, demes_x=30, demes_y=20,
                         deme_bin=False, title="Enter Title"):
    '''Function that plots fraction of Homozygotes, and best fits against pairwise Distance.
    Best Fit Params: Neighborhood Size, m, and s.'''
    
    positions = np.loadtxt(position_path, delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
    genotypes = np.loadtxt(genotype_path, delimiter='$').astype('float64')
    
    print("Dataset loaded.")
    print("Nr. of individuals: %i \nNr. of markers: %i" % np.shape(genotypes))
    
    if deme_bin == True:
        positions, genotypes, _ = group_inds(positions, genotypes,
                                                    demes_x=demes_x, demes_y=demes_y, min_ind_nr=1) 
    nr_inds, nr_loci = np.shape(genotypes)
    
    # Calculate the mean sharing in Bins
    print("Calculating Homo Matrix....")
    bin_dist, bin_h, stand_errors = calc_bin_homos(positions, genotypes, bins=bins, max_dist=max_dist / float(scale_factor))
    print("Finished Calculating.")
    
    # Extract everything according to max distance:
    # inds = np.where(bin_dist < max_dist / scale_factor)[0]
    # assert(len(inds) > 0)
    
    # bin_dist = bin_dist[inds]
    # bin_h = bin_h[inds]
    # stand_errors = stand_errors[inds]
    
    x_vec = np.linspace(min(bin_dist), max(bin_dist), 100)
    

    # Calculate the best fits
    KC = fac_kernel("DiffusionK0")
    KC.set_parameters([best_fit_params[0], best_fit_params[1], 1.0, 0.0])  # [4*np.pi*6, 0.02, 1.0, 0.04]
    f_vec = KC.calc_f_vec(x_vec)
    ss = best_fit_params[-1]
    homo_vec_fit = f_vec + (1 - f_vec) * ss
    # print(f_vec)
    
    # Rescale:
    bin_dist = bin_dist * scale_factor
    x_plot = x_vec * scale_factor
    
    # Do the Plotting:
    plt.figure()
    plt.plot(bin_dist, bin_h, "ro", label="Observed")
    plt.plot(x_plot, homo_vec_fit, label="Best Fit", color="green", linewidth=2)
    plt.ylabel("Pairwise h", fontsize=18)
    plt.xlabel(r"Pairwise Distance [$\sigma$]", fontsize=18)  # Or [m]
    plt.legend(fontsize=18, loc="upper right")
    plt.title(title, fontsize=18)
    ax = plt.gca()
    # plt.text(0.5, 0.5, "Nbh: %.3g\n m: %.4g" % (best_fit_params[0], best_fit_params[1]), fontsize=20, transform = ax.transAxes)
    plt.show()
    
def plot_homos_2(position_path, genotype_path, position_path1, genotype_path1, bins=50, max_dist=0, max_dist1=0,
                 best_fit_params=[200, 0.02, 0.52], best_fit_params1=[200, 0.02, 0.52],
                 scale_factor=50, scale_factor1=100, demes_x=30, demes_y=20, demes_x1=30, demes_y1=20, min_ind_nr=5, min_ind_nr1=1):
    
    positions = np.loadtxt(position_path, delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
    genotypes = np.loadtxt(genotype_path, delimiter='$').astype('float64')
    
    positions1 = np.loadtxt(position_path1, delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
    genotypes1 = np.loadtxt(genotype_path1, delimiter='$').astype('float64')
    
    print("Datasets loaded.")
    print("Nr. of individuals: %i \nNr. of markers: %i" % np.shape(genotypes))
    print("Nr. of individuals DS1: %i \nNr. of markers: %i" % np.shape(genotypes1))
    
    # Do the binning
    if np.min([demes_x, demes_y]) > 0:
        positions, genotypes, _ = group_inds(positions, genotypes,
                                                    demes_x=demes_x, demes_y=demes_y, min_ind_nr=min_ind_nr) 
        
    print("Nr. of Demes left: %i" % len(positions))
        
    if np.min([demes_x1, demes_y1]) > 0:
        positions1, genotypes1, _ = group_inds(positions1, genotypes1,
                                                    demes_x=demes_x1, demes_y=demes_y1, min_ind_nr=min_ind_nr1)
    print("Nr. of Demes right: %i" % len(positions1))
    
    print("Grouping Finished!")
    
    # Calculate the mean sharing in Bins
    print("Calculating Homo Matrix....")
    bin_dist, bin_h, stand_errors = calc_bin_homos(positions, genotypes, bins=bins, max_dist=max_dist / float(scale_factor))
    bin_dist1, bin_h1, stand_errors1 = calc_bin_homos(positions1, genotypes1, bins=bins, max_dist=max_dist1 / float(scale_factor1))
    print("Finished Calculating.")
    

    
    x_vec = np.linspace(min(bin_dist), max(bin_dist), 100)
    x_vec1 = np.linspace(min(bin_dist1), max(bin_dist1), 100)
    

    # Calculate the best fits
    KC = fac_kernel("DiffusionK0")
    KC.set_parameters([best_fit_params[0], best_fit_params[1], 1.0, 0.0])  # [4*np.pi*6, 0.02, 1.0, 0.04]
    f_vec = KC.calc_f_vec(x_vec)
    ss = best_fit_params[-1]
    homo_vec_fit = f_vec + (1 - f_vec) * ss
    # print(f_vec)
    
    # Rescale:
    bin_dist = bin_dist * scale_factor
    x_plot = x_vec * scale_factor
    
    # Same for the other Kernel:
    KC = fac_kernel("DiffusionK0")
    KC.set_parameters([best_fit_params1[0], best_fit_params1[1], 1.0, 0.0])  # [4*np.pi*6, 0.02, 1.0, 0.04]
    f_vec1 = KC.calc_f_vec(x_vec1)
    ss1 = best_fit_params1[-1]
    homo_vec_fit1 = f_vec1 + (1 - f_vec1) * ss1
    
    # Rescale:
    bin_dist1 = bin_dist1 * scale_factor1
    x_plot1 = x_vec1 * scale_factor1
    
    # Do the Plotting:sub
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    ax1.plot(bin_dist, bin_h, "ro", label="Observed")
    ax1.plot(x_plot, homo_vec_fit, label="Best Fit", color="green", linewidth=2)
    ax1.set_ylabel("Pairwise h", fontsize=18)
    ax1.set_xlabel("[m]", fontsize=18)
    # ax1.legend(fontsize=18, loc="upper right")
    ax1.set_title("Hybrid Zone Data", fontsize=18)
    # plt.text(0.5, 0.5, "Nbh: %.3g\n m: %.4g" % (best_fit_params[0], best_fit_params[1]), fontsize=20, transform = ax.transAxes)
    
    ax2.plot(bin_dist1, bin_h1, "ro", label="Observed")
    ax2.plot(x_plot1, homo_vec_fit1, label="Best Fit", color="green", linewidth=2)
    # ax2.set_ylabel("Pairwise h", fontsize=18)
    ax2.set_xlabel(r"[$\sigma$]", fontsize=18)
    ax2.legend(fontsize=18, loc="upper right")
    ax2.set_title("Simulated Data", fontsize=18)
    plt.show()
    
    plt.plot()
    plt.plot(bin_dist, bin_h, "ro", label="Observed")
    plt.plot(x_plot, homo_vec_fit, label="Best Fit", color="green", linewidth=2)
    plt.ylabel("Pairwise h", fontsize=18)
    plt.xlabel("Pw. Distance [m]", fontsize=18)
    # ax1.legend(fontsize=18, loc="upper right")
    ax1.set_title("Hybrid Zone IBD", fontsize=18)
    plt.legend(fontsize=18, loc="upper right")
    plt.show()
    
    # Plot Dataset left: Left: IBD for pairwise heterozygosity. Right: Pairwise F
    # Wait until Reviewers demand that!
        
    
    
def plot_IBD_across_Zone(position_path, genotype_path, bins=30, max_dist=4.0,
                         nr_bootstraps=100, scale_factor=50):
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
    power = np.array(nr_nearby_inds)  # Number of comparisons to produce this
    
    
    def get_f_vec(position_list, genotype_mat, x_bins, bin_size=5, scale_factor=50):
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
    
    ax1.set_title(r"Within %i m" % (max_dist * 50))
    ax1.plot(x_coords_mean * scale_factor, f_res[:, 1:], 'go', alpha=0.2)
    ax1.plot(x_coords_mean * scale_factor, f_res[:, 0], 'ro', label="Empirical Values")
    ax1.set_ylabel(r"Excess $F$")
    ax1.hlines(0, min(position_list[:, 0]) * scale_factor, max(position_list[:, 0]) * scale_factor, alpha=0.8)
    ax1.legend(loc="upper right")
    
    f_mean_tot = np.array([mean_kinship_coeff(genotypes[nearby_inds, :], p_mean=0.5) for nearby_inds in close_inds])  # Get F-Mean over total HZ
    ax2.scatter(position_list[:, 0] * scale_factor, position_list[:, 1] * scale_factor, c=f_mean_tot)  # c = nr_nearby_inds
    for x in x_bins:
        plt.vlines(x, min(position_list[:, 1]), max(position_list[:, 1]), alpha=0.6)
    ax2.set_xlabel("x-Position [m]")
    ax2.set_ylabel("y-Position [m]")
    plt.show()
    

def plot_IBD_anisotropy(position_path, genotype_path, scale_factor=50):
    '''Plots IBD across Hybridzone. Makes Color Plot with respect to direction'''
    # Scale Factor is there to introduce it back to Plot.
    position_list = np.loadtxt(position_path, delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
    genotypes = np.loadtxt(genotype_path, delimiter='$').astype('float64')
    
    assert(len(position_list) == len(genotypes))
    print("Nr. of individuals: %i" % np.shape(genotypes)[0])
    
    print("\nCalculating Angle Matrix...")
    tic = time()
    rel_pos = position_list[:, None] - position_list  # Make the list of relative Positions
    dist_mat = np.linalg.norm(rel_pos, axis=2)
    # print(dist_mat[0, :5])
     
    angles_rad = np.arctan(rel_pos[:, :, 1] / (rel_pos[:, :, 0] + 0.000000001))
    angles_rad = angles_rad + np.pi * (rel_pos[:, :, 0] < 0)  # Add 180 degrees if x<0
    angles = np.degrees(angles_rad)  # Add something to denominator to make it numerically stable
    toc = time()
    print("Runtime calculating Angle Matrix: %.4f " % (toc - tic))
    
    # Calculate F-Matrix:
    print("\nCalculating F-Matrix... ")
    tic = time()
    f_mat = calc_f_mat(genotypes)
    toc = time()
    print("Runtime F-Matrix: %.4f" % (toc - tic))
    
    # Define the Bins:
    angle_bins = np.linspace(-90, 270, 17)
    angle_bins_rad = np.deg2rad(angle_bins)  # Convert for printing to Rad
    dist_bins = np.array([20, 100, 400, 2000]) / float(scale_factor)  # Float so that Division works
     
    # Calculate their means:
    angle_means = (angle_bins[1:] + angle_bins[:-1]) / 2.0
    dist_means = (dist_bins[1:] + dist_bins[:-1]) / 2.0
     
    angle_inds = np.digitize(angles, angle_bins)
    geo_inds = np.digitize(dist_mat, dist_bins)
     
    print(np.min(geo_inds))
    print(np.max(geo_inds))
     
    f_means = -np.ones((len(angle_bins) - 1, len(dist_bins) - 1)) * 100  # Set to minus 100; so that errors are obvious
    nr_comps = -np.ones((len(angle_bins) - 1, len(dist_bins) - 1)) * 100  # How the Individuals are distributed)
    f_mean = np.mean(f_mat)  # Calculate the overall mean
     
    for a in xrange(1, len(angle_bins)):
        for d in xrange(1, len(dist_bins)):
            print("\nParameters")
            print(angle_bins[a])
            print(dist_bins[d])
            # Extract corresponding individuals
            inds = np.where((angle_inds == a) & (geo_inds == d))  # Is 2D Array!!
            print("Nr. of individuals in bin: %i" % len(inds[0]))
            nr_comps[a - 1, d - 1] = len(inds[0])
            f_means[a - 1, d - 1] = t = np.mean(f_mat[inds] - f_mean)  # Take the mean f of all pairwise comps in this category
            print(f_means[a - 1, d - 1])
     
     
    print("Total Number of Comparisons: %i" % np.sum(nr_comps))
    # Make three length and nine angle categories
     
    # close_inds = dist_mat < max_dist  # Calculate Boolean Matrix of all nearby Individuals
    
    # Do the Plotting:
    
    
    azimuths = angle_bins_rad
    zeniths = np.array([0, 1.5, 2.3, 3])  # np.arange(0, 70, 10)
    
    r, theta = np.meshgrid(zeniths, azimuths)
    # values = np.random.random((azimuths.size, zeniths.size))
    # values = np.ones((azimuths.size, zeniths.size))
    values = np.log10(nr_comps)
    # values[0, 2] = 0.0 # Set 0 for testing reasons
    # values = np.random.random((10,4))
    
    #-- Plot... ------------------------------------------------
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    cax = ax.pcolor(theta, r, values)
    ax.set_yticklabels([])
    cbar = fig.colorbar(cax)  # ticks=[0, 0.5, 1]
    tick_pos = zeniths
    myticks = [str(i * scale_factor) + "m" for i in dist_bins]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(myticks)
    ax.set_rlabel_position(112.5)
    # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
    plt.title("Nr. Pairwise Comparisons")
    plt.show()
    
    # Plot F-Means:
    values = f_means
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    cax = ax.pcolor(theta, r, values)
    tick_pos = zeniths
    myticks = [str(i * scale_factor) + "m" for i in dist_bins]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(myticks)
    ax.set_rlabel_position(112.5)
    cbar = fig.colorbar(cax)  # ticks=[0, 0.5, 1]
    # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
    plt.title("Pairwise F")
    plt.show()
        
    

def multi_pos_plot(folder, method_folder, res_numbers=range(0, 200), nr_bts=20, real_barrier_pos=500.5,
                   true_gamma=0.05, true_nbh=62.831, plot_hlines=1, color_path="", scale_factor=1, real_barrier=True):
    '''Plots multiple Barrier positions throughout the area.
    Upper Plot: For every Barrier-Position plot the most likely estimate -
    as well as Bootstrap Estimates around it! Lower Plot: Plot the Positions of the
    Demes/Individuals'''
        
    # Load the Results
    res_vec = np.array([load_pickle_data(folder, i, 0, subfolder=method_folder) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, subfolder=method_folder) for i in res_numbers])
    
    # Put the Barrier Estimates >1 to 1:
    res_vec[:, 2] = np.where(res_vec[:, 2] > 1, 1, res_vec[:, 2])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(4):
            print("Parameter: %i" % j)
            print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
            
    # Mean Estimates:
    mean_inds = range(0, len(res_numbers), nr_bts)
    res_mean = res_vec[mean_inds, :]  
    # print(res_mean)  
    # print(res_mean[13])
            
    # Load the Barrier Positions:
    barrier_fn = "barrier_pos.csv"
    path = folder + method_folder + barrier_fn
    barrier_pos = np.loadtxt(path, delimiter='$').astype('float64') * scale_factor
    # print(barrier_pos[13]) 
    
    
    ##############################################################
    # Add upper bound here if only lower number of results are available
    b_max = (len(res_numbers) - 1) / nr_bts
    barrier_pos = barrier_pos[:b_max + 1]   
    ##############################################################
    
    nr_barriers = len(barrier_pos)
    print("%i Barrier Positions loaded:" % nr_barriers)
    print(barrier_pos)
    
    # Do some jitter for the x-Values:
    barrier_spacing = (barrier_pos[1] - barrier_pos[0])
    x_jitter = np.linspace(-1 / 3.0 * barrier_spacing, 1 / 3.0 * barrier_spacing, nr_bts)
    x_jitter = np.tile(x_jitter, len(barrier_pos))
    mid_barriers = np.repeat(barrier_pos, nr_bts)
    barr_pos_plot = mid_barriers + x_jitter
    
    # Define the colors for the plot:
    colors = ["coral", "cyan"]
    color_vec = np.repeat(colors, nr_bts)
    color_vec = np.tile(color_vec, nr_barriers)[:len(barr_pos_plot)]  # Gets the color vector (double and extract what needed)
    
    # Calculate the offset Vector:
    # offset_tot=barrier_pos[1]-barrier_pos[2]
    # barr_pos_plot = [val for val in barrier_pos for _ in xrange(nr_bts)]
    # print(barr_pos_plot)
    
    # Load the Position File:
    path_pos = folder + "mb_pos_coords00.csv"
    position_list = np.loadtxt(path_pos, delimiter='$').astype('float64') * scale_factor
    print("Position List loaded: %i Entries" % len(position_list))
    
        # Replace Color Vector if necessary:
    if len(color_path) == 0:
        color = ["DarkGray" for _ in position_list]
    else:
        color_path_full = folder + color_path
        color = np.loadtxt(color_path_full, delimiter='$', dtype='str').astype('str')
    
    # Do the plotting:
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # Plot the Nbh Estiamtes:
    inds_sorted, true_inds = argsort_bts(res_vec[:, 0], nr_bts)
    ax1.scatter(barr_pos_plot, res_vec[inds_sorted, 0], c=color_vec, label="Bootstrap Estimate", alpha=0.8, zorder=1)
    ax1.plot(barr_pos_plot[true_inds], res_mean[:, 0], 'o', label="Estimate", color="k")
    ax1.set_ylim([0, 350])
    ax1.set_ylabel("Nbh", fontsize=18)
    if plot_hlines:
        ax1.hlines(true_nbh, min(position_list[:, 0]), max(position_list[:, 0]), linewidth=2, color=c_lines)
    # ax1.title.set_text("No Barrier")
    ax1.legend(loc="upper right")
    
    inds_sorted, true_inds = argsort_bts(res_vec[:, 2], nr_bts)
    ax2.scatter(barr_pos_plot, res_vec[inds_sorted, 2], label="Bootstrap Estimates", alpha=0.8, c=color_vec, zorder=1)
    ax2.plot(barr_pos_plot[true_inds], res_mean[:, 2], 'o', label="Estimate", color="k")
    ax2.set_ylim([0, 1])
    ax2.set_ylabel(r"$\gamma$", fontsize=18)
    if plot_hlines:
        ax2.hlines(true_gamma, min(position_list[:, 0]), max(position_list[:, 0]), linewidth=2, color=c_lines)
    # ax1.title.set_text("No Barrier")
    # ax2.legend(loc="lower right")
    
    # Plot the Positions:
    ax3.scatter(position_list[:, 0] , position_list[:, 1], c=color, s=4, zorder=1, label="Sample Position")  # c = nr_nearby_inds
    ax3.set_xlabel("x-Position", fontsize=18)
    ax3.set_ylabel("y-Position", fontsize=18)
    
    # Plot first Barrier Position (for labelling)
    ax3.vlines(barrier_pos[0], min(position_list[:, 1]), max(position_list[:, 1]), alpha=0.8, linewidth=3, label="Putative Barrier", zorder=2) 
    for x in barrier_pos[1:]:  # Plot rest
        ax3.vlines(x, min(position_list[:, 1]), max(position_list[:, 1]), alpha=0.8, linewidth=2, zorder=2)
    if real_barrier:
        ax3.vlines(real_barrier_pos, min(position_list[:, 1]), max(position_list[:, 1]), color="red", linewidth=3, label="True Barrier", zorder=3)
    ax3.legend(loc="upper right")
    plt.show()
    
def multi_pos_plot_k_only(folder, method_folder, res_numbers=range(0, 200), nr_bts=20, real_barrier_pos=500.5, plot_hlines=1):
    '''Plots multiple Barrier positions throughout the area. Plots k only.
    Upper Plot: For every Barrier-Position plot the most likely estimate -
    as well as Bootstrap Estimates around it! Lower Plot: Plot the Positions of the
    Demes/Individuals'''
    
    # Load the Results
    res_vec = np.array([load_pickle_data(folder, i, 0, subfolder=method_folder) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, subfolder=method_folder) for i in res_numbers])
    
    # Put the Barrier Estimates >1 to 1:
    res_vec[res_numbers, 0] = np.where(res_vec[res_numbers, 0] > 1, 1, res_vec[res_numbers, 0])
    
    for l in range(len(res_numbers)):
        i = res_numbers[l]
        print("\nRun: %i" % i)
        for j in range(len(res_vec[0, :])):
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
    
    ##############################################################
    # Add upper bound here if lower number of results are available
    b_max = np.max(res_numbers) / nr_bts
    barrier_pos = barrier_pos[:b_max + 1]  
    ##############################################################
    
    print("Barrier Positions loaded: ")
    print(barrier_pos)
    barr_pos_plot = [val for val in barrier_pos for _ in xrange(nr_bts)]
    # print(barr_pos_plot)
    
    # Load the Position File:
    path_pos = folder + "mb_pos_coords00.csv"
    position_list = np.loadtxt(path_pos, delimiter='$').astype('float64')
    print("Position List loaded: %i Entries" % len(position_list))
    
    
    # Do the plotting:
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Fit Barrier Only")
    ax1.plot(barr_pos_plot, res_vec[res_numbers, 0], 'ko', label="Bootstrap Estimate", alpha=0.5)
    ax1.plot(barrier_pos, res_mean[:, 0], 'go', label="Mean Estimate")
    ax1.set_ylim([0, 1])
    ax1.set_ylabel("Barrier", fontsize=18)
    if plot_hlines:
        ax1.hlines(0.05, min(position_list[:, 0]), max(position_list[:, 0]), linewidth=2, color="y")
    

    
    # Plot the Positions:
    ax2.scatter(position_list[:, 0], position_list[:, 1])  # c = nr_nearby_inds
    ax2.set_xlabel("x-Position", fontsize=18)
    ax2.set_ylabel("y-Position", fontsize=18)
    for x in barrier_pos:
        ax2.vlines(x, min(position_list[:, 1]), max(position_list[:, 1]), alpha=0.8, linewidth=3)
    ax2.vlines(real_barrier_pos, min(position_list[:, 1]), max(position_list[:, 1]), color="red", linewidth=6, label="True Barrier")
    ax2.legend()
    plt.show()
    

def sim_idea_grid(save=True, load=True, path="idea_sim.p", winter=False):
    '''Method to simulate the idea used for inference.
    Produces figure in Paper
    Use Grid Model to simulate'''
    
    # Do the simulations:
    # Parameters:
    grid_size = 1000
    mid = grid_size / 2
    sigma, ips, mu = 1.2, 1.0, 0.001
    p_mean = 0.5
    t = 5000
    
    # First: Set the position_list:
    position_list = np.array([[mid + i, mid + j] for i in xrange(-30, 30) for j in xrange(-30, 30)])
    l = int(np.sqrt(len(position_list)))  # Length of position list along given axis
    
    if save == True:
        # Do the Simulation without Barrier
        grid = Grid()  # Creates new Grid. Maybe later on use factory Method
        grid.set_parameters(grid_size, grid_size, sigma, ips, mu)
        grid.set_samples(position_list)
        grid.update_grid_t(t, p=p_mean, barrier=0)  # Uses p_mean[i] as mean allele Frequency.
        genotypes = grid.genotypes
        
        # Do the Simulations with Barrier
        grid_b = Grid()  # Creates new Grid. Maybe later on use factory Method
        grid_b.set_parameters(grid_size, grid_size, sigma, ips, mu)
        grid_b.set_barrier_parameters(mid + 0.5, barrier_strength=0.0)  # Where to set the Barrier and its strength
        grid_b.set_samples(position_list)
        grid_b.update_grid_t(t, p=p_mean, barrier=1)  # Uses p_mean[i] as mean allele Frequency.
        genotypes_b = grid_b.genotypes
        
        pickle.dump((position_list, genotypes, genotypes_b), open(path, "wb"))  # Pickle Dump the Data
        
    
    if load == True:
        res_vec = pickle.load(open(path, "rb"))  # Pickle Load the Data
        position_list, genotypes, genotypes_b = res_vec
    
    # Do the smoothing
    def smooth(positions, genotypes, sigma=1.2):
        '''Smooth out the Genotypes with Kernel'''
        # Calculate the Distance Matrix
        start = time()
        dist_mat = np.linalg.norm(positions[:, None] - positions, axis=2)
        print("Dist Mat Calculation Time: %.3f" % (time() - start))
        
        # Smmothing
        start = time()
        p_mat = np.array(genotypes)  # Just in case that not numpy array
        
        print("Dist Mat Shape:")
        print(np.shape(dist_mat))
        
        print("Shape Genotype")
        print(np.shape(genotypes))
        
        weights = 1 / (2.0 * np.pi * sigma ** 2) * np.exp(-dist_mat ** 2 / (2.0 * sigma ** 2))  # Calculate the Gaussian weights
        p_mean = np.dot(weights, p_mat) / np.sum(weights, axis=1)  # Calculate weighted mean
        print("Time taken Gaussian Smoothing: %.3f" % (time() - start))
        return p_mean
        
        # Use the Gaussian for smoothing
    
    p_mean = smooth(position_list, genotypes)
    p_mean_b = smooth(position_list, genotypes_b)
    
    
    p_mean = np.reshape(p_mean, (l, l)).T
    p_mean_b = np.reshape(p_mean_b, (l, l)).T
    
    p_dis = np.reshape(genotypes, (l, l)).T
    p_dis_b = np.reshape(genotypes_b, (l, l)).T
    
    # Make a custom Color Map:
    
    
    # Do the plotting
    x_coords, y_coords = position_list[:, 0], position_list[:, 1]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Make a custom color Map:
    # Get the colors first:
    cmap0 = cm.get_cmap('jet')
    red = cmap0(0.85)
    blue = cmap0(0.05)
    
    # Make the colormap
    cmap_dis = colors.ListedColormap([blue, red])  # ["Black", "Gainsboro"]
    bounds = [0, 0.5, 1.0]
    norm = colors.BoundaryNorm(bounds, cmap_dis.N)
    
    tf_size = 20
    label_size = 12
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    
    ax1.imshow(p_dis, cmap=cmap_dis, norm=norm)
    ax1.set_xlim(1, l - 1)
    ax1.set_ylim(1, l - 1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("No Barrier", fontsize=tf_size)
    ax1.set_ylabel("Observed Genotypes", fontsize=tf_size)

    # ax2.scatter(position_list[:, 0], position_list[:, 1], c=genotypes_b)
    im0 = ax2.imshow(p_dis_b, cmap=cmap_dis, norm=norm)
    ax2.set_title("Strong Barrier", fontsize=tf_size)
    # ax2.set_xlim([x_min, x_max])
    # ax2.set_ylim([y_min, y_max])
    ax2.set_xlim(1, l - 1)
    ax2.set_ylim(1, l - 1)
    # ax2.vlines(mid + 0.5, y_min, y_max, color="k", linewidth=5,zorder=1)
    ax2.vlines(l / 2, 0, l, color="k", linewidth=5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(location="upper right")
    
    ax3.imshow(p_mean, interpolation="bilinear", cmap="jet")
    ax3.set_xlim(1, l - 1)
    ax3.set_ylim(1, l - 1)
    # ax3.set_xlabel("x-Axis", fontsize=label_size)
    # ax3.set_ylabel("y-Axis", fontsize=label_size)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.autoscale(False)
    ax3.set_adjustable('box-forced')
    ax3.set_ylabel("Smoothed", fontsize=tf_size)
    # Some Code for coordinate arrows:
    ax3.arrow(3, 3, 0, 10, head_width=1.0, head_length=2, fc='k', ec='k')
    ax3.arrow(3, 3, 10, 0, head_width=1.0, head_length=2, fc='k', ec='k')
    ax3.annotate("x", (13, 4))
    ax3.annotate("y", (4, 13))
    if winter == True:
        # ax3.text(0.6 * l, 0.86 * l, "Winter is coming", fontsize=8, color="blue", alpha=0.2)
        ax3.text(0.4 * l, 0.86 * l, "Winter is coming", fontsize=16, color="lightblue", alpha=0.15)  # Trololol In paper 8 and blue
    
    im = ax4.imshow(p_mean_b, interpolation="bilinear", cmap="jet")
    ax4.set_xlim(1, l - 1)
    ax4.set_ylim(1, l - 1)
    ax4.vlines(l / 2, 0, l, color="k", linewidth=5)
    # ax4.vlines(l / 2, 0, l, linewidth=15, color="k")
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.autoscale(False)
    ax4.set_adjustable('box-forced')
    
    # Plot the custom colorbars:
    f.subplots_adjust(right=0.85)
    cbar_ax = f.add_axes([0.87, 0.17, 0.015, 0.3])  # left, bottom, width, height
    f.colorbar(im, cax=cbar_ax)
    f.text(0.92, 0.32, 'Allele Frequency', va='center', rotation=270, fontsize=tf_size)
    
    cbar_ax1 = f.add_axes([0.87, 0.54, 0.015, 0.3])  # left, bottom, width, height
    f.colorbar(im0, cax=cbar_ax1, cmap=cmap_dis, ticks=[0, 1])
    f.text(0.92, 0.69, 'Allele', va='center', rotation=270, fontsize=tf_size)
    # cbar = f.colorbar(cax)
    # plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.13, right=None, top=None,
                wspace=0.07, hspace=0)
    plt.show()

###############################################################################
# ## Helper Functions for Plotting Coalescence Times
# Do the analytical approximations:
# Formulas are valid for x0>0; flip if needed!  
  
def gaussian1d(t, dy, D=1):
    '''The One Dimensional Gaussian. 
    Differnce: Here dy notes the difference along the y axis'''
    return 1.0 / np.sqrt(4 * np.pi * D * t) * np.exp(-dy ** 2 / (4 * D * t))

def GS(t, y, x, k=1.0, D=1):
    '''1D Diffusion for same side of the Barrier'''
    n1 = np.exp(-(x - y) ** 2 / (4 * D * t)) + np.exp(-(x + y) ** 2 / (4 * D * t))
    d1 = np.sqrt(4 * np.pi * D * t)

    a2 = k / D * np.exp(2 * k / D * (y + x + 2 * k * t))
    b2 = erfc((y + x + 4 * k * t) / (2 * np.sqrt(D * t)))
    res = n1 / d1 - a2 * b2
    if np.isnan(res) or np.isinf(res):  # Check if numerical instability
        return gaussian1d(t, y - x, D=D)  # Fall back to Gaussian (to which one converges)
    else: return res
        
def GD(t, y, x, k=1.0, D=1):
    '''1D Diffusion for different sides of the Barrier'''
    a1 = k / D * np.exp(2 * k / D * (y - x + 2 * k * t))
    b1 = erfc((y - x + 4 * k * t) / (2 * np.sqrt(D * t)))
    res = a1 * b1
    if np.isnan(res) or np.isinf(res):  # Check if numerical instability
        return gaussian1d(t, y - x, D=D)  # Fall back to Gaussian (to which one converges)
    else: return res
    
def coal_prob_ss(t, dy, x0, x1, k=1.0, D=1.0, De=5):
    '''The integrand in case there is no barrier
    Product of 1d Gaussian along y-Axis and x-Axis Barrier Pdf.
    And a term for the long-distance migration'''
    return (gaussian1d(t, dy, D=D) * GS(t, x0, x1, k=k, D=D) / (2 * De))

def coal_prob_ds(t, dy, x0, x1, k=1.0, D=1.0, De=5):
    '''the integrand for cases of different sided of the barrier.
    Product of 1d Gaussian along y-Axis
    And a term for the long-distance migration'''
    return (gaussian1d(t, dy, D=D) * GD(t, x0, x1, k=k, D=D) / (2 * De))

def coal_prob(t, dy, x0, x1, k=1.0, D=1.0, De=5):
    '''Coal Prob at time t.
    Flip x coords if needed and choose according subfunction'''
    # Flip if needed
    if x0 < 0:
        x0 = -x0
        x1 = -x1
        
    if x1 > 0:
        prob = coal_prob_ss(t, dy, x0, x1, k, D, De)
        
    if x1 < 0:
        prob = coal_prob_ds(t, dy, x0, x1, k, D, De)
        
    return prob


###############################################################################
def plot_coal_times():
    '''Method to plot the simulated Coalescence Times'''
    data_folder = "coal_times/"  # Where to save the Results to
    # Load all the simulated Data:
    save_names = np.array([["short_1818.csv", "short_1212.csv", "short_1821.csv"],
                  ["long_1818.csv", "long_1212.csv", "long_1821.csv"],
                  ["short_barrier_1818.csv", "short_barrier_1212.csv", "short_barrier_1821.csv"],
                  ["long_barrier_1818.csv", "long_barrier_1212.csv", "long_barrier_1821.csv"]])
    
    x0s = [-1.5, -7.5, -1.5]
    x1s = [-1.5, -7.5, 1.5]
    
    k0 = 1.0  # The weak barrier
    k1 = 0.01  # The strong barrier
    
    # ## Nr of Replicates
    reps_short = 1000000
    reps_long = 100000
    
    # ##
    De = 5.0  # Density
    D = 1.0  # Diffusion
    
    # ## Define the Bins:
    # Short:
    t_min = 5
    t_max = 205
    bin_width = 10
    bins_s = np.array([t_min - 0.5 + i * bin_width for i in xrange(int((t_max - t_min) / float(bin_width)))])
    means_s = (bins_s[1:] + bins_s[:-1]) / 2.0
    bin_width_s = (bins_s[1] - bins_s[0])
    norm_s = (reps_short * bin_width_s)  # The Normalization Factor for Short
    
    # Long:
    t_min_l = 100
    t_max_l = 50100
    bin_width_l = 2000
    bins_l = np.array([t_min_l - 0.5 + i * bin_width_l for i in xrange(int((t_max_l - t_min_l) / float(bin_width_l)))])
    means_l = (bins_l[1:] + bins_l[:-1]) / 2.0
    bin_width_l = (bins_l[1] - bins_l[0])
    norm_l = (reps_long * bin_width_l)  # The Normalization Factor for Long
    
    # ## Prepare the Results Containers:
    coal_results_s = np.zeros((3, 2, len(means_s)))  # Coal Results Short
    coal_results_l = np.zeros((3, 2, len(means_l)))  # Coal Results Long
    ana_predicts_s = np.zeros((3, 2, len(means_s)))  # Analytical Predictions Short
    ana_predicts_l = np.zeros((3, 2, len(means_l)))  # Analytical Predictions Long
    mean_s = np.zeros((3, 2))
    mean_l = np.zeros((3, 2))
    
    for i in xrange(3):
        # Do all the Precalcs:
        
        # Load the Data
        path_s = data_folder + save_names[0, i]
        path_bar_s = data_folder + save_names[2, i]
        path_l = data_folder + save_names[1, i]
        path_bar_l = data_folder + save_names[3, i]
        
        s = np.loadtxt(path_s, dtype="int")
        l = np.loadtxt(path_l, dtype="int") 
        s_b = np.loadtxt(path_bar_s, dtype="int")
        l_b = np.loadtxt(path_bar_l, dtype="int")
        
        # Bin and store the data:
        ns, _ = np.histogram(s, bins=bins_s)  # Normed = True
        ns_b, _ = np.histogram(s_b, bins=bins_s)
        nl, _ = np.histogram(l, bins=bins_l)
        nl_b, _ = np.histogram(l_b, bins=bins_l)
        
        # Normalize and store
        coal_results_s[i, 0, :] = ns / norm_s
        coal_results_s[i, 1, :] = ns_b / norm_s
        coal_results_l[i, 0, :] = nl / norm_l
        coal_results_l[i, 1, :] = nl_b / norm_l
        
        # Do the analytical Approximations:
        x0 = x0s[i]
        x1 = x1s[i]
        ana_predicts_s[i, 0, :] = [coal_prob(t=t, dy=0, x0=x0, x1=x1, k=k0, D=D, De=De) for t in means_s]  # No Barrier
        ana_predicts_s[i, 1, :] = [coal_prob(t=t, dy=0, x0=x0, x1=x1, k=k1, D=D, De=De) for t in means_s]  # Barrier
        ana_predicts_l[i, 0, :] = [coal_prob(t=t, dy=0, x0=x0, x1=x1, k=k0, D=D, De=De) for t in means_l]  # No Barrier
        ana_predicts_l[i, 1, :] = [coal_prob(t=t, dy=0, x0=x0, x1=x1, k=k1, D=D, De=De) for t in means_l]  # Barrier
        
        # Calculate the Means:
        mean_s[i, 0] = np.mean(s)
        mean_s[i, 1] = np.mean(s_b)
        mean_l[i, 0] = np.mean(l)
        mean_l[i, 1] = np.mean(l_b)
        
        ##########################################
        ##########################################
        # ## Now do the plot

    f, axes = plt.subplots(3, 2, figsize=(8, 10))  # ((ax1, ax2), (ax3, ax4))
    # axes[0,0].xaxis.set_ticklabels([])
    
    for i in xrange(3):
        '''Make the three Subfigures'''
        # The Short Pic
        ax = axes[i, 0]
        ax.plot(means_s, coal_results_s[i, 0, :], "ro-", label="No Barrier")
        ax.plot(means_s, coal_results_s[i, 1, :], "bo-", label="Barrier")
    
        ax.plot(means_s, ana_predicts_s[i, 0, :], label="Approx. No Barrier", color="Orange", linewidth=4)
        ax.plot(means_s, ana_predicts_s[i, 1, :], label="Approx. Barrier", color="LightBlue", linewidth=4)
    
        # plt.xlabel("Coalescence Probability", fontsize=20)
        # plt.ylabel("Density", fontsize=20)
        # The Long Pic
        ax = axes[i, 1]
        ax.plot(means_l, coal_results_l[i, 0, :], "ro-")
        ax.plot(means_l, coal_results_l[i, 1, :], "bo-")
        
        ax.plot(means_l, ana_predicts_l[i, 0, :], color="Orange", linewidth=4)
        ax.plot(means_l, ana_predicts_l[i, 1, :], color="LightBlue", linewidth=4)
        
        # Plot the Means:
        ax.axvline(mean_l[i, 0], color='r', linestyle='dashed', linewidth=2, label="Mean No Barrier")
        ax.axvline(mean_l[i, 1], color='b', linestyle='dashed', linewidth=2, label="Mean Barrier")
        
        # Do the big Axes Labels:
        plt.text(0.6, 0.4, r"$x_0=%.1f$" % x0s[i] + "\n" + r"$x_1=%.1f$" % x1s[i], transform=ax.transAxes, fontsize=12)
        
    # Do the labelling Spice:
    # Turn xlabels of
    for i in xrange(2):
        for j in xrange(2):
            axes[i, j].xaxis.set_visible(False)
            
    for i in xrange(3):
            axes[i, 1].yaxis.tick_right()
            axes[i, 0].set_ylim([0, 0.001])
            axes[i, 1].set_ylim([0, 0.00008])
    
    axes[0, 0].legend(fontsize=12)
    axes[0, 1].legend(fontsize=12)
    axes[0, 0].set_title("Intermediate Timescale", fontsize=18)
    axes[0, 1].set_title("Long Timescale", fontsize=18)
    # axes.yaxis.set_visible(False)
    plt.gcf().text(0.5, 0.04, "Coalescence Time [Gen]", ha="center", fontsize=18)  # Set the x-Label
    plt.gcf().text(0.025, 0.5, 'Probability [per Gen]', ha='center', va='center', rotation='vertical', fontsize=18)
    # plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=0.05, hspace=0.1)
    plt.show()

def plot_variation_param_estimates(folder, method=2, res_numbers=range(300), res_folder=None,
                                   pr=True, k=0.01, reps=100, outlier=1.0):
    '''Plot what happens when various Parameters are varied
    pr  Print Results if needed'''
    array_l = 5  # How often Parameters have been varied.
    nr_varies = 3 
    nr_params = 5
    assert(reps * array_l * nr_varies == len(res_numbers))  # Sanity Check
    
    # The Values used in the Simulations
    mu_vec = np.array([0.001, 0.003, 0.005, 0.007, 0.009]) * 2
    sd_p_vec = np.array([0.04, 0.06, 0.08, 0.1, 0.12])
    ips_vec = np.array([6, 10, 14, 18, 22]) * 4 * np.pi
    
    
    batch_l = reps * array_l
    
    # Load the Results
    res_vec = np.array([load_pickle_data(folder, i, 0, method, subfolder=res_folder) for i in res_numbers])
    unc_vec = np.array([load_pickle_data(folder, i, 1, method) for i in res_numbers])
    
    # Print the results
    if pr == True:
        for l in range(len(res_numbers)):
            i = res_numbers[l]
            print("\nRun: %i" % i)
            for j in range(3):
                print("Parameter: %i" % j)
                print("Value: %f (%f,%f)" % (res_vec[l, j], unc_vec[l, j, 0], unc_vec[l, j, 1]))
                
    
    
    
    # Prepare some Containers for the Summary Results for Barrier Strength:
    means = np.zeros((nr_varies, array_l))
    stds = np.zeros((nr_varies, array_l))
    
    # Calculate the Summary Results
    barriers = res_vec[:, 2]  # Load all the estimated Barriers:
    
    param_names = ["m", r"\sigma(\bar{p})", "Nb"]
    x_arrays = np.array([mu_vec, sd_p_vec, ips_vec])  # Produces the Objects for the x-Axis
    for i in xrange(nr_varies):
        for j in xrange(array_l):
            start = batch_l * i + j * reps
            end = batch_l * i + (j + 1) * reps
            
            sub_array = barriers[start:end]  # Extract the right Subarray
            sub_array = sub_array[sub_array < outlier]  # To remove outliers
            
            means[i, j] = np.mean(sub_array)  # calculate its Mean
            stds[i, j] = np.std(sub_array)  # Calculate its Standarddeviation
            errors = 1 / np.sqrt(reps) * stds  # Calculate the Error
            
            # Plot some histograms of estimates - comment out if interested!
            # x_vec = np.arange(len(sub_array))
            # plt.figure()
            # plt.plot(x_vec, sub_array, "ro")
            # plt.title("Parameter: %s, Value: %.3f" % (param_names[i], x_arrays[i,j]))
            # plt.show()
    
    # Do the plot
    # Define the color:
    c = "blue"
    fs = 18  # Fontsize for axis labels
    
    x_arrays = np.array([mu_vec, sd_p_vec, ips_vec])  # Produces the Objects for the x-Axis
    f, axes = plt.subplots(2, 3, figsize=(10, 5), sharex='col', sharey='row')
    
    ms = 7
    for i in xrange(3):
        '''Vary over rows'''
        x_array = x_arrays[i]  # Load the right x-Array
        
        ax = axes[0, i]  # Upper Panel: Mean 
        ax.errorbar(x_array, means[i, :], yerr=errors[i, :], fmt="o-", color="red", markersize=ms)
        ax.set_ylim([0, 0.03])
        ax.axhline(k, label=r"True $\gamma$", color="green", linewidth=2, zorder=0)
        
        ax = axes[1, i]  # Lower Panel: STD
        ax.plot(x_array, stds[i, :], "bo-", markersize=ms)
    
    # Do the general Plotting
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].set_ylabel("Mean", fontsize=fs)
    axes[1, 0].set_ylabel("STD", fontsize=fs)
    
    axes[1, 0].set_xlabel("m", fontsize=fs)
    axes[1, 1].set_xlabel(r"$\sigma(\bar{p})$", fontsize=fs)
    axes[1, 2].set_xlabel("Nbh", fontsize=fs)
    f.text(0.92, 0.5, r'Estimates of $\gamma$', va='center', rotation=270, fontsize=fs)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=0.07, hspace=0.07)
    # Turn off the right axis labels
    # plt.gcf().text(0.5, 0.04, r"$\gamma=0.01$", ha="center", fontsize=18)  # Set the x-Label
    
    plt.show()
    
######################################################
if __name__ == "__main__":
    '''Here one chooses which Plot to do:'''
    # sim_idea_grid(save=False, load=True, winter=True)  # Simulate the Idea of the Grid
    # plot_coal_times()  # Plots the Distribution of Coalescence Times.
    # plot_variation_param_estimates("./multi_param_weak/", method=2, k=0.2, reps=20)  # The plot for a strong Barrier
    # plot_variation_param_estimates("./multi_param_strong/", method=2, k=0.01, reps=20)    # The plot for a weak Barrier
    # plot_variation_param_estimates("./multi_param_strong1/", method=2, k=0.02, res_numbers=range(1500), reps=100, outlier=0.08) # Plot for 100 Reps of gamma = 0.02
    # plot_variation_param_estimates("./multi_param_intermediate/", method=2, k=0.1, res_numbers=range(1500), reps=100, outlier=0.28) # The plot for the intermediate Barrier
    
    
    # multi_nbh_single(multi_nbh_folder, method=0, res_numbers=range(0,100))
    # multi_nbh_all(multi_nbh_folder, res_numbers=range(0, 100))
    # multi_nbh_single(multi_nbh_gauss_folder, method=0, res_numbers=range(0,100))
    # multi_ind_single(multi_ind_folder1, method=0) 
    # multi_ind_all(multi_ind_folder1, range(0, 100))  # 3x3 Plot of all Methods    # All three methods for Multi Ind
    # multi_loci_single(multi_loci_folder, method=0, res_numbers=range(0, 21))
    # multi_loci_all(multi_loci_folder) # 3x3 Plot of all Methods   # Partially implemented
    
    # multi_barrier_single(multi_barrier_folder, method=2)  # Mingle with the above for different Barrier Strengths.
    # multi_barrier10("./barrier_folder10/")  # Print the 10 Barrier Data Sets
    # multi_bts_barrier("./multi_barrier_bts/")  # "./multi_barrier_bts/" Plots the Bootstrap Estimates for various Barrier Strengths
    # multi_barrier_loci("./multi_loci_barrier/")  # Plots the Estimates (Including Barrier) across various Numbers of Loci (To detect Power)
    
    # plot_theory_f()  # Plots the theoretical F; from Kernel calculations. Fig. 3: Decay of IBD in data.
    plot_theory_f_local_barrier()  # For David's grant
    
    
    # multi_secondary_contact_single(secondary_contact_folder_b, method=2)
    # multi_secondary_contact_all(secondary_contact_folder, secondary_contact_folder_b, method=2)
    
    # cluster_plot(cluster_folder, method=2)
    # boots_trap("./bts_folder_test/", method=2)   # Bootstrap over Test Data Set: Dataset 00 from cluster data-set; clustered 3x3
    # ll_barrier("./barrier_folder1/")
    # multi_pos_plot(multi_pos_syn_folder, met2_folder, res_numbers=range(0, 300))
    # Plot for simulated Data with HZ Parameters:
    # multi_pos_plot("./barrier_folder_HZ_synth/", "method2/", res_numbers=range(0, 300), true_nbh=200, true_gamma=0.02)
    
    # multi_pos_plot_k_only("./multi_barrier_hz_ALL/all_v2.4/", method_folder="k_only/", res_numbers=range(0,300), nr_bts=20, real_barrier_pos=500) # k_only
    
    
    # ## Plots for Hybrid Zone Data
    # For Dataset where Demes are weighted
    # multi_pos_plot(multi_pos_hz_folder, "all/", nr_bts=20, real_barrier_pos=2, res_numbers=range(0, 460))
    
    # For Dataset where Demes are not weighted; m.d.: 4200  
    # Plot for Paper!!!
    # multi_pos_plot("./multi_barrier_hz_ALL/chr0/", "result/", nr_bts=20 , real_barrier_pos=2, res_numbers=range(0, 460), plot_hlines=0, color_path="colorsHZALL.csv",
    #               scale_factor=50, real_barrier=False)
    
    
    
    ###############################
    # Try-out Plots for all Data
    # multi_pos_plot("./multi_barrier_hz_ALL/all_v2.4/", "result/", nr_bts=10 , real_barrier_pos=2, res_numbers=range(0, 250), plot_hlines=0, color_path="colorsHZALL.csv",
    #             scale_factor=50, real_barrier=False) # The Plot for Paper
    
    # For 2014 Dataset: 
    # multi_pos_plot("./multi_barrier_hz_ALL14/min25/", "result/", nr_bts=10 , real_barrier_pos=2, res_numbers=range(0, 250), plot_hlines=0, color_path="colorsHZALL14.csv",
    #            scale_factor=50, real_barrier=False) 
    
    #
    # multi_pos_plot_k_only("./multi_barrier_hz/chr0/", method_folder="k_only/", res_numbers=range(0, 360), nr_bts=20, real_barrier_pos=2, plot_hlines=0)
    
    
    # hz_barrier_bts(hz_folder, "barrier2/")  # Bootstrap over all Parameters for Barrier Data
    # barrier_var_pos(hz_folder, "barrier18p/", "barrier2/", "barrier20m/", method=2) # Bootstrap over 3 Barrier pos
    
    # ## Bootstrap in HZ to produce IBD fig
    # plot_IBD_bootstrap("./Data/coordinatesHZALL0.csv", "./Data/genotypesHZALL0.csv", hz_folder, "barrier2/", res_number=20, nr_bootstraps=100,
    #                   plot_bootstraps=False)    
    # plot_IBD_bootstrap("./Data/coordinatesHZall2.csv", "./Data/genotypesHZall2.csv", multi_pos_hz_folder, "range_res/", res_number=100, nr_bootstraps=100,
    #                   plot_bootstraps=False)
    # plot_IBD_bootstrap("./hz_folder/hz_file_coords00.csv","./hz_folder/hz_file_genotypes00.csv", hz_folder, "barrier2/", res_number=100, nr_bootstraps=20)
    
    # plot_IBD_bootstrap("./nbh_folder/nbh_file_coords30.csv", "./nbh_folder/nbh_file_genotypes30.csv", hz_folder, "barrier2/")  # Bootstrap Random Data Set
    # plot_IBD_across_Zone("./multi_barrier_hz_ALL/chr0/mb_posHZ_coords00.csv", "./multi_barrier_hz_ALL/chr0/mb_posHZ_genotypes00.csv", bins=20, max_dist=4, nr_bootstraps=25)  # Usually the dist. factor is 50
    # plot_IBD_anisotropy("./multi_barrier_hz_ALL/chr0/mb_posHZ_coords00.csv", "./multi_barrier_hz_ALL/chr0/mb_posHZ_genotypes00.csv")
    
    # give_result_stats(hz_folder, subfolder="barrier20m/")
    
    ##### Plot pairwise homozgyosity against distance:
    # plot_homos("./Data/coordinatesHZALL2.csv", "./Data/genotypesHZALL2.csv", bins=15, max_dist=3000, best_fit_params=[218.57, 0.000038, 0.52371],
    #                     bootstrap=False, nr_bootstraps=50, scale_factor=50, title="Antirrhinum Data")  # Plot for Hybrid Zone Data
    
    # For Data from Barrier10 Dataset. Take Dataset Nr. 199
    # plot_homos("./barrier_folder10/barrier_file_coords199.csv", "./barrier_folder10/barrier_file_genotypes199.csv",
    #          bins=15, max_dist=20, best_fit_params=[67.74, 0.0107, 0.52343], bootstrap=False, nr_bootstraps=50,
    #          scale_factor=1, deme_bin=True, title="Simulated Data Set")  
    
    # Plots the two Homozygote Plots in one:
    # All Year Estimates:
    # plot_homos_2(position_path="./multi_barrier_hz_ALL/all_v2.5/mb_posHZ_coords00.csv", genotype_path="./multi_barrier_hz_ALL/all_v2.5/mb_posHZ_genotypes00.csv", 
    #            position_path1="./barrier_folder10/barrier_file_coords199.csv", genotype_path1="./barrier_folder10/barrier_file_genotypes199.csv", 
    #            bins=12, max_dist=1800, max_dist1=20, 
    #            best_fit_params=[192.203738, 0.000839, 0.528088], best_fit_params1=[67.74, 0.0107, 0.52343],
    #            scale_factor=50, scale_factor1=1, demes_x=100, demes_y=20, demes_x1=30, demes_y1=20, min_ind_nr=5)
    
    # Compared to data simulated under HZ parameters:
    # plot_homos_2(position_path="./multi_barrier_hz_ALL/chr0/mb_posHZ_coords00.csv", genotype_path="./multi_barrier_hz_ALL/chr0/mb_posHZ_genotypes00.csv", 
    #          position_path1="./barrier_folder_HZ_synth/mb_pos_coords00.csv", genotype_path1="./barrier_folder_HZ_synth/mb_pos_genotypes00.csv", 
    #          bins=15, max_dist=1800, max_dist1=25, 
    #          best_fit_params=[188.12, 0.0004426, 0.5257], best_fit_params1=[150.6, 0.0055816, 0.52735],
    #          scale_factor=50, scale_factor1=1, demes_x=50, demes_y=10, demes_x1=30, demes_y1=20, min_ind_nr=5)  #192.2, 0.000839, 0.528088
    
    
    # 2014 Estimates:
    # plot_homos_2(position_path="./multi_barrier_hz_ALL14/min25/mb_posHZ_coords00.csv", genotype_path="./multi_barrier_hz_ALL14/min25/mb_posHZ_genotypes00.csv", 
    #       position_path1="./barrier_folder10/barrier_file_coords199.csv", genotype_path1="./barrier_folder10/barrier_file_genotypes199.csv", 
    #       bins=50, max_dist=3000, max_dist1=20, 
    #       best_fit_params=[290.102813, 0.000051, 0.524790], best_fit_params1=[67.74, 0.0107, 0.52343],
    #       scale_factor=50, scale_factor1=1, demes_x=100, demes_y=20, demes_x1=30, demes_y1=20, min_ind_nr=3)
    
    # Plot IBD for Dataset used in Geneland Comparison
    # plot_homos(position_path="./barrier_folder2/barrier_file_coords60.csv", 
    #         genotype_path="./barrier_folder2/barrier_file_genotypes60.csv",
    #         bins=10, max_dist=16, best_fit_params=[50.316, 0.01857, 0.52246], bootstrap=False, nr_bootstraps=50,
    #         scale_factor=1, deme_bin=False, title="IBD Scenario") # No Binning: Too few Individuals
    
    # ## Give Stats of Results:
    # give_result_stats(multi_pos_hz_folder, subfolder="allind/")
    # give_result_stats(multi_pos_hz_folder, subfolder="noind/")
    # give_result_stats(multi_pos_hz_folder, subfolder="range_res/")  # 25-2100 m
    # give_result_stats(multi_pos_hz_folder, subfolder="range_res2/")   # 50-2500 m
    # give_result_stats(multi_pos_hz_folder, subfolder="range_res2/")   # 50-2500 m
    # give_result_stats(multi_pos_hz_folder, subfolder="chr0/result/", res_vec=range(460))
    # give_result_stats(multi_pos_syn_folder, subfolder = met2_folder)
    
    # # Print the saved Run Parameters of a Scenario:
    # print_run_params("./barrier_folder_HZ_synth/")
