'''
Created on Oct 13, 2016
Class that analyses the data produced by the grid object.
@author: hringbauer
'''
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import binned_statistic


class Analysis(object):
    '''
    General class that analyzes given Genotype matrix and position_list:
    '''
    position_list = []  # List of all positions (n individuals)
    genotypes = []  # List of all genotypes (nxk matrix. n: Individuals k: Genotypes)
    barrier = 0  # Where one can find the barrier

    def __init__(self, position_list, genotype_list, barrier=50):
        '''
        Constructor
        '''
        self.position_list = position_list  # Sets the position_list
        self.genotypes = genotype_list  # Sets the geno-type Matrix (n inds x k loci)
        self.barrier = barrier  # Where the barrier sits
        print(self.genotypes)
        print(self.position_list)
        print("Nr. of loci: %i" % len(genotype_list[0, :]))
        print("Nr. of individuals: % i" % len(genotype_list[:, 0]))
        
    def kinship_coeff(self, p1, p2, p):
        '''Takes two allele frequencies as input and calculates their correlation.'''
        f = np.mean((p1 - p) * (p2 - p) / (p * (1 - p)))
        return f
    
    def ind_correlation(self, p = 0.5):
        '''Analyze individual correlations.'''
        positions = self.position_list
        genotypes = self.genotypes
        
        distance = np.zeros(len(positions) * (len(positions) - 1) / 2.0)  # Empty container
        correlation = np.zeros(len(positions) * (len(positions) - 1) / 2.0)  # Container for correlation
        entry = 0
        
        for (i, j) in itertools.combinations(range(len(genotypes[:, 0])), r=2):
            distance[entry] = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))  # Calculate the pairwise distance
            correlation[entry] = self.kinship_coeff(genotypes[i, :], genotypes[j, :], p)  # Kinship coeff per pair, averaged over loci  
            entry += 1     
        self.vis_correlation(distance, correlation)  # Visualize the correlation
    
    def vis_correlation(self, distance, correlation):
        '''Take pairwise correlation and distances as inputs and visualizes them'''
        bin_corr, bin_edges, _ = binned_statistic(distance, correlation, bins=16, statistic='mean')  # Calculate Bin Values
        bin_dist = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Calculate the mean of the bins
        
        # Fit the data
        C, k, std_k = fit_log_linear(bin_dist[:8], bin_corr[:8])  # Fit first half of the distances bins
        Nb_est = 1 / (-k)
        Nb_std = (-std_k / k) * Nb_est
        x_plot = np.linspace(min(bin_dist), bin_dist[13], 10000)
        
        plt.plot(bin_dist[:12], bin_corr[:12], 'ro', label="Estimated Correlation per bin")
        plt.plot(x_plot, C + k * np.log(x_plot), 'g', label="Fitted decay")
        plt.axhline(np.mean(bin_corr), label="Mean Value", color='k', linewidth=2)
        plt.annotate(r'$\bar{N_b}=%.4G \pm %.2G$' % (Nb_est, Nb_std) , xy=(0.6, 0.7), xycoords='axes fraction', fontsize=25)
        plt.legend()
        plt.ylabel("Estimated Correlation")
        plt.xlabel("Distance")
        plt.show()
        
    def geo_comparison(self, mean_all_freq=0.5):
        '''Compares kinship coefficients based on geography'''
        y_comps = []  # Comparisons among yellow
        m_comps = []  # Comparisons among magenta
        h_comps = []  # Comparisons among hybrids
        ym_comps = []  # Comparisons in-between yellow/magenta
        
        genotypes, positions, barrier = self.genotypes, self.position_list, self.barrier
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
                
        
def fit_log_linear(t, y):
    '''Fitting log decay and returns parameters: y = A + B * ln(t) as (A,B)'''
    t = np.log(t)
    param, V = np.polyfit(t, y, 1, cov=True)  # Param has highest degree first
    return param[1], param[0], np.sqrt(V[0, 0])  # Returns parameters and STD
        
    

        
        
    
        
        
        
