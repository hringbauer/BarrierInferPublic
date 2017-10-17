'''
Created on Apr 15, 2015
Class containing the Data matrix
@author: Harald
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle
import re
import os
import warnings

from scipy.stats import chisquare
from matplotlib.widgets import Slider
from scipy.stats import binned_statistic, chi2_contingency
from scipy.stats import binom
from time import time

save_fig_path = "../Data/snp_slider_data.p"  # Path for saving data for SNP-Slider Pic

class Data(object):
    '''
    Class containing the Data matrix and methods to manipulate it
    '''
    li_path = "./LociInformation/shortSummarySNPsAdj.csv"  # Where to find Loci Information
    header = []
    data = []  # Field for data
    subdata = []  # Field for data under Analysis
    data_p_mean = []  # Field for matrix of estimated mean allele Frequency
    x_cords_ind = 0  # Where to find x coordinates (Integer)
    y_cords_ind = 0  # Where to find y coordinates (Integer)
    color_ind = 0  # Where to find color
    id_ind = 0  # Where to find Plant ID
    # sNP_inds = [2, 181]  # Which columns to find SNPs (python indexing)
    sNP_inds = [1, 4 * 26 + 14]  # For the SNPs in Model Course
    sNP_okay = np.array([0])  # Whether a SNP is valid: 1 for good; 0 for out of range, -1 for defect marker
    p = []  # Array of allele frequencies 
    del_list = []
    del_list_SNP = []
    max_differences = 5  # Number of maximal SNP differences for individual to be deleted in double detection
    max_failed_snps = 8
    coords = []  # Placeholder for coordinates
    names = []  # Placeholder for SNP names
    year_ind = 0

    
    
    def __init__(self, path, snp_inds=0):
        if snp_inds: self.sNP_inds = snp_inds  # In case SNP-indices given; set them
        data = np.genfromtxt(path, delimiter='$', dtype=None)
        self.header = data[0, :]
        self.data = data[1:, :]
        self.clean_data()  # Clean all the not genotyped stuff and missing coordinates data
        
        self.y_cords_ind = np.where(self.header == "DistNorthofCentre")[0][0]   
        self.x_cords_ind = np.where(self.header == "DistEastofCentre")[0][0]
        
        self.sNP_okay = np.array([0 for _ in range(0, len(self.header))])  # Set default set of sNP_okay array to 0
        self.sNP_okay[self.sNP_inds[0]:(self.sNP_inds[1] + 1)] = 1  # Flag valid SNPs
        
        self.subdata = self.data
        self.p = np.array([0.5 * np.mean(self.subdata[:, i].astype('float')[self.subdata[:, i].astype('float') > -8])
                            for i in range(self.sNP_inds[0], self.sNP_inds[1] + 1)]).astype('float')  # Calculate mean allele frequencies for every working SNP
        self.data_p_mean = 2.0 * np.array([self.p for _ in self.subdata[:, 0]])  # Create Default p_mean matrix
        self.id_ind = np.where(self.header == "ID")[0][0]
        self.color_ind = np.where(self.header == "PhenoCat")[0][0]  # Extract color index
        self.coords = self.subdata[:, (self.x_cords_ind, self.y_cords_ind)].astype('float')  # Load coordinates
        self.names = self.header[self.sNP_inds[0]:(self.sNP_inds[1] + 1)]  # Get SNP names
        self.year_ind = np.where(self.header == "year")[0][0]
        
    def set_subdata(self, xlim_down, xlim_up, ylim_down, ylim_up):
        ''' Extract data from given rectangle by using mask'''
        xcords = (self.data[:, self.x_cords_ind].astype(float) > xlim_down) & (self.data[:, self.x_cords_ind].astype(float) < xlim_up)
        ycords = (self.data[:, self.y_cords_ind].astype(float) > ylim_down) & (self.data[:, self.y_cords_ind].astype(float) < ylim_up)
        cords = np.nonzero(xcords & ycords)[0]
        self.subdata = self.data[cords, :]  # Cut out the right subdata
        self.data_p_mean = self.data_p_mean[cords, :]  # Cut out mean allele frequencies
        self.coords = self.subdata[:, (self.x_cords_ind, self.y_cords_ind)].astype('float')  # Load  according coordinates
        self.names = self.header[self.sNP_inds[0]:(self.sNP_inds[1] + 1)]  # Get SNP names
        print("Names of the SNPs in Analysis: ")
        print(self.names)
        
    def add_subdata(self, xlim_down, xlim_up, ylim_down, ylim_up):
        xcords = (self.data[:, self.x_cords_ind].astype(float) > xlim_down) & (self.data[:, self.x_cords_ind].astype(float) < xlim_up)
        ycords = (self.data[:, self.y_cords_ind].astype(float) > ylim_down) & (self.data[:, self.y_cords_ind].astype(float) < ylim_up)
        cords = np.nonzero(xcords & ycords)[0]
        self.subdata = np.append(self.subdata, self.data[cords, :], axis=0)
    
    def faulty_inds(self, max_failed):
        '''Shows and eliminates individuals with too many missing SNPs'''
        
        # Get a list with the number of missing snps
        missing_snps = np.array([np.sum(self.subdata[i, self.sNP_inds[0]:self.sNP_inds[1]].astype('float') == -9) for i in range(0, len(self.subdata[:, 0]))])
        print(missing_snps)
        plt.figure()
        plt.hist(missing_snps, range=[0, max(missing_snps)], bins=max(missing_snps))
        plt.show()
        
        # Delete failed inds
        ind_okay = np.where(missing_snps <= max_failed)[0]
        self.subdata = self.subdata[ind_okay, :]
        # self.data = self.subdata[ind_okay,:]
        
                    
    def SNP_analysis(self, i):
        ''' Do analysis of the ith SNP from Subdata'''
        arr_snp = self.subdata[:, i].astype("float")  # Load SNPS
        
        valid_ind = np.where(arr_snp > -8)  # Find measurements with valid SNPs.
        arr_snp = arr_snp[valid_ind] / 2.0
        arr_xcords = self.subdata[valid_ind[0], self.x_cords_ind].astype("float")
        arr_ycords = self.subdata[valid_ind[0], self.y_cords_ind].astype("float")
        
        return (arr_xcords, arr_ycords, arr_snp)
        
    def plot_SNP_slider(self):
        '''Does a Plot of SNPs in subdata with slider'''  
        fig = plt.figure()
        ax = plt.subplot(311)
        ax1 = plt.subplot(312)
        ax2 = plt.subplot(313)
        
        fig.subplots_adjust(left=0.25, bottom=0.25)
 
        # Do first Image:
        [arr_xcords, arr_ycords, arr_snp] = self.SNP_analysis(self.sNP_inds[0])  # Get SNP-Data for individuals with valid SNP-Entries
        distance_mean, _, _ = binned_statistic(arr_xcords, arr_snp, bins=10, statistic='mean')  # Extract data for Allele Frequency Plot
        
        ax.set_xlabel('x-Coord')
        ax.set_ylabel('y_Coord')
        ax.scatter(arr_xcords, arr_ycords, c=arr_snp, alpha=0.6)

        ax1.plot(np.arange(10), distance_mean, 'ro-')
        ax1.set_ylim([0, 1])
        
        ax2.hist(arr_snp, bins=3, range=[-0.01, 1.01], alpha=0.6)

        # Define slider
        axcolor = 'lightgoldenrodyellow'
        bx = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        
        slider = Slider(bx, 'SNP: ', self.sNP_inds[0], self.sNP_inds[1], valinit=0, valfmt='%i')
        
     
        def update(val):
            [arr_xcords, arr_ycords, arr_snp] = self.SNP_analysis([int(val)])  # Exctract Data
            distance_mean, _, _ = binned_statistic(arr_xcords, arr_snp, bins=10, statistic='mean')  # Extract data for Allele Frequency Plot
            counts, _ = np.histogram(arr_snp, bins=3, range=[-0.001, 1.001])  # Calculate counts for 0,1,2 states
            counts_freq = counts / float(len(arr_snp))
            p = np.mean(arr_snp)  # Calculate Allele Frequency
            hw_freq = np.array([(1 - p) ** 2, 2 * p * (1 - p), p ** 2])  # Calculate Expected Frequencies from HW
            
            _, pval = chisquare(counts, hw_freq * len(arr_snp), ddof=1)
            
            ax.cla()
            ax.scatter(arr_xcords, arr_ycords, c=arr_snp, alpha=0.6)
            ax.set_xlabel('x-Coord')
            ax.set_ylabel('y_Coord')
            
            ax1.cla()
            ax1.plot(np.arange(10), distance_mean, 'ro-')
            ax1.set_ylim([0, 1])
            ax1.text(0.1, 0.8, "Mean Allele Frequency: %.2f" % p)
            
            ax2.cla()
            weights = [1.0 / len(arr_snp) for i in range(0, len(arr_snp))]  # @UnusedVariable
            ax2.bar([0, 1, 2], counts_freq, 0.4, alpha=0.6, color='r', label="Measured Genotypes")
            ax2.bar([0.4, 1.4, 2.4], hw_freq, 0.4, alpha=0.6, color='b', label="Expected in HW")
            ax2.set_ylim([0, 1])
            ax2.legend()
            ax2.set_xticks([0.4, 1.4, 2.4], ('00', '01', '11'))
            ax2.text(0.1, 0.8, "Xi^2 P-Value: %.4f " % pval)
            plt.title("SNP: " + self.header[val])
            
            fig.canvas.draw()
            
        slider.on_changed(update)
     
        plt.show()

    def SNP_cleaning(self, min_maf=0.05, pval=0.01):
        '''Do quality analysis of SNPs and flag non suitable ones'''
        
        p = np.array([np.mean(self.SNP_analysis(i)[2]) for i in range(self.sNP_inds[0], self.sNP_inds[1] + 1)])  # Get mean frequencies
        
        # p_vals = np.array([self.chi2(i) for i in range(self.sNP_inds[0], self.sNP_inds[1] + 1)])  # Get p-Values
        p_vals = np.array([self.chi2_geo_struc(i) for i in range(self.sNP_inds[0], self.sNP_inds[1] + 1)])  # Get p-Values WITH EXPLICIT GEOGRAPHIC STRUCTURE
        
        print("SNPs with p-Val<%.5f:" % pval)
        print(np.where(p_vals < pval)[0] + self.sNP_inds[0])
        # Visualize results
        plt.figure()
        ax = plt.subplot(211)
        ax1 = plt.subplot(223)
        ax2 = plt.subplot(224)
        
        ax.hist(p, range=[0, 1], bins=10, alpha=0.8, color='g')  # Allele Frequency spectrum
        ax.set_title("Allele Frequency spectrum")
        ax1.hist(p_vals, range=[0, 1], bins=20, alpha=0.8, color='g')  # P-value Distribution
        ax1.set_title("P-Values")
        ax2.plot(p_vals, 'ro')
        plt.show()
        
        # Flag the failed SNPs
        fail1 = np.where(p_vals < pval)[0] + self.sNP_inds[0]  # Failed for HW
        min_p = np.fmin(p, 1 - p)  # Calculate the minimum p
        assert(np.min(min_p) > 0)  # Sanity check whether p was between 0 and 1
        
        fail2 = np.where(min_p < min_maf)[0] + self.sNP_inds[0]  # Failed for too low MAF
        self.sNP_okay[fail2] = -2  # Flag Entries with too low MAF
        self.sNP_okay[fail1] = -1  # Flag SNPs which cannot be trusted because HW
        print("Failed SNPs for HW: ")
        print(self.header[self.sNP_okay == -1])
        print("Failed SNPs for MAF: ")
        print(self.header[self.sNP_okay == -2])
        
        self.del_list_SNP = np.where((self.sNP_okay == -1) | (self.sNP_okay == -2))[0]
        print("Total SNPs flagged: %.1f" % len(self.del_list_SNP))
        return p_vals, min_p
        
        
    def chi2(self, SNP_ind):
        ''' Does chi2 analysis of SNP_indth SNP and return p-Value'''
        arr_snp = self.SNP_analysis(SNP_ind)[2]
        counts, _ = np.histogram(arr_snp, bins=3, range=[-0.001, 1.001])  # Calculate counts for 0,1,2 states
        p = np.mean(arr_snp)  # Calculate Allele Frequency
        hw_freq = np.array([(1 - p) ** 2, 2 * p * (1 - p), p ** 2])  # Calculate Expected Frequencies from HW    
        _, pval = chisquare(counts, hw_freq * len(arr_snp), ddof=1)
        return pval
    
    def chi2_geo_struc(self, SNP_ind):
        ''' Do chi2 analysis of SNP-indth SNP and return p-Value
        this time take geographic structure into account (via p_mean)'''
        arr_snp = self.subdata[:, SNP_ind].astype("float")  # Load SNPS
        valid_ind = np.where(arr_snp > -8)[0]  # Find measurements with valid SNPs.
        arr_snp = arr_snp[valid_ind] / 2.0  # Go to actual allele frequency and valid SNPs
        p_mean = self.data_p_mean[valid_ind, SNP_ind - self.sNP_inds[0]] / 2.0  # Extract the right p_mean values
        
    
        counts, _ = np.histogram(arr_snp, bins=3, range=[-0.001, 1.001])  # Calculate counts for 0,1,2 states
        
        hw_freq = np.array([[(1 - p) ** 2, 2 * p * (1 - p), p ** 2] for p in p_mean])  # Calculate expected numbers assuming HW   
        hw_freq = np.sum(hw_freq, 0)
        _, pval = chisquare(counts, hw_freq, ddof=1)
        return pval
        
        
    def save_ind_names(self, path="./plant_ids.p"):
        '''Save all Individual IDs as Pickle File.'''
        id_s = self.subdata[:, 0]
        
        pickle.dump(id_s, open(path, "wb"), protocol=2)  # Pickle the data
        print("%i Individuals successfully saved!" % len(id_s))
            
    def double_elemination(self, plot=True, save=True):
        '''Eliminate all potential double entries'''
        # Flag and delete double entries:
        
        # Some Lists of Summary Data
        homo_list = []
        true_diff_list = []  # List of Differing Loci
        delta_list = []
        nb_list = []  # List of Neighbors
        # diff_loci = [] 
        
        self.save_ind_names()
        
        for i in range(1, len(self.subdata[:, 0])):
            other_SNPs = self.subdata[0:i, self.sNP_inds[0]:self.sNP_inds[1]].astype('float')  # Matrix of sNPs to compare
            self_SNPs = self.subdata[i, self.sNP_inds[0]:self.sNP_inds[1]].astype('float')  # Own SNPs
            diffs = other_SNPs - self_SNPs  # Vector whether SNPs are different
            bool_diffs = (diffs != 0)  # Create Vector with boolean entries whether stuff is different 
            diffs = np.sum(bool_diffs, axis=1)  # Gives array of pairwise differences to individual i
            sig_diffs = np.where(diffs < (2 * self.max_failed_snps + self.max_differences))  # Where there are big enough differences warranting further investigation
            
            
            
            
            for j in sig_diffs[0]:
                true_differences = self.double_check(i, j)  # For every candidate do the actual double_check
                if sum(true_differences) < self.max_differences:  # In case of Dublicate detected
                    # Print some Output
                    opposing_homozygotes = sum([((float(self.subdata[i, k]) - float(self.subdata[j, k])) in [-2, 2]) for k in range(self.sNP_inds[0], self.sNP_inds[1])])  # Vector with number of opposing homozygotes
        
                    print("\nIndividual %i and Individual %i share %i different genotypes" % (i, j, sum(true_differences)))
                    print("Opposing Homozygotes: %i" % opposing_homozygotes)
                    delta = [self.subdata[i, self.x_cords_ind].astype(float) - self.subdata[j, self.x_cords_ind].astype(float), self.subdata[i, self.y_cords_ind].astype(float) - self.subdata[j, self.y_cords_ind].astype(float)]
                    delta = np.linalg.norm(delta)
                    print("Geographic distance: %.3f" % delta)  # Give out Euclidean Norm   
                    print("Individual 1: " + str(self.subdata[i, 0]))
                    print("Individual 2: " + str(self.subdata[j, 0])) 
                    homo_list.append(opposing_homozygotes)
                    delta_list.append(delta)
                    true_diff_list.append(np.where(true_differences == 1)[0])  # Save List of differing Positions
                    
                    self.del_list.append(j)  # Append to deletion-list
                    nb_list.append([i, j])
                    print("Double Inds found: %i" % len(true_diff_list))
        
        # Save the Results in Pickle Format
        if save == True:
            save_data = [true_diff_list, homo_list, delta_list]
            pickle.dump(save_data, open("./double_data.p", "wb"), protocol=2)  # Pickle the data
            pickle.dump(nb_list, open("./nb_list.p", "wb"))
    
        print("Finished! %i Doubles found!" % len(homo_list))
        
        
    
        # Do the actual Deleting
        print("Deleting %i entries" % len(self.del_list))
        self.del_entries(self.del_list)  # Delete double entries
        self.del_list = []  # Empty deletion list.
        
        if plot == True:
            true_diff_list = [len(i) for i in true_diff_list] # Get Number of Mismatches
            # Scatter Plot of Differences vrs Homozygotes:
            plt.figure()
            plt.scatter(true_diff_list, homo_list)
            plt.xlabel("Nr. of Mismatches")
            plt.ylabel("Nr. of opposing Homozygotes")
            plt.show()
            
            # Histogram of Pairwise Differences:
            plt.figure()
            plt.hist(true_diff_list)
            plt.xlabel("Pairwise Differences")
            plt.ylabel("Count")
            plt.show()
                
    def double_check(self, i, j):
        ''' Return the Number of True pairwise Differences 
        (not taking missing Genotypes into Account)
        '''
        differences = [self.subdata[i, k] != self.subdata[j, k] for k in range(self.sNP_inds[0], self.sNP_inds[1])]  # Vector wether entries different
        
        trusted_entry1 = [self.subdata[i, k] != "-9" for k in range(self.sNP_inds[0], self.sNP_inds[1])]  # Binary vector whether entry1 trusted
        trusted_entry2 = [self.subdata[j, k] != "-9" for k in range(self.sNP_inds[0], self.sNP_inds[1])]  # Binary vector whether entry2 trusted

        true_differences = np.array(trusted_entry1) * np.array(trusted_entry2) * np.array(differences)
        return true_differences  # Return array where Differences
    
    def forget_failed_SNPs(self):
        '''Forget about flagged SNPs'''
        self.sNP_okay[self.sNP_inds[0]:(self.sNP_inds[1] + 1)] = 1  # Set all SNPs to be valid.
    
    def extract_year(self, year=2014):
        '''Delete all individuals that are not from specified year'''
        total_nr = len(self.subdata)
        print(self.header)
        self.year_ind = np.where(self.header == "year")[0][0]
        print(self.year_ind)
        year_inds = np.where(self.subdata[:, self.year_ind] == str(year))[0]
        print(year)
        self.subdata = self.subdata[year_inds, :]
        reduced_nr = len(self.subdata)
        print("Successfully Reduced from %i to %i Individuals" % (total_nr, reduced_nr))
        
    def xi2_across_years(self, years=[2009, 2010, 2011, 2012, 2013, 2014]):
        '''Does a xi2 Test of Allele Frequencies across the years.'''
        print("Year Analyzed: %s" % str(years))
        nr_loci = len(self.names) - 2  # To exclude Chloroplast Marker
        p_vec = -np.ones((nr_loci, len(years)))  # Where the mean allele frequency will go in
        xi2_vec = -np.ones(nr_loci)  # where the xi2 value will go in
        p_val_vec = -np.ones(nr_loci)  # Where the p-Value will go in
        
        snp_ind = 0
        # nr_years=len(years)
        for snp_ind in xrange(nr_loci):  # Minus 2 - to avoid the Chloroplast Marker
            cont_table = np.array([self.extract_nr_year(snp_ind, year) for year in years])  # Get Contingency Table
            
            p_vec[snp_ind, :] = (1 * cont_table[:, 0] + 0.5 * cont_table[:, 1]) / np.sum(cont_table, axis=1)  # Calculate the mean allele Frequency per year
            
            chi2, p, _, ex = chi2_contingency(cont_table)
            
            print("Locus: %s" % self.names[snp_ind])
            print(cont_table)
            print(ex)
            print("Xi2 Value: %.4g" % chi2)
            print("p-Value: %.4g" % p)
            
            xi2_vec[snp_ind] = chi2
            p_val_vec[snp_ind] = p
            
        # Plot All Freqs Across the years
        colors = ["y", "coral", "r", "purple", "b", "navy"]

        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for i in xrange(len(years)):
            ax1.plot(p_vec[:, i], color=colors[i], label=years[i], marker='o', linestyle='', ms=6, alpha=0.7)
        
        ax1.legend()
        # ax1.xticks(np.arange(len(self.names)) + 0.5, self.names, rotation='vertical')
        
        # Plot p-Value across the years
        ax2.plot(p_val_vec, "ko")
        ax2.hlines(0.05, 0, nr_loci, color="r", label="p=0.05")
        ax2.legend()
        plt.xticks(np.arange(len(self.names)) + 0.5, self.names, rotation='vertical')
        plt.show()
            
            
    def extract_nr_year(self, snp_ind, year):
        '''Give back number contingency table of genotypes this year.
        [nr inds 0, nr inds 1, nr inds 2]'''
        year_inds = np.where(self.subdata[:, self.year_ind] == str(year))[0]
        
        snp_ind = self.sNP_inds[0] + snp_ind  # Get the right Index (with the right shift)
        # Get overview over Number of Individuals with right Values
        nr_0 = np.sum(self.subdata[year_inds, snp_ind].astype("int") == 0)
        nr_1 = np.sum(self.subdata[year_inds, snp_ind].astype("int") == 1)
        nr_2 = np.sum(self.subdata[year_inds, snp_ind].astype("int") == 2)
        
        return [nr_0, nr_1, nr_2]
    
    def produce_cleaning_table(self):
        '''Produce a table with all relevant information for SNPs.
        Save as loci_information.csv'''
        names = self.names
        print("Extracting LD-Scores...")
        ld_scores = self.ld_check(1.0)  # Run LD-Scores First; as it is only for unflagged Loci
        print("Extracting Geo-Scores...")
        geo_scores = self.geo_correlation(1.0)
        print("Extracting HW-P values...")
        hw_scores, min_all_freq = self.SNP_cleaning(0, 0)
        # Chromosome: Include Chromose Information
        
        # Now extract Linkage Group Information from Davids File
        
        def find_l_group(name):
            '''Short script that returns linkage group of a specific Locus'''
            df = pd.read_csv(self.li_path)  # Look Up Data Frame
            
            inds = np.where(df['LocusName'] == name)[0]  # LocusNameOLD
            
            if len(inds) == 0:  # In case Locus not found:
                l_group = 0
            else:
                l_group = int(df['LG'][inds[0]])
            
            return l_group
            
               
        if os.path.exists(self.li_path):
            lg_group = np.array([find_l_group(name) for name in names])  # Finds the Linkage Group
            
        else:
            warnings.warn("LG Group Information not found!", RuntimeWarning)
            lg_group = np.zeros(len(names))  # Initialize to 0
            
        # Check whether everything has the same length:
        assert(len(set(len(x) for x in [names, ld_scores,
                                        geo_scores, hw_scores, min_all_freq, lg_group])) == 1)
        
        df = pd.DataFrame(
            {'Name': names,
             'LD_Score': ld_scores,
             'Geo_Score': geo_scores,
             'HW p-Value': hw_scores,
             'Min All. Freq': min_all_freq,
             'LG': lg_group
            })
        
        # Save
        df.to_csv('loci_infoALL.csv')
        print("Successfully Saved!!") 
    
    def del_entries(self, del_list):
        '''Deletes given entries from subdata'''
        self.subdata = np.delete(self.subdata, del_list, axis=0)
        
    def del_SNPs(self):
        '''Deletes the given SNPS from the analysis and corrects necessary fields'''
        print("Deleting %.1f SNPs." % len(self.del_list_SNP))
        self.subdata = np.delete(self.subdata, self.del_list_SNP, axis=1)
        self.header = np.delete(self.header, self.del_list_SNP, axis=0)
        self.sNP_okay = np.delete(self.sNP_okay, self.del_list_SNP, axis=0)
        self.p = np.delete(self.p, self.del_list_SNP, axis=0)
        self.sNP_inds = [self.sNP_inds[0], self.sNP_inds[1] - len(self.del_list_SNP)]
        self.x_cords_ind -= len(self.del_list_SNP)
        self.y_cords_ind -= len(self.del_list_SNP)
        self.del_list_SNP = []
    
    def create_halfsibs(self, sNPs1, sNPs2, sNPs3):
        '''Pairs ind1 with ind2 and ind3 and returns SNPs'''
        new_SNPs = [-9 for _ in range(0, len(sNPs1))]  # Initialize new SNP, default is error
        new_SNPs1 = [-9 for _ in range(0, len(sNPs1))]
        
        for k in range(0, len(sNPs1)):
            if sNPs1[k] + sNPs2[k] + sNPs3[k] > -1:
                new_SNPs[k] = (np.random.binomial(1, sNPs1[k] / 2.0) + np.random.binomial(1, sNPs2[k] / 2.0))
                new_SNPs1[k] = (np.random.binomial(1, sNPs1[k] / 2.0) + np.random.binomial(1, sNPs3[k] / 2.0))
        return np.array[new_SNPs, new_SNPs1]
                           
    def update_p(self):
        '''Calculates new Allele frequencies for subdata'''
        a = self.p
        self.p = np.array([0.5 * np.mean(self.subdata[:, i].astype('float')[self.subdata[:, i].astype('float') > -8]) 
                           for i in range(self.sNP_inds[0], self.sNP_inds[1] + 1)]).astype('float')  # Calculate mean allele frequencies for every working SNP
        
        self.data_p_mean = np.array([2.0 * self.p for _ in self.subdata[:, 0]])
        plt.figure()
        plt.plot(a, 'ro')
        plt.plot(self.p, 'bo')
        plt.show()
        
    def compare_pass_p(self):
        ''' Compares Allele Frequencies from over the pass with the ones on this side of the pass'''
        self.set_subdata(-20000, -8000, -10000, 10000)
        p_pass = np.array([0.5 * np.mean(self.subdata[:, i].astype('float')[self.subdata[:, i].astype('float') > -8]) 
                           for i in range(self.sNP_inds[0], self.sNP_inds[1] + 1)]).astype('float')  # Calculate mean allele frequencies for every working SNP
        self.set_subdata(-5000, 3000, -10000, 2000)
        p_main = np.array([0.5 * np.mean(self.subdata[:, i].astype('float')[self.subdata[:, i].astype('float') > -8]) 
                           for i in range(self.sNP_inds[0], self.sNP_inds[1] + 1)]).astype('float')  
        
        # Plot:
        plt.figure()
        plt.plot(p_pass, 'ro', label="Pass Frequencies")
        plt.plot(p_main, 'bo', label="Main core Frequencies")
        plt.legend()
        plt.grid()
        plt.show()
    
    def ld_check(self, r2_score):
        '''Checks for LD via correlation'''
        good_SNPs = np.where(self.sNP_okay == 1)[0]
        data = self.subdata[:, good_SNPs].astype('float')  # Load data
        names = self.header[good_SNPs]
        
        masked_data = np.ma.array(data, mask=data == -9)  # Mask the missing data values
        r = np.ma.corrcoef(masked_data, rowvar=0)
        print("Mean Correlation: %.4f: " % np.mean(r))
        print("Standard Deviation: %.4f:" % np.std(r))
        
        for i in range(0, len(data[0, :])):
            r[i, i] = 0.15
        
        r2 = r ** 2
        r2_max_vec = np.array([0, ] + [np.max(r2[i, :i]) for i in xrange(1, len(r2))])  # Get maximum Correlation with previus Loci
        
        plt.figure()
        plt.pcolor(r)
        plt.xlim(0, len(names))
        plt.ylim(0, len(names))
        plt.colorbar()
        plt.xticks(np.arange(len(names)) + 0.5, names, rotation='vertical')
        plt.tick_params(labelsize=6)
        plt.show()
        
        paircorr = np.square([r[i, i + 1] for i in range(0, len(data[0, :]) - 1)] + [0, ])
        paircorr1 = np.square([r[i, i + 2] for i in range(0, len(data[0, :]) - 2)] + [0, 0])  # 2 Individuals ahead
        paircorr2 = np.square([r[i, i + 3] for i in range(0, len(data[0, :]) - 3)] + [0, 0, 0])  # 3 Individuals ahead
        paircorr = np.fmax(paircorr, paircorr1)  # Calculate Maximum
        paircorr = np.fmax(paircorr, paircorr2)
        plt.plot(paircorr, 'ro', label="Maximum within 3 positions")
        plt.plot(r2_max_vec, 'bo', label="Total maximum")
        plt.legend()
        plt.show()
        
        
        print("%.1f SNPs flagged." % np.sum(paircorr > r2_score))
        sNP_okay = np.where(self.sNP_okay == 1)[0]
        flag_ind = sNP_okay[paircorr > r2_score]  # Extract indices where to flag
        self.sNP_okay[flag_ind] = -3  # Flag Subset of good SNPs with high correlation
        print(self.sNP_okay)
        
        return r2_max_vec
    
    def color_correlation(self, r2_score):
        '''Gives the correlation of all loci with the color.
        Flags loci with too high correlation'''
        # Define a proper color vector...
        color_ind = np.where(self.header == "Red")[0][0]
        color = self.subdata[:, color_ind]
        print(color)
        
        # Check where only color values
        is_float = np.array([1 if re.match("^\d+(\.\d+)*$", i) else 0 for i in color]).astype(bool)  # Only entries without minus and numbers or dots
        

    
        color_cor = np.array([0 for _ in range(self.sNP_inds[0], self.sNP_inds[1] + 1)]).astype('float')
        for i in range(self.sNP_inds[0], self.sNP_inds[1]):
            inds = np.where((self.subdata[:, i] > -1) & (is_float == 1))[0]  # Where the SNPs are okay
            
            p1 = self.subdata[inds, i].astype('float')
            c1 = color[inds].astype('float')
            color_cor[i - self.sNP_inds[0]] = np.corrcoef(p1, c1)[0, 1]
            
        
        plt.figure()
        plt.plot(color_cor, 'ro')
        plt.xlabel("Locus")
        plt.ylabel("Pearson R")
        plt.show()  
        
        color_cor = np.square(color_cor)
        
        plt.hist(color_cor, range=[0, 0.1])
        plt.ylabel("R^2")
        plt.show()
        
        print("%.1f SNPs flagged." % np.sum(color_cor > r2_score))
        self.sNP_okay[np.where(color_cor > r2_score)[0] + self.sNP_inds[0]] = -4  # Flag Subset of good SNPs with high correlation to color
    
    def geo_correlation(self, r2_score):
        '''Gives the correlation of all loci with x- and y-Axis.
        Flag loci with too high correlation
        r2: R^2 Pearson Correlation'''
        # First extract x- and y-Coordinates
        x_coords = self.subdata[:, self.x_cords_ind].astype('float')
        y_coords = self.subdata[:, self.y_cords_ind].astype('float')
        
        
        # Create empty Vector describing x and y-Correlation
        x_cor = np.array([0 for _ in range(self.sNP_inds[0], self.sNP_inds[1] + 1)]).astype('float')
        y_cor = np.array([0 for _ in range(self.sNP_inds[0], self.sNP_inds[1] + 1)]).astype('float')
        
        for i in range(self.sNP_inds[0], self.sNP_inds[1]):
            p1 = self.subdata[:, i].astype('float')
            # Only extract individuals where SNP did not fail.
            good_inds = np.where(p1 > -1)[0] 
            p1 = p1[good_inds]
            x1 = x_coords[good_inds]
            y1 = y_coords[good_inds]
            
            x_cor[i - self.sNP_inds[0]] = np.corrcoef(p1, x1)[0, 1] ** 2  # Sets the values to r^2
            y_cor[i - self.sNP_inds[0]] = np.corrcoef(p1, y1)[0, 1] ** 2  # Sets the values to r^2
        
        # Do Plotting:
        names = self.names
        print(len(names))
        assert len(names) == len(x_cor)
        plt.figure()
        plt.plot(x_cor, 'ro', label="Correlation x-Axis")
        plt.plot(y_cor, 'bo', label="Correlation y-Axis")
        plt.xticks(np.arange(len(names)) + 0.5, names, rotation='vertical')  # Creates the Ticks on the x-Axis
        plt.tick_params(labelsize=6)
        plt.hlines(r2_score, 0, len(names), 'g', linewidth=3, label="Cut Off")
        plt.xlabel("Locus")
        plt.ylabel(r"Pearson $R^2$")
        plt.legend()
        plt.show()  
        
        tot_cor = np.maximum(x_cor, y_cor)
        print("%.1f SNPs flagged." % np.sum(tot_cor > r2_score))
        bad_snps = np.where(tot_cor > r2_score)[0]
        print(names[bad_snps])
        self.sNP_okay[np.where(tot_cor > r2_score)[0] + self.sNP_inds[0]] = -4  # Flag Subset of good SNPs with high correlation to Geography    
        return tot_cor
        
    def extract_good_SNPs(self):
        '''Extracts good SNPs: Returns SNP-Matrix, their allele frequency, their coordinates and name as numpy float arrays'''
        good_SNPs = np.where(self.sNP_okay == 1)[0]
        data = self.subdata[:, good_SNPs].astype('float')  # Load data
        p = self.p[good_SNPs - self.sNP_inds[0]].astype('float')  # Load allele frequencies
        coords = self.subdata[:, (self.x_cords_ind, self.y_cords_ind)].astype('float')
        names = self.header[good_SNPs]
        color = self.subdata[:, self.sNP_inds[1] + 1].astype(float)
        return(data, p, coords, names, color)    
        
    def kernel_estimation(self, sigma=500):
        '''Given sigma, estimates the mean allele frequencies per individual'''
        self.coords = self.subdata[:, (self.x_cords_ind, self.y_cords_ind)].astype('float')  # Load coordinates    # WORKAROUND - ADDRESS
        data, p, coords = self.subdata[:, self.sNP_inds[0]:(self.sNP_inds[1] + 1)].astype('float'), self.p, self.coords  # Load necessary Data
        
        # Impute missing genotypes
        for (ind, sNP), value in np.ndenumerate(data):  # Iterate over all data points
            if value == -9:  # In case of missing data
                data[ind, sNP] = np.random.binomial(2, p[sNP])  # Draw random genotype for missing data
        
        # Calculate Pairwise distance (Overkill - but only by a factor of 2 - I cannot be bothered.
        # dist_mat1 = np.array([[np.linalg.norm(coords[i, :] - coords[j, :]) for i in range(n)] for j in range(n)]) # The old way
        dist_mat = self.calc_distance_matrix(coords)
        # np.fill_diagonal(dist_mat, 10000)  # Set self distance to very far away (comp. trick instead of deleting it)
        
        
        # Calculate likelihood for various sigmas
#         y=[]
#         sigmas=[i for i in range(150,801,75)]
#         for sigma in sigmas:
#             print("Doing Sigma: %.0f " % sigma)
#             p_mean = np.array([[self.calc_p_mean(dist_mat[i, :], data[:, j], sigma) for j in range(len(p))] for i in range(n)])
#             y.append(self.calc_ll_pmean(data, p_mean/2.0))
#             
#         plt.figure()
#         plt.plot(sigmas,y,'r-')
#         plt.xlabel("Sigma")
#         plt.ylabel("Log Likelihood")
#         plt.show() 
        
        self.picture_kernel_smoothing(dist_mat, data, coords, p)  # Call picture function for smoothing the kernel

        p_mean = self.calc_p_mean(dist_mat, data, sigma)
        self.data_p_mean = p_mean
        
    def kernel_estimation_rare(self, sigma=500, rare_factor=20):
        '''Given sigma, estimates the mean allele frequencies per individual - from rarified data'''
        self.coords = self.subdata[:, (self.x_cords_ind, self.y_cords_ind)].astype('float')  # Load coordinates
        data, p, coords = self.subdata[:, self.sNP_inds[0]:(self.sNP_inds[1] + 1)].astype('float'), self.p, self.coords  # Load necessary Data
        
        # Impute missing genotypes
        for (ind, sNP), value in np.ndenumerate(data):  # Iterate over all data points
            if value == -9:  # In case of missing data
                data[ind, sNP] = np.random.binomial(2, p[sNP])  # Draw random genotype for missing data
        
        dist_mat = self.calc_distance_matrix(coords)
        keep_inds = self.rarefy(dist_mat, rare_factor)
        
        plt.figure()
        plt.scatter(coords[keep_inds, 0], coords[keep_inds, 1])
        plt.title("Rarefied inds")
        plt.show()
        
        
        
        p_mean = self.calc_p_mean(dist_mat[:, keep_inds], data[keep_inds, :], sigma)
        self.data_p_mean = p_mean
        
    def rarefy(self, dist_mat, rare_dist, factor=10):
        '''Rarefies according to distance_matrix; such that on average 1 ind in radius rare_dist
        Return list of rarefied individuals'''
        print("Rarefying...")
        near_dist_mat = dist_mat < rare_dist * factor
        near_nbrs = np.sum(near_dist_mat, axis=1)
        chance = np.random.random(len(near_nbrs))
        keep_inds = np.where(chance < (factor ** 2 / near_nbrs))[0]  # Print where the nearest neighbours are to be found
        print("Original inds: %i " % len(near_nbrs))
        print("Rarefied inds: %i " % len(keep_inds))
        return keep_inds

        
    def picture_kernel_smoothing(self, dist_mat, data, coords, p):
        '''Calculates Kernel smoothing with 3 kernel distances and give back a picture for every SNP'''
        # Calculate the mean matrix
        names = self.names  # Loads the names of all SNPs
        print("Calculate Mean allele frequs for every individual")
        
        
        sigma = 200
        p_mean200 = self.calc_p_mean(dist_mat, data, sigma)
        print(self.calc_ll_pmean(data, p_mean200 / 2.0))
        
        sigma = 500
        p_mean = self.calc_p_mean(dist_mat, data, sigma)
        print(self.calc_ll_pmean(data, p_mean / 2.0))

        sigma = 1000
        p_mean1000 = self.calc_p_mean(dist_mat, data, sigma)
        print(self.calc_ll_pmean(data, p_mean1000 / 2.0))
        
        print("Size of empirical Data: %i %i" % (len(data[0, :]), len(data[:, 0])))
        print("Size of estimated all-freq Data: %i %i" % (len(p_mean[0, :]), len(p_mean[:, 0])))
        
        fig = plt.figure()
        # Do first Image:
        l, = plt.plot(coords[:, 0], data[:, 0], 'ko', label='Real Allele Frequency')
        l200, = plt.plot(coords[:, 0], p_mean200[:, 0], 'yo', label='200m')
        l500, = plt.plot(coords[:, 0], p_mean[:, 0], 'ro', label='500m')
        l1000, = plt.plot(coords[:, 0], p_mean1000[:, 0], 'bo', label='1000m')
        
        plt.legend()
        
        plt.xlabel('x-Coord')
        plt.ylabel('Allele Freq')
        plt.ylim([-0.3, 2.3])

        # Define slider
        axcolor = 'lightgoldenrodyellow'
        bx = plt.axes([0.25, 0, 0.65, 0.03], axisbg=axcolor)
        plt.axes
        slider = Slider(bx, 'SNP: ', 0, len(p) - 1, valinit=0, valfmt='%i')  # Python indexing!
        print("Datatype of data:")
        print(type(data))
     
        def update(val):
            val = int(val)
            l.set_ydata(data[:, val])
            l200.set_ydata(p_mean200[:, val])
            l500.set_ydata(p_mean[:, val])
            l1000.set_ydata(p_mean1000[:, val])
            fig.canvas.draw_idle()
            plt.title("SNP: " + names[val])  # Sets the Value to the SNPs
            
        slider.on_changed(update)
        plt.show()
        
        print("Saving Data for Picture...")
        data_fig = (coords, names, data, p_mean, p_mean200, p_mean1000)
        pickle.dump(data_fig, open(save_fig_path, "wb"), protocol=2)  # Pickle the data
        print("Successfully Saved.")
        
        
#     def calc_p_mean_or(self, dists, p, sigma):
#         '''Given a distance matrix and the matrix of allele frequencies,
#         calculate the mean allele frequency'''
#         p, dists = np.array(p), np.array(dists)  # Just in case that not numpy array
#         
#         weights = 1 / (2.0 * np.pi * sigma ** 2) * np.exp(-dists ** 2 / (2.0 * sigma ** 2))  # Calculate the Gaussian weights
#         p_mean = np.dot(weights * p) / np.sum(weights)  # Calculate weighted mean
#         return(p_mean)
    
    def calc_p_mean(self, dist_mat, p_mat, sigma):
        '''Given a distance matrix and the matrix of allele frequencies,
        calculate the mean allele frequency'''
        print("Smoothing out...")
        start = time()
        p_mat, dist_mat = np.array(p_mat), np.array(dist_mat)  # Just in case that not numpy array
        
        weights = 1 / (2.0 * np.pi * sigma ** 2) * np.exp(-dist_mat ** 2 / (2.0 * sigma ** 2))  # Calculate the Gaussian weights
        p_mean = np.dot(weights, p_mat) / np.sum(weights, axis=1)[:, None]  # Calculate weighted mean
        print("Time taken %.2f" % (time() - start))
        return(p_mean)
    
    
    def calc_ll_pmean(self, x, p_mean):
        '''Calculates the likelihood of having this exact p-mean. x is empirical data - p_mean estimated mean'''
        print("Calculating likelihood")
        l_matrix = binom.pmf(x, 2, p_mean)
        ll = np.sum(np.log(l_matrix))
        return ll
    
    def calc_distance_matrix(self, coords):
        '''Calculate the distance matrix between all coords. Requires numpy array of 2d coordinates'''
        print("Calculating distance matrix")
        start = time()
        dist_mat = np.linalg.norm(coords[:, None] - coords, axis=2)
        print("Time taken %.2f" % (time() - start))
        return dist_mat
        
    def clean_data(self):
        '''Method that cleans data. Gets rid of "NA GPS Values
        and NGY genotypes'''
        data, header = self.data, self.header
        y_cords_ind = np.where(self.header == "DistNorthofCentre")[0][0]   
        x_cords_ind = np.where(self.header == "DistEastofCentre")[0][0]
        print("Raw data Nr. individuals: %i " % len(data[:, 0]))
        
        gps_good_ind = np.where((data[:, x_cords_ind] != 'NA') * (data[:, y_cords_ind] != 'NA'))[0]
        data = data[gps_good_ind, :]  # Delete Entries without GPS
        

        # Now to NGY errors
        
        no_gen = np.array([np.sum(data[:, i] == 'NGY') for i in np.arange(self.sNP_inds[0], self.sNP_inds[1] + 1)])
        
        print("Number of loci with <10 NGY SNPs: % i" % np.sum(no_gen < 10))
        bad_snps = np.where(no_gen > 10)[0]  # Detect bad loci; i.e. loci with more than 10 non-genotyped individuals
        
        data = np.delete(data, bad_snps + self.sNP_inds[0], axis=1)  # Delete bad SNPs
        header = np.delete(header, bad_snps + self.sNP_inds[0])  # Delete header SNPs
        self.sNP_inds[1] -= len(bad_snps)
        
        data[data[:, :] == 'NGY'] = -9  # Replace the few not genotyped individuals with failed SNPs
        self.data, self.header = data, header  # Save what was done
        
    def save_data(self, path):
        '''Method that pickles the data.'''
        temp = self.subdata
        self.data = self.subdata  # Save subdata
        self.subdata = []  # Temporary delete subdata
        pickle.dump(self, open(path, "wb"), protocol=2)  # Pickle the data
        print("Data saved to: %s" % path)
        self.subdata = temp  # Restore subdata
        
    def save_p(self):
        '''Method that saves allele-frequencies. Written to extract for Nick'''
        np.savetxt("allele_freqs.csv", self.p, delimiter=",")
        
        
        
        
        
        

    
    
    
if __name__ == "__main__":
    loci_information = "./LociInformation/shortSummarySNPsAdj.csv"  # Where to find Loci Information
    # a=np.loadtxt(loci_information, header=True)
    df = pd.read_csv(loci_information)
    print(df.dtypes)
    name = "s787_264617"
    print("waaait a second")
    inds = np.where(df['LocusNameOLD'] == name)[0]
    print(inds)
    print(df['LG'][inds[0]])



        

        
