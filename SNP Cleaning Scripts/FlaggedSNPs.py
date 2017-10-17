'''
Created on May 15, 2015
@author: Harald Ringbauer
'''
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors
from scipy.stats import norm
from scipy.stats import binned_statistic 
from scipy.misc import factorial  # @UnresolvedImport
import itertools
from time import time
from random import choice
from collections import Counter
from scipy.optimize import curve_fit
from pyproj import Proj  # @UnresolvedImport
from colorpins import extract_colored_kml

draw_parent = False  # Whether simulated data is shown in correlogram
min_relatedness = 0.2  # Kinship threshold for PO detection
max_opp_homos = 1  # Maximal number of opposing homos for P0 detection (strictly <)
error_rate = 0.05  # Estimated error rate for SNP genotyping

lat_center = 42.32271733
long_center = 2.074444088



class FlaggedSNPs(object):
    '''
    Class for Flagged SNPs and other relevant Data. Has methods to analyze the data
    '''
    data, p, coords, names, color, id, maf = 0, 0, 0, 0, 0, 0, 0
    po_list = []  # List of Parent-Offspring coordinates

    def __init__(self, Data):
        '''Initialize '''
        good_SNPs = np.where(Data.sNP_okay == 1)[0]
        self.data = Data.subdata[:, good_SNPs].astype('float')  # Load data from Subdata (cut out the good snps)
        self.data_p_mean = Data.data_p_mean[:, good_SNPs - Data.sNP_inds[0]]  # Load matrix of estimated mean allele frequencies (good snps already cut out)
        self.p = Data.p[good_SNPs - Data.sNP_inds[0]].astype('float')  # Load allele frequencies
        # self.coords = Data.subdata[:, Data.x_cords_ind:Data.y_cords_ind + 1].astype('float')
        self.coords = Data.coords
        self.names = Data.header[good_SNPs]
        self.color = Data.subdata[:, Data.color_ind]  # What is the color ind
        self.id = Data.subdata[:, Data.id_ind]  # Assumes id-field is the first
        self.set_colors()
        print("Analyzing %i SNPs from %i individuals" % (len(self.data[0, :]), len(self.data[:, 0])))
        
    def set_colors(self):
        color_vec = self.color
        self.color=set_colors(color_vec)
        return 0
       
    
    def xy_to_gps(self, delta_x, delta_y):
        '''Extract GPS data relative to center
        Takes vector as input and outputs vector - Return lat and long'''
        my_proj = Proj(proj='utm', zone="31T",
            ellps='WGS84', units='m')  # Prepare the Longitude/Latidude to Easting/Northing transformer
        center_x, center_y = my_proj(long_center, lat_center)  # Transform to Easting Northing
        
        longs, lats = [], []
        for i in range(len(delta_x)):  # Iterate
            lon, lat = my_proj(center_x + delta_x[i], center_y + delta_y[i], inverse=True)[:2]
            longs.append(lon)
            lats.append(lat)
        return longs, lats
            
    def principal_components(self):
        '''Does a quick PC-analysis'''
        data, p, coords = self.data, self.p, self.coords  # Extract filtered SNPs.
        
        # Impute missing genotypes
        for (ind, sNP), value in np.ndenumerate(data):  # Iterate over all data points
            if value == -9:  # In case of missing data
                data[ind, sNP] = np.random.binomial(2, p[sNP])  # Draw random genotype for missing data
                
        _, score, _ = princomp(data)
        
        pca1_mean, bin_edges, _ = binned_statistic(coords[:, 0], score[0, :], bins=15, statistic='mean')  # Calculate Bin Values
    
        plt.figure()
        plt.title("PCA")
        plt.scatter(score[0, :], score[1, :], color=self.color, alpha=0.7)
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.axis('equal')
        plt.show()
         
        # Scatter-Plot of 1st principal component vrs distance
        plt.figure()
        plt.scatter(coords[:, 0], score[0, :], color=self.color, alpha=0.7)
        plt.xlabel("x-Coordinate")
        plt.ylabel("1st Principal Component")
        plt.hlines(pca1_mean, bin_edges[:-1], bin_edges[1:], colors='g', lw=5, label='Binned data')
        plt.axhline(0, label="Null line", color='k', linewidth=1)
        plt.legend()
        plt.show()
        
        
        plt.scatter(coords[:, 0], score[1, :], color=self.color, alpha=0.7)
        plt.xlabel("x-Coordinate")
        plt.ylabel("2nd Principal Component")
        plt.show()        
        
        # Plot Geography vrs first component
        plt.scatter(coords[:, 0], coords[:, 1], c=score[0, :], alpha=0.4)
        plt.show()
        # Plot Geography vrs second component
        plt.scatter(coords[:, 0], coords[:, 1], c=score[1, :], alpha=0.4)
        plt.show()
        
        # Some special code for filtering out what strange things are going on with PCA2:
#         inds = np.where((score[1, :] > 2.5) * (coords[:, 0] > 0))[0] # Trololol
#         inds = np.where((score[1, :] > 2.5) * (coords[:, 0] > 0))[0]
#         print(inds)
#         print(data[inds, :])
#         print(self.id[inds])
#         np.savetxt("special_inds.csv", data[inds, :], delimiter='$')
#         self.color[inds] = "k"
        
        print("Extracting Google Maps file")
        longs, lats = self.xy_to_gps(coords[:, 0], coords[:, 1])
        extract_colored_kml(score[0, :], lats, longs, self.id, "pca.kml")
        extract_colored_kml(score[1, :], lats, longs, self.id, "pca2.kml")  # Second PCA axis
        print("Extraction complete")

    def analyze_correlations(self, max_dist):
        ''' Method which analyzes allele correlations for every sample pair and creates plots'''
        # Preprocessing of SNPs
        print("Extract SNPs...")
        data, p, coords = self.data, self.p, self.coords  # Extract filtered SNPs.
        p_mean = self.data_p_mean
        nr_inds = len(data[:, 0])
        n = nr_inds * (nr_inds - 1) / 2  # Number of all pairwise comparisons        
        prelim_results = np.zeros(n)            
        homos = np.zeros(n)
        pair_distance = np.zeros(n)
        
        # Randomly permute geographic labels:
        # i=np.random.permutation(np.size(coords,0))
        # coords=coords[i,:]
        
        print("Analysing %i suitable SNPs. " % len(data[0, :]))
        
        # Do the correlation analysis:
        dist_mat = calc_distance_matrix(coords)  # Calculate the distance matrix
        
        t = time()
        for (i, j) in itertools.combinations(np.arange(np.size(data, 0)), r=2):
            ind = j * (j - 1) / 2 + i  # Index in a linearized array
            # estimator, homo = self.kinship_coeff(data[i, :], data[j, :], p)  # Kinship coeff per pair, averaged over loci + Number of opposing homozygotes
            estimator, homo = self.kinship_coeff_loc_freq(data[i, :], data[j, :], p_mean[i, :], p_mean[j, :])  # Kinship coeff. p.p, averaged over loci + Number of opp. homos   
            prelim_results[ind] = estimator
            homos[ind] = homo
                             
            # Calculate Pairwise distance
            pair_distance[ind] = dist_mat[i, j]  # Load the distance matrix
                
            # Append Lines of detected PO to P0-List
            if estimator > min_relatedness and homo < max_opp_homos:
                self.po_list.append([coords[i, :], coords[j, :]])
        print("Elapsed Time: %2f" % (time() - t))
        
        
        
        # Quicker, vectorized way of doing calculations
#         t= time()
#         f_ufunc = np.frompyfunc(self.kinship_coeff_loc_freq, 4, 2)  # 4 inputs, 2 output
#         estimator, homo = f_ufunc(data[:,None],data, p_mean[:,None],p_mean)
#         dist_mat,estimator,homo = dist_mat.flatten(), estimator.flatten(), homo.flatten()
#         
#         
#         print("Elapsed Time: %2f" % (time() - t))
#         print(prelim_results[:10]-estimator[:10])
        
        
        
        
        # Create some artificial offsprings and measure their value with their parents. Don't forget to delete.
        parent1 = np.random.randint(len(data[:, 0]), size=1000)
        parent2 = np.random.randint(len(data[:, 0]), size=1000)
        distance_parents = [np.linalg.norm(coords[parent1[i], :] - coords[parent2[i], :]) for i in range(0, len(parent1))]  # Pairwise distance of parents
        offspring = [self.create_offspring(data[parent1[i], :], data[parent2[i], :]) for i in range(0, len(parent1))]  # Create Offspring
        # Now calculate relationship between parent1 and offspring
        kinship = []
        for i in range(0, len(parent1)):
                par1 = data[parent1[i], :]  # Extract all SNPs
                kid = offspring[i]
                estimator, _ = self.kinship_coeff(par1, kid, p)
                kinship.append(np.mean(estimator))   
        
        
        mean_het = np.mean(prelim_results)
        distance_mean, bin_edges, _ = binned_statistic(pair_distance, prelim_results, bins=20, statistic='mean', range=[0, max_dist])  # Calculate Bin Values
        
        # Histogram of heterozygosity
        gaussian_params = fit_gaussian(prelim_results)
        print("Mean kinship coefficient: %.6f" % mean_het)
        print("STD of kinship coefficients: %.6f" % gaussian_params[1])
        print("Mean kinship of simulated PO-pairs: %.4f" % np.mean(kinship))
        x = np.linspace(min(prelim_results), max(prelim_results), 10000)
        pdf_fitted = norm.pdf(x, loc=gaussian_params[0], scale=gaussian_params[1])
        plt.figure()
        plt.hist(prelim_results, bins=50, label='Data', color='g', alpha=0.6)
        plt.plot(x, pdf_fitted * len(prelim_results) * (max(prelim_results) - min(prelim_results)) / 50.0, 'r-', label='Gaussian Fit', linewidth=2)
        plt.legend()
        plt.xlabel("Estimated kinship coefficient per pair")
        plt.ylabel("Count")
        plt.show()
        
        # Scatter-plot of heterozygosity vrs. distance
        homos = np.array(homos)
        colors = np.array(['r' for i in range(0, len(homos))])  # Adjust color
        colors[homos == 0] = 'b'
        colors[homos == 1] = 'y'

        plt.figure()
        plt.scatter(pair_distance, prelim_results, alpha=0.5, c=colors, label="Pairwise Data")
        plt.xlabel("Distance")
        plt.ylabel("Estimated shared loci")
        plt.hlines(distance_mean, bin_edges[:-1], bin_edges[1:], colors='g', lw=5, label='Binned data')
        plt.xlim([0, max_dist])
        plt.axhline(mean_het, label="Mean Value", color='k', linewidth=2)
        plt.axhline(0.25, label="Expected P0/Siblings", color='y')
        if draw_parent:
            plt.scatter(distance_parents, kinship, alpha=1, c='k', label="Simulated PO-pairs")
        plt.legend()
        plt.show()
        
        # Plot Correlations
        bin_mean = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Calculate the mean of the bins
        index = np.where(bin_mean > 50)  # Only take distances bigger than sigma
        C, k, std = fit_log_linear(bin_mean[index], distance_mean[index])  # Fit with exponential fit
        Nb_est = 1 / (-k)
        Nb_std = (-std / k) * Nb_est
        x_plot = np.linspace(min(bin_mean), max(bin_mean), 10000)
        # fit = [C * np.exp(r * t) for t in x]
        print("Fit: \nC: %.4G \nk: %.4G" % (C, k))
        
        plt.plot(bin_mean, distance_mean, 'ro', label="Estimated Correlation per bin")
        plt.plot(x_plot, C + k * np.log(x_plot), 'g', label="Fitted decay")
        plt.axhline(mean_het, label="Mean Value", color='k', linewidth=2)
        plt.annotate(r'$\bar{N_b}=%.4G \pm %.2G$' % (Nb_est, Nb_std) , xy=(0.6, 0.7), xycoords='axes fraction', fontsize=25)
        plt.legend()
        plt.ylabel("Estimated Correlation")
        plt.xlabel("Distance")
        # plt.xscale('log')
        plt.show()
        
        print("Samples with no opposing homozygotes: %.1f: " % np.sum(homos == 0))
        print("Samples with one opposing homozygote: %.1f: " % np.sum(homos == 1))
        
        # Plot Opposing-Homozygotes distribution
        counts = Counter(homos)
        hits = np.array(counts.keys())  
        values = np.array(counts.values())
        values = values / float(sum(values))  # To get probabilities
        
        parameters, cov_matrix = curve_fit(poisson, hits, values, p0=np.mean(homos))  # @UnusedVariable
        print("\nParameters %.2f\n" % parameters[0])
        # plot poisson-deviation with fitted parameter
        x_plot = np.linspace(0, hits.max(), 1000)
          
        plt.figure()
        plt.plot(x_plot, len(homos) * poisson(x_plot, *parameters), 'r-', lw=2, label="Poisson Fit")
        plt.hist(homos, bins=np.linspace(-0.5, 59.5, 61), label="Histogram")
        plt.plot(hits, values * len(homos), 'go', label="True data points")
        plt.legend()
        plt.title("Opposing homozygotes per pair")
        plt.show()
        
        x_plot = np.linspace(-0.5, 3, 500)
        plt.figure()
        plt.plot(x_plot, len(homos) * poisson(x_plot, *parameters), 'r-', lw=2, label="Poisson Fit")
        plt.hist(homos, bins=np.linspace(-0.5, 3.5, 5), label="Histogram")
        plt.plot(hits[:4], values[:4] * len(homos), 'go', label="True data points")
        plt.legend()
        plt.title("Opposing homozygotes per pair")
        plt.show()
        
    
    def ym_comparison(self, max_dist=1000, true_color=True):
        '''Compares yellow, magenta and inbetween Kinship coefficients along distance classes'''
        data, p_all, coords, color = self.data, self.p, self.coords, self.color  # Extract filtered SNPs. @UnusedVariable
        p_mean = self.data_p_mean
        print("Analysing %.1f suitable SNPs. " % len(data[0, :]))
        
        print("Allele Frequs:")
        print(p_all)
        # p_del=int(input("What SNP do you want to treat as color?"))
        # Pretend color is some other SNP:
        p_del_vec = np.arange(100) + 5  # Try out 100 random SNPs.
        
        distance_mean_y = np.zeros(20)
        distance_mean_m = np.zeros(20)
        distance_mean_ym = np.zeros(20)
        if true_color == True:  # "Forget about the pretending"-variable.
            p_del_vec = [-1]
        
        for p_del in p_del_vec:  # For every locus in p_del_vec do the deletion thingy
            if p_del != -1:  # Pretend SNP is "color" 
                color = self.data[:, p_del]
                data = np.delete(self.data, (p_del), axis=1)  # Delete this SNP from data to not bias kinship estimates
                p_mean = np.delete(p_all, (p_del), axis=0)
            else: p_mean = p_mean
         
            print(np.mean(color[color > -1]) / 2.0)
            
            y_comps = []  # Comparisons among yellow
            m_comps = []  # Comparisons among magenta
            ym_comps = []  # Comparisons inbetween yellow/magenta
            
            for (i, j) in itertools.combinations(np.arange(np.size(data, 0)), r=2):
                delta = coords[i, :] - coords[j, :]  # Calculate Pairwise distance
                pair_distance = np.linalg.norm(delta)
                estimator, _ = self.kinship_coeff_loc_freq(data[i, :], data[j, :], p_mean[i, :], p_mean[j, :])  # Kinship coeff per pair, averaged over loci
                
                if color[i] == 2 and color[j] == 2:  # Case: Two magentas
                    # estimator, _ = self.kinship_coeff(data[i, :], data[j, :], p)  # Kinship coeff per pair, averaged over loci
                    y_comps.append([estimator, pair_distance])  # Append Kinship coefff and pairwise distance         
                
                if color[i] == 0 and color[j] == 0:  # Case: Two yellows
                    # estimator, _ = self.kinship_coeff(data[i, :], data[j, :], p)
                    m_comps.append([estimator, pair_distance])
                    
                if (color[i] == 0 and color[j] == 2) or (color[i] == 2 and color[j] == 0):  # Case: One yellow, one magenta
                    # estimator, _ = self.kinship_coeff(data[i, :], data[j, :], p)
                    ym_comps.append([estimator, pair_distance])
                    # if pair_distance>200 and pair_distance<300:       # Mark Comparisons in right distance classe DELETE
                        # self.po_list.append([coords[i, :], coords[j, :]])
                    
            y_comps = np.array(y_comps)  # Make everything a numpy array
            m_comps = np.array(m_comps)
            ym_comps = np.array(ym_comps)

            print("Y-mean Kinship: %.6f" % np.mean(y_comps[:, 0]))
            print("M-mean Kinship: %.6f" % np.mean(m_comps[:, 0]))
            print("YM-mean Kinship: %.6f" % np.mean(ym_comps[:, 0]))
            
            distance_mean_yt, bin_edges, _ = binned_statistic(y_comps[:, 1], y_comps[:, 0], bins=20, statistic='mean', range=[0, max_dist])  # Calculate Bin Values
            distance_mean_mt, _, _ = binned_statistic(m_comps[:, 1], m_comps[:, 0], bins=20, statistic='mean', range=[0, max_dist])  # Calculate Bin Values
            distance_mean_ymt, _, _ = binned_statistic(ym_comps[:, 1], ym_comps[:, 0], bins=20, statistic='mean', range=[0, max_dist])  # Calculate Bin Values
        
            distance_mean_y += distance_mean_yt / float(len(p_del_vec))
            distance_mean_m += distance_mean_mt / float(len(p_del_vec))
            distance_mean_ym += distance_mean_ymt / float(len(p_del_vec))
            
        bin_mean = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Calculate the mean of the bins
        
        plt.figure()
        plt.scatter(y_comps[:, 1], y_comps[:, 0], alpha=0.5, c='y', label="Yellow-Yellow")
        plt.scatter(m_comps[:, 1], m_comps[:, 0], alpha=0.5, c='m', label="Magenta-Magenta")
        plt.scatter(ym_comps[:, 1], ym_comps[:, 0], alpha=0.5, c='g', label="Yellow-Magenta")
        plt.xlabel("Distance")
        plt.ylabel("Estimated shared loci")
        plt.xlim([0, max(bin_edges)])
        plt.axhline(0.25, label="Expected P0/Siblings", color='y')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(bin_mean, distance_mean_y, 'yo', label="Yellow-Yellow")
        plt.plot(bin_mean, distance_mean_m, 'mo', label="Magenta-Magenta")
        plt.plot(bin_mean, distance_mean_ym, 'go', label="Magenta-Yellow")
        plt.legend()
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Mean Kinship coefficient")
        plt.show()
        
    def geo_comparison(self, max_dist=1000):
        '''Compares kinship coefficients based on geography'''
        data, p, coords, color = self.data, self.p, self.coords, self.color  # Extract filtered SNPs. @UnusedVariable
        p_mean = self.data_p_mean
        
        print("Analysing %.1f suitable SNPs. " % len(data[0, :]))
        # p_del=int(input("What SNP do you want to treat as color?"))
        # Pretend color is some other SNP:
            
        y_comps = []  # Comparisons among yellow
        m_comps = []  # Comparisons among magenta
        h_comps = []  # Comparisons among hybrids
        ym_comps = []  # Comparisons inbetween yellow/magenta
        
        color = [1 for i in coords[:, 0]]
        
        for i in range(0, len(color)):  # Index everything via geographic position.
            if coords[i, 0] < -250:  # Set the color border (geographic)
                color[i] = 0
                
            elif coords[i, 0] > 400:  # Set the right color border (geographic)
                color[i] = 2
                
            else: color[i] = 1
            
        # p_y = 0.5 * np.array([np.mean([data[j, i] for j in range(0, len(color)) if color[j] == 0 and data[j, i] > -8]) for i in range(0, len(data[0, :]))])
        # p_m = 0.5 * np.array([np.mean([data[j, i] for j in range(0, len(color)) if color[j] == 2 and data[j, i] > -8]) for i in range(0, len(data[0, :]))])
        # p_h = 0.5 * np.array([np.mean([data[j, i] for j in range(0, len(color)) if color[j] == 1 and data[j, i] > -8]) for i in range(0, len(data[0, :]))])
        
        
        for (i, j) in itertools.combinations(np.arange(np.size(data, 0)), r=2):
            delta = coords[i, :] - coords[j, :]  # Calculate Pairwise distance
            pair_distance = np.linalg.norm(delta)
            estimator, _ = self.kinship_coeff_loc_freq(data[i, :], data[j, :], p_mean[i, :], p_mean[j, :])  # Kinship coeff per pair, averaged over loci
            # estimator, _ = self.kinship_coeff(data[i, :], data[j, :], p)  # Kinship coeff per pair, averaged over loci
                
            if color[i] == 2 and color[j] == 2:  # Case: Two magentas
                # estimator, _ = self.kinship_coeff(data[i, :], data[j, :], p)  # Kinship coeff per pair, averaged over loci
                m_comps.append([estimator, pair_distance])  # Append Kinship coefff and pairwise distance         
                
            elif color[i] == 0 and color[j] == 0:  # Case: Two yellows
                # estimator, _ = self.kinship_coeff(data[i, :], data[j, :], p)
                y_comps.append([estimator, pair_distance])
            
            elif color[i] == 1 and color[j] == 1:  # Case: Two hybrids
                estimator, _ = self.kinship_coeff(data[i, :], data[j, :], p)
                h_comps.append([estimator, pair_distance])
                    
            elif (color[i] == 0 and color[j] == 2) or (color[i] == 2 and color[j] == 0):  # Case: One yellow, one magenta
                # estimator, _ = self.kinship_coeff(data[i, :], data[j, :], p)
                ym_comps.append([estimator, pair_distance])
                    # if pair_distance>200 and pair_distance<300:       # Mark Comparisons in right distance classe DELETE
                        # self.po_list.append([coords[i, :], coords[j, :]])
                    
        y_comps = np.array(y_comps)  # Make everything a numpy array
        m_comps = np.array(m_comps)
        ym_comps = np.array(ym_comps)
        h_comps = np.array(h_comps)

        print("Y-mean Kinship: %.6f" % np.mean(y_comps[:, 0]))
        print("M-mean Kinship: %.6f" % np.mean(m_comps[:, 0]))
        print("YM-mean Kinship: %.6f" % np.mean(ym_comps[:, 0]))
            
        distance_mean_y, bin_edges, _ = binned_statistic(y_comps[:, 1], y_comps[:, 0], bins=15, statistic='mean', range=[0, 1500])  # Calculate Bin Values
        distance_mean_m, bin_edges_m, _ = binned_statistic(m_comps[:, 1], m_comps[:, 0], bins=15, statistic='mean', range=[0, 1500])  # Calculate Bin Values
        distance_mean_ym, bin_edges_ym, _ = binned_statistic(ym_comps[:, 1], ym_comps[:, 0], bins=9, statistic='mean', range=[800, max_dist])  # Calculate Bin Values
        distance_mean_h, bin_edges_h, _ = binned_statistic(h_comps[:, 1], h_comps[:, 0], bins=6, statistic='mean', range=[0, 600])  # Calculate Bin Values
        
        # Calculate the Variances
        dist_std_y, _, _ = binned_statistic(y_comps[:, 1], y_comps[:, 0], bins=15, statistic=np.std, range=[0, 1500])  # Calculate Bin Values
        dist_std_m, _, _ = binned_statistic(m_comps[:, 1], m_comps[:, 0], bins=15, statistic=np.std, range=[0, 1500])  # Calculate Bin Values
        dist_std_h, _, _ = binned_statistic(h_comps[:, 1], h_comps[:, 0], bins=6, statistic=np.std, range=[0, 600])  # Calculate Bin Values
        
        bin_mean = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Calculate the geographic mean of the bins
        bin_mean_m = (bin_edges_m[:-1] + bin_edges_m[1:]) / 2.0
        bin_mean_ym = (bin_edges_ym[:-1] + bin_edges_ym[1:]) / 2.0
        bin_mean_h = (bin_edges_h[:-1] + bin_edges_h[1:]) / 2.0
        
#         plt.figure()
#         plt.scatter(y_comps[:, 1], y_comps[:, 0], alpha=0.5, c='y', label="Yellow-Yellow")
#         plt.scatter(m_comps[:, 1], m_comps[:, 0], alpha=0.5, c='m', label="Magenta-Magenta")
#         plt.scatter(ym_comps[:, 1], ym_comps[:, 0], alpha=0.5, c='g', label="Yellow-Magenta")
#         plt.xlabel("Pairwise Euclidean distance")
#         plt.ylabel("Kinship coefficient ")
#         plt.xlim([0, max(bin_edges)])
#         plt.axhline(0.25, label="Expected kinship PO/Full Sibs", color='y')
#         plt.legend()
#         plt.show()
        
        plt.figure()
        plt.plot(bin_mean, distance_mean_y, 'yo', label="Yellow-Yellow", markersize=10)
        plt.plot(bin_mean_m, distance_mean_m, 'mo', label="Magenta-Magenta", markersize=10)
        plt.plot(bin_mean_ym, distance_mean_ym, 'go', label="Magenta-Yellow", markersize=10)
        plt.plot(bin_mean_h, distance_mean_h, 'wo', label="Hybrid-Hybrid", markersize=10)
        plt.tick_params(labelsize=20)
        plt.xlabel("Euclidean Distance", fontsize=28)
        plt.ylabel("f", fontsize=28)
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(bin_mean, dist_std_y, 'ys', label="Yellow-Yellow", markersize=10)
        plt.plot(bin_mean_m, dist_std_m, 'ms', label="Magenta-Magenta", markersize=10)
        plt.plot(bin_mean_h, dist_std_h, 'ws', label="Hybrid-Hybrid", markersize=10)
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Standard Deviation Kinship")
        plt.legend()
        plt.tick_params(labelsize=20)
        plt.show()
       
    def kinship_coeff(self, sample1, sample2, p):
        '''Takes two samples as input(SNP-numpy arrays) and calculates the kinship coefficient
        Additionally return the number of opposing homozygotes'''
        working_SNPs = np.where(sample1[:] + sample2[:] > -1)[0]  # Which entries to actually use

        pi = 0.5 * sample1[working_SNPs]
        # qi = 1 - pi
        pj = 0.5 * sample2[working_SNPs]
        # qj = 1 - pj
        pt = p[working_SNPs]  # Temporary p
        # qt = 1 - pt
                
        # estimator = np.mean((pi - pt) * (pj - pt) / pt + (qi - qt) * (qj - qt) / qt)  # Multi allelic version
        estimator = np.mean((pi - pt) * (pj - pt) / (pt * (1 - pt)))
        homos = np.sum(np.absolute(pi - pj) == 1)
        return (estimator, homos)
    
    def kinship_coeff_loc_freq(self, sample1, sample2, p1, p2):
        '''Implement an updated version of the kinship coefficient - assuming better knowledge of allele frequencies
        and calculate pairwise relationship based on updated frequencies '''
        p1_okay = np.minimum(p1, 2 - p1) > (2.0 * self.maf)  # Everything times 2 since given all-freqs always times 2
        p2_okay = np.minimum(p2, 2 - p2) > (2.0 * self.maf)
        working_snps = sample1[:] + sample2[:] > -1  # Since -9 on bad entries
        working_SNPs = np.where(p1_okay * p2_okay * working_snps)[0]  # Which entries to actually use - all three categories have to be good (i.e.= 1)
        pi = 0.5 * sample1[working_SNPs]  # Get allele freqs per sample (normalize to 0,1)
        pj = 0.5 * sample2[working_SNPs]
        pt1, pt2 = 0.5 * p1[working_SNPs], 0.5 * p2[working_SNPs]  # Estimated parental all frequs
        
        pc = (pt1 + pt2) / 2.0  # Calculate the mean allele frequency (i.e. the "common" pool)
        
                
        estimator = np.mean((pi - pt1) * (pj - pt2) / (pc * (1 - pc)))  # Calculate the mean relatedness
        homos = np.sum(np.absolute(pi - pj) == 1)
        return (estimator, homos)
    
    def create_offspring(self, sNPs1, sNPs2):
        '''Combines two individuals randomly to a new individual'''
        new_SNPs = [-9 for _ in range(0, len(sNPs1))]  # Initialize new SNP, default is error
        
        for k in range(0, len(sNPs1)):
            if sNPs1[k] + sNPs2[k] > -1:
                # Print draw random number with Prob. sNP1 and random number with Prob sNp2:
                new_SNPs[k] = (np.random.binomial(1, sNPs1[k] / 2.0) + np.random.binomial(1, sNPs2[k] / 2.0))
        return np.array(new_SNPs)
    
    def plot_geography(self):
        '''Plots geography of the data. Now with flower color!!!'''    
        plt.figure()
        plt.scatter(self.coords[:, 0], self.coords[:, 1], c=self.color, alpha=0.5)
        plt.ylim([-1000, 2000])
        plt.axis("equal")
        # plt.ylim([-500, 500])
        # Draw PO-lines:
        for i in self.po_list:
            plt.plot([i[0][0], i[1][0]], [i[0][1], i[1][1]], c='r', linewidth=2)
        plt.axvline(-250, color='k', linestyle='solid')
        plt.axvline(400, color='k', linestyle='solid')
        plt.tick_params(labelsize=22)
        plt.grid()
        plt.show()    

    def permute_colors(self, dist=60, neighbr_nr=100):
        '''Permutes color values with values from neighbors. Save it in self.colors'''
        color1 = [0 for _ in self.color]  # New color vector
        pool_cand = []  # Pool for possible candidates
        
        for i in range(0, len(self.color)):
            for j in range(max(i - neighbr_nr, 0), min(i + neighbr_nr, len(self.color) - 1)):  # Iterate through possible neighbours
                dist_ij = np.linalg.norm(self.coords[i, :] - self.coords[j, :])
                if dist_ij < dist:  # If Pairwise distance small enough, add to pool
                    pool_cand.append(j)  # Append possible candidate
            color1[i] = self.color[choice(pool_cand)]  # Replace color; at least element itself is in List
            pool_cand = []  # Delete Candidate List
        
        self.color = np.array(color1)           
        print("Colors permuted")
        
    def nbh_analysis(self, nbh_dist, min_nbh=6):
        '''Determines for every individual its neighbors max. nbh_dist away and calculates their mean kinship coefficient.'''
        data, p = self.data, self.p
        nbh = [[] for _ in data[:, 0]]  # Generates the neighborhood list
        mean_corr = []  # Placeholder array
        
        for i in range(0, len(nbh)):  # Generate the neighborhood for every plant
            if i % 100 == 0:  # Do let programmer know where program is
                print("Doing: %.1f" % i)
            j = i  # Index for other individual
            while j > -1:
                dist_ij = np.linalg.norm(self.coords[i, :] - self.coords[j, :])
                if dist_ij < nbh_dist:
                    nbh[i].append(j)
                if (self.coords[i, 0] - self.coords[j, 0]) > nbh_dist: break  # If x-distance bigger than max distance.
                else: j -= 1
            j = i + 1  # Now go x-distances up (assuming everything is sorted by x-axis)
            while j < len(nbh):
                dist_ij = np.linalg.norm(self.coords[i, :] - self.coords[j, :])
                if dist_ij < nbh_dist:
                    nbh[i].append(j)
                if (self.coords[j, 0] - self.coords[i, 0]) > nbh_dist:    break
                else: j += 1
                
        # Plot Size of the Nbhs:
        nbh_tot = np.array([len(i) for i in nbh]).astype('int')
        plt.figure()
        plt.plot(self.coords[:, 0], nbh_tot, 'ro')
        plt.xlabel("X-Coord")
        plt.ylabel("Neighbors")
        plt.show()  
        
        plt.figure()
        plt.scatter(self.coords[:, 0], self.coords[:, 1], c=nbh_tot, alpha=0.5)
        plt.colorbar()
        plt.show()
        
        for k in range(0, len(nbh)):
            if k % 100 == 0:  # To let programmer know where program is
                print("Doing: %.1f" % k)
            
            pair_comps = []  # Array for all pairwise Kinship estimates
            for (i, j) in itertools.combinations(nbh[k], r=2):
                estimator, _ = self.kinship_coeff(data[i, :], data[j, :], p)  # Kinship estimate per pair, averaged over loci
                pair_comps.append(estimator)  # Append Kinship estimate
                
            if len(nbh[k]) > min_nbh:
                mean_corr.append(np.mean(pair_comps))  # Mean Kinship
            else: mean_corr.append(-9)  # Not enough individuals for reliable estimate
        
        # Plot the Neighborhood-Kinship estimates:
        mean_corr = np.array(mean_corr)
        good_inds = mean_corr != -9
        plt.figure()
        plt.plot(self.coords[good_inds, 0], mean_corr[good_inds], 'ro')
        plt.xlabel("X-Coord")
        plt.ylabel("Kinship-Estimate")
        plt.show()  
        
        mean_correlation, bin_edges, _ = binned_statistic(self.coords[good_inds, 0], mean_corr[good_inds], bins=20, statistic='mean')  # Calculate Bin Values   
        
        
        plt.figure()
        plt.scatter(self.coords[good_inds, 0], self.coords[good_inds, 1], c=mean_corr[good_inds], alpha=0.5, vmin=-0.01, vmax=0.05)
        plt.xlim([min(self.coords[good_inds, 0]) - 50, max(self.coords[good_inds, 0]) + 50])
        plt.ylim([min(self.coords[good_inds, 1]) - 50, max(self.coords[good_inds, 1]) + 60])
        plt.colorbar()
        plt.show()
        
        bin_mean = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Calculate the mean of the bins
        print(bin_mean)
        print(mean_correlation)
        plt.figure()
        plt.plot(bin_mean, mean_correlation, 'ro')
        plt.xlabel("x-Axis")
        plt.ylabel("Mean correlation")
        plt.grid()
        plt.show()
        
    def loc_all_freq_covariances(self, min_dist=30, max_dist=200):
        '''Calculate and plot local allele frequency correlations'''
        data, coords, p_mean = self.data, self.coords, self.data_p_mean  # Extract filtered SNPs.
        n = len(data[:, 0])
        
        print("Calculate pairwise distances...")  # Calculate pairwise distances
        dist_mat = calc_distance_matrix(coords)
        indices = (dist_mat < max_dist) * (dist_mat > min_dist)  # Mark the individuals to do
        
        neighbors = np.sum(indices, axis=1)  # Quicker vectorized way
        
        print("Neighbors: ")
        print(neighbors)
        
        print("Calculate average f per relevant pair...")
        x = np.zeros(n)  # Create empty vector for the data
        
        has_nb = np.where(neighbors > 0)[0]  # Create list where actually neighbors
        
        for i in has_nb:
            inds = np.where(indices[i, :] == 1)[0]  # Where the distances are right
            x[i] = np.mean([self.kinship_coeff_loc_freq(data[i, :], data[j, :], p_mean[i, :], p_mean[j, :])[0] for j in inds])  # Calculate the mean value per individual
        print("Mean value: %.5f" % np.mean(x))
        print("Smoothing...") 
        self.smooth_n_print(x, dist_mat, 150, coords, weights=neighbors, yval="Local f")  # Smooth out
    
    def x_dist_cov(self, min_dist=500, max_dist=1000):
        '''Calculate and plot allele frequency correlations'''
        data, coords, p_mean = self.data, self.coords, self.data_p_mean  # Extract filtered SNPs.
        n = len(data[:, 0])
        
        dist_mat = calc_distance_matrix(coords)
        dist_mat1 = np.array([[-coords[i, 0] + coords[j, 0] for i in range(n)] for j in range(n)])
        
        # Mark the individuals for for comparison: Smaller than maximal distance, greater than min x-distance
        indices = (dist_mat < max_dist) * (dist_mat1 > min_dist)
        
        neighbors = np.sum(indices, axis=1)  # Number of neighbours per ind
        
        print("Calculate average f per relevant pair...")
        x = np.zeros(n)  # Create empty vector for the data
        
        has_nb = np.where(neighbors > 0)[0]  # Create list where actually neighbors
        
        for i in has_nb:
            inds = np.where(indices[i, :] == 1)[0]  # Where the distances are right
            x[i] = np.mean([self.kinship_coeff_loc_freq(data[i, :], data[j, :], p_mean[i, :], p_mean[j, :])[0] for j in inds])  # Calculate the mean value per individual
        print("Mean value: %.5f" % np.mean(x))
        print("Smoothing...") 
        self.smooth_n_print(x, dist_mat, 150, coords, weights=neighbors, yval="Mean Covariance")
          
                  
    def smooth_n_print(self, x, dist_mat, sigma, coords, weights=0, plot=True, yval="y-Val"):
        '''Smooth out the given statistics x and plots against x-axis'''
        indices_good = np.array(weights) > 10  # Only if enough neighbors (>10)
        
        x_mean = self.calc_weighted_mean(dist_mat, x, weights, sigma)
        
        if plot == False: return x_mean
        
        plt.figure()  # Plot against x-axis
        plt.scatter(coords[indices_good, 0], x_mean[indices_good], color=self.color[indices_good], alpha=0.7, s=25)
        plt.tick_params(labelsize=20)
        plt.xlabel("x-Axis", fontsize=20)
        plt.ylabel(yval, fontsize=20)
        plt.show()
        
        plt.scatter(self.coords[indices_good, 0], self.coords[indices_good, 1], c=x_mean[indices_good], alpha=0.7, s=25)
        plt.tick_params(labelsize=20)
        plt.colorbar()
        plt.show()
        
    def calc_weighted_mean(self, dist_mat, x, w, sigma):
        '''Given a distance matrix and a vector of values,
        calculate the weighted mean according to a kernel sigma.
        Also weight with w'''
        x, dist_mat, w = np.array(x), np.array(dist_mat), np.array(w)  # Just in case that not numpy array
        
        weights = 1 / (2.0 * np.pi * sigma ** 2) * np.exp(-dist_mat ** 2 / (2.0 * sigma ** 2))  # Calculate the Gaussian distance weights
        weights = w[None, :] * weights  # Bring in the number of neighbor weights
        print(weights[:10, :10])
        x_mean = np.dot(weights, x) / np.sum(weights, axis=1)  # Calculate weighted mean
        print(x_mean[:10])
        return(x_mean)
    
          
    def homozygosity_analysis(self):
        '''Plots homozygosity per sample over all samples'''
        data, p = self.data, self.p
        homozyg = np.zeros(len(data[:, 0]))  # Vector for homozygosity per sample
        
        for i in range(0, len(data[:, 0])):
            working_SNPs = np.where((data[i, :] > -1) & (np.minimum(p, 1 - p) > 0.05))[0]  # Which entries to actually use; only snps with maf>0.5

            p_i = data[i, working_SNPs] / 2.0
            p_t = p[working_SNPs]  # Temporal allele freq vector @UnusedVariable
            # heteros = np.sum(p_i == 0.5) / float(len(p_i))  # Calculate mean heterozygosity
            
            heteros = np.mean(2 * p_i * (1 - p_i) / (p_t * (1 - p_t)))
            homozyg[i] = 1 - heteros
        
        hom_mean, bin_edges, _ = binned_statistic(self.coords[:, 0], homozyg, bins=10, statistic='mean')  # Calculate Bin Values
        
        # Do the plot 
        print("Mean homozygosity %.6f:" % np.mean(homozyg))   
        
        plt.figure()
        plt.scatter(self.coords[:, 0], self.coords[:, 1], c=homozyg, alpha=0.5)
        # plt.xlim([min(self.coords[:, 0]) - 50, max(self.coords[:, 0]) + 50])
        # plt.ylim([min(self.coords[:, 1]) - 50, max(self.coords[:, 1]) + 60])
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.scatter(self.coords[:, 0], homozyg, c=self.color, alpha=0.7)
        plt.xlabel("x-Coordinate")
        plt.ylabel("Mean Homozygosity")
        plt.hlines(hom_mean, bin_edges[:-1], bin_edges[1:], colors='g', lw=5, label='Binned data')
        plt.axhline(np.mean(homozyg), label="Null line", color='k', linewidth=1)
        plt.show()
        
    
def set_colors(color_vec):
    '''Helper Function that transforms colors to Python Colors.
    Return color vector'''
    color = np.array(['grey' for _ in range(0, len(color_vec))]).astype('|S16') # Default Value
    color[color_vec == 'Y'] = 'y'
    color[color_vec == 'FR'] = 'm'
    color[color_vec == 'W'] = 'azure'
    color[color_vec == 'FO'] = 'darkorange'
    color[color_vec == 'WO'] = "sandybrown"
    color[color_vec == 'WR'] = "lightsalmon"
    return color
                                
def princomp(A):
    """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables. 
    Returns :  coeff is a p-by-p matrix, each column containing coefficients 
    for one principal component. score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
  latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A."""
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A - np.mean(A.T, axis=1)).T  # subtract the mean (along columns)
    [latent, coeff] = np.linalg.eig(np.corrcoef(M))  # attention:not always sorted
    score = np.dot(coeff.T, M)  # projection of the data in the new space
    return coeff, np.real(score), latent  

             
def fit_log_linear(t, y):
    '''Fitting exponential decay and returns parameters: y=A*Exp(-kt)'''
    t = np.log(t)
    param, V = np.polyfit(t, y, 1, cov=True)  # Param has highest degree first
    return param[1], param[0], np.sqrt(V[0, 0])

def fit_gaussian(values):
    ''' Fits a Gaussian to array containing values'''
    param = norm.fit(values)  # distribution fitting
    return param  # param[0] and param[1] are the mean and the standard deviation of the fitted distribution   

def fit_poisson(values):
    '''Fits a poisson distribution to array containing values'''
    param = poisson.fit(values)
    print(param)
    
def poisson(k, lamb):
    # Poisson function, parameter lamb is the fit parameter
    return (lamb ** k / factorial(k)) * np.exp(-lamb)

def calc_distance_matrix(coords):
    '''Calculate the distance matrix between all coords. Requires numpy array of 2d coordinates'''
    print("Calculating distance matrix")
    start = time()
    dist_mat = np.linalg.norm(coords[:, None] - coords, axis=2)
    print("Time taken %.2f" % (time() - start))
    return dist_mat


    
    
        
