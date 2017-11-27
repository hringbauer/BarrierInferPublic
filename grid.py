'''
Created on 27.01.2015
The Grid class
Contains grid to run my simulations; 
it can run no barrier and a barrier
There is also a SecondaryGrid class that inherits from it and can
simulate a scenario of secondary contact
@author: hringbauer
'''

import numpy as np
import matplotlib.pyplot as plt
import itertools
import bisect
from scipy.stats import binned_statistic
from scipy.special import kv as kv
from scipy.optimize.minpack import curve_fit
from kernels import fac_kernel  # Factory Method which yields Kernel Object
from time import time
import multiprocessing as mp
# from kernels import fac 

class Grid(object):
# Object for the Data-Grid. Contains matrix of lists for chromosomal pieces and methods to update it.    
    test = 100
    gridsize_x = 1000
    gridsize_y = 1000
    ind_list = []  # List of lists to which the initial genotypes correspond
    update_list = []  # List of individuals do update. Contains x and y positions
    position_list = []  # List of initial positions.
    ancestry = []  # List of individuals that currently descent from individual i
    final_ancestry = []  # List of final ancestry
    genotypes = []  # List of the genotypes of every individuals
    genotype_mat = []  # List of the genotype matrix. In case multiple multiple markers are simulated
    p_mean = 0.5
    t = 0  # Current time back in generations.
    barrier = 500.5
    barrier_strength = 1  # The strength of the barrier # 1 Everything migrates; 0 Nothing Migrates
    sigma = 0.965  # 0.965  # 1.98 # 0.965
    ips = 10  # Number of haploid Individuals per Node (For D_e divide by 2)
    mu = 0.003  # The Mutation/Long Distance Migration rate.
    
    def __init__(self):  # Initializes an empty grid
        print("Initializing...")  # Actually all relevant things are set with set_samples
        
    def set_parameters(self, gridsize_x, gridsize_y, sigma, ips, mu):
        '''Sets all important Grid Parameters'''
        self.gridsize_x = gridsize_x
        self.gridsize_y = gridsize_y
        self.sigma = sigma
        self.ips = ips
        self.mu = mu
        
    def set_barrier_parameters(self, barrier, barrier_strength):
        '''Sets important Barrier Parameters'''
        self.barrier = barrier  # Where to actually find the Barrier.
        self.barrier_strength = barrier_strength  # How strong the barrier actually is.
      
    def set_samples(self, position_list):
        '''Sets samples to where they belong. THE ONLY WAY TO SET SAMPLES'''
        # print(position_list)  # For Debugging
        self.update_list = position_list  # Set the update List
        print("Ancestry List initialized.")
        self.ancestry = [[i] for i in range(len(position_list))]  # Set the list of ancestors
        self.genotypes = [-1 for _ in range(len(position_list))]  # -1 for not set
        self.position_list = position_list
        self.final_ancestry = []
        
    def update_grid_t(self, t, coalesce=1, barrier=0, p=0.5):
        '''Updates the grid for t generations'''
        for i in range(t):
            t += 1  # Adds time to the generation clock!
            self.drop_mutations()  # Drops mutations that moves lineages into their pension
            
            if i % 100 == 0:
                print("Doing generation: %i " % i)
            
            if barrier == 0:
                self.update_grid()  # Updates the update list
            
            elif barrier == 1:  # In case of barrier)
                self.update_grid_barrier()
                
            else:
                raise Exception("Only 1/0 Input allowed!")
            
            if coalesce == 1:
                self.coal_inds()  # Coalesces individuals in the same position; and merges the off-spring indices
        self.ancestry = self.ancestry + self.final_ancestry  # Gets non-active lineages back from Pension
        self.draw_genotypes(p)  # Draw genotypes
        
        # Some output for debugging:
        # print("Number of individuals hit by mutation: %i" % np.sum([len(i) for i in self.final_ancestry]))
        # print("Total length of ancestry: %i" % np.sum([len(i) for i in self.ancestry]))
        print("Run complete Lol\n ")  
        
    def update_grid(self):
        '''Update the grid for a single generation'''
        new_coords = [self.update_individual_pos(pos[0], pos[1]) for pos in self.update_list]
        self.update_list = new_coords
        
    def update_grid_barrier(self):
        '''Update the grid in case of a barrier'''
        new_coords = [self.update_individual_pos_barrier(pos[0], pos[1]) for pos in self.update_list]
        self.update_list = new_coords
    
    def draw_delta(self):
        '''Draws the individual axial position Offsets'''
        scale = self.sigma / np.sqrt(2)  # To scale it right for Laplace Dispersal.
        return np.around(np.random.laplace(scale=scale))  # Draw random off-set
           
    def update_individual_pos(self, x, y):
        '''Method that updates the position of an individual'''
        scale = self.sigma / np.sqrt(2)  # For Laplace Dispersal
        x1 = (x + np.around(np.random.laplace(scale=scale))) % self.gridsize_x
        y1 = (y + np.around(np.random.laplace(scale=scale))) % self.gridsize_y
        return(x1, y1)
    
    def update_individual_pos_barrier(self, x, y):
        '''Method that updates individual positions with barrier'''
        delta_x = self.draw_delta() # Draw the x off-set
        delta_y = self.draw_delta()  # Draw the y off-set
        
        x1 = (x + delta_x)
        # print("Old/New: %.2f %.2f" % (x,x1))   For Debugging...
        y1 = (y + delta_y)
        
        if (x > self.barrier and x1 <= self.barrier):
            if np.random.random() > self.barrier_strength:  # In case of reflection
                x1 = x  # Nothing happens
                # x1 = self.barrier + (self.barrier - x1)  # x1 gets reflected#
                # x1 = self.barrier + 1 
            
        elif (x < self.barrier and x1 >= self.barrier):
            if np.random.random() > self.barrier_strength:  # In case of reflection 
                x1 = x  # Nothing happens       
                # x1 = self.barrier - (x1 - self.barrier)  # x1 gets reflected
                # x1 = self.barrier - 1 
        
        x1 = x1 % self.gridsize_x
        y1 = y1 % self.gridsize_y    
        return(x1, y1)
    
    def drop_mutations(self):
        '''Drops mutations. Moves lineages to final ancestry and deletes them from update List'''
        inds = self.update_list
        n = len(inds)
        # print("Currently: %i Individuals" % n)
        inds_mut = np.random.random(n) < self.mu  # Where mutations happen. mu is mutation rate per generation
        del_list = np.where(inds_mut)[0]  # Get a list of Entries where mutations actually happen
        
        # Take the individuals where mutations happen and saves final ancestry
        for i in del_list:
            self.final_ancestry.append(self.ancestry[i])  # Saves the ancestry information to final Ancestry List
         
        # Sends lineages were mutation happen into Pension   
        self.update_list = [i for j, i in enumerate(self.update_list) if j not in del_list]
        self.ancestry = [i for j, i in enumerate(self.ancestry) if j not in del_list] 
        
    def coal_inds(self):
        '''Method to coalesce individuals.'''
        inds = self.update_list
        
        dub_inds = list_duplicates(inds)
        del_list = []  # List of individuals to delete in the end
        
        for inds in dub_inds:
            '''Iterate over all individuals in dub_inds'''
            i, j = inds[0], inds[1]  # i < j. Always
            if np.random.random() < 1.0 / (self.ips):  # Only in case if individuals fall an the same ancestor
                self.ancestry[i] = self.ancestry[i] + self.ancestry[j]  # Pool the ancestry list 
                del_list += [j]  # Update the list of which individuals to delete

        # Remove Individuals in Delete_List from update and ancestry List:    
        self.update_list = [i for j, i in enumerate(self.update_list) if j not in del_list]
        self.ancestry = [i for j, i in enumerate(self.ancestry) if j not in del_list]
    
    def draw_genotypes(self, p=0.5):
        '''Method that draws genotypes from pool of allele freqs.
        p is the frequency of the allele from which one has to draw.
        Sets genotype i to a certain allele.'''
        self.p_mean = p  # Sets the mean allele Frequency
        print(self.ancestry)
        for lists in self.ancestry:
            all_freq = np.random.random() < p  # Draws the allele Frequency
            for i in lists:
                self.genotypes[i] = all_freq  # Sets the Genotype to that allele Frequency              
        print(self.genotypes)
    
    def plot_grid(self):
        '''Function that plots the grid.'''
        x_cords = [pos[0] for pos in self.update_list]
        y_cords = [pos[1] for pos in self.update_list]
        
        print(self.update_list)
        print(x_cords)
        print(y_cords)
        
        plt.figure()  # Do the plotting
        plt.scatter(x_cords, y_cords)  
        plt.show() 
        
    def plot_all_freqs(self):
        '''Plots local allele frequency genotypes'''
        plt.figure()
        x_cords = [pos[0] for pos in self.position_list]
        y_cords = [pos[1] for pos in self.position_list]
        
        plt.scatter(x_cords, y_cords, c=self.genotypes, s=50) 
        plt.show()
        
    def extract_F(self, n_bins=10, show=False):
        '''Extracts a vector containing the fraction of Fs.
        Group successful  F via np.hist in bins and calculate mean distance and probabilities'''
        # Create array with the pairwise distance of all detected elements
        
        pairlist = []  # Creates List of all Pairs that coalesced (Pairs of indices)
        for lis in self.ancestry:  # Ancestry List
            for combs in itertools.combinations(lis, r=2):
                pairlist.append(combs)
                
        pair_distance = [torus_distance(self.position_list[ind[0]], self.position_list[ind[1]], self.gridsize_x, self.gridsize_y) for ind in pairlist]
                  
        # Plot the result in a histogram:
        # counts, bins, patches = plt.hist(pair_distance, n_bins, facecolor='g', alpha=0.9)  # @UnusedVariable Old Version (Matplotlib)
        counts, bins = np.histogram(pair_distance, n_bins)
        
        # Find proper normalization factors:
        distance_bins = np.zeros(len(bins) - 1)  # Create bins for every element in List; len(bins)=len(counts)+1
        bins[-1] += 0.000001  # Hack to make sure that the distance exactly matching the max are counted
#         for i in self.start_list[1:]:  # Calculate distance to the first element in start_list for all elements to get proxy for number of comparisons
#             dist = torus_distance(i[0], i[1], self.start_list[0][0], self.start_list[0][1], self.gridsize)
#                 
#             j = bisect.bisect_right(bins, dist)
#             if j < len(bins) and j > 0:  # So it actually falls into somewhere
#                 distance_bins[j - 1] += 1
                
        # Calculate Distance for every possible pair to get proper normalization factor:
        for (m, n) in itertools.combinations(self.position_list, r=2):
            dist = torus_distance(m, n, self.gridsize_x, self.gridsize_y)   
            j = bisect.bisect_right(bins, dist)
            if j < len(bins) and j > 0:  # So it actually falls into somewhere
                distance_bins[j - 1] += 1
        
        distance_mean, _, _ = binned_statistic(pair_distance, pair_distance, bins=n_bins, statistic='mean')  # Calculate mean distances for distance bins
        # distance_mean=(bins[1:]+bins[:-1])/2.0
        
        distance_mean = distance_mean[counts != 0]  # Remove bins with no values / MAYBE UPDATE?
        distance_bins = distance_bins[counts != 0]
        counts = counts[counts != 0]
        distance_mean[distance_mean == 0] = 1  # In deme case, to account for possibility of bins with dist=0    
        
        # Poisson-Error per bin:
        error = np.sqrt(counts)
        results = [counts[i] / distance_bins[i] for i in range(0, len(counts))]
        error = [error[i] / distance_bins[i] for i in range(0, len(counts))]  # STD
        
        if show == True:
            plt.figure()
            plt.errorbar(distance_mean, results, error, fmt='o')
            plt.ylabel("Fraction of F")
            plt.xlabel("Distance")
            # plt.xscale("log")
            plt.show()
        
        return (distance_mean, results, error)   
     
    def fit_F(self, show=False):
        '''Fits the underlying F-vector: '''
        x, y, error = self.extract_F(20, show=False)  # First calculates the F Vector
        
        parameters, cov_matrix = curve_fit(bessel0, x, y, absolute_sigma=True, sigma=error)  # @UnusedVariable p0=(C / 10.0, -r)
        std_param = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
        C1, r1 = parameters  # Fit with curve_fit:

        print("Fitted Parameters + Errors")
        print(parameters)
        print(std_param)
        
        # Fit the Diffusion Kernel
        KC = fac_kernel("DiffusionK0")
        KC.set_parameters([4 * np.pi * 5.0, 0.006, 1.0, 0.0])  # Diffusion; t0, mutation, density 
        print(KC.give_parameters())
        x_plot = np.linspace(min(x), max(x), 100)    
        coords = [[0, 0], ] + [[0, i] for i in x_plot]  # Coords-Vector. Begin with [0,0]!!
        kernel = KC.calc_kernel_mat(coords)
        
        # y_vec= [KC.num_integral(r) for r in x_plot] # 0 Difference along the y-Axis ;
        
        if show == True:  # Do a plot of the fit:
            
            plt.figure()
            # plt.yscale('log')
            plt.errorbar(x, y, yerr=error, fmt='go', label="Observed F", linewidth=2)
            # plt.semilogy(x, fit, 'y-.', label="Fitted exponential decay")  # Plot of exponential fit
            plt.plot(x_plot, bessel0(x_plot, C1, r1), 'r-.', label="Fitted Bessel decay", linewidth=2)  # Plot of exact fit
            plt.plot(x_plot, kernel[1:, 0], label="From Kernel Function")
            plt.plot(x_plot, bessel0(x_plot, 1 / (np.pi * 4 * 5.0), np.sqrt(2 * 0.003)), 'g-.', label="Ideal Bessel decay", linewidth=2)
            # plt.plot(x_plot, y_vec, 'm-.', label="Direct numerical Integration")
            
            plt.xlabel('Pairwise Distance', fontsize=25)
            plt.ylabel('F', fontsize=25)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.legend(prop={'size':15})
            plt.show()

    def draw_correlated_genotypes(self, nr_genotypes, l, a, c, p_mean, show=False, fluc_mean=0, coords=None):
        '''Draw correlated genotypes
        l: Typical correlation length, a: Absolute correlation, 
        c: Strength of the Barrier, fluc_mean: Whether varying mean all. frequency is simulated'''
        
        # Set Sample Coordinates:
        if coords == None:
            coords = np.array([(i, j) for i in range(-19, 20, 2) for j in range(-25, 25, 2)])  # To have dense sampling on both sides of the HZ

        f_mean = 2.0 * np.arcsin(np.sqrt(p_mean))  # Do the Arc Sin Transformation (Reverse of the Link Function)
        mean_p = np.array([f_mean for _ in range(len(coords))])  # Calculate the mean allele frequency
        f_delta = np.zeros(nr_genotypes)  # Initialize 0 deviations.
        
        if fluc_mean > 0:
            fluc_mean = float(input("What should the standard deviation around the mean f be?\n"))
            f_delta = np.random.normal(scale=fluc_mean, size=nr_genotypes)  # Draw some random Delta F from a normal distribution
            # f_delta = np.random.laplace(scale=v / np.sqrt(2.0), size=nr_genotypes)  # Draw some random Delta f from a Laplace distribution 
            # f_delta = np.random.uniform(low=-v * np.sqrt(3), high=v * np.sqrt(3), size=nr_genotypes)  # Draw from Uniform Distribution
            # f_delta = np.random.uniform(0, high=v * np.sqrt(3), size=nr_genotypes)  # Draw from one-sided uniform Distribution!
            
            print("Observed Standard Deviation: %.4f" % np.std(f_delta))
            print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(f_delta ** 2)))
            
        if show == True:
            print("Mean f: %.4f" % f_mean)
        
        
        r = np.linalg.norm(coords[:, None] - coords, axis=2)
        # Add Identity matrix for numerical stability
        
        cov_mat = full_kernel_function(coords, l, a, c) + 0.000001 * np.identity(len(coords))  # Calculate the covariance matrix. Added diagonal term for numerical stability
        
        # Function for Covariance Matrix from Diffusion Kernel:
        # #KC = fac_kernel("DiffusionK0")
        # #KC.set_parameters([62.8, 0.002, 1.0])
        # #start = time()
        # #cov_mat = KC.calc_kernel_mat(coords)
        # #end = time()
        # #print("Runtime: %.4f: " % (end - start))
        
        
        
        # if show == True:
            # print(np.linalg.eig(cov_mat)[0])
        
        data = np.random.multivariate_normal(mean_p, cov_mat, nr_genotypes)  # Do the random draws of the deviations from the mean
        data = np.transpose(data)  # Transpose, so that individual x locus matrix
        data = data + f_delta[None, :]  # Add the allele frequency fluctuations
        # print(np.mean(data, axis=0))
        
        p = arc_sin_lin(data)  # Do an arc-sin transform
        
        genotypes = np.random.binomial(1, p)  # Draw the genotypes
        
        print("Mean Allele Frequencies: \n")
        print(np.mean(genotypes, axis=0))
        
        if show == True:
            plt.figure()
            plt.subplot(211)
            plt.scatter(coords[:, 0], coords[:, 1], c=data[:, 0], s=100)
            plt.colorbar()
            
            plt.subplot(212)
            plt.scatter(coords[:, 0], coords[:, 1], c=genotypes[:, 0], s=100)
            plt.colorbar()
            plt.show()
            
            plt.figure()
            plt.title("Sample Distribution")
            plt.scatter(coords[:, 0], coords[:, 1], label="Samples")
            plt.vlines(0, min(coords[:, 1]), max(coords[:, 1]), linewidth=2, color="red", label="Barrier")
            plt.xlabel("X-Coordinate")
            plt.ylabel("Y-Coordinate")
            plt.legend()
            plt.show()
        return coords, genotypes[:]  # Returns the geographic list + Data 
    
        
    def draw_corr_genotypes_replicates(self, coords, nr_genotypes, l=25, a=0.1, replicate_nr=100):
        '''Draws a number of replicates of correlated_genotypes'''
        genotypes = []
        for i in range(replicate_nr):  # Do independent draws
            genotype, coords = self.draw_correlated_genotypes(self, coords, nr_genotypes, l, a)
            genotypes.append(genotype)
        return coords, np.array(genotypes)
    
    def draw_corr_genotypes_kernel(self, kernel_mat, p_mean=0.5):
        '''Draw correlated genotypes given a Kernel Matrix and coords, as well as mean allele frequency and standard deviation.
        Give back Genotype Matrix.'''
        assert(len(np.shape(kernel_mat)) == 2)  # Assert kernel mat is really a matrix
        assert(np.shape(kernel_mat)[0] == np.shape(kernel_mat)[1])  # Check whether Kernel Matrix is actually quadratic.
        
        f_mean = 2.0 * np.arcsin(np.sqrt(p_mean))  # Do the Arc Sin Transformation (Reverse of the Link Function)
        mean_p = np.array([f_mean for _ in range(len(kernel_mat))])  # Calculate the mean allele frequency
        
        data = np.random.multivariate_normal(mean_p, kernel_mat)  # Do the random draws of the deviations from the mean
        data = np.transpose(data)  # Transpose, so that individual x locus matrix
        
        p = arc_sin_lin(data)  # Do an arc-sin transform
        genotypes = np.random.binomial(1, p)  # Draw genotypes
        return genotypes  # Returns the geographic list + Data

 ###################################################################################################################   
    
class Secondary_Grid(Grid):
    # Grid that can simulate secondary contact
    test = 100
    gridsize_x = 1000
    gridsize_y = 1000
    ind_list = []  # List of lists to which the initial genotypes correspond
    update_list = []  # List of individuals do update. Contains x and y positions
    position_list = []  # List of initial positions.
    ancestry = []  # List of individuals that currently descent from individual i
    final_ancestry_left = []  # List of final ancestry from left-hand side
    final_ancestry_right = []  # List of final ancestry from right-hand side
    genotypes = []  # List of the genotypes of every individuals
    genotype_mat = []  # List of the genotype matrix. In case multiple multiple markers are simulated
    t = 0  # Current time back in generations.
    barrier = 500.5
    barrier_strength = 1  # The strength of the barrier # 1 Everything migrates; 0 Nothing Migrates
    sigma = 0.965  # 0.965  # 1.98 # 0.965
    ips = 10  # Number of haploid Individuals per Node (For D_e divide by 2)
    mu = 0.003  # The Mutation/Long Distance Migration rate.
    
    def __init__(self):  # Initializes an empty grid
        print("Initializing Grid for secondary contact...")  # Actually all relevant things are set with set_samples
        
        
    def update_grid_t(self, t, coalesce=1, barrier=0, p1=0.5, p2=0.5):
        '''Updates the grid for t generations'''
        for i in range(t):
            t += 1  # Adds time to the generation clock!
            self.drop_mutations()  # Drops mutations that moves lineages into their pension
            
            if i % 100 == 0:
                print("Doing generation: %i " % i)
            
            if barrier == 0:
                self.update_grid()  # Updates the update list
            
            elif barrier == 1:  # In case of barrier
                self.update_grid_barrier()
                
            else:
                raise Exception("Only 1/0 Input allowed!")
            
            if coalesce == 1:
                self.coal_inds()  # Coalesces individuals in the same position; and merges the off-spring indices
                
        self.assign_ancestry()  # Assigns ancestry to left and right; gets back Lineages from pension
        # And assigns them randomly
        
        self.draw_genotypes(p1, p2)  # Draw genotypes
    
    # Some output for debugging:
    # print("Number of individuals hit by mutation: %i" % np.sum([len(i) for i in self.final_ancestry]))
    # print("Total length of ancestry: %i" % np.sum([len(i) for i in self.ancestry]))
        print("Run complete\n ") 
    
    def assign_ancestry(self):
        '''Assign Ancestry to left or to the right of barrier; i.e. 
        secondary contact zone. Use axis one.'''
        
        # Assign ancestry to left if x-Axis position smaller than where the barrier is
        self.ancestry_left = [self.ancestry[i] for i, pos in enumerate(self.update_list) if pos[0] < self.barrier]
        
        # Assign it to the right in if x-Axis position bigger than where the barrier is
        self.ancestry_right = [self.ancestry[i] for i, pos in enumerate(self.update_list) if pos[0] >= self.barrier]
        
        assert(len(self.ancestry_left) + len(self.ancestry_right) == len(self.ancestry))  # Assert that everything worked
        
        # Randomly split up final_ancestry:
        n = len(self.final_ancestry)
        left = np.random.random(n)
        self.final_ancestry_left = [self.final_ancestry[i] for i in xrange(n) if left[i] < 0.5]
        self.final_ancestry_right = [self.final_ancestry[i] for i in xrange(n) if left[i] >= 0.5]
        
        assert(len(self.final_ancestry_left) + len(self.final_ancestry_right) == len(self.final_ancestry))  # Check whether the splitting worked
        
        self.ancestry_left = self.ancestry_left + self.final_ancestry_left  # Gets non-active lineages back from Pension
        self.ancestry_right = self.ancestry_right + self.final_ancestry_right  # Gets non-active lineages back from Pension
        
        self.ancestry = self.ancestry_left + self.ancestry_right  # For calculation of full paiwrise F-Statistics
        
        return self.ancestry_left, self.ancestry_right
        
        
    def draw_genotypes(self, p1=0.5, p2=0.5):
        '''Method that draws genotypes from pool of allele freqs.
        p is the frequency of the allele from which one has to draw.
        p1: left. p2: right
        Sets genotype i to a certain allele.'''

        
        for lists in self.ancestry_left:
            all_freq = np.random.random() < p1  # Draws the allele Frequency
            for i in lists:
                self.genotypes[i] = all_freq  # Sets the Genotype to that allele Frequency              
        for i in lists:
                self.genotypes[i] = all_freq  # Sets the Genotype to that allele Frequency     
        
        for lists in self.ancestry_right:
            all_freq = np.random.random() < p2
            for i in lists:
                self.genotypes[i] = all_freq  # Sets the Genotype to that allele Frequency  
                
        assert(np.sum(self.genotypes < 0) == 0)  # Check whether all Genotypes are set   
                
        print(self.genotypes)
            

#############################################################################################
class Coalescence_Grid(Grid):
    # Grid that simulates coalescence of two sample
    gridsize_x = 20
    gridsize_y = 20
    position_list = []  # List of initial positions.
    start_pos1 = [5, 5]  # Position of first Individual
    start_pos2 = [14, 5]  # Position of second Individual
    multi_processing = False
    
    offset_drawer = 0  # An object that draws the Laplace offsets.
    
    t = 0  # Current time back in generations.
    t_max = 30000  # How long to run the simulations for 
    
    barrier = 9.5
    barrier_strength = 1.0  # The strength of the barrier # 1 Everything migrates; 0 Nothing Migrates
    sigma = 0.965  # 0.965  # 1.98 # 0.965
    ips = 10  # Number of haploid Individuals per Node (For D_e divide by 2)
    mu = 0.003  # The Mutation/Long Distance Migration rate.
    
    def __init__(self, start_pos1=[], start_pos2=[], drawer=0):  # Initializes an empty grid
        '''Allows to set start positions.'''
        if len(start_pos1) == 2:
            self.start_pos1 = start_pos1
            
        if len(start_pos2) == 2:
            self.start_pos2 = start_pos2
            
        if drawer == 0:
            print("Initializing Drawer...")
            self.offset_drawer = Laplace_Offsets(1000, sigma=self.sigma)
            
        else:
            self.offset_drawer = drawer
        print("Grid Successfully initiliazed")  
            
    def reset_positions(self):
        '''Resets the Grid so that Positions agree with Start Positions'''
        self.pos1 = self.start_pos1[:]
        self.pos2 = self.start_pos2[:]
        
    def run_until_coalescence(self, warning=False):
        '''Run the samples until coalescence. Return coalescence time'''
        t = 0  # When to start the coalescence simulations
        self.reset_positions()
        
        while t < self.t_max:
            t += 1
            # Update the two individuals positions
            if (self.pos1 == self.pos2) and (np.random.random() < 1.0 / (self.ips)):  # Only in case if individuals fall an the same ancestor
                return t
            
            # update the individual positions:
            self.pos1 = self.update_individual_pos_barrier(self.pos1[0], self.pos1[1])
            self.pos2 = self.update_individual_pos_barrier(self.pos2[0], self.pos2[1])
        
        if warning == True:
            print("Warning: Maximum time hit")
        t = np.nan  # So that no counted in statistics
        return t  # In case
    
    def return_coalescence_times(self, n=1, report_int=1, warning=False):
        '''Run n coalescence simulations. Return array of n coalescence times.'''
        coal_times = np.zeros(n)
        
        # Does not work actually... Cannot run class functions from map
        if self.multi_processing == True:
            warnings = [warning for _ in xrange(n)] # Just a vector of dummy variables
            pool_size = mp.cpu_count() * 2
            pool = mp.Pool(processes=pool_size)
            coal_times = pool.map(self.run_until_coalescence, warnings)  # map
            pool.close()
            pool.join()
        
        # If not Multiprocessing:
        else:
            for i in xrange(n):
                if i % report_int == 0:
                    print("Doing run: %i" % i)
                coal_times[i] = self.run_until_coalescence()
        return coal_times
    
    def draw_delta(self):
        '''Overwrites the drawing of the Offsets to make it quicker here.'''
        offset = self.offset_drawer.give_offset()   # Gives the actual Offset
        return offset
    
        
class Laplace_Offsets(object):
    '''This class draws the Laplace offsets. 
    It makes a list, and then draws from it.'''
    offsets = [] # Array of Offsets
    n = 1          # Size of the List
    sigma = 0.965  # The STD of Laplacian
    i=0            # Counter where one stands
    
    def __init__(self, list_size, sigma=0.965):
        '''Draws the Offsets. The idea is to draw them together'''
        self.n = list_size
        self.sigma = sigma
        self.draw_offsets() # Draw all offsets.
        self.i=0 # Set the counter to 0.
        print("Drawer initialized!")
    
    def draw_offsets(self):
        '''Draw all n offsets'''
        scale = self.sigma / np.sqrt(2)  # To scale it right for Laplace-Dispersal.
        offsets =  np.around(np.random.laplace(scale=scale, size=self.n))  # Draws all the Offsets
        self.offsets = offsets
    
    def give_offset(self):
        '''Give the Offset. Redraw if necessary'''
        # Redraw if needed:
        if self.i>=self.n:
            self.draw_offsets()
            self.i = 0
        
        # Give the offset
        offset = self.offsets[self.i]
        self.i = self.i + 1
        return offset
        
def arc_sin_lin(x):
    '''Arcus-Sinus Link function'''
    x = np.where(x < 0, 0, x)  # If x smaller 0 make it 0
    x = np.where(x > np.pi, np.pi, x)  # If x bigger Pi keep it Pi
    y = np.sin(x / 2.0) ** 2
    return y   
               
def list_duplicates(seq):
    '''Returns list of indices of all duplicate entries.
    Only the first entry is '''
    seen, result = [], []
    for idx, item in enumerate(seq):
        if item not in seen:
            seen.append(item)  # First time seeing the element
        else:
            idx1 = seq.index(item)
            result.append([idx1, idx])  # Already seen, add the index to the result
    return result

def full_kernel_function(coords, l, a, c):
    '''Return barrier Kernel - describing reduced correlation across barrier
    and increased correlation next to barrier. Coords is nx2 Numpy array'''
    x = coords[:, 0]  # Extracts x-coords
    coords_refl = np.copy(coords)
    # print(coords[:10])
    coords_refl[:, 0] = -coords_refl[:, 0]  # Reflects the samples
    
    g = np.sign(x)  # Calculates Signum of x
    same_side = (g[:, None] * g + 1) / 2  # Whether the x-Values are on the same side
    
    r = np.linalg.norm(coords[:, None] - coords, axis=2)  # Calculates pairwise Distance
    r_refl = np.linalg.norm(coords_refl[:, None] - coords, axis=2)  # Calculates the reflected Distance 
    
    # Calculate the normal Kernel:
    cov_mat = a * np.exp((-r ** 2) / (2. * l ** 2))  # Calculate the co-variance matrix. Added diagonal term
    cov_mat_refl = a * np.exp((-r_refl ** 2) / (2. * l ** 2))  # Calculate the covariance matrix for reflected coordinates.
    
    cov_tot = same_side * (cov_mat + c * cov_mat_refl) + (1 - same_side) * (1 - c) * cov_mat + 0.000001 * np.identity(len(coords))
    return cov_tot

def torus_distance(pos1, pos2, torus_size_x, torus_size_y):
    '''Calculates Torus Distance.
    Takes 2d Position lists/arrays as Input'''
    x0, y0 = pos1
    x1, y1 = pos2
    # Calculates the Euclidean distance on a Torus
    dist_x = abs(x0 - x1)
    dist_y = abs(y0 - y1)
    distance = np.sqrt(min(dist_x, torus_size_x - dist_x) ** 2 + min(dist_y, torus_size_y - dist_y) ** 2)
    return(distance)

def bessel0(x, C, a):
    '''Bessel Decay for IBD. Used for fitting.'''
    return (C * kv(0, a * x)) 

############################################################################################################

def test_fit_f():
    '''Function to fit F'''
    position_list = [(i, j) for i in range(502, 600, 4) for j in range(502, 600, 4)]  # Position_List describing individual positions
    # position_list = [(i, j) for i in range(2, 100, 4) for j in range(2, 100, 4)]  # Position_List describing individual positions
    
    grid = Grid()
    grid.set_samples(position_list)  # Sets the samples
    grid.update_grid_t(5000)  # Updates for t Generations
    grid.fit_F(show=True)
    
def test_secondary_contact():
    '''Test secondary contact scenario.'''
    position_list = [(i, j) for i in range(502, 600, 4) for j in range(502, 600, 4)]  # Position_List describing individual positions
    grid = Secondary_Grid()
    grid.barrier = 550.5
    grid.set_samples(position_list)  # Sets the samples
    grid.update_grid_t(10, p1=0.0, p2=1.0)  # Updates for t Generations
    grid.fit_F(show=True)
    np.savetxt("./coordinates_snd.csv", position_list, delimiter="$")  # Save the coordinates
    genotypes = np.reshape(grid.genotypes, (len(grid.genotypes)), 1)
    np.savetxt("./data_genotypes_snd.csv", genotypes, delimiter="$")  # Save the data 
    print("Run Complete; data saved.")
        
    
def test_coalescence_sim():
    '''Tests whether the coalescene framework actually works'''
    
    grid = Coalescence_Grid()  # Calls the coalescence Grid
    start = time()
    # t = grid.run_until_coalescence()
    t_coals = grid.return_coalescence_times(100)
    end = time()
    print("Runtime %.4f: " % (end - start))
    print(t_coals)
    

if __name__ == "__main__":
    # test_fit_f()
    # test_secondary_contact()
    test_coalescence_sim() 
    # grid.extract_F(20, show=True)

# test_fit_f()   # Runs the actual testing Function



