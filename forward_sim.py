'''
Created on Oct 17, 2016

@author: hringbauer
'''
import numpy as np
import matplotlib.pyplot as plt
from time import time

class Forward_sim(object):
    '''
    Class for simulating positions of genotypes forward in time.
    Needs position list
    '''
    genotypes = []  # List of all genotypes
    curr_genotypes = []  # Current genotype matrix: n individuals x k loci
    positions = []  # List of all positions
    disp_kernel = []
    dist_mat = []
    weights = []
    ploidy = 0  # How many alleles in a local individual

    def __init__(self, positions, genotypes, sigma=100, mode="laplace", diploid=True):
        '''
        Position List
        '''
        self.positions = positions  # Set all positions
        self.sigma = sigma  # Updates sigma
        if diploid == True:
            self.ploidy = 2  # Diploids
        else: self.ploidy = 1  # Monoploids
        
        self.dist_mat = self.calculate_distance_mat(positions)  # Calculate Matrix of pairwise Distances
        self.disp_kernel = self.calc_kernel(mode)  # Calculates the dispersal kernel  
        self.genotypes = genotypes
        self.curr_genotypes = self.genotypes
        self.plot_genotypes()
        
    def set_dispersal_kernel(self):
        '''Sets the forward dispersal kernel.'''
        self.dispersal_kernel = 1  # Sets the dispersal Kernel to a chosen one
    
    def update_t_generations(self, t):
        '''Updates for t generations.'''
        for i in range(t):
            print("Mean allele frequency: %.2f" % (np.mean(self.curr_genotypes) / self.ploidy))
            if i % 10 == 0:
                print("Doing Generation: %i" % i)
            self.update_single_gen()           
        
    def update_single_gen(self):
        '''Updates a single generation.'''
        temp_g = self.curr_genotypes  # List of all current genotypes
        weights = self.weights  # Loads weights for dispersal
        
        p_mean = np.dot(weights, temp_g)  # Calculate expected allele frequency
        new_genotypes = np.random.binomial(self.ploidy, p_mean / float(self.ploidy))  # Adjust for ploidy
        self.curr_genotypes = new_genotypes  # Updates Genotypes      
        
    def calculate_distance_mat(self, coords):
        '''Calculate the distance matrix between all coords. Requires Numpy array of 2d coordinates'''
        print("Calculating distance matrix")
        start = time()
        dist_mat = np.linalg.norm(coords[:, None] - coords, axis=2)
        print("Time taken %.2f" % (time() - start))
        return dist_mat
    
    def calc_kernel(self, mode="normal"):
        '''Gives the mode of the dispersal kernel.'''
        sigma = self.sigma
        dist_mat = np.array(self.dist_mat)
        
        if mode == "laplace":
            scale = self.sigma / np.sqrt(2)
            weights = np.exp(-abs(dist_mat)/scale)/(2.*scale)
            self.weights = weights / np.sum(weights, axis=1)[:, None]  # Normalize Weights
            
        elif mode == "normal":
            weights = 1 / (2.0 * np.pi * sigma ** 2) * np.exp(-dist_mat ** 2 / (2.0 * sigma ** 2))  # Calculate the Gaussian weights
            self.weights = weights / np.sum(weights, axis=1)[:, None]  # Normalize Weights
            
        else:
            print("Invalid Mode!!")
    
    def plot_genotypes(self,i=0):
        '''Plots the genotypes on a map.'''
        plt.figure()
        plt.scatter(self.positions[:, 0], self.positions[:, 1], c=self.curr_genotypes[:,i], s=40)
        plt.show()
        
        
        
