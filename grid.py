'''
Created on 27.01.2015
The Grid class
@author: hringbauer
'''

import numpy as np
import matplotlib.pyplot as plt


class Grid(object):
# Object for the Data-Grid. Contains matrix of lists for chromosomal pieces and methods to update it.    
    loci = 200
    test = 100
    gridsize_x = 105
    gridsize_y = 105
    ind_list = []  # List of lists to which the initial genotypes correspond
    update_list = []  # List of individuals do update. Contains x and y positions
    position_list = []  # List of initial positions.
    genotypes = []  # List of the genotypes of every individuals
    genotype_mat = []  # List of the genotype matrix. In case multiple multiple markers are simulated
    barrier = 50
    barrier_strength = 1  # The strength of the barrier
    sigma = 1.98
    
    def __init__(self):  # Initializes an empty grid
        print("Initializing...")  # Actually all relevant things are set with set_samples
      
    def set_samples(self, position_list):
        '''Sets samples to where they belong'''
        self.update_list = position_list  # Set the update List
        print("Ancestry List initialized: ")
        print(self.update_list)
        self.ancestry = [[i] for i in range(len(position_list))]  # Set the list of ancestors
        self.genotypes = [-1 for i in range(len(position_list))]  # -1 for not set
        self.position_list = position_list
        
    def update_grid_t(self, t, coalesce=1, barrier=0):
        '''Updates the grid for t generations'''
        for i in range(t):
            if i % 10 == 0:
                print("Doing generation: %i " % i)
            
            if barrier == 0:
                self.update_grid()  # Updates the update list
            
            elif barrier == 1:  # In case of barrier
                self.update_grid_barrier()
            
            if coalesce == 1:
                self.coal_inds()  # Coalesces individuals in the same position; and merges the off-spring indices
        self.draw_genotypes()  # Draw genotypes
        print("Run complete\n ")  
        
    def update_grid(self):
        '''Update the grid for a single generation'''
        new_coords = [self.update_individual_pos(pos[0], pos[1]) for pos in self.update_list]
        self.update_list = new_coords
        
    def update_grid_barrier(self):
        '''Update the grid in case of a barrier'''
        new_coords = [self.update_individual_pos_barrier(pos[0], pos[1]) for pos in self.update_list]
        self.update_list = new_coords
        

    def update_individual_pos(self, x, y):
        '''Method that updates the position of an individual'''
        scale = self.sigma / np.sqrt(2)  # For Laplace Dispersal
        x1 = (x + np.around(np.random.laplace(scale=scale))) % self.gridsize_x
        y1 = (y + np.around(np.random.laplace(scale=scale))) % self.gridsize_y
        return(x1, y1)
    
    def update_individual_pos_barrier(self, x, y):
        '''Method that updates individual positions with barrier'''
        scale = self.sigma / np.sqrt(2)
        delta_x = np.around(np.random.laplace(scale=scale))  # Draw the random off-set
        delta_y = np.around(np.random.laplace(scale=scale))  # Draw the ranodm off-set
        
        x1 = (x + delta_x)  
        y1 = (y + delta_y)
        
        if (x > self.barrier and x1 <= self.barrier):
            if np.random.random() < self.barrier_strength:  # In case of reflection
                # x1 = self.barrier + (self.barrier - x1)  # x1 gets reflected#
                x1 = self.barrier + 1 
            
        elif (x < self.barrier and x1 >= self.barrier):
            if np.random.random() < self.barrier_strength:  # In case of reflection           
                # x1 = self.barrier - (x1 - self.barrier)  # x1 gets reflected
                x1 = self.barrier - 1 
        
        x1 = x1 % self.gridsize_x
        y1 = y1 % self.gridsize_y    
        return(x1, y1)
    
    def coal_inds(self):
        '''Method to coalesce individuals'''
        inds = self.update_list
        
        dub_inds = list_duplicates(inds)
        del_list = []  # List of individuals to delete in the end
        
        for inds in dub_inds:
            '''Iterate over all individuals in dub_inds'''
            i, j = inds[0], inds[1]  # i < j. Always
            self.ancestry[i] = self.ancestry[i] + self.ancestry[j]  # Pool the ancestry list 
            del_list += [j]  # Update the list of which individuals to delete

        # Remove Individuals in Delete_List from update and ancestry List:    
        self.update_list = [i for j, i in enumerate(self.update_list) if j not in del_list]
        self.ancestry = [i for j, i in enumerate(self.ancestry) if j not in del_list]
    
    def draw_genotypes(self, p=0.5):
        '''Method that draws genotypes from pool of allele freqs.
        p is the frequency of the allele from which one has to draw.
        Sets genotype i to a certain allele.'''
        
        for lists in self.ancestry:
            all_freq = np.random.random() < p
            for i in lists:
                self.genotypes[i] = all_freq                  
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
        
    def simulate_correlated_data(self, position_list, cov_func):
        '''Given a position list, simulate correlated data.
        Simulate as draws from a random Gaussian.'''
        print("Todo")      

    def draw_correlated_genotypes(self):
        '''Draws correlated genotypes. l: Typical correlation length'''
        nr_genotypes = int(input("How many genotypes?\n "))  # Nr of genotypes
        l = float(input("What length scale? \n"))
        a = float(input("What absolute correlation?\n"))
        p_mean = float(input("What mean allele frequency? (p) \n"))
        f_mean = np.log(p_mean) - np.log(1 - p_mean)  # Do the logit transform
        
        coords = np.array([(i, j) for i in range(0, 201, 10) for j in range(0, 201, 10)])  # To have denser sampling: Originally 301
        
        r = np.linalg.norm(coords[:, None] - coords, axis=2)
        mean_p = np.array([f_mean for _ in range(len(coords))])  # Calculate the mean allele frequency
        # Add Identity matrix for numerical stability
        cov_mat = a * np.exp((-r ** 2) / (2. * l ** 2)) + 0.000001 * np.identity(len(mean_p))  # Calculate the covariance matrix. Added diagonal term
        # to make covariance matrix positive semidefinite.
        print(np.linalg.eig(cov_mat)[0])
        
        data = np.random.multivariate_normal(mean_p, cov_mat, nr_genotypes)  # Do the random draws
        data = np.transpose(data)  # Transpose, so that individual x locus matrix
        p = 1.0 / (1.0 + np.exp(-data))  # Create the mean from which to draw
        
        genotypes = np.random.binomial(1, p)  # Draw the genotypes
        
        
        plt.figure()
        plt.subplot(211)
        plt.scatter(coords[:, 0], coords[:, 1], c=data[:, 0], s=100)
        plt.colorbar()
        
        plt.subplot(212)
        plt.scatter(coords[:, 0], coords[:, 1], c=genotypes[:, 0], s=100)
        plt.colorbar()
        plt.show()
        return coords, genotypes[:]  # Returns the geographic list + Data 
    
    def draw_correlated_genotypes_var_p(self):
        '''Draws correlated genotypes with varying p.'''
        nr_genotypes = int(input("How many genotypes?\n "))  # Nr of genotypes
        l = float(input("What length scale? \n"))
        a = float(input("What absolute correlation?\n"))
        v = float(input("What should the standard deviation around the mean f be?\n"))
        
        f_mean = np.random.normal(scale=v, size=nr_genotypes)
        print("Observed Standard Deviation: %.4f" % np.std(f_mean))
        
        coords = np.array([(i, j) for i in range(0, 201, 10) for j in range(0, 201, 10)])  # To have denser sampling
        r = np.linalg.norm(coords[:, None] - coords, axis=2)
        
        cov_mat = a * np.exp((-r ** 2) / (2. * l ** 2)) + 0.000001 * np.identity(len(r[:, 0]))  # Calculate the covariance matrix.
        # Add Identity matrix for numerical stability
        # to make covariance matrix positive semidefinite.
        print(np.linalg.eig(cov_mat)[0])
        
        data = np.zeros((len(coords), nr_genotypes))   # Create individual x locus matrix
        
        for i in range(nr_genotypes):
            mean= np.array([f_mean[i] for _ in range(len(coords))])         # Create the mean    
            data[:,i] = np.random.multivariate_normal(mean, cov_mat)  # Do the random draws for the ith locus

        p = 1.0 / (1.0 + np.exp(-data))  # Create the mean from which to draw
        
        genotypes = np.random.binomial(1, p)  # Draw the genotypes
        
        plt.figure()
        plt.subplot(211)
        plt.scatter(coords[:, 0], coords[:, 1], c=data[:, 0], s=100)
        plt.colorbar()
        
        plt.subplot(212)
        plt.scatter(coords[:, 0], coords[:, 1], c=genotypes[:, 0], s=100)
        plt.colorbar()
        plt.show()
        np.savetxt("mean_f6.csv", f_mean, delimiter="$")  # Save the coordinates
        return coords, genotypes[:]  # Returns the geographic list + Data 
        
        
    def draw_corr_genotypes_replicates(self, coords, nr_genotypes, l=25, a=0.1, replicate_nr=100):
        '''Draws a number of replicates of correlated_genotypes'''
        genotypes = []
        for i in range(replicate_nr):  # Do independent draws
            genotype, coords = self.draw_correlated_genotypes(self, coords, nr_genotypes, l, a)
            genotypes.append(genotype)
        return coords, np.array(genotypes)
    
               
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

