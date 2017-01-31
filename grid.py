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

    def draw_correlated_genotypes(self, nr_genotypes, l, a, c, p_mean, show=False, fluc_mean=False):
        '''Draw correlated genotypes
        l: Typical correlation length, a: Absolute correlation, 
        c: Strength of the Barrier, fluc_mean: Whether varying mean all. frequency is simulated'''
        
        # Set Sample Coordinates:
        coords = np.array([(i, j) for i in range(-19, 20, 2) for j in range(-25, 25, 2)])  # To have dense sampling on both sides of the HZ

        f_mean = 2.0 * np.arcsin(np.sqrt(p_mean))  # Do the Arc Sin Transformation (Reverse of the Link Function)
        mean_p = np.array([f_mean for _ in range(len(coords))])  # Calculate the mean allele frequency
        f_delta = 0  # Initialize 0 deviations.
        
        if fluc_mean == True:
            v = float(input("What should the standard deviation around the mean f be?\n"))
            # f_delta = np.random.normal(scale=v, size=nr_genotypes)  # Draw some random Delta F from a normal distribution
            # f_delta = np.random.laplace(scale=v / np.sqrt(2.0), size=nr_genotypes)  # Draw some random Delta f from a Laplace distribution 
            f_delta = np.random.uniform(low=-v * np.sqrt(3), high=v * np.sqrt(3), size=nr_genotypes)  # Draw from Uniform Distribution
            # f_delta = np.random.uniform(0, high=v * np.sqrt(3), size=nr_genotypes)  # Draw from one-sided uniform Distribution!
            
            print("Observed Standard Deviation: %.4f" % np.std(f_delta))
            
        if show == True:
            print("Mean f: %.4f" % f_mean)
        
        
        r = np.linalg.norm(coords[:, None] - coords, axis=2)
        # Add Identity matrix for numerical stability
        
        cov_mat = full_kernel_function(coords, l, a, c) + 0.000001 * np.identity(len(coords))  # Calculate the covariance matrix. Added diagonal term for numerical stability
        
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

