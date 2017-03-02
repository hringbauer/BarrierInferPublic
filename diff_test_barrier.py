'''
Created on Oct 13, 2016
Class that contains methods to simulate and analyze random walk with a
barrier
@author: hringbauer
'''
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import binned_statistic
from scipy.stats import sem
# from kernels import fac_kernel  # Factory Method which yields Kernel Object
from random import shuffle
from scipy.optimize.minpack import curve_fit
from time import time
from scipy.special import erfc
from scipy.stats import norm

# Some Parameters for the test run:
reduced_migration =  0.1
initial_position = -10
nr_steps        = 1000


class diffusion_test(object):
    '''Has methods to simulate and analyze random walk with a barrier.
    Checks whether 1D formula is correct.'''
    initial_position = -10
    reduced_migration = 0.0  # Nagylaki Factor between 0 and 1. 1: Everything Migrates 0: Nothing
    nr_steps = 500  # Nr of Steps to take for a single realization
    
    def __init__(self, initial_position=-10, reduced_migration=1.0, nr_steps=1000):
        self.intial_position = initial_position
        self.reduced_migration = reduced_migration
        self.nr_steps = nr_steps
        print("I BECAME AWARE")
    
    def set_c(self, c):
        self.reduced_migration= c
        
    def single_run(self):
        '''Runs a single run of the Random Walker.
        Barrier sits between 0 and 1'''
        # Iterate over all random Numbers
        position = self.intial_position
        steps = np.sign(np.random.random(self.nr_steps) - 0.5)  # Which direction in case of no barrier
        
        for i in steps:
            if position == 0:
                a = np.random.random() - 0.5
                if a < 0:
                    position += -1  # Move Left
                elif a < self.reduced_migration / 2.0:  # Move right (through Barrier)
                    position += 1
                
            elif position == 1:
                a = np.random.random() - 0.5
                if a > 0:
                    position += 1  # Move Right
                    
                elif a > self.reduced_migration / -2.0:  # Move Left (through Barrier)
                    position += -1
                
            else:
                position += i  # Normal +/- 1 Update
        
        return position  # Return End Position
    
        
    def create_pdf(self, nr_replicates):
        '''Creates the Point Density Funcion'''
        print("Doing Runs...")
        end_points = [self.single_run() for _ in xrange(nr_replicates)]  # Creates Vector for Endpoints
        self.end_points = end_points
        print("Runs Finished.")
        return end_points
    
    def analyse_results(self, k=0.01):
        '''Method that analyses the results'''
        k = self.reduced_migration # This is an empirical Finding
        end_points = self.end_points
        plt.figure()
        bins = np.arange(-99, 101, 4) - 0.5  # Choose intervall so that 
        n, bins = np.histogram(end_points, normed=0, bins=bins)
        
        mean_dist = (bins[1:] + bins[:-1]) / 2.0  # Get the mean Distance
        
        norm_factor = (bins[1] - bins[0]) * len(end_points) # Factor so that histogram is normalized
        errors = np.sqrt(n) / float(norm_factor)  # Get the poisson error
        nn = n / float(norm_factor)  # Norm the end points
        
        # print(np.sum(n)*(bins[1]-bins[0])) # Sums up t0 1. Check
        x_vec = np.linspace(min(bins), max(bins), 1000)  # Creates the positions for x-Array
        y_vec_g = norm.pdf(x_vec, scale=1.0 * np.sqrt(self.nr_steps), loc=self.initial_position)
        y_vec_barrier = [barrier_function(self.initial_position, x, self.nr_steps/2.0, k) for x in x_vec]
        
        plt.axvline(0.5, color="r", linewidth=2, label="Barrier")
        plt.legend()
        plt.plot(x_vec, y_vec_g, label="Gaussian No Barrier")
        plt.plot(x_vec, y_vec_barrier, label="Barrier Fit")
        plt.errorbar(mean_dist, nn, yerr=errors, label="Observations", fmt="go")
        plt.xlabel("End Point")
        plt.ylabel("PDF")
        plt.legend()
        plt.show()
        


def barrier_function(y, x, t, k=0.5, D=1):
    '''Function for 1D Barrier PDF. y is starting point; x the end point'''
    if y<0:
        x,y = -x,-y  # Swap signs. Formula assumes that y<0!
    if x>0:
        # Same side
        n1 = np.exp(-(x - y) ** 2 / (4 * D * t)) + np.exp(-(x + y) ** 2 / (4 * D * t))
        d1 = np.sqrt(4 * np.pi * D * t)

        a2 = k / D * np.exp(2 * k / D * (y + x + 2 * k * t))
        b2 = erfc((y + x + 4 * k * t) / (2 * np.sqrt(D * t)))
        res = n1 / d1 - a2 * b2
        # if np.isnan(res) or np.isinf(res):  # Check if numerical instability
        #    return self.gaussian(t, y, x)  # Fall back to Gaussian (to which one converges)
        return res
    
    # Different Side
    if x<0:
        a1 = k / D * np.exp(2 * k / D * (y - x + 2 * k * t))
        b1 = erfc((y - x + 4 * k * t) / (2 * np.sqrt(D * t)))
        res = a1 * b1
        # if np.isnan(res) or np.isinf(res):  # Check if numerical instability
        #    return self.gaussian(t, y, x)  # Fall back to Gaussian (to which one converges)
        return res

def test_diffusion():
    '''Method to test the Diffusion'''
    diff_tester = diffusion_test(initial_position=initial_position, 
                                 reduced_migration=reduced_migration, nr_steps=nr_steps)
    diff_tester.create_pdf(20000)
    diff_tester.analyse_results()
    
def create_2x2_plot(c_values, initial_position=-15, 
                    nr_steps=1000, nr_replicates=2000):
    '''Create a plot of Diffusion PDF vrs. 
    fit for four c_values
    '''
    assert(len(c_values)==4) # So that 4 plots are done
    results=[] # Empty vector for results
    
    # Create the diff_tester Object
    diff_tester = diffusion_test(initial_position=initial_position, 
                        reduced_migration=reduced_migration, nr_steps=nr_steps)
    
    # Actually do the runs:
    for c in c_values:
        diff_tester.set_c(c)
        pdf=diff_tester.create_pdf(nr_replicates)
        results.append(pdf)
        
    # Do the actual plot:
    x_vec = np.linspace(-100, 100, 1000)  # Creates the positions for x-Array: -100 to 100
    f, axarr = plt.subplots(2, 2, sharex=True, sharey=True)  # Create sub-plots
    for i in range(4):  # Loop through interval list
        curr_plot = axarr[i / 2, i % 2]  # Set current plot
        #curr_plot.set_yscale('log')  # Set Log-Scale
        k = c_values[i]      # Load the right Barrier Strength
        end_points = results[i]  # Load the right PDF
        
        # Bin the data
        bins = np.arange(-101, 104, 6) - 0.5  # Choose intervall so that 
        n, bins = np.histogram(end_points, normed=0, bins=bins)
        mean_dist = (bins[1:] + bins[:-1]) / 2.0  # Get the mean Distance
        norm_factor = (bins[1] - bins[0]) * len(end_points) # Factor so that histogram is normalized
        errors = np.sqrt(n) / float(norm_factor)  # Get the poisson error
        nn = n / float(norm_factor)  # Norm the end points
        # print(np.sum(n)*(bins[1]-bins[0])) # Sums up t0 1. Check
        
        y_vec_g = norm.pdf(x_vec, scale=1.0 * np.sqrt(nr_steps), loc=initial_position)
        y_vec_barrier = [barrier_function(initial_position, x, nr_steps/2.0, k) for x in x_vec]
        curr_plot.axvline(0.5, color="r", linewidth=4, label="Barrier")
        
        
        l1,=curr_plot.plot(x_vec, y_vec_g)
        l2,=curr_plot.plot(x_vec, y_vec_barrier)
        l3=curr_plot.errorbar(mean_dist, nn, yerr=errors, fmt="go")
                
        curr_plot.text(-95, 0.02, "Barrier c: " + str(k), fontsize=14)
        curr_plot.tick_params(axis='x', labelsize=12)
        curr_plot.tick_params(axis='y', labelsize=12)
        
        # curr_plot.annotate("Blocks: %.0f" % self.total_bl_nr, xy=(0.4, 0.7), xycoords='axes fraction', fontsize=16)
    f.text(0.5, 0.02, 'End Point', ha='center', va='center', fontsize=16)
    f.text(0.025, 0.5, 'Probability', ha='center', va='center', rotation='vertical', fontsize=16)
    f.legend((l1, l2, l3), ('Gaussian No Barrier', 'Barrier Function', 'Random Walk'), 
             fontsize=12, loc=(0.78, 0.36))
    plt.tight_layout()
    plt.show() 

    
    

#test_diffusion()     # Single test
create_2x2_plot([0.001,0.01,0.1,0.5], nr_steps=1000, nr_replicates=500000)    # Creates 2x2 Plot

    




    







