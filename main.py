'''
Created on 17.10.2014
This is the main file.
TEST
TEST1
@author: Harald Ringbauer
'''
from grid import Grid
from analysis import Analysis
from forward_sim import Forward_sim
from GPR_kernelfit import GPR_kernelfit
from tf_analysis import TF_Analysis
import numpy as np


def main():
    '''Main loop of the program. Here we can control everything.'''
    grid = Grid()
    position_list = [(i, j) for i in range(0, 101, 5) for j in range(0, 101, 5)]  # Position_List describing individual positions
    genotype_matrix = []  # Matrix of multiple geno-types  
    print(position_list)

    print("Welcome back!")
    
    while True:
        print("\nWhat do you want to do?")
        inp = int(input("\n (2) Do Tensorflow Analysis \n (3) Simulate Bunch of correlated allele frequencies \n (4) Simulate correlated allele frequencies"
                        "\n (5) Run analysis for multiple Genotypes \n (6) Run analysis for multiple Genotypes with barrier"
                        "\n (7) Analyze Samples \n (8) Do forward simulations \n (9) Load Samples \n (10) Exit Program\n "))   
        
        if inp == 2:
            tf_analysis = TF_Analysis()
            print("\nWhat do you want to do?")
            inp3 = int(input("\n (1) Do Test Run \n (2) Do stuff \n (3) Generate Likelihood-Surface \n (4) Exit \n"))  
            
            if inp3 == 1:
                tf_analysis.calc_likelihood(params=[0.1, 25])  # Does a test run of the likelihood function
                
            if inp3 == 2:
                print("To implement")
                
            if inp3 == 3:
                tf_analysis.generate_likelihood_surface()
                
            if inp3 == 4:
                break
            
        if inp == 4:
            nr_genotypes = int(input("How many genotypes?\n "))  # Nr of genotypes
            position_list, genotype_matrix = grid.draw_correlated_genotypes(position_list, nr_genotypes)  # Simulate draws from)
            
        if inp == 5:
            nr_loci = int(input("\nFor how many loci?\n")) 
            t = int(input("\nFor how long?\n"))
            genotype_matrix = np.zeros((len(position_list), nr_loci))
            
            for i in range(nr_loci):
                print("Doing run %i: " % i)
                grid.set_samples(position_list)
                grid.update_grid_t(t)
                genotype_matrix[:, i] = grid.genotypes
        
        if inp == 6:
            nr_loci = int(input("\nFor how many loci?\n")) 
            t = int(input("\nFor how long?\n"))
            genotype_matrix = np.zeros((len(position_list), nr_loci))
            
            for i in range(nr_loci):
                print("Doing run %i: " % i)
                grid.set_samples(position_list)
                grid.update_grid_t(t, barrier=1)
                genotype_matrix[:, i] = grid.genotypes
                
        if inp == 7:
            analysis = Analysis(position_list, genotype_matrix)
            while True:
                inp1 = int(input("What analysis?\n (1) Correlation Analysis\n (2) Barrier Analysis\n (3) Geographic Comparison "
                                 "\n (4) Extract Data\n (5) Correlation Analysis where mean is also estimated\n "
                                 "(6) Gaussian Process analysis\n (7) Back to main menu\n "))
                
                if inp1 == 1:
                    analysis.ind_correlation()
                    
                if inp1 == 2:
                    print("To implement")
                    
                if inp1 == 3:
                    analysis.geo_comparison()
                    
                if inp1 == 4:
                    np.savetxt("coordinates2.csv", analysis.position_list, delimiter="$")  # Save the coordinates
                    np.savetxt("data_genotypes2.csv", analysis.genotypes, delimiter="$")  # Save the data 
                    print("Saving Complete...")
                    
                if inp1 == 5:
                    print(np.mean(genotype_matrix, axis=0))
                    analysis.ind_correlation(np.mean(genotype_matrix, axis=0))
                    
                if inp1 == 6:
                    print("Loading GPR-analysis")
                    gpr = GPR_kernelfit(analysis.position_list, analysis.genotypes[:, 0])
                    
                    
                    while True:
                        inp2 = int(input("What do you want to do?\n (1) Run Fit \n (2) Show Fit\n"
                                         " (3) Show Likelihood Surface\n (4) Back\n"))
                        
                        if inp2 == 1:
                            gpr.run_fit()
                        if inp2 == 2:
                            gpr.show_fit()
                        if inp2 == 3:
                            gpr.plot_loglike_sf()
                        if inp2 == 4:
                            break
                    
                if inp1 == 7:
                    break                                     
        
        if inp == 8:
            positions = np.loadtxt("coordinates.csv", delimiter="$")
            nr_loci = int(input("How many loci?\n"))
            p_s = np.ones((len(positions), nr_loci)) * 0.5  # How many loci to draw
            genotypes = np.random.binomial(2, p_s)  # Draw the initial genotypes
            
            f_sim = Forward_sim(positions, genotypes)
            print("Forward Simulator loaded. Nr. of positions: %i" % len(positions))
            
            while True:
                inp2 = int(input("What do you want to do?\n (1) Run forward simulations \n (2) Plot Genotypes \n (3) Save Data for Analysis \n"
                                 " (4) Go back\n"))
                if inp2 == 1:
                    t = int(input("How many generations?\n"))
                    print("Updating...")
                    f_sim.update_t_generations(t) 
                    
                elif inp2 == 2:
                    f_sim.plot_genotypes()
                    
                elif inp2 == 3:
                    position_list = f_sim.positions
                    genotype_matrix = f_sim.curr_genotypes / float(f_sim.ploidy)  # Assumes that there is no 2nd axis 
                elif inp2 == 4:
                    break
        if inp == 9:
            position_list = np.loadtxt('./coordinates2.csv', delimiter='$').astype('float64')
            genotype_matrix = np.loadtxt('./data_genotypes2.csv', delimiter='$').astype('float64')
            
            
        if inp == 10:
            print("Future Harald: Keep your shit together - I believe in you. Do not think too much - that just hurts") 
            break
        
        
if __name__ == '__main__':
    main()
    
