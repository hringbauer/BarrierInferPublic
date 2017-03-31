'''
Created on 17.10.2014
This is the main file.
@author: Harald Ringbauer
'''
from grid import Grid
from analysis import Analysis
from forward_sim import Forward_sim
from GPR_kernelfit import GPR_kernelfit
from tf_analysis import TF_Analysis
import numpy as np
import cPickle as pickle  # @UnusedImport
import matplotlib.pyplot as plt


def main():
    '''Main loop of the program. Here we can control everything.'''
    grid = Grid()
    # position_list = [(i, j) for i in range(502, 600, 4) for j in range(502, 600, 4)]  # Position_List describing individual positions
    position_list = np.array([(500 + i, 500 + j) for i in range(-19, 21, 2) for j in range(-49, 51, 2)])
    print(position_list)
    barrier_position = 500.5  # Where is the Barrier
    # position_list = np.array([(500 + i, 500 + j) for i in range(-19, 20, 1) for j in range(-25, 25, 1)])
    genotype_matrix = []  # Matrix of multiple genotypes  

    print("Welcome back!")
    
    while True:
        print("\nWhat do you want to do?")
        inp = int(input("\n (2) Do Tensor-flow Analysis \n (3) Simulate multiple repl. of correlated allele frequencies \n (4) Simulate correlated allele frequencies"
                        "\n (5) Run grid simulations; multiple Genotypes \n (6) Run grid simulations; multiple Genotypes with barrier"
                        "\n (7) Analyze Samples \n (8) Do forward simulations \n (9) Load / Save \n (10) Exit Program\n "))   
        
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
        
        if inp == 3:
            nr_genotypes = int(input("How many genotypes?\n "))  # Nr of genotypes
            l = float(input("What length scale? \n"))
            a = float(input("What absolute correlation?\n"))
            c = float(input("What reflection factor? (c) \n"))
            p_mean = float(input("What mean allele frequency? (p) \n"))
            
            x_data_list = []
            y_data_list = []
            
            print("Doing: c=0.0")
            for i in range(20):
                position_list, genotype_matrix = grid.draw_correlated_genotypes(nr_genotypes, l, a, 0.0, p_mean)  # No Barrier
                x_data_list.append(position_list)
                y_data_list.append(genotype_matrix)
                
            print("Doing: c=0.3")
            for i in range(20):
                position_list, genotype_matrix = grid.draw_correlated_genotypes(nr_genotypes, l, a, 0.3, p_mean)  # A bit barrier
                x_data_list.append(position_list)
                y_data_list.append(genotype_matrix)
                
            print("Doing: c=0.6")
            for i in range(20):
                position_list, genotype_matrix = grid.draw_correlated_genotypes(nr_genotypes, l, a, 0.6, p_mean)
                x_data_list.append(position_list)
                y_data_list.append(genotype_matrix)
             
            print("Doing: c=0.95")   
            for i in range(20):
                position_list, genotype_matrix = grid.draw_correlated_genotypes(nr_genotypes, l, a, 0.95, p_mean)
                x_data_list.append(position_list)
                y_data_list.append(genotype_matrix)
            
            save_string = "data_lists.p"
            print("Saving")
            pickle.dump((x_data_list, y_data_list), open(save_string, "wb"))  # Pickle the data    
            print("Saving Complete")
            
        if inp == 4:
            nr_genotypes = int(input("How many genotypes? \n"))  # Nr of genotypes
            l = float(input("What length scale? \n"))
            a = float(input("What absolute correlation? \n"))
            c = float(input("What reflection factor? (c) \n"))
            p_mean = float(input("What mean allele frequency? (p) \n"))
            f_m = int(input("Do you want to have fluctuating means? \n(1) Yes \n(0) No\n"))
            position_list, genotype_matrix = grid.draw_correlated_genotypes(nr_genotypes, l, a, c, p_mean, show=True, fluc_mean=f_m)
            
        if inp == 5:
            nr_loci = int(input("\nFor how many loci?\n")) 
            t = int(input("\nFor how long?\n"))
            p_m = float(input("Mean allele frequency?\n"))
            f_m = int(input("Do you want to have fluctuating means? \n(1) Yes \n(0) No\n"))
            
            p_mean = np.ones(nr_loci) * p_m  # Sets the mean allele Frequency
            
            if f_m == 1:
                if f_m == True:
                    v = float(input("What should the standard deviation around the mean p be?\n"))
                    p_delta = np.random.normal(scale=v, size=nr_loci)  # Draw some random Delta F from a normal distribution
                    # p_delta = np.random.laplace(scale=v / np.sqrt(2.0), size=nr_genotypes)  # Draw some random Delta f from a Laplace distribution 
                    # p_delta = np.random.uniform(low=-v * np.sqrt(3), high=v * np.sqrt(3), size=nr_genotypes)  # Draw from Uniform Distribution
                    # p_delta = np.random.uniform(0, high=v * np.sqrt(3), size=nr_genotypes)  # Draw from one-sided uniform Distribution!
                    print("Observed Standard Deviation: %.4f" % np.std(p_delta))
                    print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
                    p_mean = p_mean + p_delta
            
            print("Mean Allele Frequencies:")
            print(p_mean)
            genotype_matrix = np.zeros((len(position_list), nr_loci))
            
            for i in range(nr_loci):
                print("Doing run %i: " % i)
                grid.set_samples(position_list)
                grid.update_grid_t(t, p=p_mean[i])  # Uses p_mean[i] as mean allele Frequency.
                genotype_matrix[:, i] = grid.genotypes
        
        if inp == 6:
            nr_loci = int(input("\nFor how many loci?\n")) 
            t = int(input("\nFor how long?\n"))
            p_m = float(input("Mean allele frequency?\n"))
            c = float(input("Strength of the barrier?\n"))
            f_m = int(input("Do you want to have fluctuating means? \n(1) Yes \n(0) No\n"))
            
            p_mean = np.ones(nr_loci) * p_m  # Sets the mean allele Frequency
            
            if f_m == 1:
                if f_m == True:
                    v = float(input("What should the standard deviation around the mean p be?\n"))
                    p_delta = np.random.normal(scale=v, size=nr_loci)  # Draw some random Delta F from a normal distribution
                    # p_delta = np.random.laplace(scale=v / np.sqrt(2.0), size=nr_genotypes)  # Draw some random Delta f from a Laplace distribution 
                    # p_delta = np.random.uniform(low=-v * np.sqrt(3), high=v * np.sqrt(3), size=nr_genotypes)  # Draw from Uniform Distribution
                    # p_delta = np.random.uniform(0, high=v * np.sqrt(3), size=nr_genotypes)  # Draw from one-sided uniform Distribution!
                    print("Observed Standard Deviation: %.4f" % np.std(p_delta))
                    print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
                    p_mean = p_mean + p_delta
            
            print("Mean Allele Frequencies:")
            print(p_mean)
            genotype_matrix = np.zeros((len(position_list), nr_loci))
            
            for i in range(nr_loci):
                print("Doing run %i: " % i)
                grid.set_barrier_parameters(barrier_position, c)  # Where to set the Barrier and its strength
                print(position_list)
                grid.set_samples(position_list)
                grid.update_grid_t(t, p=p_mean[i], barrier=1)  # Uses p_mean[i] as mean allele Frequency.
                genotype_matrix[:, i] = grid.genotypes
                
            # # Modify position List so that barrier at 0 (as in the model):
            position_list = position_list.astype("float")  # Make it float so that subtraction with float barrier works
            position_list[:, 0] = position_list[:, 0]  # Only works once!!
                
        if inp == 7:
            analysis = Analysis(position_list, genotype_matrix)
            while True:
                inp1 = int(input("What analysis?\n (1) Correlation Analysis\n (2) Group Individuals\n (3) Geographic Comparison "
                                 "\n (4) Extract Data\n (5) Correlation Analysis where mean is also estimated\n "
                                 "(6) Gaussian Process analysis\n (7) Plot Positions \n (8) Plot Mean allele freq. Distribution "
                                 "\n (9) Back to main menu\n "))
                
                if inp1 == 1:
                    nr_inds = int(input("How many random individuals?\n"))
                    analysis.ind_correlation(nr_inds=nr_inds)
                    
                if inp1 == 2:
                    x_demes = int(input("How many Demes along x-axis?\n"))
                    y_demes = int(input("How many Demes along y-axis?\n"))
                    analysis.group_inds(analysis.position_list, analysis.genotypes, x_demes, y_demes)
                    
                if inp1 == 3:
                    barrier_pos = float(input("Where is the barrier?\n"))
                    analysis.geo_comparison(barrier=barrier_pos)
                    
                if inp1 == 4:
                    np.savetxt("coordinates00.csv", analysis.position_list, delimiter="$")  # Save the coordinates
                    np.savetxt("data_genotypes00.csv", analysis.genotypes, delimiter="$")  # Save the data 
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
                    # Plot the data
                    row = int(input("What genotype raw? \n"))
                    analysis.plot_positions(row)

                if inp1 == 8:
                    analysis.plot_all_freqs()
  
                if inp1 == 9:
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
            inp9 = int(input("(1) Save data \n(2) Load data \n"))
            
            if inp9 == 1:
                np.savetxt("./Data/coordinates01b.csv", position_list, delimiter="$")  # Save the coordinates
                np.savetxt("./Data/data_genotypes01b.csv", genotype_matrix, delimiter="$")  # Save the data 
                print("Saving Complete.")
                
            elif inp9 == 2:
                # position_list = np.loadtxt('./Data/coordinates00b.csv', delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
                # genotype_matrix = np.loadtxt('./Data/data_genotypes00b.csv', delimiter='$').astype('float64')     
                # Some commonly used file paths: nbh_file_coords30.csv, ./Data/coordinates00b.csv
                # ./nbh_folder/nbh_file_coords200.csv    ./nbh_folder/nbh_file_genotypes200.csv
                
                position_list = np.loadtxt('./Data/coordinates01b.csv', delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
                genotype_matrix = np.loadtxt('./Data/data_genotypes01b.csv', delimiter='$').astype('float64')
                print("Loading Complete.")   
                print("Nr. of Samples:\t\t %i" % np.shape(genotype_matrix)[0])
                print("Nr. of Genotypes:\t %i" % np.shape(genotype_matrix)[1])   
            
        if inp == 10:
            print("Future Harald: Keep your shit together - I believe in you. Do not think too much - that just hurts") 
            break
        
        
if __name__ == '__main__':
    main()
    
