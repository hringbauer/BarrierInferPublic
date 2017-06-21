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
from mle_pairwise import analyze_barrier  # Methods to fit individual data
from mle_pairwise import analyze_normal
from mle_class import calculate_ss
from random import shuffle 
import numpy as np
import pickle as pickle  # @UnusedImport
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
    inds_per_deme = []  #  Vector storing the Number of Individuals per Deme

    print("Welcome back!")
    
    while True:
        print("\nWhat do you want to do?")
        inp = int(input("\n (2) Do Tensor-flow Analysis \n (3) Simulate multiple repl. of correlated allele frequencies \n (4) Simulate correlated allele frequencies"
                        "\n (5) Run grid simulations; multiple Genotypes \n (6) Run grid simulations; multiple Genotypes with barrier"
                        "\n (7) Analyze Samples \n (8) Do forward simulations \n (9) Load / Save" 
                        "\n (10) Load HZ Data \n (11) Exit Program\n "))   
        
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
                inp1 = int(input("\nWhat analysis?\n (1) Correlation Analysis (Binned) \n (2) Fit Individual Correlation \n (3) Group Individuals"
                                 "\n (4) Geographic Comparison \n (5) Correlation Analysis where mean is also estimated\n "
                                 "(6) Flip Genotypes\n (7) Plot Positions \n (8) Plot Mean allele freq. Distribution "
                                 "\n (9) Back to main menu\n "))
                
                if inp1 == 1:
                    nr_inds = int(input("How many random individuals?\n"))
                    bins = int(input("How many bins?\n"))
                    analysis.ind_correlation(nr_inds=nr_inds, bins=bins)
                    
                    
                if inp1 == 2: 
                    nr_inds = int(input("How many random individuals?\n"))
                    fit_t0 = bool(input("Do you want to fit t0? (1=yes/0=no)\n"))
                    if fit_t0 == 1:
                        start_params = [150, 0.005, 1.0, 0.5]
                    
                    elif fit_t0 == 0:
                        start_params = [150, 0.005, 0.5]
                        
                    analyze_normal(position_list, genotype_matrix, nr_inds=nr_inds, fit_t0=fit_t0, start_params=start_params)
                    
                    # bootstrap=int(input("How often do you want to bootstrap? (0: Skip)"))
                    
                if inp1 == 3:
                    x_demes = int(input("How many Demes along x-axis?\n"))
                    y_demes = int(input("How many Demes along y-axis?\n"))
                    nr_inds = int(input("What is the minimum Nr of individuals?\n"))
                    position_list, genotype_matrix, inds_per_deme = analysis.group_inds(
                        analysis.position_list, analysis.genotypes, x_demes, y_demes, min_ind_nr=nr_inds)
                    
                if inp1 == 4:
                    barrier_pos = float(input("Where is the barrier?\n"))
                    analysis.geo_comparison(barrier=barrier_pos)
                    
                if inp1 == 5:
                    print(np.mean(genotype_matrix, axis=0))
                    analysis.ind_correlation(np.mean(genotype_matrix, axis=0))
                    
                if inp1 == 6:
                    genotype_matrix = analysis.flip_gtps(analysis.genotypes)
                
                if inp1 == 7:
                    # Plot the data
                    row = int(input("What genotype raw? \n"))
                    analysis.plot_positions(row)  # inds_per_deme

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
                np.savetxt("./Data/coordinatesHZall2.csv", position_list, delimiter="$")  # Save the coordinates
                np.savetxt("./Data/genotypesHZall2.csv", genotype_matrix, delimiter="$")  # Save the data 
                if(len(inds_per_deme > 0)):
                    np.savetxt("./Data/inds_per_deme_HZall2.csv", inds_per_deme, delimiter="$")
                print("Saving Complete.")
                
            elif inp9 == 2:
                # position_list = np.loadtxt('./Data/coordinates00b.csv', delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
                # genotype_matrix = np.loadtxt('./Data/data_genotypes00b.csv', delimiter='$').astype('float64')     
                # Some commonly used file paths: nbh_file_coords30.csv, ./Data/coordinates00b.csv
                # ./nbh_folder/nbh_file_coords200.csv    ./nbh_folder/nbh_file_genotypes200.csv
                # ./Data/coordinatesHZ.csv ./Data/genotypesHZ.csv
                # ./nbh_folder_gauss/nbh_file_coords30.csv   # ./nbh_folder_gauss/nbh_file_genotypes30.csv
                # './cluster_folder/barrier_file_coords01.csv'
                # ./cluster_folder/barrier_file_coords01.csv
                # ./hz_folder/hz_file_coords04.csv  ./hz_folder/hz_file_genotypes04.csv
                
                
                position_list = np.loadtxt('./Data/coordinates400i200l.csv', delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
                # position_list = position_list / 50.0  # Normalize; for position_list and genotype Matrix of HZ data!
                genotype_matrix = np.loadtxt('./Data/genotypes400i200l.csv', delimiter='$').astype('float64')
                #genotype_matrix = genotype_matrix / 2.0  # In case of Diploids
                
                
                # genotype_matrix = np.reshape(genotype_matrix, (len(genotype_matrix), 1))
                print("Loading Complete!")   
                print("Nr. of Samples:\t\t %i" % np.shape(genotype_matrix)[0])
                print("Nr. of Genotypes:\t %i" % np.shape(genotype_matrix)[1]) 
         
        elif inp == 10:
            while True:
                inp10 = int(input("\n(1) Load HZ Data \n(2) Save HZ Data \n(3) Clean HZ Data \n(4) Go back\n"))
            
                if inp10 == 1:
                    position_list = np.loadtxt('./Data/coordinatesHZALL.csv', delimiter='$').astype('float64')  # nbh_file_coords30.csv # ./Data/coordinates00.csv
                    scale_factor = float(input("Scale Factor?\n"))
                    position_list = position_list / scale_factor  # Normalize; for position_list and genotype Matrix of HZ data!
                    genotype_matrix = np.loadtxt('./Data/genotypesHZALL.csv', delimiter='$').astype('float64')
                    
                    print("Successfully Loaded!")
                    print("Nr. of Loci: %i" % np.shape(genotype_matrix)[1])
                    print("Nr. of Individuals: %i" % np.shape(genotype_matrix)[0])
                    
                elif inp10 == 2:
                    np.savetxt("./Data/coordinatesHZALL2.csv", position_list, delimiter="$")  # Save the coordinates
                    np.savetxt("./Data/genotypesHZALL2.csv", genotype_matrix, delimiter="$")  # Save the data 
                    print("Successfully Saved!")
                    
                elif inp10 == 3:
                    loci_path = "./Data/loci_infoALL.csv"
                    analysis = Analysis(position_list, genotype_matrix, loci_path=loci_path)
                    genotype_matrix, position_list = analysis.clean_hz_data(plot=True)
                
                elif inp10 == 4:
                    break         
            
        elif inp == 11:
            print("Future Harald: Keep your shit together - I believe in you. Do not think too much - that just hurts") 
            break
        
        
if __name__ == '__main__':
    main()
    
