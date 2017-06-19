'''
Created on June 19th, 2017
This is a function used to simulate Data on both sides of a barrier;
where the allele Frequencies are panmictic under the model
@author: hringbauer
'''
import numpy as np
import matplotlib.pyplot as plt

def simulate_panmictic_barrier(save=False, plot=False):
    '''Method to simulate Data for both sides of a barrier. Both sides are panmictic.'''
    position_list = np.array([(500 + i, 500 + j) 
                              for i in range(-29, 31, 1) for j in range(-19, 21, 1)])  # 1000 Individuals.
    nr_loci = 200
    std_pl = 0.1  # Standard Deviation of Loci on the left side
    std_pr = 0.1  # Standard Deviation of Loci on the right side
    p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
    barrier_pos = 500.5  # Where to set the Barrier
            
    # Define left and right allele Frequencies
    p_delta_l = np.random.normal(scale=std_pl, size=nr_loci)  # Draw some random Delta p from a normal distribution
    p_delta_r = np.random.normal(scale=std_pr, size=nr_loci)  # Draw some random Delta p from a normal distribution
    p_mean_l = p_mean + p_delta_l
    p_mean_r = p_mean + p_delta_r
    
    
    inds_left = np.where(position_list[:, 0] < barrier_pos)[0]
    inds_right = np.where(position_list[:, 0] > barrier_pos)[0]
    assert(len(inds_left) + len(inds_right) == len(position_list))
    
    print("Nr. of Individuals left: %i" % len(inds_left))
    print("Nr. of Individuals right: %i" % len(inds_right))
    
    p = -np.ones((len(position_list), nr_loci))
    p[inds_left, :] = p_mean_l[None, :]  # Set allele Frequencies left
    p[inds_right, :] = p_mean_r[None, :]  # Set allele Frequencies right
    
    
    genotypes = np.random.binomial(2, p)  # Draw the genotypes
    
    if plot == True:
        plt.figure()
        plt.scatter(position_list[:, 0], position_list[:, 1], c=genotypes[:, -1])
        plt.colorbar()
        plt.show()
    
    # Save the Data
    if save == True:
        np.savetxt("./coords_2demes.csv", position_list.astype("int"), delimiter="$", fmt='%i')  # Save the coordinates
        np.savetxt("./genotypes_2demes.csv", genotypes.astype("int"), delimiter="$", fmt='%i')  # Save the data 
        
    print("\nSimulated Genotypes:\n Nr Inds: % i \n Nr. Loci: % i" % np.shape(genotypes))
    return position_list, genotypes
    
def transform_to_geneland(genotype_path, save_path="geneland_deme2.csv", plot=True):
    '''Transform Genotype File to Geneland Format'''
    genotype_matrix = np.loadtxt(genotype_path, delimiter='$')
    
    # Save the Genotype Files:
    file = open(save_path, "w")
    for individual in genotype_matrix:
        line="" # The String that will be written in this line.
        for l in individual:
            if l == 0:
                line += "0 0 "
                
            elif l == 1:
                line += "1 0 "
                
            elif l == 2:
                line += "1 1 "
                
            else:
                raise RuntimeError("Invalid Genotype")
            
        file.write(line+"\n") # Write the full line
    file.close() 
    print("File Successfully Written")
    
    # Sanity Check:
    saved_data=np.loadtxt(save_path) # Delimiter is white space by default
    print("\nShape of old Data:\n Nr Inds: % i \n Nr. Loci: % i" % np.shape(genotype_matrix))
    print("\nShape of saved Data:\n Nr Inds: % i \n Nr. Loci: % i" % np.shape(saved_data))
    
    

    
# Run the code
# simulate_panmictic_barrier(save=True, plot=True)
transform_to_geneland("./genotypes_2demes.csv")








