'''
Created on June 19th, 2017
This is a function used to simulate Data on both sides of a barrier;
where the allele Frequencies are panmictic under the model
@author: hringbauer
'''
import numpy as np
import matplotlib.pyplot as plt


def simulate_panmictic_barrier(folder, save=False, plot=False):
    '''Method to simulate Data for both sides of a barrier. Both sides are panmictic.'''
    position_list = np.array([(500 + i, 500 + j) 
                              for i in range(-24, 26, 1) for j in range(-9, 11, 1)])  # Space 1: 2400 Individuals. -29,31 -19,21
    nr_loci = 200
    std_overall = 0.1  # Standard Deviation of Loci on the left side.
    std_pr = 0.02  # Standard Deviation of Loci on the right side - what is the deviation.
    p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
    barrier_pos = 500.5  # Where to set the Barrier
            
    # Define left and right allele Frequencies
    p_delta_overall = np.random.normal(scale=std_overall, size=nr_loci)  # Draw some random Delta p from a normal distribution
    p_delta_r = np.random.normal(scale=std_pr, size=nr_loci)  # Draw some random Delta p from a normal distribution
    p_mean_l = p_mean + p_delta_overall
    p_mean_r = p_mean + p_delta_overall + p_delta_r
    
    
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
    
    # Save the Data. Use Space as Delimiter!
    if save == True:
        output_coords_path = folder + "coords.csv"  # Output Name for Coordinates
        output_genotype_path = folder + "genotypes.csv"       # Output Name for Genotypes
        
        transform_to_geneland(genotypes, output_genotype_path) # Transform and save Genotypes
        np.savetxt(output_coords_path, position_list, fmt='%i')  # Saves Coordinates
        
    print("\nSimulated Genotypes:\n Nr Inds: % i \n Nr. Loci: % i" % np.shape(genotypes))
    return position_list, genotypes
    
def transform_to_geneland(genotype_matrix, save_path):
    '''Transform Genotype File to Geneland Format. Save it to Save Path
    Assumes Genotype Mat is in Format nxl; with genotype values ranging from 0 to 2'''
    
    # Save the Genotype Files:
    file = open(save_path, "w")
    for individual in genotype_matrix:
        line = ""  # The String that will be written in this line.
        for l in individual:
            if l == 0:
                line += "0 0 "
                
            elif l == 1:
                line += "1 0 "
                
            elif l == 2:
                line += "1 1 "
                
            else:
                raise RuntimeError("Invalid Genotype")
            
        file.write(line + "\n")  # Write the full line
    file.close() 
    print("File Successfully Written")
    
    # Sanity Check:
    saved_data = np.loadtxt(save_path)  # Delimiter is white space by default
    #print("\nShape of old Data:\n Nr Inds: % i \n Nr. Loci: % i" % np.shape(genotype_matrix))
    print("\nShape of saved Data:\n Nr Inds: % i \n Nr. Loci: % i" % np.shape(saved_data))    
    
    
def data_to_geneland_folder(coords_path, genotype_path, save_folder, delimiter="$", p=False): 
    '''Takes data that have been generated elsewhere, 
    and transforms it to the right format and saves it to the righ folder
    in R_CODE. Delimiter: What delimiter was used in input files.
    p: whether data is 0,0.5,1'''
    
    coords = np.loadtxt(coords_path, delimiter=delimiter)  # Delimiter is white space by default
    genotypes = np.loadtxt(genotype_path, delimiter=delimiter)  # Delimiter is white space by default 
    if p==True:
        genotypes=genotypes*2
    
    assert(len(coords)==len(genotypes))
    print("Loaded:\n Nr. of Individuals: %i Nr. of Genotypes: %i" % np.shape(genotypes))
    
    genotype_path_out = save_folder + "genotypes.csv"
    coords_path_out = save_folder + "coords.csv"
    
    
    # Transform and Save Genotypes.
    transform_to_geneland(genotypes, genotype_path_out)
    
    # Save Coordinates.
    np.savetxt(coords_path_out, coords, fmt='%i')  # Save the coordinates
    print("Done.")
    
# Run the code
simulate_panmictic_barrier(folder ="./2Deme1000i200l/", save=True, plot=True)
#data_to_geneland_folder("./ExternalFiles/barrier_file_coords32.csv","./ExternalFiles/barrier_file_genotypes32.csv", "./SynthBarrier0/")
# data_to_geneland_folder("./ExternalFiles/coordinatesHZALL0.csv","./ExternalFiles/genotypesHZALL0.csv", "./HZData/", p=True)









