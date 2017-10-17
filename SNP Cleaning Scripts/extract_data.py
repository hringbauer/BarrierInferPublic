'''
Created on Oct 14, 2015

@author: Harald
'''

import numpy as np
import itertools
import matplotlib.pyplot as plt  # @UnusedImport
from time import time
from FlaggedSNPs import set_colors

error_rate = 0.005  # Estimated error rate for SNP genotyping PER BASE
min_relatedness = 0.12  # Minimum required relatedness
max_opp_homo = 9  # Maximum nbr of opposing homozygotes

class Extractor(object):
    '''
    This class contains methods to extract data; mainly for Coancestry.
    28.3.2017: Also update to extract Position and Genotype Data
    for Barrier scheme.
    '''
    loci_information = "./LociInformation/shortSummarySNPsAdj.csv"  # Where to find Loci Information
    good_SNPs = []
    data = []
    p = []
    coords = []
    names = []  # Vector for Name of the Loci
    color = []  # Vector for the Color (in Python readable Form)

    def __init__(self, Data):
        '''Initialize; same as for FlaggedSNPs '''
        good_SNPs = np.where(Data.sNP_okay == 1)[0]  # Extracts only the good SNPs
        self.data = Data.subdata[:, good_SNPs].astype('float')  # Load data from Subdata
        self.p = Data.p[good_SNPs - Data.sNP_inds[0]].astype('float')  # Load allele frequencies
        self.coords = Data.subdata[:, (Data.x_cords_ind, Data.y_cords_ind)].astype('float')
        self.names = Data.header[good_SNPs]  # Get SNP names
        self.color = set_colors(Data.subdata[:, Data.color_ind])  # Sets the color vector
        # self.color = Data.subdata[:, Data.sNP_inds[1] + 1].astype(float)
        print("Analyzing %.0f SNPs from %.0f individuals" % (len(self.data[0, :]), len(self.data[:, 0])))
        
        
    def extract_pairs(self, po_pairs=300, full_sibs=300, half_sibs=300, unrelated=300):
        '''Extract pairs of simulated SNPs with defined relationship AND Error-Rate'''
        data, coords = self.data, self.coords  # @UnusedVariable
        kinship = []
        
        ind1 = []  # List for individuals (their full genotypes)
        ind2 = []
        
        offspring, parent1, _ = self.extract_po_pairs(po_pairs)  # First do the parent-pairs
        kinship += [[0, 1, 0] for _ in range(po_pairs)]
        ind1 += offspring
        ind2 += parent1
        
        offspring1, offspring2, _ = self.extract_full_sibs(full_sibs)  # And then full sibs
        kinship += [[0.25, 0.5, 0.25] for _ in range(full_sibs)]
        ind1 += offspring1
        ind2 += offspring2
        
        offspring1, offspring2, _ = self.extract_half_sibs(half_sibs)  # And now the half-sibs
        kinship += [[0, 0.5, 0] for _ in range(half_sibs)]
        ind1 += offspring1
        ind2 += offspring2
        
        inds1, inds2, _ = self.extract_unrelated_pairs(unrelated)  # And now the unrelated individuals
        kinship += [[0, 0, 1] for _ in range(unrelated)]
        ind1 += inds1
        ind2 += inds2
        
        ind1 = [add_gtp_error(self, p, error_rate) for p in ind1]  # Add genotyping error:
        ind2 = [add_gtp_error(self, p, error_rate) for p in ind2]

        print("%.0f pairs created!" % len(ind2))
        self.write_coancestry_gtps(ind1, ind2)  
          
        
    def extract_po_pairs(self, nr_po_pairs):
        '''Simulate po-pairs based on real data and return their genotypes and relative positions'''
        data, coords = self.data, self.coords
        parent1 = np.random.randint(len(data[:, 0]), size=nr_po_pairs)
        parent2 = np.random.randint(len(data[:, 0]), size=nr_po_pairs)
        distance_parents = [np.linalg.norm(coords[parent1[i], :] - coords[parent2[i], :]) for i in range(nr_po_pairs)]  # Pairwise distance of parents
        offspring = [self.create_offspring(data[parent1[i], :], data[parent2[i], :]) for i in range(nr_po_pairs)]  # Create Offspring
        parent1 = [data[parent1[i], :] for i in range(nr_po_pairs)]
        return (offspring, parent1, distance_parents) 
    
    def extract_full_sibs(self, nr_sibs):
        '''Extract full sibs based on real data and return their genotypes and relative positions'''
        data, coords = self.data, self.coords
        parent1 = np.random.randint(len(data[:, 0]), size=nr_sibs)
        parent2 = np.random.randint(len(data[:, 0]), size=nr_sibs)
        distance_parents = [np.linalg.norm(coords[parent1[i], :] - coords[parent2[i], :]) for i in range(nr_sibs)]  # Pairwise distance of parents
        offspring1 = [self.create_offspring(data[parent1[i], :], data[parent2[i], :]) for i in range(nr_sibs)]  # Create Offspring 1
        offspring2 = [self.create_offspring(data[parent1[i], :], data[parent2[i], :]) for i in range(nr_sibs)]  # Create Offspring 2
        return(offspring1, offspring2, distance_parents)
        
    def extract_unrelated_pairs(self, nr_pairs):
        '''Extract random putatively unrelated pairs from real data and return genotypes and relative positions'''
        data, coords = self.data, self.coords
        ind1 = np.random.randint(len(data[:, 0]), size=nr_pairs)
        ind2 = np.random.randint(len(data[:, 0]), size=nr_pairs)
        distance_parents = [np.linalg.norm(coords[ind1[i], :] - coords[ind2[i], :]) for i in range(0, nr_pairs)]  # Pairwise distance of parents
        offspring1 = [data[ind1[i], :] for i in range(0, nr_pairs)]  # Create Offspring 1
        offspring2 = [data[ind2[i], :] for i in range(0, nr_pairs)]  # Create Offspring 2
        return(offspring1, offspring2, distance_parents)
    
    def extract_half_sibs(self, nr_pairs):
        '''Extract half sibs based on real data and return their genotypes and relative positions of different parents'''
        data, coords = self.data, self.coords
        parent1 = np.random.randint(len(data[:, 0]), size=nr_pairs)
        parent2 = np.random.randint(len(data[:, 0]), size=nr_pairs)
        parent_s = np.random.randint(len(data[:, 0]), size=nr_pairs)
        distance_parents = [np.linalg.norm(coords[parent1[i], :] - coords[parent2[i], :]) for i in range(nr_pairs)]  # Pairwise distance of parents
        offspring1 = [self.create_offspring(data[parent_s[i], :], data[parent1[i], :]) for i in range(nr_pairs)]  # Create Offspring 1
        offspring2 = [self.create_offspring(data[parent_s[i], :], data[parent2[i], :]) for i in range(nr_pairs)]  # Create Offspring 2
        return(offspring1, offspring2, distance_parents)     
                    
    def create_offspring(self, sNPs1, sNPs2):
        '''Combines two individuals randomly to a new individual'''
        new_SNPs = [-9 for _ in range(0, len(sNPs1))]  # Initialize new SNP, default is error
        
        for k in range(0, len(sNPs1)):
            if sNPs1[k] + sNPs2[k] > -1:
                # Print draw random number with Prob. sNP1 and random number with Prob sNp2:
                new_SNPs[k] = np.random.binomial(1, sNPs1[k] / 2.0) + np.random.binomial(1, sNPs2[k] / 2.0)
        return np.array(new_SNPs).astype(np.int)
    
    def write_coancestry_gtps(self, list1, list2):
        '''Method which is taking two lists of Gtps and produces output ready for Coancestry'''
        file_name = input("\nWhich file do you want to write to?")
        txt = open(file_name, "w")
        for i in range(len(list1)):
            j = i + 1
            txt.write("R001F%03d" % j + str("M0") + gtp_to_coancestry(list1[i]) + "\n")
            txt.write("R001F%03d" % j + str("M1") + gtp_to_coancestry(list2[i]) + "\n")
        txt.close()
        print("New file created: " + file_name)
        
    def write_distance_list(self, dist_list):
        '''Write and save Distance-List as TXT'''
        file_name = "Distance_list.txt"
        txt = open(file_name, "w")
        for i in dist_list:
            txt.write(str(i) + "\n")
        txt.close()
        print("New file created: " + file_name)
           
    def save_coancestry_data(self):
        '''Saves data for coancestry analysis'''
        # First save the data for allele frequencies:
        p = self.p  # Load the allele frequency of good loci
        
        txt = open("allfreqs.txt", "w")
        for p_i in p:
            txt.write("1 2\n")
            txt.write(str(1 - p_i) + " " + str(p_i) + "\n")
        txt.close()
        
        txt = open("miss_error_rate.txt", "w")
        for i in range(0, len(p)):
            errors = np.sum(self.data[:, i] < 0)
            err_rate = errors / float(len(self.data[:, i]))
            txt.write("0.01 " + str(err_rate) + "\n")
        txt.close()
        
    def extract_high_rel_pairs(self):
        '''Extracts pairs of high relatedness + 500 random pairs'''
        print("Extract SNPs...")
        data, p, coords = self.data, self.p, self.coords  # Extract filtered SNPs.
        print("Analysing %.1f suitable SNPs. " % len(data[0, :]))
        ind_list, ind_list1, dist_list = [], [], []
        # Do the correlation analysis:
        t = time()
        for (i, j) in itertools.combinations(np.arange(np.size(data, 0)), r=2):
            estimator, homo = kinship_coeff(data[i, :], data[j, :], p)  # Kinship coeff per pair, averaged over loci + Number of opposing homozygotes
            pair_distance = np.linalg.norm(coords[i, :] - coords[j, :])  # Calculate Pairwise distance
                
            if estimator > min_relatedness or homo < max_opp_homo:  # Append highly related individuals to list
                ind_list.append(data[i, :])
                ind_list1.append(data[j, :])
                dist_list.append(pair_distance)
        print("Elapsed Time: %2f" % (time() - t))
        print("Highly related pairs extracted: %.0f" % len(ind_list))
        
        # Add 500 unrelated pairs
        inds1, inds2, distance = self.extract_unrelated_pairs(500)
        ind_list += inds1
        ind_list1 += inds2
        dist_list += distance
        
        self.write_coancestry_gtps(ind_list, ind_list1)  # Save the file
        self.write_distance_list(dist_list)
        
    def extract_snp_data(self, cut_off_corr=1, haploid=1, color_save=True):
        '''Method to extract SNP-data that can be analyzed by Mr. AK
        haploid: Whether to draw haploid data.'''
        genotypes, coords = self.data.astype('int'), self.coords
        genotypes = self.repick_failed_snps(genotypes)  # Repicks all failed SNPs
        
        # In case of haploidy repick Individuals:
        if haploid == 1:
            genotypes = self.pick_haploid_inds(genotypes)
            
        elif haploid == 0:
            genotypes = genotypes / 2.0  # In case of diploid data normalize it
            
        # x_corrs, y_corrs=self.geographic_correlation(genotypes, coords, self.names)
    
        
        np.savetxt("coordinatesHZALL.csv", coords, delimiter="$")  # Save the coordinates
        np.savetxt("genotypesHZALL.csv", genotypes, delimiter="$")  # Save the data
        
        # Extract color if needed.
        if color_save:
            np.savetxt("colorsHZALL.csv", self.color, delimiter="$", fmt="%s")  # Save the Color
            print("Color Successfully Saved!")
            
        print("Data successfully extracted!!")
      
    def repick_failed_snps(self, genotypes, p=None):
        '''Locate failed snps; i.e. -9 entries. Repick them from mean allele frequency'''  
        nr_genotypes = np.shape(genotypes)[1]
        
        # Calculate mean allele frequency if none given
        if p == None:
            p = np.zeros(nr_genotypes) - 1  # Set it to -1 so that error immediately evident
            for i in xrange(nr_genotypes):
                good_gtps = genotypes[:, i] > -1  # Extract only good genotypes
                p[i] = 0.5 * np.mean(good_gtps)  # To account for diploid Data!!
                
        # Find all Faile SNPs:
        failed = genotypes < 0
        print("Nr. of failed SNPs: %i" % np.sum(failed))
        print("Fraction: %.4f " % (np.sum(failed) / float(np.size(genotypes))))
           
        # Impute missing genotypes
        for (ind, sNP), value in np.ndenumerate(genotypes):  # Iterate over all data points
            if value < 0:  # In case of missing data
                genotypes[ind, sNP] = np.random.binomial(2, p[sNP])  # Draw random Genotypes for missing data
        return genotypes
        
    def pick_haploid_inds(self, genotypes):
        '''Picks only a haploid individual.'''
        # First overwrites -9s
        
        # Then picks randomly
        mean_freqs = genotypes / 2.0  # Calculates the Mean Frequencies to pick from
        hap_genotypes = np.random.binomial(1, mean_freqs)  # Then picks the Genotypes at random (0->0, 1->1, 2->1)
        return hap_genotypes
 
    
def add_gtp_error(self, p, mu):
    '''Add error mu to genotyping p'''
    error = np.random.choice([-1, 0, 1], len(p), p=[mu , 1 - 2 * mu, mu])  # 2mu chance that something chances at this locus
    for i in range(len(p)):
        if p[i] > -1:  # Only do something if there is no error already
            if error[i] != 0:  # If there is no error do nothing
                if p[i] == 0 or p[1] == 2:  # Change to heterozygote
                    p[i] = 1
                if p[i] == 1:
                    p[i] = p[i] + error[i]  # Change to homozgygote
    return p

def gtp_to_coancestry(p):
    '''Converts Int 0/1/2 array to string of single allele values saved as string'''
    string = ""
    for i in p:
        if i == -9:
            string += " 0 0"
        if i == 0:
            string += " 1 1"
        if i == 1:
            string += " 1 2"
        if i == 2:
            string += " 2 2"
    return string

def kinship_coeff(sample1, sample2, p):
    '''Takes two samples as input(SNP-numpy arrays) and calculates the kinship coefficient'''
    working_SNPs = np.where(sample1[:] + sample2[:] > -1)[0]  # Which entries to actually use
    pi = 0.5 * sample1[working_SNPs]
    qi = 1 - pi
    pj = 0.5 * sample2[working_SNPs]
    qj = 1 - pj
    pt = p[working_SNPs]  # Temporary p
    qt = 1 - pt
                
    estimator = np.mean((pi - pt) * (pj - pt) / pt + (qi - qt) * (qj - qt) / qt)  # Estimator per per-individual pair, averaged over loci
    homos = np.sum(np.absolute(pi - pj) == 1)
    return (estimator, homos)
        







