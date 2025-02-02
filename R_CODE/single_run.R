# This is a R script for the cluster.
# Author: hringbauer
# Date: June 20th
###############################################################################

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
# test if there is at least one argument: if not, return an error
if (length(args)!=1) {
	stop("Exactly one argument must be supplied. Stupid Nishiprod.", call.=FALSE)
	}

# Load Libraries:
library(Geneland)

# Set Working Directory:
getwd()  # Assumes R-Script is run from the R_CODE folder!!
#wd = "./R_CODE"
#setwd(wd)

data_set_nr = args[1]  # Dont forget: Not Python Indexing!!

#data_set_nr = 1
folder = "./SynthKVar/"
input_file_g = "genotypes.csv"
input_file_c = "coords.csv"
output_folder = "output" # End is appended later on

output_strings = c(folder, output_folder, data_set_nr,"/")
output_folder = paste(output_strings, collapse = '')

geno_path = paste(folder, input_file_g,  sep = "")
coord_path = paste(folder, input_file_c,  sep = "")

# Create Output-Folder
dir.create(output_folder)
		
# Read the Data:
coord <- read.table(coord_path)
geno <- read.table(geno_path)

dim(coord)
nrow(geno)
ncol(geno)
#plot(coord,xlab="Eastings",ylab="Northings",asp=1)



#####################################################
# Run the actual Analysis

start <- Sys.time ()  # Tic
MCMC(coordinates=coord,
		geno.dip.codom=geno,
		varnpop=TRUE,     # If K is variable. Originally FALSE
		npopmin=2,
		npopmax=10,
		spatial=TRUE,
		freq.model="Uncorrelated",
		nit=500000,        # Number of Iterations
		thinning=500,
		path.mcmc=output_folder)
print("MCMC Run Done!")
Sys.time () - start # Toc

#####################################################
### Post Processing

PostProcessChain(coordinates=coord,
		path.mcmc=output_folder,
		nxdom=100,
		nydom=100,
		burnin=200)

print("Full Run Done. Success!")


