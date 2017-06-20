# TODO: This is a hello world script
# 
# Author: hringbauer
###############################################################################
# 0) Test of R-Shell
print("Hello World")

geneland_path = "Geneland_4.0.8.tar.gz"


# 1) Install Geneland
install.packages(geneland_path, repos=NULL)

# 1.1) Install some Multi Processing Packages:
# For MPI:
#install.packages("snow")
#install.packages("Rmpi")   # Installation does not work
# FOR RPVM:
#install.packages("rpvm")

# SUCCESSFULLY INSTALLED!


# 2) Inference

####################################################
# Load packages
library(Geneland)
# Geneland.GUI()
# Script for Processing the Data
# Set Working Directory:
wd = "/home/hringbauer/git/Harald/R_CODE"
setwd(wd)
#getwd()


folder = "./2Deme/"
input_file_g = "i300l100_genotypes.csv"
input_file_c = "i300l100_coords.csv"
output_folder = "output/"

geno_path = paste(folder, input_file_g,  sep = "")
coord_path = paste(folder, input_file_c,  sep = "")

output_folder = paste(folder, output_folder, sep = "")
print(output_file)
		
# Read the Data:
coord <- read.table(coord_path)
geno <- read.table(geno_path)
#####################################################

fix(coord)
fix(geno)

# Some checks on Data. One can skip it if needed.
dim(coord)
nrow(geno)
ncol(geno)
plot(coord,xlab="Eastings",ylab="Northings",asp=1)

? MCMC  # Some help on MCMC

start <- Sys.time ()  # Tic
MCMC(coordinates=coord,
		geno.dip.codom=geno,
		varnpop=FALSE,     # If K is variable
		npopmin=2,
		npopmax=2,
		spatial=TRUE,
		freq.model="Uncorrelated",
		nit=100000,        # Number of Iterations
		thinning=100,
		path.mcmc=folder)
Sys.time () - start # Toc

#####################################################
### Post Processing

PostProcessChain(coordinates=coord,
		geno.dip.codom=geno,
		path.mcmc=folder,
		nxdom=100,
		nydom=100,
		burnin=200)

Plotnpop(path.mcmc=folder,
		burnin=200)

PosteriorMode(coordinates=coord,
		path.mcmc="./",
		file="map.pdf")




