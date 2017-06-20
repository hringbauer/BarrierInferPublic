# TODO: This is an R Script to install Geneland on 
# the cluster.
# Author: hringbauer
# Date: June 20th 2017
###############################################################################

## Create the personal library if it doesn't exist. Ignore a warning if the directory already exists.
dir.create(Sys.getenv("R_LIBS_USER"), showWarnings = FALSE, recursive = TRUE)


## Install one package.
## install.packages("timeDate", Sys.getenv("R_LIBS_USER"), repos = "http://cran.case.edu" )
## Install a package that you have copied to the remote system.
install.packages("Geneland_4.0.8.tar.gz", Sys.getenv("R_LIBS_USER"), repos=NULL)


## Install multiple packages.
## install.packages(c("timeDate","robustbase"), Sys.getenv("R_LIBS_USER"), repos = "http://cran.case.edu" )


#################################################################################
# Instructions:
# 1) Submit Command is: R CMD BATCH install-packages.R

# 2) Confirm installation by listing the contents of ~/R

# 3) Retrieve any error messages from install-packages.Rout, which is generated as a result of running install-packages.R.