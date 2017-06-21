# TODO: This is a script that contains R functions to visualize 
# the Geneland Output
# 
# Author: hringbauer

plot_output <- function(folder, data_set_nr=1, post_process=FALSE, save=FALSE){
    #Post Process: Whether to postprocess the Data
	wd = "/home/hringbauer/git/Harald/R_CODE"
	setwd(wd)
	# Load the Data:
	input_file_g = "genotypes.csv"
	input_file_c = "coords.csv"
	output_folder = "output" # End is appended later on
	output_strings = c(folder, output_folder, data_set_nr,"/")
	output_folder = paste(output_strings, collapse = '')
	
	# Read the Data
	geno_path = paste(folder, input_file_g,  sep = "")
	coord_path = paste(folder, input_file_c,  sep = "")
	
	coord <- read.table(coord_path)
	geno <- read.table(geno_path)
	
	print(dim(coord))
	print(nrow(geno))
	print(ncol(geno))
	
	# Do Post-Processing if needed:
	if (post_process == TRUE) {
		start <- Sys.time ()  # Tic
		PostProcessChain(coordinates=coord,
				path.mcmc=output_folder,
				nxdom=100,
				nydom=100,
				burnin=200)
		print(Sys.time () - start) # Toc
	}
	
	# Plot Number of Tile (Convergence Criterium)
	dev.new()
	Plotntile(path.mcmc=output_folder,
			burnin=200,
			printit=save,
			file="nr_tiles.pdf")
	
	
	invisible(readline(prompt="Press [enter] to continue"))
	dev.off()
	dev.new()
	# Plot PosteriorMode:
	PosteriorMode(coordinates=coord,
			path.mcmc=output_folder,
			file="posterior.pdf",
			printit=save,
			format="pdf")
	
	
	invisible(readline(prompt="Press [enter] to continue"))
	dev.off()
	dev.new()
	# Plot Tesselation:
	PlotTessellation(coordinates=coord, 
			path.mcmc=output_folder,
			path=folder,
			printit=save)
	print("Plotting Complete!")
	invisible(readline(prompt="Press [enter] to leave"))
	dev.off()
	dev.new()
	graphics.off()
	
}



library(Geneland)
# plot_output("./2Deme1000i200l/", data_set_nr=9, post_process=TRUE)
# plot_output("./2Deme400i200l/", data_set_nr=1, post_process=FALSE, save=FALSE) 
plot_output("./SynthWeak/", data_set_nr=6, post_process=FALSE, save=TRUE)  # 6
# plot_output("./SynthFull/", data_set_nr=9, post_process=FALSE, save=TRUE) # 9. 3 is interesting


