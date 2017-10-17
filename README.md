# Instructions
This is the source code that was used to simulate and analyze datasets with a barrier; and to visualize the results.

The simulation engine is implemented in grid.py
It contains a class Grid.py to simulate data on a grid, and a subclass Secondary_Grid to simulate data on a grid with secondary contact.

The analysis engine is implemented in mle_pairwise.py. The key method, that fits pairwise heterozygosity, is implemented in  the class MLE_f_emp.

The Kernel to calculate 2D Diffusion with a barrier is implemented in kernels.py.
There, a general Kernel class to calculate covariance matrices between sampling points is implemented, and several subclasses inherit from it. Importantly DiffusionBarrierK0 implements the 2D Diffusion with a barrier.

Importantly, classes to run replicates of various scenarious are implemented in multirun.py. Various subclasses for different scenarios are implemented, with a common framework to simulate and analyze these runs (see bottom of source code). Every subclass there inherits a method to create and analyze a dataset (create_data_set resp. analyze_data_set). A factory method (fac_method) returns these various classes.

The methods from multirun can be run on a cluster, for instance in combination of lines in script_multirun.py and the shell script nickscript.sh.

To visualize the results, figs_paper.py contains various methods to plot data and diagrams.


Also, in analysis.py, several helper methods are implemented, that are called from the other classes (for instance, binning of individual genotype data).

## Data Availability
The Scripts used for filtering the hybrid zone data can be found in the subfolder "./SNP Cleaning Scripts"

The filtered Antirrhinum hybrid zone data used for inference (positions and genotypes) can be found in the folder "./DataHZ"
