
# coding: utf-8

# ## Classes for Kernels
# Classes for Covariance Kernels. They all inherit from one abstract class, which contains methods they all should have.
# Also has a factory method, which can be used for producing the instance of the wished class.

# In[111]:

import numpy as np
import cProfile  # @UnusedImport
from scipy.special import kv as kv  # Import Bessel Function
from scipy.integrate import quad  # Import function to do the integral
from scipy.special import erfc
from time import time
from functools import partial

import matplotlib.pyplot as plt
import multiprocessing as mp  # Library for Multiprocessing
import intergand  # Import the Integrand Watch the type
import time
# get_ipython().magic(u'matplotlib notebook')

# In[49]:

class Kernel(object):
    '''The Class for the Kernels.
    At least should have a calc_kernel method that calculates the 
    Covariance Matrix '''
    nr_parameters = 0  # The Number of Parameters to optimize
    multi_processing = 0  # Whether to use multi-processing or not (advantegous for Kernel)
    
    def __init__(self):
        print("Sets the parameters of the respective Kernel")
        
    def draw_from_kernel(self, mean, coords):
        '''Draw correlated Deviations.'''
        mean_vec = np.ones(len(coords)) * mean
        cov_mat = self.calc_kernel_mat(coords)
        data = np.random.multivariate_normal(mean_vec, cov_mat)  # Do the random draws of the deviations from the mean
        return data
    
    def time_kernel(self, coords):
        '''Time the runtime of the Kernel. '''
        start = time.time()
        cov_mat = self.calc_kernel_mat_old(coords)
        end = time.time()
        return (end - start)  # Returns the Runtime of the Kernel
    
    def calc_kernel_mat(self, coords):
        raise NotImplementedError("Implement this you lazy fuck")
        
    def set_parameters(self):
        raise NotImplementedError("Implement this you lazy fuck")
        
    def give_nr_parameters(self):
        raise NotImplementedError("Implement this you lazy fuck")
        
    def give_parameter_names(self):
        raise NotImplementedError("Implement this you lazy fuck")
        
    def give_parameters(self):
        raise NotImplementedError("Implement this you lazy fuck")


# In[79]:

class DiffusionBarrierK(Kernel):
    '''A whole class which is designed to 
    calculate covariance kernels from the Barrier Diffusion Model
    It assumes Diffusion along x- and y- axis; with a Barrier at x=0
    Numerically integrates what f should be.
    Throughout y is the STARTING point and x the end point'''
    # Parameters of the Covariance
    nr_parameters = 5
    k = 0.5  # Permeability of the Barrier
    D = 1.0  # Diffusion Constant; equals sigma**2. Sets how quickly stuff decays
    mu = 0.001  # Long Distance/Migration rate; sets the max. scale of identity
    t0 = 1  # Starting Point of Integration; i.e. where to start integration. Sets the "local" scale.
    density = 1.0  # Density of Diploids
    
    def __init__(self, k=0.5, D=1, t0=1.0, mu=0.001, density=1.0):
        '''Initialize to set desired values!'''
        self.k = k  # Sets the constaints to their required values
        self.D = D
        self.t0 = t0
        self.mu = mu
        self.density = density
        
    def set_parameters(self, params=[0.5, 1.0, 0.001, 1, 1]):
        assert(len(params) == self.nr_parameters)  # Check whether right length
        self.k = params[0]
        self.D = params[1]
        self.t0 = params[2]
        self.mu = params[3]
        self.density = params[4]
        
    def give_nr_parameters(self):
        return(5)
    
    def give_parameter_names(self):
        return(["k", "D", "mu", "t0", "loc_rate"])
        
    def give_parameters(self):
        return([self.k, self.D, self.t0, self.mu, self.density]) 
    
    def GS(self, t, y, x):
        '''1D Diffusion for same side of the Barrier'''
        n1 = np.exp(-(x - y) ** 2 / (4 * self.D * t)) + np.exp(-(x + y) ** 2 / (4 * self.D * t))
        d1 = np.sqrt(4 * np.pi * self.D * t)
        
        a2 = self.k / self.D * np.exp(2 * self.k / self.D * (y + x + 2 * self.k * t))
        b2 = erfc((y + x + 4 * self.k * t) / (2 * np.sqrt(self.D * t)))
        res = n1 / d1 - a2 * b2
        if np.isnan(res) or np.isinf(res):  # Check if numerical instability
            return self.gaussian(t, y, x)  # Fall back to Gaussian (to which one converges)
        else: return res

    def GD(self, t, y, x):
        '''1D Diffusion for different sides of the Barrier'''
        a1 = self.k / self.D * np.exp(2 * self.k / self.D * (y - x + 2 * self.k * t))
        b1 = erfc((y - x + 4 * self.k * t) / (2 * np.sqrt(self.D * t)))
        res = a1 * b1
        if np.isnan(res) or np.isinf(res):  # Check if numerical instability
            return self.gaussian(t, y, x)  # Fall back to Gaussian (to which one converges)
        else: return res

    def gaussian(self, t, y, x):
        '''The normal thing without a barrier'''
        return np.exp(-(x - y) ** 2 / (4 * self.D * t)) / np.sqrt(4 * np.pi * self.D * t)

    def gaussian1d(self, t, dy):
        '''The One Dimensional Gaussian. 
        Differnce: Here dy notes the difference along the y axis'''
        return 1.0 / np.sqrt(4 * np.pi * self.D * t) * np.exp(-dy ** 2 / (4 * self.D * t))

    def integrand_barrier_ss(self, t, dy, x0, x1):
        '''The integrand in case there is no barrier
        Product of 1d Gaussian along y-Axis and x-Axis Barrier Pdf.
        And a term for the long-distance migration'''
        return (self.gaussian1d(t, dy) * self.GS(t, x0, x1) * np.exp(-2 * self.mu * t))

    def integrand_barrier_ds(self, t, dy, x0, x1):
        '''the integrand for cases of different sided of the barrier.
        Product of 1d Gaussian along y-Axis
        And a term for the long-distance migration'''
        return (self.gaussian1d(t, dy) * self.GD(t, x0, x1) * np.exp(-2 * self.mu * t))
        

    def num_integral_barrier(self, dy, x0, x1):
        '''Calculate numerically what the identity 
        due to shared ancestry should be. 
        dy: Difference along y-Axis
        x0: Starting point on x-Axis 
        x1: Ending point on x-Axis
        Integrate from t0 to Infinity'''  
        if x0 < 0:  # Formulas are only valid for x0>0; but simply flip at barrier if otherwise!
            x0 = -x0
            x1 = -x1
        
        if x1 > 0:  # Same side of Barrier
            return (1.0 / (2.0 * self.density)) * quad(self.integrand_barrier_ss,
                self.t0, np.inf, args=(dy, x0, x1))[0] 
        
        if x1 < 0:  # Different side of Barrier
            return (1.0 / (2.0 * self.density)) * quad(self.integrand_barrier_ds,
                self.t0, np.inf, args=(dy, x0, x1))[0]
   
    def calc_kernel_mat(self, coords):
        '''Given List of Coordinates; calculate the full covariance Kernel'''
        kernel_mat = [[self.num_integral_barrier(i[1] - j[1],
                            i[0], j[0]) for i in coords] for j in coords]
        
        K1 = 0.0000001 * np.eye(len(coords))  # Add Identity Matrix to make numerically stable
        return np.array(kernel_mat) + K1


class DiffusionBarrierK0(Kernel):
    '''A whole class which is designed to 
    calculate covariance kernels from the Barrier Diffusion Model
    It assumes Diffusion along x- and y- axis; with a Barrier at x=0
    Numerically integrates what f should be.
    Throughout y is the STARTING point and x the end point.
    Here: Integral transformed after dimension analysis to get rid of redundant parameters'''
    # Parameters of the Covariance
    nr_parameters = 5
    k = 0.2  # 2k/D
    t0 = 1  # Starting Point of Integration; i.e. where to start integration. Sets the "local" scale.
    nbh = 100  # 1/(4 Pi De D)
    L = 0.002  # mu/D
    ss = 0
    
    def __init__(self, k=0.5, t0=1.0, nbh=50.0, L=0.002, ss=0, multi_processing=0):
        '''Initialize to set desired values!'''
        self.set_parameters([k, t0, nbh, L, ss])
        self.multi_processing = multi_processing

        
    def set_parameters(self, params=[0.5, 1.0, 0.001, 1, 1]):
        assert(len(params) == self.nr_parameters)  # Check whether right length
        self.k = params[0]
        self.t0 = params[1]  # Starting Point of Integration; i.e. where to start integration. Sets the "local" scale.
        self.nbh = params[2]
        self.L = params[3]
        self.ss = params[4]
        
    def give_nr_parameters(self):
        return(5)
    
    def give_parameter_names(self):
        return(["k", "D", "mu", "t0", "loc_rate"])
        
    def give_parameters(self):
        return([self.k, self.D, self.t0, self.mu, self.density]) 

    def gaussian(self, t, y, x):
        '''The normal thing without a barrier'''
        return np.exp(-(x - y) ** 2 / (4 * t)) / np.sqrt(4 * np.pi * t)  # New one t->D*t
        # return np.exp(-(x - y) ** 2 / (4 * self.D * t)) / np.sqrt(4 * np.pi * self.D * t)  # Old one

    def gaussian1d(self, t, dy):
        '''The One Dimensional Gaussian. 
        Differnce: Here dy notes the difference along the y axis
        Not used inside class; but it is here for historical reasons.'''
        return 1.0 / np.sqrt(4 * np.pi * t) * np.exp(-dy ** 2 / (4 * t))  # Again t-> D*t

    def integrand_barrier_ds(self, t, dy, x0, x1):
        '''The integrand in case there is no barrier
        Product of 1d Gaussian along y-Axis and x-Axis Barrier Pdf.
        And a term for the long-distance migration.
        Everything transformed so that t->Dt and one gets coordinate independent constants'''
        
        # 1D Contribution from Gaussian along y-axis
        pre_fac = 1.0 / self.nbh * np.sqrt(np.pi / t)  # Prefactor from first Gaussian
        gaussiany = pre_fac * np.exp(-dy ** 2 / (4.0 * t))
        
        # 1D Contribution from x-Axis (barrier)
        a1 = self.k * np.exp(2 * self.k * (x0 - x1 + 2 * self.k * t))  # First Term Barrier
        b1 = erfc((x0 - x1 + 4 * self.k * t) / (2 * np.sqrt(t)))  # Second Term Barrier
        pdfx = a1 * b1
        
        if np.isnan(pdfx) or np.isinf(pdfx):  # Check if numerical instability
            pdfx = self.gaussian(t, x0, x1)  # Fall back to Gaussian (to which one converges)
        
        # Mutation/Long Distance Migration:
        mig = np.exp(-self.L * t)  # Long Distance Migrationg
        res = gaussiany * pdfx * mig  # Multiply everything together
        return res

    def integrand_barrier_ss(self, t, dy, x0, x1):
        '''The integrand for cases of different sides on the barrier.
        Product of 1d Gaussian along y-Axis
        And a term for the long-distance migration.
        Everything transformed so that t->Dt and one gets coordinate independent constants'''
                # 1D Contribution from Gaussian along y-axis
        pre_fac = 1.0 / self.nbh * np.sqrt(np.pi / t)  # Prefactor from first Gaussian
        gaussiany = pre_fac * np.exp(-dy ** 2 / (4 * t))
        
        # 1D Contribution from x-Axis (barrier)
        n1 = np.exp(-(x0 - x1) ** 2 / (4 * t)) + np.exp(-(x0 + x1) ** 2 / (4 * t))
        d1 = np.sqrt(4 * np.pi * t)
        
        a2 = self.k * np.exp(2 * self.k * (x0 + x1 + 2 * self.k * t))
        b2 = erfc((x0 + x1 + 4 * self.k * t) / (2 * np.sqrt(t)))
        pdfx = n1 / d1 - a2 * b2
        
        if np.isnan(pdfx) or np.isinf(pdfx):  # Check if numerical instability
            pdfx = self.gaussian(t, x0, x1)  # Fall back to Gaussian (to which one converges)
        
        # Mutation/Long Distance Migration:
        mig = np.exp(-self.L * t)  # Long Distance Migrationg
        res = gaussiany * pdfx * mig  # Multiply everything together
        return res
    
    def num_integral_barrier(self, dy, x0, x1):
        '''Calculates numerically what the identity due to shared ancestry should be'''  
        res = numerical_integration_barrier(self.t0, np.inf, dy, x0, x1, self.nbh, self.L, self.k)
        return res

    def num_integral_barrier_old(self, dy, x0, x1):
        '''Calculate numerically what the identity 
        due to shared ancestry should be. 
        dy: Difference along y-Axis
        x0: Starting point on x-Axis 
        x1: Ending point on x-Axis
        Integrate from t0 to Infinity'''
         
        if x0 < 0:  # Formulas are only valid for x0>0; but simply flip at barrier if otherwise!
            x0 = -x0
            x1 = -x1
        
        if x1 > 0:  # Same side of Barrier
            return quad(self.integrand_barrier_ss,
                self.t0, np.inf, args=(dy, x0, x1))[0] 
        
        if x1 < 0:  # Different side of Barrier
            return quad(self.integrand_barrier_ds,
                self.t0, np.inf, args=(dy, x0, x1))[0]
   
    def calc_kernel_mat_old(self, coords):
        '''Given List of Coordinates; calculate the full covariance Kernel'''
        kernel_mat = [[self.num_integral_barrier_old(i[1] - j[1],
                            i[0], j[0]) for i in coords] for j in coords]
        
        K1 = 0.0000001 * np.eye(len(coords))  # Add something to make it positive definite
        K2 = self.ss * np.ones(np.shape(K1))  # Add the kernel from Deviations from the Mean:
        return kernel_mat + K1 + K2
        
        
    def calc_kernel_mat(self, coords):
        # Do the Multiprocessing Action
        coords = np.array(coords)  # Make Coordinates a Numpy Array!
        nr_inds = len(coords)
        
        # Gets the upper triangular Indices
        inds0, inds1 = np.triu_indices(nr_inds)  # Gets all the indices which are needed.
        inds = np.triu_indices(nr_inds)
        
        y_coords = coords[:, 1]  # Array of all the y-Coordinates
        rel_ycoord_mat = y_coords[:, None] - y_coords[None, :]  # Matrix of relative coordinates
        
       
        x_vec0 = coords[inds0, 0]  # Calculate the complete vector of x0 vals
        x_vec1 = coords[inds1, 0]  # Calculate the complete vector of x1 vals
        rel_vec_y = rel_ycoord_mat[inds]  # Calculates the vector of relative y-indices      
        
        argument_vec = [[self.t0, np.inf, rel_vec_y[i], x_vec0[i], x_vec1[i], self.nbh, self.L, self.k] 
                          for i in xrange(len(x_vec0))]  # Create vector with all arguments    #  lb, ub, dy, x0, x1, nbh, L, k
        
        # Do the Multiprocessing Action
        if self.multi_processing==1:
            pool_size = mp.cpu_count() * 2
            pool = mp.Pool(processes=pool_size)
            pool_outputs = pool.map(numerical_integration_barrier_mr, argument_vec)  # map
            pool.close()
            pool.join()
            
        else: pool_outputs = map(numerical_integration_barrier_mr, argument_vec)
        
        # Fills up upper triangle again
        kernel = np.zeros((nr_inds, nr_inds))
        kernel[inds] = pool_outputs  
        
        # Symmetrizes again and fill up everything:
        kernel = np.triu(kernel) + np.triu(kernel, -1).T - np.diag(np.diag(kernel))   
        K1 = 0.0000001 * np.eye(len(coords))  # Add something to make it positive definite
        K2 = self.ss * np.ones(np.shape(kernel))  # Add the kernel from Deviations from the Mean:
        return (kernel + K1 + K2)


class DiffusionK(Kernel):
    '''A whole class which is designed to 
    calculate covariance kernels from the simple Diffusion Model
    It assumes Diffusion along x- and y- axis; with a Barrier at x=0
    Numerically integrates what f should be.
    Throughout y is the STARTING point and x the end point'''
    # Parameters of the Covariance
    nr_parameters = 5
    D = 1.0  # Diffusion Constant; equals sigma**2. Sets how quickly stuff decays
    t0 = 1  # Starting Point of Integration; i.e. where to start integration. Sets the minimum "local" scale.
    mu = 0.001  # Long Distance/Migration rate; sets the max scale of identity
    density = 1.0  # Density of Diploids
    ss = 0
    
    def __init__(self, D=1, t0=1.0, mu=0.001, density=1.0, ss=0):
        '''Initialize to set desired values!'''
        self.D = D
        self.t0 = t0
        self.mu = mu
        self.density = density
        self.ss = ss
        
    def set_parameters(self, params=[1.0, 0.001, 1, 1, 0]):
        assert(len(params) == self.nr_parameters)  # Check whether right length
        self.D = params[0]
        self.t0 = params[1]
        self.mu = params[2]
        self.density = params[3]
        self.ss = params[4]
        
    def give_nr_parameters(self):
        return(4)
    
    def give_parameter_names(self):
        return(["D", "t0", "mu", "Density"])
        
    def give_parameters(self):
        return([self.D, self.t0, self.mu, self.density]) 
    
    def integrand(self, t, r):
        '''the integrand for cases of no barrier.
        Product of 1d Gaussian along both Axis
        And a term for the long-distance migration'''
        diff = 1.0 / (4 * np.pi * self.D * t) * np.exp(-(r ** 2) / (4 * self.D * t))  # Diffusion
        ld_migration = np.exp(-t * 2 * self.mu)  # Long Distance Migration
        return (diff * ld_migration)
        
    def num_integral(self, r):
        '''Calculate numerically what the identity 
        due to shared ancestry should be. 
        Integrate from t0 to Infinity'''  
        return (1.0 / (2.0 * self.density) * quad(self.integrand,
            self.t0, np.inf, args=(r))[0])
        
    def calc_kernel_mat(self, coords):
        '''Given List of Coordinates; calculate the full covariance Kernel'''
        coords = np.array(coords)  # Make Coordinates a Numpy Array!
        dist_mat = np.sqrt(np.sum((coords[:, None] - coords[None, :]) ** 2, axis=2))  # First set up the Distance Matrix
        
        num_integral_v = np.vectorize(self.num_integral)  # Vectorizes the integral; maybe later parallelize
        kernel = num_integral_v(dist_mat)  # Calculate the kernel via vectorized function   
        
        K1 = 0.0000001 * np.eye(len(coords))  # Add Identity Matrix to make numerically stable    
        K2 = self.ss * np.ones(np.shape(kernel))  # Add the kernel from Deviations from the Mean:
        return kernel + K1 + K2



class DiffusionK0(Kernel):
    '''Dimensionless Diffusion Kernel:
    A whole class which is designed to 
    calculate covariance kernels from the simple Diffusion Model
    It assumes Diffusion along x- and y- axis; with a Barrier at x=0
    Numerically integrates what f should be.
    Throughout y is the STARTING point and x the end point'''
    # Parameters of the Covariance
    nr_parameters = 4
    nbh = 100.0  # Neighborhood Size: 4*pi*De*D
    L = 0.002  # 2mu/D
    t0 = 1  # Starting Point of Integration; i.e. where to start integration. Sets the minimum "local" scale.
    ss = 0
    
    def __init__(self, nbh=100, L=0.002, t0=1, ss=0, multi_processing=0):
        '''Initialize to set desired values!'''
        self.nbh = nbh
        self.L = L
        self.ss = ss
        self.t0 = t0
        self.multi_processing = multi_processing
        
    def set_parameters(self, params=[100, 0.002, 1, 0]):
        '''Nbh, L, t0, ss'''
        assert(len(params) == self.nr_parameters)  # Check whether right length
        self.nbh = params[0]
        self.L = params[1]
        self.t0 = params[2]
        self.ss = params[3]
        
    def give_nr_parameters(self):
        return(self.nr_parameters)
    
    def give_parameter_names(self):
        return(["nbh", "L", "t0", "ss"])
        
    def give_parameters(self):
        return([self.nbh, self.L, self.t0, self.ss]) 
    
    def integrand(self, t, r):
        '''Transformed integrand t'= D t'''
        res = 1 / (2 * t * self.nbh) * np.exp(-(r ** 2) / (4 * t) - self.L * t)
        return res
        
    def num_integral(self, r):
        '''The transformed Integral'''
        return quad(self.integrand,
            self.t0, np.inf, args=(r))[0]         
        
    def calc_kernel_mat_old(self, coords):
        '''Given List of Coordinates; calculate the full covariance Kernel'''
        coords = np.array(coords)  # Make Coordinates a Numpy Array!
        dist_mat = np.sqrt(np.sum((coords[:, None] - coords[None, :]) ** 2, axis=2))  # First set up the Distance Matrix
        
        num_integral_v = np.vectorize(self.num_integral)  # Vectorizes the integral; maybe later parallelize
        kernel = num_integral_v(dist_mat)  # Calculate the kernel via vectorized function  
        K1 = 0.0000001 * np.eye(len(coords))  # Add Identity Matrix to make numerically stable
        K2 = self.ss * np.ones(np.shape(kernel))  # Add the kernel from Deviations from the Mean
        return kernel + K1 + K2
    
    def calc_kernel_mat(self, coords):
        '''Calculates Full Covariance Kernel
        Calculates only upper triangular Matrix; and used multi-processing'''
        coords = np.array(coords)  # Make Coordinates a Numpy Array!
        nr_inds = len(coords)
        
        dist_mat = np.sqrt(np.sum((coords[:, None] - coords[None, :]) ** 2, axis=2))  # Calculates the distances matrix
        
        ind_ut = np.triu_indices(nr_inds)  # Indices for upper triangular array
        dist_vec = dist_mat[ind_ut]
        argument_vec = [[self.t0, np.inf, self.nbh, self.L, r] for r in dist_vec]  # Create vector with all arguments
        # argument_vec = zip([self.t0] * nr_inds, [np.inf] * nr_inds, , ,dist_vec)  
        
        # Do the Multiprocessing Action
        if self.multi_processing == 1:
            pool_size = mp.cpu_count() * 2
            pool = mp.Pool(processes=pool_size)
            pool_outputs = pool.map(numerical_integration_mr, argument_vec)
            pool.close()
            pool.join()
            
        else: pool_outputs = map(numerical_integration_mr, argument_vec)
        
        # Fills up upper triangle again
        kernel = np.zeros((nr_inds, nr_inds))
        kernel[ind_ut] = pool_outputs  
        
        # Symmetrizes again and fill up everything:
        kernel = np.triu(kernel) + np.triu(kernel, -1).T - np.diag(np.diag(kernel))   
        K1 = 0.0000001 * np.eye(len(coords))  # Add something to make it positive definite
        K2 = self.ss * np.ones(np.shape(kernel))  # Add the kernel from Deviations from the Mean
        return kernel + K1 + K2
        
        
# In[98]:

class RBFBarrierK(Kernel):
    '''Class for the radial base function kernel'''
    # Parameters
    nr_parameters = 4
    l = 15  # Length Scale
    a = 0.02  # Absolute Correlation
    c = 0.5  # Set Barrier Strength
    sigma_sqr = 0.01  # The Deviation From the mean

    
    def __init__(self, l=15, a=0.02, c=0.5, ss=0.01):
        self.l = l
        self.a = a
        self.c = c
        self.sigma_sqr = ss
    
    def give_nr_parameters(self):
        return 4
        
    def set_parameters(self, params=[15.0, 0.02, 0.5, 0.1]):
        '''Method to set Parameters'''
        self.l = params[0]
        self.a = params[1]
        self.c = params[2]
        self.sigma_sqr = params[3]

    def give_parameters(self):
        return [self.l, self.a, self.c, self.sigma_sqr]
    
    def give_parameter_names(self):
        return(["Length Scale", "Absolute Correlation", "Boundary Reduction, Sigma Sqr"])
    
    def calc_r(self, r):
        '''Calculate the RBF-Kernel (NO BARRIER) for a given distance r'''
        return self.a * np.exp(-r ** 2 / (2. * self.l ** 2))  # Calculate the RBF Kernel!
    
    def calc_kernel_mat(self, coords):
        '''Return Kernel for Individuals at Coords
        Barrier is assumed to be at x=0'''
        x = coords[:, 0]  # Extracts x-coords
        nr_inds = len(x)

        mask = np.array([-1.0, 1.0])

        coords_refl = coords * mask[None, :]  # Reflects the x-Coordinate

        g = np.sign(x)  # Calculates Signum of x
        same_side = (g[:, None] * g + 1) / 2.0  # Whether the x-Values are on the same side

        r_sqrd = np.sum(((coords[:, None] - coords[None, :]) ** 2), axis=2)  # Calculates pairwise Distance
        r_refl_sqrd = np.sum(((coords_refl[:, None] - coords[None, :]) ** 2), axis=2)  # Calculates the reflected Distance 

        # Calculate the normal Kernel:
        cov_mat = self.a * np.exp(-r_sqrd / (2. * self.l ** 2))  # Calculate the co-variance matrix. Added diagonal term
        cov_mat_refl = self.a * np.exp(-r_refl_sqrd / (2. * self.l ** 2))  # Calculate the covariance matrix for reflected coordinates.

        # Introduce the Covariance Function due to complete correlation
        # K1 = sigma_sqrd * np.ones((nr_inds,nr_inds),dtype=tf.float64)

        # Calculate the full Covariance Matrix
        K1 = self.sigma_sqr * np.ones(np.shape(cov_mat)) + 0.0000001 * np.eye(nr_inds)  # Due to Deviations from the mean
        K = same_side * (cov_mat + self.c * cov_mat_refl) + (1 - same_side) * (1 - self.c) * cov_mat + K1
        return K

def integrand(t, r, nbh, L):
    '''Integrand for numerical integration'''
    res = 1 / (2 * t * nbh) * np.exp(-(r ** 2) / (4 * t) - L * t)
    return res

def numerical_integration(lb, ub, nbh, L, r):
    '''Function for numerical Integration.
    Main reason: It is outside a class so multiprocessing can access it.'''
    # print([lb, ub, r, nbh, L]))
    return quad(intergand.integrand_c,  # # intergand.integrand_c   for c function. Old one was integrand
        lb, ub, args=(r, nbh, L))[0]  # Returns only the Integration, not the uncertainty

def numerical_integration_mr(arg):
    '''Wrapper for numerical integration; so that it works with one argument
    Needed for parallelization.'''
    return numerical_integration(*arg)

def numerical_integration_barrier(lb, ub, dy, x0, x1, nbh, L, k):
    '''Does the numerical Integration if there is a Barrier.
    '''
    return quad(intergand.integrand_barrier_c, lb, ub,
                args=(dy, x0, x1, nbh, L, k))[0]  # Returns only the Integration, not the uncertainty  

def numerical_integration_barrier_mr(arg):
    '''Does the numerical Integration; so that it works with one argument.
    Needed for Parallelization'''
    return numerical_integration_barrier(*arg)
        
    

# Factory Method that produces the Kernel:
def fac_kernel(kernel_type):
    '''Options DiffusionBarrierK, DiffusionK, 
    '''
    if kernel_type == "DiffusionBarrierK":
        return DiffusionBarrierK()
    
    elif kernel_type == "DiffusionBarrierK0":
        return DiffusionBarrierK0()
    
    elif kernel_type == "DiffusionK":
        return DiffusionK()
    
    elif kernel_type == "DiffusionK0":
        return DiffusionK0()
    
    elif kernel_type == "RBFBarrierK":
        return RBFBarrierK()
    
    else:
        raise Exception('Give a valid Kernel - you idiot.')
    
    
#################################################################################
# Test the factory method and everything.
def kernel_test():
    '''Method to test the Kernel'''
    kc = fac_kernel("DiffusionBarrierK")
    kc.set_parameters([0, 1.0, 1.0, 0.001, 5.0])  # k, Diff, t0, mu, dens
    k0 = fac_kernel("DiffusionK")
    k0.set_parameters([1.0, 1.0, 0.001, 5.0, 0.0])  # Diffusion; t0; mutation; density; ss 
    
    print("Parameters Barrier: ")
    print(kc.give_parameter_names())
    print(kc.give_parameters())
    
    print("Parameters No Barrier: ")
    print(k0.give_parameter_names())
    print(k0.give_parameters())
    mu = k0.give_parameters()[2]  # Set Mutation Rate
    # dens = k0.give_parameters
    
    # x_vec = np.logspace(-2, 2.0, 100) + 2.0
    x_vec = np.linspace(1.0, 10, 100)
    y_vec = [kc.num_integral_barrier(0, -1, -1 + x1) for x1 in x_vec]  # 0 Difference along the y-Axis ; 
    y_vec2 = [kc.num_integral_barrier(0, 1, 1 + x1) for x1 in x_vec]  # 0 Difference along the y-Axis ; 
     
    y_vec01 = np.array([k0.num_integral(r) for r in x_vec])  # Numerical Integral no barrier
    # y_vec1=np.array([num_integral(x, t0=1, sigma=sigma, mu=mu) for x in x_vec])
    # y_vec20=np.array([num_integral(x, t0=2, sigma=sigma, mu=mu) for x in x_vec])
    
    y_bessel = 1 / (4 * np.pi * 5) * kv(0, np.sqrt(2 * mu) * x_vec)  # The numerical Result from Mathematica
    
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_vec, y_vec01, label="Numerical Integral no barrier")
    plt.plot(x_vec, y_bessel, alpha=0.8, label="Bessel Decay K0")
    plt.plot(x_vec, y_vec, alpha=0.8, label="Num. Integral DS")
    plt.plot(x_vec, y_vec2, alpha=0.8, label="Num. Integral SS")
    
    # plt.xscale("log")
    plt.legend()
    plt.show()
    
def test_diffusion_barrier():
    '''Method to test Diffusion across a barrier from the numerical Integration'''
    kc = fac_kernel("DiffusionBarrierK")
    kc.set_parameters([0, 2.0, 1.0, 0.01, 5.0])  # k, Diff, t0, mu, dens
    k0 = fac_kernel("DiffusionBarrierK0")
    k0.set_parameters([0.0, 1.0 * 2.0, 4 * np.pi * 5 * 2.0, 0.02 / 2.0, 0.0])  # k', t0, nbh, L, ss
    
    delta_y = 0  # Some Parameters to play around with
    x0 = -5
    x_vec = np.linspace(-20, 20, 200)
    
    # y_vec = [kc.num_integral_barrier(delta_y, x0, x1) for x1 in x_vec]  # 0 Difference along y-Axis
    y_vec0 = [k0.num_integral_barrier(delta_y, x0, x1) for x1 in x_vec]  # 0 Difference along y-Axis
    
    
    kc.set_parameters([0.5, 2.0, 1.0, 0.01, 5.0])  # k, Diff, t0, mu, dens; set weaker barrier
    k0.set_parameters([0.5 / 2.0, 1 * 2.0, 4 * np.pi * 5 * 2.0, 0.02 / 2.0, 0.0])
    # y_vec1 = [kc.num_integral_barrier(delta_y, x0, x1) for x1 in x_vec]  # 0 Difference along y-Axis
    y_vec10 = [k0.num_integral_barrier(delta_y, x0, x1) for x1 in x_vec]  # 0 Difference along y-Axis
    
    kc.set_parameters([10.0, 2.0, 1.0, 0.01, 5.0])  # k, Diff, t0, mu, dens; set weaker barrier
    k0.set_parameters([10.0 / 2.0, 1 * 2.0, 4 * np.pi * 5 * 2.0, 0.02 / 2.0, 0.0])
    # y_vec2 = [kc.num_integral_barrier(delta_y, x0, x1) for x1 in x_vec]   # 0 Difference along y-Axis
    y_vec20 = [k0.num_integral_barrier(delta_y, x0, x1) for x1 in x_vec]  # 0 Difference along y-Axis
    
    
    plt.figure()
    # plt.plot(x_vec, y_vec, label="K=0", alpha=0.8)
    plt.plot(x_vec, y_vec0, label="k0=0", alpha=0.8)
    # plt.plot(x_vec, y_vec1, label="K=0.5", alpha=0.8)
    plt.plot(x_vec, y_vec10, label="K0=0.5", alpha=0.8)
    # plt.plot(x_vec, y_vec2, label="K=10", alpha=0.8)
    plt.plot(x_vec, y_vec20, label="K0=10", alpha=0.8)
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("F / Correlation")
    plt.show()
    

def test_parallel():
    '''Tests old vrs new Kernel'''
    k1 = fac_kernel("DiffusionBarrierK0")
    k1.set_parameters([5, 1.0, 100, 0.005, 0.04])  # k, t0, L, Nbh, ss
    
    # k1 = fac_kernel("DiffusionK0")
    # k1.set_parameters([100, 0.005, 1.0, 0.04])
    # [0.5 / 2.0, 1 * 2.0, 4 * np.pi * 5 * 2.0, 0.02 / 2.0, 0.0]
    coords = [[i, i] for i in np.linspace(-3, 3, 100)]
    
    start = time.time()
    res1 = k1.calc_kernel_mat_old(coords)
    end = time.time()
    print("Runtime Old: %.6f" % (end - start))
    
    start = time.time()
    res2 = k1.calc_kernel_mat(coords)
    end = time.time()
    print("Runtime New: %.6f" % (end - start))
    print("Max. Diff.: %.8f" % np.max(res2 - res1))
    
    
def plot_samples():
    '''Draws from Kernel and plots samples'''
    mean = 0
    kc = fac_kernel("DiffusionK0")
    kc.set_parameters([1000, 0.001, 1.0, 0.04])
    position_list = np.array([(500 + i, 500 + j) for i in range(-19, 20, 2) for j in range(-25, 25, 2)])
    draws = kc.draw_from_kernel(mean, position_list)
    
    # Plots all samples:
    plt.figure()
    plt.scatter(position_list[:, 0], position_list[:, 1], c=draws)
    plt.colorbar()
    plt.show()
    
    # Plot all samples in a line to see whether there is correlated Deviation from Mean
    plt.figure()
    plt.plot(draws, 'ro')
    plt.show()
    
def timer_kernel_diff(nr_runs):
    position_list = np.array([(500 + i, 500 + j) for i in range(-19, 0, 4) for j in range(-25, 0, 4)])
    kc = fac_kernel("DiffusionBarrierK0")
    kc.set_parameters([0.0, 1.0 * 2.0, 4 * np.pi * 5 * 2.0, 0.02 / 2.0, 0.0])  # k', t0, nbh, L, ss
    print("Calculating Kernel...")
    run_times = [kc.time_kernel(position_list) for _ in xrange(nr_runs)]
    
    print(run_times)
    print("Mean Runtime: %.6f" % np.mean(run_times))
    

if __name__ == "__main__":
    # kernel_test()
    test_parallel()  # Tests; but also has timer
    # test_diffusion_barrier()
    # plot_samples()
    
    # Time the Kernel
    # position_list = np.array([(500 + i, 500 + j) for i in range(-19, 20, 2) for j in range(-25, 25, 2)])
    # timer_kernel_diff(1)
    # cProfile.run('timer_kernel_diff(1)')
    
  
