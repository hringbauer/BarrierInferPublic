
# coding: utf-8

# ## Classes for Kernels
# Classes for Covariance Kernels. They all inherit from one abstract class, which contains methods they all should have.
# Also has a factory method, which can be used for producing the instance of the wished class.

# In[111]:

import numpy as np
from scipy.special import kv as kv  # Import Bessel Function
from scipy.integrate import quad  # Import function to do the integral
from scipy.special import erfc
from time import time
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib notebook')


# In[49]:

class Kernel(object):
    '''The Class for the Kernels.
    At least should have a calc_kernel method that calculates the 
    Covariance Matrix '''
    nr_parameters = 0  # The Number of Parameters to optimize
    
    def __init__(self):
        print("Sets the parameters of the respective Kernel")
    
    def calc_kernel_mat(self, coords):
        print("Not Implemented: Implement this you lazy shit.")
        
    def set_parameters(self):
        print("Not Implemented: Method to set the parameters. Takes list as input")
        
    def give_nr_parameters(self):
        print("Not Implemented: Method that returns number of Parameters")
        
    def give_parameter_names(self):
        print("Not Implemented: Gives List of Names of Parameters")
        
    def give_parameters(self):
        print("Not Implemented: Returns List of Parameters Values")


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
    loc_rate = 1  # Local density; giving the rate of local coalescence.
    
    def __init__(self, k=0.5, D=1, t0=1.0, mu=0.001, loc_rate=1):
        '''Initialize to set desired values!'''
        self.k = k  # Sets the constaints to their required values
        self.D = D
        self.t0 = t0
        self.mu = mu
        self.loc_rate = loc_rate
        
    def set_parameters(self, params=[0.5, 1.0, 0.001, 1, 1]):
        assert(len(params) == self.nr_parameters)  # Check whether right length
        self.k = params[0]
        self.D = params[1]
        self.t0 = params[2]
        self.mu = params[3]
        self.loc_rate = params[4]
        
    def give_nr_parameters(self):
        return(5)
    
    def give_parameter_names(self):
        return(["k", "D", "mu", "t0", "loc_rate"])
        
    def give_parameters(self):
        return([self.k, self.D, self.t0, self.mu, self.loc_rate]) 
    
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
            return self.loc_rate * quad(self.integrand_barrier_ss,
                self.t0, np.inf, args=(dy, x0, x1))[0] 
        
        if x1 < 0:  # Different side of Barrier
            return self.loc_rate * quad(self.integrand_barrier_ds,
                self.t0, np.inf, args=(dy, x0, x1))[0]
   
    def calc_kernel_mat(self, coords):
        '''Given List of Coordinates; calculate the full covariance Kernel'''
        kernel_mat = [[self.num_integral_barrier(i[1] - j[1],
                            i[0], j[0]) for i in coords] for j in coords]
        
        K1 = 0.0000001 * np.eye(len(coords))  # Add Identity Matrix to make numerically stable
        return np.array(kernel_mat) + K1


# In[86]:

class DiffusionK(Kernel):
    '''A whole class which is designed to 
    calculate covariance kernels from the simple Diffusion Model
    It assumes Diffusion along x- and y- axis; with a Barrier at x=0
    Numerically integrates what f should be.
    Throughout y is the STARTING point and x the end point'''
    # Parameters of the Covariance
    nr_parameters = 4
    D = 1.0  # Diffusion Constant; equals sigma**2. Sets how quickly stuff decays
    t0 = 1  # Starting Point of Integration; i.e. where to start integration. Sets the minimum "local" scale.
    mu = 0.001  # Long Distance/Migration rate; sets the max scale of identity
    density = 1.0  # Density of Diploids
    
    def __init__(self, D=1, t0=1.0, mu=0.001, density=1.0):
        '''Initialize to set desired values!'''
        self.D = D
        self.t0 = t0
        self.mu = mu
        self.density = density
        
    def set_parameters(self, params=[1.0, 0.001, 1, 1]):
        assert(len(params) == self.nr_parameters)  # Check whether right length
        self.D = params[0]
        self.t0 = params[1]
        self.mu = params[2]
        self.density = params[3]
        
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
        return kernel + K1



class DiffusionK0(Kernel):
    '''Dimensionless Diffusion Kernel:
    A whole class which is designed to 
    calculate covariance kernels from the simple Diffusion Model
    It assumes Diffusion along x- and y- axis; with a Barrier at x=0
    Numerically integrates what f should be.
    Throughout y is the STARTING point and x the end point'''
    # Parameters of the Covariance
    nr_parameters = 3
    nbh = 100.0  # Neighborhood Size: 4 pi De D
    L = 0.002  # 2mu/D
    t0 = 1  # Starting Point of Integration; i.e. where to start integration. Sets the minimum "local" scale.
    
    def __init__(self, nbh=100, L=0.002, t0=1):
        '''Initialize to set desired values!'''
        self.nbh = nbh
        self.L = L
        self.t0 = t0
        
    def set_parameters(self, params=[100, 0.002, 1]):
        '''Nbh, L, t0'''
        assert(len(params) == self.nr_parameters)  # Check whether right length
        self.nbh = params[0]
        self.L = params[1]
        self.t0 = params[2]
        
    def give_nr_parameters(self):
        return(3)
    
    def give_parameter_names(self):
        return(["Neighborhood Size", "L", "t0"])
        
    def give_parameters(self):
        return([self.nbh, self.L, self.t0]) 
    
    def integrand(self, t, r):
        '''Transformed integrand t'= D t'''
        res = 1 / (2 * t * self.nbh) * np.exp(-(r ** 2) / (4 * t) - self.L * t)
        return res
        
    def num_integral(self, r):
        '''The transformed Integral'''
        return quad(self.integrand,
            self.t0, np.inf, args=(r))[0]         
        
    def calc_kernel_mat(self, coords):
        '''Given List of Coordinates; calculate the full covariance Kernel'''
        coords = np.array(coords)  # Make Coordinates a Numpy Array!
        dist_mat = np.sqrt(np.sum((coords[:, None] - coords[None, :]) ** 2, axis=2))  # First set up the Distance Matrix
        
        num_integral_v = np.vectorize(self.num_integral)  # Vectorizes the integral; maybe later parallelize
        kernel = num_integral_v(dist_mat)  # Calculate the kernel via vectorized function  
        K1 = 0.0000001 * np.eye(len(coords))  # Add Identity Matrix to make numerically stable
        return kernel + K1
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


# In[112]:

# Factory Method that produces the Kernel:
def fac_kernel(kernel_type):
    '''Options DiffusionBarrierK, DiffusionK, 
    '''
    if kernel_type == "DiffusionBarrierK":
        return DiffusionBarrierK()
    
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
    k0 = fac_kernel("DiffusionK")
    k0.set_parameters([1.0, 1.0, 0.001, 5.0])  # Diffusion; t0, mutation, density 
    
    print("Parameters Barrier: ")
    print(kc.give_parameter_names())
    print(kc.give_parameters())
    
    print("Parameters No Barrier: ")
    print(k0.give_parameter_names())
    print(k0.give_parameters())
    mu = k0.give_parameters()[2]  # Set Mutation Rate
    # dens = k0.give_parameters
    
    x_vec = np.logspace(-0.1, 2.0, 100) + 2
    y_vec = [kc.num_integral_barrier(0, -1, -1 + x1) for x1 in x_vec]  # 0 Difference along the y-Axis ; 
    y_vec2 = [kc.num_integral_barrier(0, 1, 1 + x1) for x1 in x_vec]  # 0 Difference along the y-Axis ; 
     
    
    y_vec01 = np.array([k0.num_integral(r) for r in x_vec])  # Numerical Integral no barrier
    # y_vec1=np.array([num_integral(x, t0=1, sigma=sigma, mu=mu) for x in x_vec])
    # y_vec20=np.array([num_integral(x, t0=2, sigma=sigma, mu=mu) for x in x_vec])
    
    y_bessel = 1 / (4 * np.pi * 5) * kv(0, np.sqrt(2 * mu) * x_vec)  # The numerical Result from Mathematica
    
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_vec, y_vec01, label="Numerical Integral no barrier")
    plt.plot(x_vec, y_bessel, alpha=0.8, label="Bessel Decay K0")
    # plt.plot(x_vec,y_vec, alpha=0.8, label="Num. Integral DS")
    # plt.plot(x_vec,y_vec2, alpha=0.8, label="Num. Integral SS")
    
    plt.xscale("log")
    plt.legend()
    plt.show()

# kernel_test()
# k0 = fac_kernel("DiffusionK")
# k0.set_parameters([2.0, 2.0, 0.001, 5.0])  # Diffusion; t0, mutation, density
# 
# 
# k1 = fac_kernel("DiffusionK0")
# k1.set_parameters([4 * np.pi * 2.0 * 5, 0.002 / 2.0, 2 * 2.0])
# start = time()
# 
# k0.num_integral(15)
# end = time()
# print(end - start)
# start = time()
# k1.num_integral(15)
# end = time()
# print(end - start)
#  
# print(k0.num_integral(0.1))
# print(k1.num_integral(0.1))
