import numpy as np
from scipy.special import erfc


def integrand_c(t, r, nbh, L):
    '''Integrand for numerical integration'''
    res = 1 / (2 * t * nbh) * np.exp( - (r ** 2) / (4 * t) - L * t)
    return res

def gaussian(t, y, x):
    '''The normal thing without a barrier'''
    return np.exp(-(x - y) ** 2 / (4 * t)) / np.sqrt(4 * np.pi * t)  # New one t->D*t


def integrand_barrier_c(t, dy, x0, x1, nbh, L, k):
    '''Calculate numerically what the identity 
    due to shared ancestry should be. 
    dy: Difference along y-Axis
    x0: Starting point on x-Axis 
    x1: Ending point on x-Axis
    Integrate from t0 to Infinity'''  

    if x0 < 0:  # Formulas are only valid for x0>0; but simply flip if otherwise!
        x0 = -x0
        x1 = -x1

    #######################################################################################
    if x1 > 0:  # Same side of Barrier
        '''The integrand for cases of different sides on the barrier.
        Product of 1d Gaussian along y-Axis
        And a term for the long - distance migration.
        Everything transformed so that t->Dt and one gets coordinate independent constants'''
        # 1D Contribution from Gaussian along y-axis
        pre_fac = 1.0 / nbh * np.sqrt(np.pi / t)  # Prefactor from first Gaussian
        gaussiany = pre_fac * np.exp(-dy ** 2 / (4 * t))
        
        # 1D Contribution from x-Axis (barrier)
        exponent = 2 * k * (x0 + x1 + 2 * k * t)
        if exponent > 700:
            pdfx = gaussian(t, x0, x1)  # Fall back to Gaussian (to which one converges)

        else:
            n1 = np.exp(-(x0 - x1) ** 2 / (4 * t)) + np.exp(-(x0 + x1) ** 2 / (4 * t))
            d1 = np.sqrt(4 * np.pi * t)
            
            a2 = k * np.exp(exponent)
            b2 = erfc((x0 + x1 + 4 * k * t) / (2 * np.sqrt(t)))
            pdfx = n1 / d1 - a2 * b2
        
        #if np.isnan(pdfx) or np.isinf(pdfx):  # Check if numerical instability
        #    pdfx = gaussian(t, x0, x1)  # Fall back to Gaussian (to which one converges)
        
        # Mutation/Long Distance Migration:
        mig = np.exp(-L * t)  # Long Distance Migrationg
        res = gaussiany * pdfx * mig  # Multiply everything together
        

    #######################################################################################


    elif x1 < 0:  # Different side of Barrier
        # 1D Contribution from Gaussian along y-axis
        pre_fac = 1.0 / nbh * np.sqrt(np.pi / t)  # Prefactor from first Gaussian
        gaussiany = pre_fac * np.exp(-dy ** 2 / (4.0 * t))
        
        # 1D Contribution from x-Axis (barrier)
        exponent = 2 * k * (x0 - x1 + 2 * k * t)

        if exponent>700:
            pdfx = gaussian(t, x0, x1)        # Fall back to Gaussian (to which one converges)

        else:
            a1 = k * np.exp(exponent)  # First Term Barrier
            b1 = erfc((x0 - x1 + 4 * k * t) / (2 * np.sqrt(t)))  # Second Term Barrier
            pdfx = a1 * b1
        
        #if np.isnan(pdfx) or np.isinf(pdfx):  # Check if numerical instability
        #    pdfx = gaussian(t, x0, x1)        # Fall back to Gaussian (to which one converges)
        
        # Mutation/Long Distance Migration:
        mig = np.exp(-L * t)            # Long Distance Migration
        res = gaussiany * pdfx * mig    # Multiply everything together

    return res





