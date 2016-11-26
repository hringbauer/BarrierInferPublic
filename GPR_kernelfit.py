'''
Created on Oct 24, 2016
Class that is used for fitting Gaussian Kernels.
This uses an implementation from Scikit that uses the 
Laplace approximation to expand around the posterior
@author: hringbauer
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from time import time



class GPR_kernelfit(object):
    '''
    Class that is used for fitting Gaussian Kernels.
    This uses an implementation from Scikit that uses the 
    Laplace approximation to expand around the posterior
    '''

    X = []  # x-data
    y = []  # y-data
    kernel = 0.1 * RBF([25.0]) # Implicitly sets the initial values
    fit = 0


    def __init__(self, x, y, kernel=0):
        '''
        Constructor
        '''
        self.X = x
        self.y = y
        if kernel != 0:
            self.kernel = kernel
            
    def run_fit(self):
        '''Runs a fit for the Gaussian Process Classifier'''
        print("Starting Run...")
        print(self.X)
        print(self.y)
        start = time()
        self.fit = GaussianProcessClassifier(kernel=self.kernel).fit(self.X, self.y)
        print("Run time: %.4f" % (time() - start))
        print("Theta-Estimates: ")
        print(self.fit.kernel_)
        print(self.fit.kernel_.theta)
        print("Finished Run!!")
        
        
    def show_fit(self):
        '''Visualizes the fit'''  
        # create a mesh to plot in
        h = 5
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Put the result into a color plot
        Z = self.fit.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        
        plt.figure()
        plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")
        image = plt.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)  # @UndefinedVariable
        contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,  # @UnusedVariable
                           linetypes='--')
        plt.colorbar(image)
        
        # Plot also the training points
        plt.scatter(self.X[:, 0], self.X[:, 1], s=30, c=self.y, cmap=plt.cm.Paired)  # @UndefinedVariable
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title("%s, LML: %.3f" % 
                  ("Log Likelihood Surface", self.fit.log_marginal_likelihood(self.fit.kernel_.theta)))
    
        plt.show() 
        
    def plot_loglike_sf(self):
        '''Plots the log likelihood surface'''
        theta0 = np.logspace(-2, 0, 20)  # Manually set values
        theta1 = np.logspace(0, 2, 20)    # Manually set values
        Theta0, Theta1 = np.meshgrid(theta0, theta1)
        LML = [[self.fit.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]]))
                for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
        LML = np.array(LML).T
        plt.plot(np.exp(self.fit.kernel_.theta)[0], np.exp(self.fit.kernel_.theta)[1],
                 'ko', zorder=10)
        plt.plot(np.exp(self.fit.kernel_.theta)[0], np.exp(self.fit.kernel_.theta)[1],
                 'ko', zorder=10)
        plt.pcolor(Theta0, Theta1, LML)
        plt.xscale("log")
        plt.yscale("log")
        plt.colorbar()
        plt.xlabel("Magnitude")
        plt.ylabel("Length-scale")
        plt.title("Log-marginal-likelihood")
        plt.show()
        
    