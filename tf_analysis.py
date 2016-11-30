'''
Created on 29.11.2016
Class for doing and handling the Tensorflow analysis.
Use Laplace approximation for Likelihood
@author: hringbauer
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cPickle as pickle

path1 = '../Data/SNP_data2014.p'  # SNP-data from 2014

class TF_Analysis(object):
    '''This is the class that handles the Tensor-Flow analysis.
    It has custom methods to run the TF-analysis'''
    x_data = []  # x-Data Geographical positions. In the form of Nr. of Individuals x 2
    y_data = []  # y-Data: In the form of Nr. of Individuals x Nr. of loci

    def __init__(self, x_data=[], y_data=[]):  # Initialize the Analysis Object
        if np.size(x_data>0):
            self.x_data = x_data
            self.y_data = y_data
        else:
            self.x_data = np.loadtxt('./coordinates2.csv', delimiter='$').astype('float64')[:,:]
            self.y_data = np.loadtxt('./data_genotypes2.csv', delimiter='$').astype('float64')[:, :]
        
    def load_data_from_path(self, x_data_string, y_data_string):
        '''Method that loads data'''
        self.x_data = np.loadtxt('./coordinates2.csv', delimiter='$').astype('float64')[:, :]  # Only load first 400 entries
        self.y_data = np.loadtxt('./data_genotypes2.csv', delimiter='$').astype('float64')[:, :]  # Load 400
        # self.x_data /= np.max(self.x_data)  # Divide by the maximum WHY??
        
    def print_word(self, string):
        '''Test method that prints a test string'''
        print(string)
        
    def set_kernel(self):
        '''Method to set the kernel'''
        print("ToDo")
        
        
    def set_model(self):
        '''Method that calculates Likelihood.'''
        with tf.device('/cpu:0'):
            X = tf.Variable(dtype=tf.float64, initial_value=self.x_data, trainable=False)  # Set x-Values
            Y = tf.Variable(dtype=tf.float64, initial_value=self.y_data, trainable=False)  # Set y-Values
            F = tf.Variable(dtype=tf.float64, initial_value=np.random.normal(0.0, 0.1, np.shape(self.y_data)).astype('float64'),
                            trainable=True)  # Set place-holder for latent Variables.
        
            l = tf.placeholder(shape=[], dtype=tf.float64)
            a = tf.placeholder(shape=[], dtype=tf.float64)
            
            N = np.size(self.x_data, axis=0)
            eye = tf.eye(N, dtype=tf.float64)
        
            K = a * tf.exp(-tf.reduce_sum(((X[:, None] - X[None, :]) ** 2) / (2 * l), reduction_indices=[2])) + 0.0001 * tf.eye(N, dtype=tf.float64)  # Calculate Matrix of Covariances
            
            # K = a * tf.exp(-tf.reduce_sum(((X[:, None] - X[None, :]) ** 2) / (2 * l), reduction_indices=[2])) + \
               # 0.0001 * eye                
        
            p = tf.nn.sigmoid(F)  # Calculate probabilities(assuming sigmoid link function)
            Kinv_F = tf.matrix_solve(K, F)  # Calculate K^(-1)*F
            
            data_fit = -tf.reduce_sum(tf.log(1 + tf.exp(-Y * F)), reduction_indices=[0])  # Calculate Data Fit: Check  
            prior_fit = -0.5 * tf.reduce_sum(F * Kinv_F, reduction_indices=[0])  # Calcualte 1st term of prior probability: Check
            logL = tf.reduce_sum(data_fit + prior_fit)  # Modulo terms not depending on f: Check
            g = (0.5 * (Y + 1) - p) - Kinv_F  # Calculate Matrix for Gradients Check
            W = p * (1 - p)  # Calculate first part of Hessia. It is diagonal (but here diagonal in every column)
            lhs = -K[:, :, None] * W[None, :, :] - tf.eye(N, dtype=tf.float64)[:, :, None]  # Check (HEADACHE)
            # lhs = -K[:, :, None] * W[None, :, :] - eye[:, :, None] 
            rhs = tf.matmul(K, g)
            update = tf.matrix_solve(tf.transpose(lhs, [2, 0, 1]), tf.transpose(rhs, [1, 0])[:, :, None])[:, :, 0]
            update = tf.transpose(update)
            opt_op = F.assign(F - update)
            
            #####
            
            B = (W[:, None, :] ** 0.5) * K[:, :, None] * (W[None, :, :] ** 0.5) + tf.eye(N, dtype=tf.float64)[:, :, None]  # Check
            # B = (W[:, None, :] ** 0.5) * K[:, :, None] * (W[None, :, :] ** 0.5) + eye[:, :, None] 
            det = tf.reduce_sum(tf.log(tf.matrix_diag_part(tf.cholesky(tf.transpose(B, [2, 0, 1])))))  # Factor of 2??
            margL = logL - det
            print("Model Set Up")
            
    def calc_likelihood(self, params):
        '''Calculates the likelihood of a given model.'''
        self.load_data_from_path(0, 0)  # Load Data
        # self.set_model()
        ###################################################
        with tf.device('/cpu:0'):
            X = tf.Variable(dtype=tf.float64, initial_value=self.x_data, trainable=False)  # Set x-Values
            Y = tf.Variable(dtype=tf.float64, initial_value=self.y_data, trainable=False)  # Set y-Values
            F = tf.Variable(dtype=tf.float64, initial_value=np.random.normal(0.0, 0.1, np.shape(self.y_data)).astype('float64'),
                            trainable=True)  # Set place-holder for latent Variables.
         
            l = tf.placeholder(shape=[], dtype=tf.float64)
            a = tf.placeholder(shape=[], dtype=tf.float64)
             
            N = np.size(self.x_data, axis=0)
            eye = tf.eye(N, dtype=tf.float64)
         
            K = a * tf.exp(-tf.reduce_sum(((X[:, None] - X[None, :]) ** 2) / (2 * l ** 2),
                                          reduction_indices=[2])) + 0.0001 * tf.eye(N, dtype=tf.float64)  # Calculate Matrix of Covariances
               
            p = tf.nn.sigmoid(F)  # Calculate probabilities(assuming sigmoid link function)
            Kinv_F = tf.matrix_solve(K, F)  # Calculate K^(-1)*F
             
            data_fit = -tf.reduce_sum(tf.log(1 + tf.exp(-Y * F)), reduction_indices=[0])  # Calculate Data Fit: Check  
            prior_fit = -0.5 * tf.reduce_sum(F * Kinv_F, reduction_indices=[0])  # Calcualte 1st term of prior probability: Check
            logL = tf.reduce_sum(data_fit + prior_fit)  # Modulo terms not depending on f: Check
            g = (0.5 * (Y + 1) - p) - Kinv_F  # Calculate Matrix for Gradients Check
            W = p * (1 - p)  # Calculate first part of Hessia. It is diagonal (but here diagonal in every column)
            lhs = -K[:, :, None] * W[None, :, :] - tf.eye(N, dtype=tf.float64)[:, :, None]  # Check (HEADACHE)
            # lhs = -K[:, :, None] * W[None, :, :] - eye[:, :, None] 
            rhs = tf.matmul(K, g)
            update = tf.matrix_solve(tf.transpose(lhs, [2, 0, 1]), tf.transpose(rhs, [1, 0])[:, :, None])[:, :, 0]
            update = tf.transpose(update)
            opt_op = F.assign(F - update)
             
            #####
             
            B = (W[:, None, :] ** 0.5) * K[:, :, None] * (W[None, :, :] ** 0.5) + tf.eye(N, dtype=tf.float64)[:, :, None]  # Check
            # B = (W[:, None, :] ** 0.5) * K[:, :, None] * (W[None, :, :] ** 0.5) + eye[:, :, None] 
            det = tf.reduce_sum(tf.log(tf.matrix_diag_part(tf.cholesky(tf.transpose(B, [2, 0, 1])))))  # Factor of 2??
            margL = logL - det
            print("Model Set Up")
            
            ###################################################
         
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.01
        
#         with tf.Session(config=config) as sess:
#             sess.run(tf.global_variables_initializer())  # Update by Harald
#             
#             for i in range(3):
#                 sess.run(opt_op, {a: 0.1, l:0.01})
#             r = sess.run(F, {a: 0.1, l: 0.01})
#             print("Run Finished!")
#             print(r)
        # The actual run of the Likelihood:
        aa = params[0]
        ll = params[1]
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            prev = None
            for i in range(20):
                r = sess.run([opt_op, update, logL, margL], {a: aa, l:ll})
                if prev and np.abs(prev - r[-1]) < 0.01:
                    break
                prev = r[-1]
        return(r)  # Returns the result
    
    def set_kernel(self):
        '''Function to set the kernel'''
        print("ToDo")
    
        
                        
    def generate_likelihood_surface(self):
        '''Generates a likelihood surface'''
        a_list = np.linspace(0.05, 0.5, 10)
        l_list = np.linspace(5, 50, 10)

        res = []
        j = 0
        for aa in a_list:
            for ll in l_list:
                print("Doing run: %i" % j)
                j += 1                                            
                r = self.calc_likelihood([aa, ll])  # Changed to global_variables_initializer
                print("Marginal Likelihood: %.4f" % r[-1])
                res.append(r[-1])  # Append result
        
        pickle.dump(res, open("./test.p", "wb"))  # Pickle the data
        print("Results successfully saved!")
        return res
        
    def plot_likelihood_surface(self):
        '''Plots the Log Likelihood Surface'''
        data = pickle.load(open("./test.p", "rb"))  # Loads the data
        
        a_list = np.linspace(0.05, 0.5, 10)  # Sets the Lists
        l_list = np.linspace(5, 50, 10)  # Sets the Lists
        
        x_vec = a_list
        y_vec = l_list
        xv, yv = np.meshgrid(x_vec, y_vec)
        z = np.ceil(data)  # Round up for better plotting   
        # levels = np.arange(max(z) - 30, max(z) + 1, 2)  # Every two likelihood units
        
        plt.figure()
        ax=plt.contourf(xv, yv, z.reshape((len(a_list), len(l_list))), alpha=0.8)
        
        # plt.clabel(ax, inline=1, fontsize=10)
        plt.colorbar(ax, format="%i")
        plt.title("Log Likelihood Surface", fontsize=20)
        plt.xlabel("a", fontsize=20)
        plt.ylabel("l", fontsize=20)
        plt.show()
        
        plt.figure()        #2nd Figure for Comparison
        plt.pcolormesh(a_list, l_list, z.reshape(10, 10))
        plt.colorbar()
        plt.title("Log Likelihood Surface", fontsize=20)
        plt.xlabel("a", fontsize=20)
        plt.ylabel("l", fontsize=20)
        plt.show()
        
             
if __name__ == '__main__':
    tf = TF_Analysis()
    tf.plot_likelihood_surface()
            
        
