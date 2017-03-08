'''
Created on 29.11.2016
Class for doing and handling the Tensorflow analysis.
Use Laplace approximation for Likelihood
@author: hringbauer
'''
x_data_string = './coordinates1.csv'  # Where to find the data
y_data_string = './data_genotypes1.csv'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cPickle as pickle

# Sections for Kernels:  
def k_rbf(a, l , X, N):
    return a * tf.exp(-tf.reduce_sum(((X[:, None] - X[None, :]) ** 2) / (2 * l ** 2),
                                        reduction_indices=[2])) + 0.0001 * tf.eye(N, dtype=tf.float64)  # Calculate Matrix of Covariances
                                          
def k_classical(a, l, X, N):
    return a - (1 / l) * tf.log(tf.reduce_sum(((X[:, None] - X[None, :]) ** 2), 
                                        reduction_indices=[2])**(0.5)) + 0.0001 * tf.eye(N, dtype=tf.float64)  # Classical IBD-Kernel (Short Scales)
                                          
##########################################################################################################################################                                          

class TF_Analysis(object):
    '''This is the class that handles the Tensor-Flow analysis.
    It has custom methods to run the analysis'''
    x_data = []  # x-Data Geographical positions. In the form of Nr. of Individuals x 2
    y_data = []  # y-Data: In the form of Nr. of Individuals x Nr. of loci
    true_val = [10, 0.15]
    nr_inds = 100  # Nr of Individuals which one wants to initialize
    

    def __init__(self):  # Initialize the Analysis Object
        self.load_data_from_path()  # In case there is not
        
    def load_data_from_path(self):
        '''Method that loads data'''
        self.x_data = np.loadtxt(x_data_string, delimiter='$').astype('float64')[:self.nr_inds, :]  # Only load first 400 entries
        self.y_data = np.loadtxt(y_data_string, delimiter='$').astype('float64')[:self.nr_inds, :]  # Load 400
        # self.x_data /= np.max(self.x_data)  # Divide by the maximum WHY ALEX WHY??
        self.y_data[self.y_data < 0.5] = -1.0  # Set y-data to -1
        assert(len(self.x_data) == len(self.y_data))
        N = self.x_data.shape[0]
        k = self.x_data.shape[1]
        print("Dim. of X-data: %i x %i" % (N, k))
        M1 = self.y_data.shape[0]
        M = self.y_data.shape[1]
        print("Dim. of Y-data: %i x %i" % (M1, M))

    def print_word(self, string):
        '''Test method that prints a test string'''
        print(string)
        
    def set_kernel(self):
        '''Method to set the kernel'''
        print("ToDo")
            
            
#     def calc_likelihood(self, params):
#         '''Calculates the likelihood of a given model.'''
#         # self.load_data_from_path(0, 0)  # Load Data
#         # self.set_model()
#         ###################################################
#         with tf.device('/cpu:0'):
#             X = tf.Variable(dtype=tf.float64, initial_value=self.x_data, trainable=False)  # Set x-Values
#             Y = tf.Variable(dtype=tf.float64, initial_value=self.y_data, trainable=False)  # Set y-Values
#             F = tf.Variable(dtype=tf.float64, initial_value=np.random.normal(0.0, 0.1, np.shape(self.y_data)).astype('float64'),
#                             trainable=True)  # Set place-holder for latent Variables.
#             
#          
#             l = tf.placeholder(shape=[], dtype=tf.float64)
#             a = tf.placeholder(shape=[], dtype=tf.float64)
#              
#             N = np.size(self.x_data, axis=0)
#             eye = tf.eye(N, dtype=tf.float64)
#          
#             K = a * tf.exp(-tf.reduce_sum(((X[:, None] - X[None, :]) ** 2) / (2 * l ** 2),
#                                           reduction_indices=[2])) + 0.0001 * tf.eye(N, dtype=tf.float64)  # Calculate Matrix of Covariances
#                
#             p = tf.nn.sigmoid(F)  # Calculate probabilities(assuming sigmoid link function)
#             Kinv_F = tf.matrix_solve(K, F)  # Calculate K^(-1)*F
#              
#             data_fit = -tf.reduce_sum(tf.log(1 + tf.exp(-Y * F)), reduction_indices=[0])  # Calculate Data Fit: Check  
#             prior_fit = -0.5 * tf.reduce_sum(F * Kinv_F, reduction_indices=[0])  # Calcualte 1st term of prior probability: Check
#             logL = tf.reduce_sum(data_fit + prior_fit)  # Modulo terms not depending on f: Check
#             g = (0.5 * (Y + 1) - p) - Kinv_F  # Calculate Matrix for Gradients Check
#             W = p * (1 - p)  # Calculate first part of Hessia. It is diagonal (but here diagonal in every column)
#             lhs = -K[:, :, None] * W[None, :, :] - tf.eye(N, dtype=tf.float64)[:, :, None]  # Check (HEADACHE)
#             # lhs = -K[:, :, None] * W[None, :, :] - eye[:, :, None] 
#             rhs = tf.matmul(K, g)
#             update = tf.matrix_solve(tf.transpose(lhs, [2, 0, 1]), tf.transpose(rhs, [1, 0])[:, :, None])[:, :, 0]
#             update = tf.transpose(update)
#             opt_op = F.assign(F - update)
#              
#             #####
#              
#             B = (W[:, None, :] ** 0.5) * K[:, :, None] * (W[None, :, :] ** 0.5) + tf.eye(N, dtype=tf.float64)[:, :, None]  # Check 
#             det = tf.reduce_sum(tf.log(tf.matrix_diag_part(tf.cholesky(tf.transpose(B, [2, 0, 1])))))  # Factor of 2??
#             margL = logL - det
#             print("Model Set Up")
#             
#             ###################################################
#          
#         config = tf.ConfigProto()
#         config.gpu_options.per_process_gpu_memory_fraction = 0.01
#         
#         # The actual run of the Likelihood:
#         aa, ll = params[0], params[1]  # Set the parameters
# 
#         with tf.Session(config=config) as sess:
#             sess.run(tf.global_variables_initializer())
#             
#             prev = None
#             for i in range(20):  # Outer Loop.
#                 r = sess.run([opt_op, update, logL, margL], {a: aa, l: ll})
#                 if prev and np.abs(prev - r[-1]) < 0.01:
#                     break
#                 prev = r[-1]
#         print(r[-1])
#         return(r)  # Returns the result
    
    def set_tf_model(self):
        '''Sets the likelihood.'''
        with tf.device('/cpu:0'):
            X = tf.Variable(dtype=tf.float64, initial_value=self.x_data, trainable=False)  # Set x-Values
            Y = tf.Variable(dtype=tf.float64, initial_value=self.y_data, trainable=False)  # Set y-Values
            F = tf.Variable(dtype=tf.float64, initial_value=np.random.normal(0.0, 0.1, np.shape(self.y_data)).astype('float64'),
                            trainable=True)  # Set place-holder for latent Variables.
            
         
            l = tf.placeholder(shape=[], dtype=tf.float64)
            a = tf.placeholder(shape=[], dtype=tf.float64)
             
            N = np.size(self.x_data, axis=0)
            eye = tf.eye(N, dtype=tf.float64)
         
            # K = a * tf.exp(-tf.reduce_sum(((X[:, None] - X[None, :]) ** 2) / (2 * l ** 2),
            #                              reduction_indices=[2])) + 0.0001 * tf.eye(N, dtype=tf.float64)  # Calculate Matrix of Covariances
            
            K = k_rbf(a, l , X, N)  # Test
            #K = k_classical(a, l , X, N)
            
            p = tf.nn.sigmoid(F)            # Calculate probabilities(assuming sigmoid link function)
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
            det = tf.reduce_sum(tf.log(tf.matrix_diag_part(tf.cholesky(tf.transpose(B, [2, 0, 1])))))  # Factor of 2??
            margL = logL - det
            
            config = tf.ConfigProto()  # Sets up configuration
            config.gpu_options.per_process_gpu_memory_fraction = 0.01
            
            print("Model Set Up")
            return((config, update, opt_op, logL, margL, a, l, F))
            ###################################################
        
    def run_ll(self, params):
        '''Method that actually runs the Model (until convergence)'''
        aa, ll = params[0], params[1]  # Set the parameters

        (config, update, opt_op, logL, margL, a, l, F) = self.set_tf_model()

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            prev = None
            for i in range(20):  # Outer Loop.
                r = sess.run([F, opt_op, update, logL, margL], {a: aa, l: ll})
                if prev and np.abs(prev - r[-1]) < 0.01:  # Convergence Criteria
                    break
                prev = r[-1]
        return(r)  # Returns the result
          
    def set_kernel(self):
        '''Function to set the kernel'''
        print("ToDo")
                        
    def generate_likelihood_surface(self, a_list, l_list):
        '''Generates a likelihood surface'''
        res = []
        j = 0
        for aa in a_list:
            for ll in l_list:
                print("Doing run: %i" % j)
                j += 1                                            
                r = self.run_ll([aa, ll])  # Changed to global_variables_initializer
                print("Marginal Likelihood: %.4f" % r[-1])
                res.append(r[-1])  # Append result
        
        pickle.dump(res, open("./test.p", "wb"))  # Pickle the data
        print("Results successfully saved!")
        return res
        
    def plot_likelihood_surface(self, a_list, l_list):
        '''Plots the Log Likelihood Surface'''
        res = pickle.load(open("./test.p", "rb"))  # Loads the data
        print(res)
        
        surface = np.array(res).reshape((10, 10))
        
        plt.figure()
        levels = np.arange(max(res) - 30, max(res) + 1, 2)  # Every two likelihood units
        ax = plt.contourf(l_list, a_list, surface, alpha=0.9, levels=levels)
        
        # plt.clabel(ax, inline=1, fontsize=10)
        plt.colorbar(ax, format="%i")
        plt.title("Log Likelihood Surface", fontsize=20)
        plt.xlabel("l", fontsize=20)
        plt.ylabel("a", fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(self.true_val[0], self.true_val[1], 'ko', linewidth=5, label="True Value")
        plt.legend()
        plt.show()

             
if __name__ == '__main__':
    tf1 = TF_Analysis()  # Always do this two steps
    a_list = np.logspace(-1.5, 0, 10)
    l_list = np.logspace(0, 2, 10)
    
    # tf1.generate_likelihood_surface(a_list, l_list)
    # tf1.plot_likelihood_surface(a_list, l_list)
    
    params = [0.1, 25]  # [a, l]
    #tf1.calc_likelihood(params)
    r=tf1.run_ll(params)
    print(r[-1])

#########################################################################################

                                          


    
            
        
