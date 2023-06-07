# -------------------------------------------------------------
# Statistical metrics module
#
# Author: Ziad Ghauch
#
# Reference
# https://en.wikipedia.org/wiki/Statistical_distance
# -------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import energy_distance
from scipy.stats import wasserstein_distance
from sklearn import metrics
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde


class StatDist:
    
    ''' Statistical distance class
    
    Class containing methods for computing statistical distance metrics
    between two random variables, two probability distributions, or two 
    samples, or a distance between a sample point and a population. 
    
    Supported distance measures include:
        
        o Energy distance
        o Mahalanobis distance
        o Kullback–Leibler divergence
        o Renyi divergence
        o Variational distance
        o Kantorovich-Rubinstein distance
        o Maximum Mean Discrepancy (MMD) distance
        o Hellinger distance
        o Hilbert-Schmidt Independence Criterion
        
    Reference:
        https://en.wikipedia.org/wiki/Statistical_distance
    '''

    def __init__(self, P, Q, shared_axis, grid=200):
        self.P = P
        self.Q = Q
        self.shared_axis = shared_axis
        self.grid = grid
    
    def set_p(self, P):
        self.P=P
        
    def set_q(self, Q):
        self.Q=Q

    def set_shared_axis(self, shared_axis):
        self.shared_axis=shared_axis

    def set_grid(self, grid):
        self.grid=grid
        
    def get_p(self):
        return self.P
    
    def get_q(self):
        return self.Q
    
    def get_shared_axis(self):
        return self.shared_axis
    
    def get_grid(self):
        return self.grid

    @staticmethod
    def compute_empirical_pdf_grid(x, new_grid):
        ''' Compute empirical PDF of 1D data 
        
        Input
        -----
        x : 1D array of data from which to compute the empirical PDF
        new_grid : new grid on which to evaluate the empirical PDF of x

        Return
        -----
        emp_pdf : empirical PDF at the new_grid

        Example
        -------
        x = np.random.normal(0,1,1000)
        new_grid = np.arange(-3.,3., 0.05)
        compute_empirical_pdf_grid(x, new_grid)
        '''

        assert x.ndim == 1 # 1D array
    
        ecdf = ECDF(x)

        x_new = new_grid
        y_new = ecdf(x_new)

        emp_pdf = np.zeros((new_grid.shape[0]))
        for i in range(1,new_grid.shape[0]):
            emp_pdf[i] = (y_new[i] - y_new[i-1]) / (x_new[i] - x_new[i-1])

        return emp_pdf

    @staticmethod
    def compute_density_kde(x, cov_factor=0.1):
        ''' PDF using KDE   
        
        Compute PDF of 1D data x using Gaussian KDE method
        
        Input
        -----
        x : n-dimensional random vector from distribution P
        
        Returns
        -------
        density : Gaussian kernel density estimate function of x
        '''

        density = gaussian_kde(x)
        density.covariance_factor = lambda:cov_factor
        density._compute_covariance()

        return density


    @staticmethod
    def plot_density_kde(x, grid_len=200):
        ''' Plot data and KDE fit
        
        Plot histogram of random variable along with Gaussian KDE estimate
        
        Input
        -----
        x : n-dimensional random vector from distribution P
        '''
        
        fit_func = StatDist.compute_density_kde(x)
        plot_grid = np.linspace(np.min(x), np.max(x), grid_len)
        fit = fit_func(plot_grid)
        plt.hist(x, density=True)
        plt.plot(plot_grid, fit, linewidth=2)
        plt.legend(['Data','KDE Fit'], loc='best')
        plt.title('Data vs Gaussian KDE fit')
        plt.ylabel('Density')
        
    
class Metrics(StatDist):
    
    def __init__(self, P, Q, shared_axis, grid):
        super().__init__(P, Q, shared_axis, grid)

    def compute_var_distance(self):
        ''' Total Variation distance
        
        Function that computes the Variational distance 
        between two probability distributions P and Q. 
        The variational distance measures the largest possible
        difference between the probabilities that the two 
        probability distributions P and Q can assign to 
        same event. 
    
        Input
        -----
        P : n-dimensional random vector from distribution P
        Q : n-dimensional random vector from distribution Q
        
        Returns
        ------
        Variational distance between distributions P and Q
        '''
        
        assert self.P.shape[1] == self.Q.shape[1]
        assert self.P.shape[1] == 1

        epsilon = 1e-13

        # get density functions:
        d1 = self.compute_density_kde(self.P[:,0])
        d2 = self.compute_density_kde(self.Q[:,0])

        if self.shared_axis == True:
            # combined axis (shared)
            cX = np.concatenate((self.P,self.Q))
            xs = np.linspace(min(cX),max(cX),self.grid)
            tvd = 0.0
            for x in xs:
                p1 = d1(x) + epsilon
                p2 = d2(x) + epsilon
                tvd += np.abs(p1 - p2) 
            return ( 0.5 * tvd )

        elif self.shared_axis == False:
            x1 = np.linspace(min(self.P),max(self.P),self.grid)
            x2 = np.linspace(min(self.Q),max(self.Q),self.grid)
            tvd = 0.0
            for i in range(self.grid):
                p1 = d1(x1[i]) + epsilon
                p2 = d2(x2[i]) + epsilon
                tvd += np.abs(p1 - p2) 
            return ( 0.5 * tvd )

    def compute_kantorovich_distance(self):
        ''' Kantorovich-Rubinstein distance
        
        Function that computes the Kantorovich-Rubinstein distance (aka Wasserstein
        metric) between two probability distributions P and Q.
        
        Input
        -----
        P : n-dimensional random vector from distribution P
        Q : n-dimensional random vector from distribution Q
        
        Returns
        -----
        Kantorovich-Rubinstein distance between distributions P and Q
        '''
        
        assert self.P.shape[1] == self.Q.shape[1]
        assert self.P.shape[1] == 1

        # get density functions:
        d1 = self.compute_density_kde(self.P[:,0])
        d2 = self.compute_density_kde(self.Q[:,0])

        if self.shared_axis == True:
            # combined axis (shared)
            cX = np.concatenate((self.P,self.Q))
            xs = np.linspace(min(cX),max(cX),self.grid)    
            p1 = np.zeros((self.grid,1))
            p2 = np.zeros((self.grid,1))
            for count,x in enumerate(xs):
                p1[count,0] = d1(x) 
                p2[count,0] = d2(x) 
            return (wasserstein_distance(p1.flatten(), p2.flatten()))

        elif self.shared_axis == False:
            x1 = np.linspace(min(self.P),max(self.P),self.grid)
            x2 = np.linspace(min(self.Q),max(self.Q),self.grid)
            p1 = np.zeros((self.grid,1))
            p2 = np.zeros((self.grid,1))
            for i in range(self.grid):
                p1[i,0] = d1(x1[i])
                p2[i,0] = d2(x2[i]) 
            return (wasserstein_distance(p1.flatten(), p2.flatten()))   

    def compute_maha_distance(self):
        ''' Mahalanobis distance
        
        Function that computes Mahalanobis distance between two probability 
        distributions P and Q. If each of these axes is re-scaled to have unit variance, 
        then the Mahalanobis distance corresponds to standard Euclidean distance 
        in the transformed space. The Mahalanobis distance is unitless, 
        scale-invariant, and takes into account correlations of the data set.
        
        Input
        -----
        P : {x_1, x_2, ... , x_n} is a (n,1) random vector from P
        Q : {y_1, y_2, ... , y_n} of size (n,1) random vector from Q

        Returns
        -----
        Mahalanobis distance DM(P,Q) between distributions P and Q
        '''    
        
        assert self.P.shape[1] == self.Q.shape[1]
        assert self.P.shape[1] == 1

        # get density functions:
        d1 = self.compute_density_kde(self.P[:,0])
        d2 = self.compute_density_kde(self.Q[:,0])

        if self.shared_axis == True:
            # combined axis (shared)
            cX = np.concatenate((self.P,self.Q))
            xs = np.linspace(min(cX),max(cX),self.grid)

            diff = np.zeros((self.grid,1))
            p1_cov = np.zeros((self.grid,1))

            for count,x in enumerate(xs):
                p1 = d1(x)
                p2 = d2(x)
                diff[count,0] = (p1 - p2)
                p1_cov[count,0] = p1

            diff = diff.T
            S = np.eye(self.grid)
            mh = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(S) ), diff.T))

            return (mh)

        elif self.shared_axis == False:
            x1 = np.linspace(min(self.P),max(self.P),self.grid)
            x2 = np.linspace(min(self.Q),max(self.Q),self.grid)
            diff = np.zeros((self.grid,1))
            p1_cov = np.zeros((self.grid,1))

            for i in range(self.grid):
                p1 = d1(x1[i])
                p2 = d2(x2[i])
                diff[i,0] = (p1 - p2)
                p1_cov[i,0] = p1

            diff = diff.T
            S = np.eye(self.grid)
            mh = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(S) ), diff.T))

            return (mh)


    def compute_e_distance(self):
        ''' Energy Distance
        
        Function that computes the Energy distance between two samples P and Q.
        The Energy distance represents a statistical distance
        between two probability distributions P and Q.

        Input
        -----
        P : {x_1, x_2, ... , x_n} of size (n,1) 
        Q : {y_1, y_2, ... , y_m} of size (m,1)

        Returns
        -----
        Energy distance E(P,Q) between random variables P and Q
        '''
        
        assert self.P.shape[1] == self.Q.shape[1]

        # get density functions:
        d1 = self.compute_density_kde(self.P[:,0])
        d2 = self.compute_density_kde(self.Q[:,0])

        if self.shared_axis == True:
            cX = np.concatenate((self.P,self.Q))

            # combined axis (shared)
            xs = np.linspace(min(cX),max(cX),self.grid)    
            
            p1 = np.zeros((self.grid,1))
            p2 = np.zeros((self.grid,1))

            for count,x in enumerate(xs):
                p1[count,0] = d1(x) 
                p2[count,0] = d2(x) 

            return (energy_distance(p1.flatten(), p2.flatten()))
        
        elif self.shared_axis == False:
            x1 = np.linspace(min(self.P),max(self.P),self.grid)
            x2 = np.linspace(min(self.Q),max(self.Q),self.grid)

            p1 = np.zeros((self.grid,1))
            p2 = np.zeros((self.grid,1))

            for i in range(self.grid):
                p1[i,0] = d1(x1[i])
                p2[i,0] = d2(x2[i])
            
            return (energy_distance(p1.flatten(), p2.flatten()))

    
    def compute_mmd_distance(self, type_kern='rbf', kernel_par=[]):
        ''' Maximum Mean Discrepancy (MMD) distance
        
        Function that computes Maximum Mean Discrepancy (MMD) distance between 
        two probability distributions P and Q. The MMD is a distance-measure 
        between P and Q which is defined as the squared distance between their
        embeddings in the reproducing kernel Hilbert space
        
        Input
        -----
        P : {x_1, x_2, ... , x_m} of size (m,d), with x_i d-dimensional random vector
        Q : {y_1, y_2, ... , y_n} of size (n,d), with y_j d-dimensional random vector
        type_kern : kernel type, in ['linear', 'rbf', 'polyn']
        Optional Input
        if type=='rbf'
            theta = args[0] : RBF kernel parameter
        
        if type=='polyn'
            order = args[0] : order of polynomial kernel
            coef0 = args[1] : zero'th coefficient of polynomial kernel
            coef1 = args[2] : first coefficient of polynomial kernel
        
        Returns
        -------
        MMD distance between P and Q
        '''

        assert type_kern in ['linear', 'rbf', 'polyn']
        assert self.P.shape[1] == self.Q.shape[1]

        # get density functions:
        d1 = self.compute_density_kde(self.P[:,0])
        d2 = self.compute_density_kde(self.Q[:,0])

        if self.shared_axis == True:
            # combined axis (shared)
            cX = np.concatenate((self.P,self.Q))
            xs = np.linspace(min(cX),max(cX),self.grid)    
            p1 = np.zeros((self.grid,1))
            p2 = np.zeros((self.grid,1))
            for count,x in enumerate(xs):
                p1[count,0] = d1(x) 
                p2[count,0] = d2(x)
        elif self.shared_axis == False:
            x1 = np.linspace(min(self.P),max(self.P),self.grid)
            x2 = np.linspace(min(self.Q),max(self.Q),self.grid)
            p1 = np.zeros((self.grid,1))
            p2 = np.zeros((self.grid,1))
            for i in range(self.grid):
                p1[i,0] = d1(x1[i]) 
                p2[i,0] = d2(x2[i])

        if type_kern == 'linear':
            # linear kernel
            PP = np.dot(p1, p1.T)
            QQ = np.dot(p2, p2.T)
            PQ = np.dot(p1, p2.T)
            return ( np.mean(PP) - 2 * np.mean(PQ) + np.mean(QQ) )

        elif type_kern == 'rbf':
            # radial basis function kernel 
            theta = kernel_par[0]
            
            PP = metrics.pairwise.rbf_kernel(p1, p1, theta)
            QQ = metrics.pairwise.rbf_kernel(p2, p2, theta)
            PQ = metrics.pairwise.rbf_kernel(p1, p2, theta)
            return ( np.mean(PP) - 2 * np.mean(PQ) + np.mean(QQ) )
                
        elif type_kern == 'polyn':
            # polynomial kernel
            order = kernel_par[0]
            coef0 = kernel_par[1]
            coef1 = kernel_par[2]
            
            PP = metrics.pairwise.polynomial_kernel(p1, p1, order, coef1, coef0)
            QQ = metrics.pairwise.polynomial_kernel(p2, p2, order, coef1, coef0)
            PQ = metrics.pairwise.polynomial_kernel(p1, p2, order, coef1, coef0)
            return ( np.mean(PP) - 2 * np.mean(PQ) + np.mean(QQ) )
            
        else:
            print ('kernel type not supported')


    def compute_hsic_measure(self,ker_par_P, ker_par_Q):
        ''' Hilbert-Schmidt Independence Criterion

        Function that computes the Hilbert-Schmidt Independence Criterion (Hsic)
        measure between two probability distributions P and Q. The Hsic function 
        provides a kernel-based independence test designed to measure multivariate
        nonlinear dependence between distributions P and Q.

        Supported kernel include RBF only. 

        Input
        -----
        P : {x_1, x_2, ..., x_n} of size (n,d) with x_i d-dimensional random vector
        Q : {y_1, y_2, ..., y_n} of size (n,d) with y_j d-dimensional random vector
        ker_par_P : parameter of RBF kernel for P distribution
        ker_par_Q : parameter of RBF kernel for Q distribution

        Returns
        -------
        Hsic measure of nonlinear association between P and Q
        '''

        assert self.P.shape[1] == self.Q.shape[1]
        
        # get density functions:
        d1 = self.compute_density_kde(self.P[:,0])
        d2 = self.compute_density_kde(self.Q[:,0])

        if self.shared_axis == True:
            # combined axis (shared)
            cX = np.concatenate((self.P,self.Q))
            xs = np.linspace(min(cX),max(cX),self.grid)    
            p1 = np.zeros((self.grid,1))
            p2 = np.zeros((self.grid,1))
            for count,x in enumerate(xs):
                p1[count,0] = d1(x) 
                p2[count,0] = d2(x)
            n = self.grid
            PP = metrics.pairwise.rbf_kernel(p1, p1, ker_par_P)
            QQ = metrics.pairwise.rbf_kernel(p2, p2, ker_par_Q)
            H = np.eye(n) - 1/n * np.dot(np.ones((n,1)), np.ones((n,1)).T)
            PPc = np.dot(np.dot(H, PP), H)
            QQc = np.dot(np.dot(H, QQ), H)
            return ( np.trace( np.dot(PPc, QQc) ) / n )

        elif self.shared_axis == False:
            x1 = np.linspace(min(self.P),max(self.P),self.grid)
            x2 = np.linspace(min(self.Q),max(self.Q),self.grid)
            p1 = np.zeros((self.grid,1))
            p2 = np.zeros((self.grid,1))
            for i in range(self.grid):
                p1[i,0] = d1(x1[i]) 
                p2[i,0] = d2(x2[i])
            n = self.grid
            PP = metrics.pairwise.rbf_kernel(p1, p1, ker_par_P)
            QQ = metrics.pairwise.rbf_kernel(p2, p2, ker_par_Q)
            H = np.eye(n) - 1/n * np.dot(np.ones((n,1)), np.ones((n,1)).T)
            PPc = np.dot(np.dot(H, PP), H)
            QQc = np.dot(np.dot(H, QQ), H)
            return ( np.trace( np.dot(PPc, QQc) ) / n )


