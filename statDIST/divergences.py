# -------------------------------------------------------------
# Statistical divergences module
#
# Author: Ziad Ghauch
#
# Reference
# https://en.wikipedia.org/wiki/Statistical_distance
# -------------------------------------------------------------



import numpy as np
from metrics import StatDist



class Divergence(StatDist):
    
    def __init__(self, P, Q, shared_axis, grid):
        super().__init__(P, Q, shared_axis, grid)

    def compute_bhat_distance(self):
        ''' Bhattacharyya distance
        
        Function that computes Bhattacharyya distance between a two probability 
        distributions P and Q. The distance measure can be used to determine the
        relative closeness of the two samples being considered. It is used to 
        measure the separability of classes in classification and it is 
        considered to be more reliable than the Mahalanobis distance, 
        as the Mahalanobis distance is a particular case of the Bhattacharyya 
        distance when the standard deviations of the two classes are the same.
        
        Input
        -----
        P : n-dimensional random vector from distribution P
        Q : n-dimensional random vector from distribution Q
        
        Returns
        -----
        Bhattacharyya distance between samples P and Q
        '''
        
        epsilon = 1e-14
        
        assert self.P.shape[1] == self.Q.shape[1]
        assert self.P.shape[1] == 1

        # get density functions:
        d1 = self.compute_density_kde(self.P[:,0])
        d2 = self.compute_density_kde(self.Q[:,0])

        if self.shared_axis == True:
            # combined axis (shared)
            cX = np.concatenate((self.P,self.Q))
            xs = np.linspace(min(cX),max(cX),self.grid)
        
            bc = 0.0
            for x in xs:
                p1 = d1(x) + epsilon
                p2 = d2(x) + epsilon
                bc += np.sqrt(p1 * p2)
            return ( -np.log(bc) )

        elif self.shared_axis == False:
            x1 = np.linspace(min(self.P),max(self.P),self.grid)
            x2 = np.linspace(min(self.Q),max(self.Q),self.grid)
            bc = 0.0
            for i in range(self.grid):
                p1 = d1(x1[i]) + epsilon
                p2 = d2(x2[i]) + epsilon
                bc += np.sqrt(p1 * p2) 
            return ( -np.log(bc) )        

    def compute_kl_divergence(self):
        ''' Kullback–Leibler divergence
        
        Function that computes Kullback–Leibler divergence between two probability 
        distributions P and Q. It is a distribution-wise asymmetric measure and thus
        does not qualify as a statistical metric of spread. Note that the KL
        divergence is undefined for distributions where some event Q[i] has a 
        probability of zero.

        Input
        -----
        P : n-dimensional random vector from distribution P
        Q : n-dimensional random vector from distribution Q
        
        Returns
        -----
        Kullback–Leibler divergence between distributions P and Q
        ''' 

        assert self.P.shape[1] == self.Q.shape[1]
        assert self.P.shape[1] == 1
        
        # get density functions:
        d1 = self.compute_density_kde(self.P[:,0])
        d2 = self.compute_density_kde(self.Q[:,0])

        dkl = 0.0
        epsilon = 1e-13

        if self.shared_axis == True:
            cX = np.concatenate((self.P,self.Q))

            # combined axis (shared)
            xs = np.linspace(min(cX),max(cX),self.grid)

            for x in xs:
                p1 = d1(x) + epsilon 
                p2 = d2(x) + epsilon
                if p1 != 0 and p2 != 0:
                    delta_dkl = p1 * np.log10(p1 / p2)
                    if np.isfinite(delta_dkl):
                        dkl += delta_dkl

            return dkl

        elif self.shared_axis == False:
            x1 = np.linspace(min(self.P),max(self.P),self.grid)
            x2 = np.linspace(min(self.Q),max(self.Q),self.grid)

            for i in range(self.grid):
                p1 = d1(x1[i]) + epsilon
                p2 = d2(x2[i]) + epsilon 
                if p1 != 0 and p2 != 0:
                    delta_dkl = p1 * np.log10(p1 / p2)
                    if np.isfinite(delta_dkl):
                        dkl += delta_dkl
            return dkl


    def compute_jensen_divergence(self):
        ''' Jensen-Shannon divergence
        
        Function that computes the Jensen-Shannon divergence between two probability 
        distributions P and Q. The Jensen-Shannon inequality is based on the 
        Kullback–Leibler divergence with some notable, useful differences such as
        being symmetric, smooth, and having a finite value.
        
        Input
        -----
        P : n-dimensional random vector from distribution P
        Q : n-dimensional random vector from distribution Q
        
        Returns
        -----
        Jensen-Shannon divergence between distributions P and Q
        '''
        
        assert self.P.shape[1] == self.Q.shape[1]
        assert self.P.shape[1] == 1

        if (self.P.shape[0]) < (self.Q.shape[0]):
            self.Q = self.Q[:self.P.shape[0],:]
        elif (self.P.shape[0]) > (self.Q.shape[0]):
            self.P = self.P[:self.Q.shape[0],:]
        else:
            assert self.P.shape[0] == self.Q.shape[0]

        M = 0.5 * (self.P[:,0] + self.Q[:,0])
        M = M.reshape(M.shape[0],1)
        self.P = self.P
        self.Q = M
        p1 = self.compute_kl_divergence()
        self.P = self.Q
        self.Q = M
        p2 = self.compute_kl_divergence()
        jsd = 0.5 * (p1 + p2)
        
        return jsd


    def compute_renyi_divergence(self, alpha):
        ''' Renyi divergence
        
        Function that computes the Renyi divergence or order alpha
        between two probability distributions P and Q.
        
        Input
        -----
        P : n-dimensional random vector from distribution P
        Q : n-dimensional random vector from distribution Q
        alpha : order of Renyi divergence
        
        Returns
        ------
        Renyi divergence (order alpha) between distributions P and Q
        '''
        
        assert alpha > 0
        assert alpha != 1
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

            dr = 0.0
            for x in xs:
                p1 = d1(x) + epsilon
                p2 = d2(x) + epsilon
                if p1 != 0 and p2 != 0:
                    dr += ( p1**(alpha) / p2**(alpha-1.) ) 
            return (np.log10(dr) / (alpha-1.))
        
        elif self.shared_axis == False:
            x1 = np.linspace(min(self.P),max(self.P),self.grid)
            x2 = np.linspace(min(self.Q),max(self.Q),self.grid)
            dr = 0.0
            for i in range(self.grid):
                p1 = d1(x1[i]) + epsilon
                p2 = d2(x2[i]) + epsilon
                if p1 != 0 and p2 != 0:
                    dr += ( p1**(alpha) / p2**(alpha-1.) ) 
            return (np.log10(dr) / (alpha-1.))




