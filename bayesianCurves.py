# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import special

#x1 = np.arange(9.0).reshape((3, 3))
#x2 = np.arange(3.0).reshape((3,1))
#print(x1)
#
#print(np.array(x1)) 
#print(x2)
#print(x1*x2)
#print(x2*x1)

class StochasticCurve:
    def __init__(self, basis, sigma, init_curve):
        assert basis.shape[0] == sigma.shape[0]
        assert basis.shape[1] == init_curve.shape[0]
        self.nb = basis.shape[0]
        self.basis = basis
        self.sigma = sigma
        self.init_curve = init_curve

    def generatePath(self, ts):
        path = [self.init_curve]
        tp = 0.0
        for t in ts:
            ws = np.sqrt(t-tp)*self.sigma*np.random.normal(size=self.nb)
            path.append(path[-1] + ws @ self.basis)
            tp = t
        path.pop(0)
        return path

def randomObservations(path, pillars, uncertainties):
     return np.array([[curve[p]+u*np.random.normal() for p,u in zip(pillars,uncertainties)] for curve in path])




def main():
    np.random.seed(1)
    ts = np.array(range(1,366))/365.0
    pillars = np.array([1.0/12, 1.0/6, 1.0/4, 1.0/2, 3.0/4, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    mid = 2.0
    def t_trans(t): return (t-mid)/(t+mid)
    nbasis = 2
    sigma = (50.0/10000)/np.array(range(1,nbasis+1)) #scaled to 50 bps annual normal vol for first component
    print(sigma)
    init_curve=np.zeros(len(pillars))
    basis_eval = special.eval_chebyt    
    basis = np.array([[basis_eval(i,t) for t in t_trans(pillars)] for i in range(nbasis)])
    sc = StochasticCurve(basis,sigma,init_curve)
    path = sc.generatePath(ts)
    print(len(path))
    print(path[0])
    print(path[-1])

    obs_pillars = [2,5,7,12]
    uncertainties = np.array([2,1.5,1,0.5])/10000.0 
    observations = randomObservations(path, obs_pillars, uncertainties)
    for i,p in enumerate(obs_pillars):
        series = np.diff(observations[:,i])
        autocor = np.dot(series[:-1], series[1:])/np.dot(series,series)
        print (i, p, autocor)

#    x = np.array([[1, 2], [3, 4], [5, 6]])
#    print(x[0:3,[0]])
#    timeseries = []
    

if __name__== "__main__":
    main()
