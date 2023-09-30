# -*- coding: utf-8 -*-
"""
@author: Frank Gao
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import scipy.special

'''
Gilliespie Simulations for one state and the burst model
'''
@jit(nopython=True)
def OneStateTranscription(alpha, beta, gamma, num_reactions):
    '''
    Parameters
    ----------
    alpha : transcription rate u -> u+1
    beta : Stochastic rate constant u -> s
    gamma : Stochastic rate constant for s -> null
    num_reactions: total reactions allowed
    
    Returns
    -------
    Timestamps and trjectories of U and S
    '''
    #create trajectories storage
    U = np.zeros(num_reactions+1, dtype=np.int16)
    S = np.zeros(num_reactions+1, dtype=np.int16)
    T = np.zeros(num_reactions+1, dtype=np.float32)
    dt = np.zeros(num_reactions+1, dtype=np.float32)
    #generate random numbers
    r1_array = np.random.random(size=num_reactions)
    r2_array = np.random.random(size=num_reactions)
    
    #loop till termination time
    for i in range(num_reactions):
        #current population
        Ucur = U[i]
        Scur = S[i]
        #propensity for reactions
        
        #splicing
        a1 = Ucur*beta
        #degradation
        a2 = Scur*gamma
        #total propensity
        a0 = alpha+a1+a2
        
        #find and update arrival time
        r1 = r1_array[i] 
        tau = np.log(1/r1)/a0
        dt[i] = tau
        T[i+1] = T[i]+tau
        # Threshold for selecting a reaction
        r2a0 = r2_array[i]*a0
        
        #choose a reaction
        if r2a0 < alpha:
            U[i+1] = Ucur+1
            S[i+1] = Scur
        elif alpha <= r2a0 <= (alpha+a1):
            U[i+1] = Ucur-1
            S[i+1] = Scur+1
        else:
            U[i+1] = Ucur
            S[i+1] = Scur-1
    return U, S, dt

@jit(nopython=True)
def GeometricBurstTranscription(kon, b, beta, gamma, num_reactions):
    '''
    Parameters
    ----------
    kon: rate of firing
    b: mean of the geometric burst 
    beta : Stochastic rate constant u -> s
    gamma : Stochastic rate constant for s -> null
    num_reactions: total reactions allowed
    
    Returns
    -------
    Timestamps and trjectories of U and S
    '''
    #parameter for geometric distribution
    p = 1/(b+1)
    #create trajectories storage
    U = np.zeros(num_reactions+1, dtype=np.int16)
    S = np.zeros(num_reactions+1, dtype=np.int16)

    dt = np.zeros(num_reactions+1, dtype=np.float32)
    #generate random numbers
    r1_array = np.random.random(size=num_reactions)
    r2_array = np.random.random(size=num_reactions)
    
    for i in range(num_reactions):
        #current population
        Ucur = U[i]
        Scur = S[i]
        #propensity for reactions
        a1 = Ucur*beta #splicing
        a2 = Scur*gamma #degradation
        a0 = kon+a1+a2 #total propensity
        
        #find and update arrival time
        r1 = r1_array[i] 
        tau = np.log(1/r1)/a0
        dt[i] = tau

        # Threshold for selecting a reaction
        r2a0 = r2_array[i]*a0
        
        #choose a reaction
        if r2a0 < kon:
            U[i+1] = Ucur + np.random.geometric(p) - 1
            S[i+1] = Scur
        #splicing
        elif kon <= r2a0 <= kon+a1:
            U[i+1] = Ucur-1
            S[i+1] = Scur+1
        #degradation
        else:
            U[i+1] = Ucur
            S[i+1] = Scur-1
    return U, S, dt

@jit(nopython=True)
def MomentConvergence(x, dt):
    '''
    Check the time convergence of x as a function of time (T)
    '''
    return np.cumsum(np.multiply(x, dt))/np.cumsum(dt)

'''
Analytical solutions for the one state model 
'''
def OneState_SteadyState_JD_us(alpha, beta, gamma, u, s):
    a = alpha/beta
    b = alpha/gamma
    Pus = np.float_power(a, u)*np.float_power(b, s)*np.exp(-a-b)/ scipy.special.factorial(u) / scipy.special.factorial(s)  
    return Pus
def OneState_SteadyState_JD(alpha, beta, gamma, umax, smax):
    ana_p = np.zeros((umax, smax)) 
    for u in range(umax):
        for s in range(smax):
            ana_p[u,s] = OneState_SteadyState_JD_us(alpha, beta, gamma, u, s)
    return ana_p

'''
Analysis for constructing JD and obtaining moments
'''
@jit(nopython=True)
def JointDistributionAnalysis(U, S, dt):
    '''
    Construct the joint distribution from a list of (U, S) generated from simulation
    '''
    N = len(U)
    A = np.zeros( ( int(max(U))+1,int(max(S))+1) )
    for i in range(N):
        A[U[i], S[i]] += dt[i]
    return A/np.sum(dt)
    

@jit(nopython=True)
def JointDistributionAnalysis_exp(U, S):
    '''
    Construct the joint distribution from a list of (U, S)
    '''
    N = len(U)
    A = np.zeros( (max(U)+1,max(S)+1) )
    for i in range(N):
        A[U[i], S[i]] += 1
    return A/N

def MarginalDistributionFromJD(JD, margin = 'S'):
    '''
    Given a joint distribution (row, column) = (unspliced, spliced)
    compute the marginal distribution 
    '''
    if margin == 'S':
        return np.sum(JD, axis=0)
    else:
        return np.sum(JD, axis=1)

def FirstMomentsFromJD(p):
    '''
    p should be U (row) by S (column)
    '''
    nrow, ncol = p.shape
    EU = 0
    ES = 0
    for i in range(nrow):
        for j in range(ncol):
            pij = p[i,j]
            EU += pij*i
            ES += pij*j
    return EU, ES
