import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsela
import scipy.linalg as sla
import math
from tqdm import tqdm
import time
import numpy.linalg as la 
import pandas as pd
import random
random.seed(7)

import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def initializer(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, lambda_type, lambda_duration, p_0):
    before_p =  np.tile(p_0, Nt)
    before_u = np.zeros((n_blocks*n_states, Nt)) 
    if lambda_type == 0: #same for every block time independent
        lambda_s = np.tile(lambda_s_in, n_blocks) #format: [1, 1, 1]
        lambda_i = np.tile(lambda_i_in, n_blocks)
        lambda_r = np.tile(lambda_r_in, n_blocks)
    if lambda_type == 1: #different for each block time independent
        lambda_s = lambda_s_in #format: [1, 1, 1]
        lambda_i = lambda_i_in
        lambda_r = lambda_r_in
    if lambda_type == 2: #different for each block time dependent       
        lambda_s = np.repeat(lambda_s_in, lambda_duration[0].astype(int), axis=1) #format[1,1,1,...,3,3,3];
        lambda_i = np.repeat(lambda_i_in, lambda_duration[1].astype(int), axis=1) #format[1,1,1,...,3,3,3];
        lambda_r = np.repeat(lambda_r_in, lambda_duration[2].astype(int), axis=1) #format[1,1,1,...,3,3,3];
    return (before_p, before_u, lambda_s, lambda_i, lambda_r)    


def Z_calculator(n_blocks, block_dens, lambda_i, graphon, p, lambda_type):
    if (lambda_type==0 or lambda_type==1):
        Z = np.matmul(graphon, np.multiply(np.reshape(np.multiply(block_dens, lambda_i), (n_blocks,1)),p[n_blocks:2*n_blocks,:]))
    if lambda_type==2:
        Z = np.matmul(graphon,np.multiply(np.reshape(block_dens,(n_blocks,1)), lambda_i)*p[n_blocks:2*n_blocks,:])
    return Z    

def opt_control_calculator(lambda_s, lambda_i, lambda_r, beta, c_lambda, c_nu, Z, u, n_blocks, lambda_type, kappa, Nt, V):
    if (lambda_type==0 or lambda_type==1):
        alpha_s = np.reshape(lambda_s, (n_blocks,1)) + np.reshape(beta/(2*c_lambda),(n_blocks,1)) * Z * (u[0:n_blocks,:]-u[n_blocks:2*n_blocks,:])
        alpha_i = np.tile(np.reshape(lambda_i,(n_blocks,1)),Nt)
        alpha_r = np.tile(np.reshape(lambda_r,(n_blocks,1)),Nt)
        # nu      = np.reshape(kappa/(2*c_nu),(n_blocks,1)) * (u[0:n_blocks,:]-u[2*n_blocks:3*n_blocks,:])
    if lambda_type==2:
        alpha_s = lambda_s + np.reshape(beta/c_lambda,(n_blocks,1)) * Z * (u[0:n_blocks,:]-u[n_blocks:2*n_blocks,:])
        alpha_i = lambda_i
        alpha_r = lambda_r
        # nu      = np.reshape(kappa/(2*c_lambda),(n_blocks,1)) * (u[0:n_blocks,:]-u[2*n_blocks:3*n_blocks,:])

    diff_SR = u[0:n_blocks,:] - u[2*n_blocks:3*n_blocks,:]     # u(S)-u(R)
    thresh  = (kappa.reshape(-1,1)) * diff_SR                   # κ * (uS - uR)
    nu = (thresh > c_nu.reshape(-1,1)).astype(float) * V  # assume V = 1 for now
    # print(nu)
    return (alpha_s, alpha_i, alpha_r, nu)

def rate_ODE_p(t, p, death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks):
    alpha_s = inter_alpha_s(t)
    nu      = inter_nu(t)
    Z = inter_Z(t)
    rate_p_S = -beta*alpha_s*Z*p[0:n_blocks]- kappa * nu * p[0:n_blocks]
    # p[(n_states-1)*n_blocks:n_states*n_blocks]
    rate_p_I = beta*alpha_s*Z*p[0:n_blocks] - gamma * p[n_blocks:2*n_blocks]
    rate = []
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_p_S[k])
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_p_I[k])  
    if death==0:
        rate_p_R = gamma * p[n_blocks:2*n_blocks] + kappa * nu * p[0:n_blocks]
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_p_R[k])   
    if death==1:
        #can be changed after CDC submission
        rate_p_R = rho * gamma * p[n_blocks:2*n_blocks] - kappa * p[(n_states-1)*n_blocks:n_states*n_blocks]
        rate_p_D = (1-rho) * gamma * p[n_blocks:2*n_blocks]
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_p_R[k]) 
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_p_D[k])             
    return rate


def solver_KFP(death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks, p_0, t_grid, T):
    sol_p = solve_ivp(rate_ODE_p, [0,T], p_0, t_eval = t_grid, args = (death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks))    
    p = sol_p.y
    return p

def rate_ODE_u(t, u, death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks, block_dens, inter_pI, cp_S, cp_I, cp_R):
    Z  = inter_Z(t) 
    nu = inter_nu(t)
    alpha_s = inter_alpha_s(t)
    if lambda_type==2:
        lambda_s = lambda_s(t)
    # print(np.array(block_dens) @ p[n_blocks:2*n_blocks])
    # print(u)
    # p_I_all = np.array(block_dens) @ p[n_blocks:2*n_blocks]
    # print(t)
    # print(inter_pI(t))
    rate_u_S = beta * alpha_s * Z * (u[0:n_blocks]-u[n_blocks:2*n_blocks]) - c_lambda*((lambda_s-alpha_s)**2) + kappa*nu*(u[0:n_blocks]-u[2*n_blocks:3*n_blocks]) - c_nu*nu - cp_S*inter_pI(t)
                # ((beta**2)/(2*c_lambda)) * (Z**2) * (u[n_blocks:2*n_blocks]- u[0:n_blocks])**2)
    rate_u_I = -(gamma * (u[2*n_blocks:3*n_blocks]- u[n_blocks:2*n_blocks]) + c_inf) - cp_I*inter_pI(t)
    rate = []
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_u_S[k])
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_u_I[k]) 
    if death==0:
        # rate_u_R = -(kappa*(u[0:n_blocks] - u[2*n_blocks:3*n_blocks]))
        rate_u_R = 0 * alpha_s - cp_R*inter_pI(t)
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_u_R[k])        
    if death==1:   
#         change after CDC 2025 submission
        rate_u_R = -gamma*(kappa*(u[0:n_blocks] - u[2*n_blocks:3*n_blocks]))
        rate_u_D = - c_dead * np.ones((n_blocks))
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_u_R[k])  
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_u_D[k])  
    return rate


def solver_HJB(death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks, u_T, t_grid, T, block_dens, inter_pI, cp_S, cp_I, cp_R):
    backward_t_grid = T-t_grid
    sol_u = solve_ivp(rate_ODE_u, [T, 0], u_T, t_eval = backward_t_grid, \
                  args = (death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks, block_dens, inter_pI, cp_S, cp_I, cp_R)) 
    u = sol_u.y
    return np.flip(u,axis=1)

def stoch_block_check_pop_aware(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, \
                      graphon, beta, kappa, gamma, rho, c_lambda, c_inf, c_dead, c_nu, \
                      t_grid, T, p_0, u_T, n_print, exp_id, block_dens, lambda_type, lambda_duration, death, epsilon, V, cp_S, cp_I, cp_R):
    #Algorithm for constant lambda
    if (lambda_type==0 or lambda_type==1):
        before_p, before_u, lambda_s, lambda_i, lambda_r  =  initializer(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, lambda_type, lambda_duration, p_0)
        Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, before_p, lambda_type)
        inter_Z = interp1d(t_grid, Z)
        alpha_s, alpha_i, alpha_r, nu = opt_control_calculator(lambda_s, lambda_i, lambda_r, beta, c_lambda, c_nu, Z, before_u, n_blocks, lambda_type, kappa, Nt, V)
        inter_alpha_s = interp1d(t_grid, alpha_s)
        inter_nu      = interp1d(t_grid, nu, kind='previous', axis=1, bounds_error=False,
                    fill_value=(nu[:,0], nu[:,-1])) # interp1d(t_grid, nu)
        after_p = solver_KFP(death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
        p_I_all = np.array(block_dens) @ after_p[n_blocks:2*n_blocks]
        inter_pI = interp1d(t_grid, p_I_all)
        after_u = solver_HJB(death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T, block_dens, inter_pI, cp_S, cp_I, cp_R)
        convergence_p = la.norm(after_p - before_p)
        convergence_u = la.norm(after_u - before_u)
        i=0
        print("iter: ", i, "p conv: ", convergence_p, "u conv:", convergence_u)
        while ((la.norm(after_p - before_p) > epsilon) or (la.norm(after_u - before_u) > epsilon)):
            Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, after_p, lambda_type)
            inter_Z = interp1d(t_grid, Z)
            alpha_s, alpha_i, alpha_r, nu = opt_control_calculator(lambda_s, lambda_i, lambda_r, beta, c_lambda, c_nu, Z, after_u, n_blocks, lambda_type, kappa, Nt, V)
            inter_alpha_s = interp1d(t_grid, alpha_s)
            inter_nu      = interp1d(t_grid, nu, kind='previous', axis=1, bounds_error=False,
                    fill_value=(nu[:,0], nu[:,-1])) # interp1d(t_grid, nu)
            before_p = after_p.copy()
            before_u = after_u.copy()
            after_p = solver_KFP(death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
            p_I_all = np.array(block_dens) @ after_p[n_blocks:2*n_blocks]
            inter_pI = interp1d(t_grid, p_I_all)
            after_u = solver_HJB(death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T, block_dens, inter_pI, cp_S, cp_I, cp_R)
            convergence_p = np.append(convergence_p, la.norm(after_p - before_p))
            convergence_u = np.append(convergence_u, la.norm(after_u - before_u))  
            i +=1
            if i % n_print == 0:
                print("iter: ", i, "p conv: ", convergence_p[-1], "u conv:", convergence_u[-1])    
    return (after_u, after_p, alpha_s, alpha_i, alpha_r, nu, Z, convergence_p, convergence_u) 