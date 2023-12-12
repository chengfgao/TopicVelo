"""
@author: Frank Gao
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kl_div
import scipy.special
from numba import jit
from scipy.optimize import minimize
import time
from functools import partial

from .transcription_simulation import GeometricBurstTranscription, JointDistributionAnalysis, MarginalDistributionFromJD, FirstMomentsFromJD, JointDistributionAnalysis_exp

'''
miscellaneous tools
'''
def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
    
'''
One State inference tools
'''
def OneState_SteadyState_JD(alpha, beta, gamma, umax, smax):
    def OneState_SteadyState_JD_us(alpha, beta, gamma, u, s):
        a = alpha/beta
        b = alpha/gamma
        Pus = np.float_power(a, u)*np.float_power(b, s)*np.exp(-a-b)/ scipy.special.factorial(u) / scipy.special.factorial(s)  
        return Pus
    ana_p = np.zeros((umax, smax)) 
    for u in range(umax):
        for s in range(smax):
            ana_p[u,s] = OneState_SteadyState_JD_us(alpha, beta, gamma, u, s)
    return ana_p

def OneState_Inference(adata,  ukey = 'unspliced', xkey='spliced', vkey = 'one_state_velocity', savestring='OneState_Inferences.npz'):
    n = adata.n_vars
    
    os_gamma = np.zeros(n)
    os_alpha = np.zeros(n)
    minKL = np.zeros(n)
    
    for i in range(n):   
        EU_i = np.mean(adata.layers[ukey][:, i].toarray().flatten())
        ES_i = np.mean(adata.layers[xkey][:, i].toarray().flatten())
        
        #skip infesible optimization
        if EU_i == 0 or ES_i == 0:
            continue;
        else:
            #save the parameters
            alpha = EU_i
            gamma = EU_i/ES_i
            os_alpha[i] = alpha
            os_gamma[i] = gamma
            #Compute the joint distribution
            model_JD = OneState_SteadyState_JD(alpha, 1, gamma, np.max(gene_i_U), np.max(gene_i_S))
            minKL[i] = KLdivergence(gene_i_JD, modelJD)
    vals_to_save = {'alpha': os_alpha, 'gamma':os_gamma, 'KLdiv':minKL}
    np.savez(savestring, **vals_to_save)
    gamma_key = vkey+'_gamma'
    alpha_key = vkey+'_alpha'
    KL_key = vkey+'_KL'
    adata.var[gamma_key] = os_gamma
    adata.var[alpha_key] = os_alpha
    adata.var[KL_key] = minKL
    return adata

'''
KL divergence tools
'''
def KLdivergence(obsJD, modelJD, support=1e-10):    
    '''
    Compute the KL divergence between 
    obsJD (a matrix): observed joint distribution, the reference distribution since we want to find anaJD as close to obsJD as possible
    anaJD (a matrix): analytical joint distribution
    '''
    umax_obs, smax_obs = obsJD.shape
    umax_mod, smax_mod = modelJD.shape
    #Ensure the modelJD to be in the correct dimensions as obsJD
    mJD = None
    oJD = None
    if umax_mod >= umax_obs:
        #model has all information for observations
        if smax_mod >= smax_obs:
            oJD = np.zeros((umax_mod, smax_mod))
            oJD[0:umax_obs, 0:smax_obs] = obsJD
            mJD = modelJD
        #model has partial info for s
        else:
            oJD = np.zeros((umax_mod, smax_obs))
            oJD[0:umax_obs, 0:smax_obs] = obsJD
            mJD = np.zeros((umax_mod, smax_obs))
            mJD[0:umax_mod, 0:smax_mod] = modelJD
    else:
        #model has partial infor u 
        if smax_mod >= smax_obs:
            oJD = np.zeros((umax_obs, smax_mod))
            oJD[0:umax_obs, 0:smax_obs] = obsJD
            mJD = np.zeros((umax_obs, smax_mod))
            mJD[0:umax_mod, 0:smax_mod] = modelJD
        #model has partial info for u and s
        else:
            oJD = obsJD
            mJD = np.zeros((umax_obs, smax_obs))
            mJD[0:umax_mod, 0:smax_mod] = modelJD
    nrow, ncol = mJD.shape
    #remove negative artifacts 0 values if any
    for i in range(nrow):
        for j in range(ncol):
            if mJD[i, j] <= 0:
                mJD[i, j] = support
    return np.sum(kl_div(oJD, mJD))

def KLdivergence_simulations(obsJD, kon, b, beta, gamma, burnin = 50000, num_reactions = 500000): 
    '''
    GeometricBurstTranscription
    '''
    U, S, dt = GeometricBurstTranscription(kon, b, beta, gamma, num_reactions)
    U = U[burnin:]
    S = S[burnin:]
    dt = dt[burnin:]
    JD_ij = JointDistributionAnalysis(U, S, dt)
    return KLdivergence(obsJD, JD_ij)
    
'''
Burst Inference Tools
'''
def Burst_Inference_Obj(EU, ES, EU2, JD_obs, init_type='MoM', 
                        burnin = 50000, num_reactions = 1000000, mf = 50, 
                        xt = 0.0001, ft = 0.0001):
    
    def KLdiv_obj(x, *args):
        '''
        x : the parameter (kon, b, gamma)
        *args = (JD_obs, burnin, num_reactions)
        '''
        # prevent evaluation of nonphysical regime
        JD_obs, burnin, num_reactions = args
        if (x<0).any():
            return np.inf
        #kon, b, beta, gamma
        #setting beta 1
        evaluation = KLdivergence_simulations(JD_obs, x[0], x[1], 1, x[2], 
                                        burnin = burnin, num_reactions = num_reactions)
        return evaluation

    #method of moments initialization
    if init_type == 'MoM':
        b0 = EU2/EU - 1
        if b0 < 0:
            b0 = EU2/EU
        kon0 = EU/b0
        beta0 = 1
        gamma0 = EU/ES
        
    else:
        #approximate method of moments initialization
        p0 = np.sqrt(EU)
        b0 = p0*2.5
        kon0 = p0/3
        gamma0 = EU/ES*0.8
    
    x0 = [kon0, b0, gamma0]
    
    #bounds 10^-1<b<10^4, 10^-2<beta<10^2.5, 10^-2<gamma<10^2.5,
    res = minimize(KLdiv_obj, x0, args=(JD_obs, burnin, num_reactions), 
                                                bounds=((0.003, 100) , (0.1, 1e4), (1e-4, 1e3)),
                   method='nelder-mead', options={'maxfev':mf,'return_all':True, 'adaptive':True,   
                                                  'xatol':xt, 'fatol':ft})
    return res

def Burst_Inference(adata, savestring = 'Burst_Inferences.npz', xkey = 'raw_spliced', ukey = 'raw_unspliced', report_freq = 50,
                      vkey = 'burst_velocity', burnin=50000, num_reactions=500000, mf = 50, inference_method = 'Nelder-Mead'):
    '''
    inference methods:
     1. 'MoM': Analytical estimate of parameters with the method of moments. 
     2. 'Nelder-Mead': Use NM to find global minimum.
    '''
    n = adata.n_vars
    B_InferredParameters = np.zeros((n, 3))
    B_minKL = np.zeros(n)
    start = time.time()
    for i in range(n):
        if i%report_freq == 0:
            progressBar(i, n, barLength = 20)
        #round to the nearest integer
        gene_i_S = np.round(adata.layers[xkey][:,i].toarray().flatten().astype(np.uint64))
        gene_i_U = np.round(adata.layers[ukey][:, i].toarray().flatten().astype(np.uint64))
        EU_i = np.mean(gene_i_U)
        ES_i = np.mean(gene_i_S)
        EU2_i = np.var(gene_i_U)
        gene_i_JD = JointDistributionAnalysis_exp(gene_i_U , gene_i_S)
        #skip infesible optimization
        if EU_i == 0 or ES_i == 0:
            continue;
        else:
            #catch extremely sparse genes
            try:
                if inference_method == 'Nelder-Mead':
                    res_i = Burst_Inference_Obj(EU_i, ES_i, EU2_i, gene_i_JD,
                                           burnin=burnin, num_reactions = num_reactions, mf=mf)
                else:
                    b0 = EU2/EU - 1
                    if b0 < 0:
                        b0 = EU2/EU
                    kon0 = EU/b0
                    gamma0 = EU/ES
                    res_i = [kon0, b0, gamma0]
            except ZeroDivisionError:
                print('insufficient data for some genes')
                continue;
            else:
                B_InferredParameters[i] = res_i.x
                B_minKL[i] = res_i.fun
    progressBar(n, n, barLength = 20)
    end = time.time()
    print('Time Taken:')
    print(end - start)
    vals_to_save = {'Optimzal Parameters': B_InferredParameters, 'KLdiv':B_minKL}
    np.savez(savestring, **vals_to_save)
    KLdiv_key = vkey+'_'+'KLdiv'
    adata.var[KLdiv_key] = B_minKL
    gamma_key = vkey+'_gamma'
    adata.var[gamma_key] = B_InferredParameters[:, 2]
    return B_InferredParameters, B_minKL

def Burst_Inference_Gene(adata, gene, xkey = 'raw_spliced',  burnin=500000, num_reactions=5000000, mf = 50):
    '''
    Burst inference for one gene
    '''
    i = adata.var.index.get_loc(gene)
    if xkey == 'raw_spliced':
        gene_i_S = adata.layers['raw_spliced'][:, i].toarray().flatten().astype(np.uint64)
        gene_i_U = adata.layers['raw_unspliced'][:, i].toarray().flatten().astype(np.uint64)
    else:
        #for size normalized data
        gene_i_S = np.round(adata.layers['spliced'][:, i].toarray().flatten()).astype(np.uint64)
        gene_i_U = np.round(adata.layers['unspliced'][:,i].toarray().flatten()).astype(np.uint64)
    EU_i = np.mean(gene_i_U)
    ES_i = np.mean(gene_i_S)
    EU2_i = np.var(gene_i_U)
    gene_i_JD = JointDistributionAnalysis_exp(gene_i_U , gene_i_S)
    #print warning message for infesible optimization
    if EU_i == 0 or ES_i == 0:
        print('No spliced or unspliced RNA observed')
        return      
    try:
        res_i = Burst_Inference_Obj(EU_i, ES_i, EU2_i, gene_i_JD,
                                           burnin=burnin, num_reactions = num_reactions, mf=mf)
    except ZeroDivisionError:
        print('Insufficient Data')
        return
    else:
        InferredParameters = res_i.x
        minKL = res_i.fun
        return InferredParameters, minKL

'''
Tools for selecting topic threshold
'''     
def get_cells_indices(adata, topics, topic_weights_th_percentile = None, above_or_below = 'above'):
    '''
    'above_or_below': pick cells above the th or below the threshold
    '''
    ttc_indices = []
    
    #if topic_weights_th_percentile is a scalar, all topics will have the threshold
    if type(topic_weights_th_percentile) is not list:
        topic_weights_th_percentile = np.ones(len(topics))*topic_weights_th_percentile
    for i in range(len(topics)):
        if topic_weights_th_percentile is None:
            ttc_indices.append([j for j in range(adata.n_obs) if adata.obs['fastTopics_cluster'][j] == topics[i]])
        else:
            #get the threshold for topic k 
            k_str = 'fastTopics_'+str(topics[i])
            th_k = np.percentile(adata.obs[k_str], topic_weights_th_percentile[i])
            if above_or_below == 'above':
                ttc_indices.append([j for j in range(adata.n_obs) if adata.obs[k_str][j] >= th_k])
            elif above_or_below == 'below':
                ttc_indices.append([j for j in range(adata.n_obs) if adata.obs[k_str][j] < th_k])
            else:
                print('Error: Please choose if the percentiles are for above or below')
    
    other_cells_indices = np.array(list(set(np.arange(adata.n_obs))-set([x for xs in ttc_indices for x in xs])))
    return ttc_indices, other_cells_indices

def gene_threshold_heuristic(adata, topic, topic_gene,
                    xkey = 'spliced', ukey='unspliced', rescale =True):
    '''
    For a given topic, compute the sum of KL-divergence of all genes at given thresholds comparing to when th=0
    
    topic is an integer 
    
    topic_gene
    
    th: the threshold, a positive real number between 0 and 100
    '''
    ths = np.linspace(1, 99, num=99)
    KL_Ank_ps = np.zeros(len(ths))
    KL_Ank_ms = np.zeros(len(ths))
    gene_id = adata.var.index.get_loc(topic_gene)
    #distributions over all cells
    gene_S = np.round(adata.layers[xkey][:, gene_id ].toarray().flatten()).astype(np.uint64)
    gene_U = np.round(adata.layers[ukey][:, gene_id ].toarray().flatten()).astype(np.uint64)
    gene_JD = JointDistributionAnalysis_exp(gene_U, gene_S)
    for i in range(len(ths)):
        #get cells above a topic threshold
        ttc_p_indices, other_cells_indices = get_cells_indices(adata, [topic], topic_weights_th_percentile = ths[i], above_or_below='above')
        #subset data
        adata_p_subset = adata[ttc_p_indices[0], :]
        #distributions over cells above a topic threshold
        gene_S_th_p = np.round(adata_p_subset.layers[xkey][:, gene_id ].toarray().flatten()).astype(np.uint64)
        gene_U_th_p = np.round(adata_p_subset.layers[ukey][:, gene_id ].toarray().flatten()).astype(np.uint64)
        gene_JD_th_p = JointDistributionAnalysis_exp(gene_U_th_p, gene_S_th_p)
        KL_Ank_ps[i] = KLdivergence(gene_JD, gene_JD_th_p)
        ttc_m_indices, other_cells_indices = get_cells_indices(adata, [topic], topic_weights_th_percentile = ths[i], above_or_below='below')
        #subset data
        adata_m_subset = adata[ttc_m_indices[0], :]
        #distributions over cells above a topic threshold
        gene_S_th_m = np.round(adata_m_subset.layers[xkey][:, gene_id ].toarray().flatten()).astype(np.uint64)
        gene_U_th_m = np.round(adata_m_subset.layers[ukey][:, gene_id ].toarray().flatten()).astype(np.uint64)
        gene_JD_th_m = JointDistributionAnalysis_exp(gene_U_th_m, gene_S_th_m)
        KL_Ank_ms[i] = KLdivergence(gene_JD, gene_JD_th_m)
    if rescale:
        KL_Ank_ps = (KL_Ank_ps-np.min(KL_Ank_ps))/np.max(KL_Ank_ps)
        KL_Ank_ms = (KL_Ank_ms-np.min(KL_Ank_ms))/np.max(KL_Ank_ms)
    return KL_Ank_ps, KL_Ank_ms
    
def topic_threshold_heuristic(adata, topic, xkey = 'spliced', ukey='unspliced', rescale =True):
    '''
    For a given topic, compute the sum of KL-divergence of all genes at given thresholds comparing to when th=0
    
    topic is an integer 
    
    topic_gene
    
    th: the threshold, a positive real number between 0 and 100
    '''
    ths = np.linspace(1, 99, num=99)
    #get topic genes
    topic_genes = adata.uns['top_genes'][topic]
    mean_KL_Ank_ps = np.zeros(len(ths)) 
    mean_KL_Ank_ms = np.zeros(len(ths)) 
    for i in range(len(topic_genes)):
        gene = topic_genes[i]
    KL_Ank_ps, KL_Ank_ms = gene_threshold_heuristic(adata, topic, gene, xkey = xkey, ukey=ukey, rescale = rescale)
    mean_KL_Ank_ps += KL_Ank_ps
    mean_KL_Ank_ms += KL_Ank_ms
    return mean_KL_Ank_ps, mean_KL_Ank_ms

def topic_threshold_heuristic_plot(adata, topic, xkey = 'spliced', ukey='unspliced', rescale =True, yscale='linear'):
    mean_KL_Ank_ps, mean_KL_Ank_ms = topic_threshold_heuristic(adata, topic, xkey = xkey, ukey=ukey, rescale = rescale)
    
    ths = np.linspace(1, 99, num=99)
    
    plt.plot(ths, mean_KL_Ank_ps, label=r'$A_n^+$')
    plt.plot(ths, mean_KL_Ank_ms, label=r'$A_n^-$')
    
    if yscale == 'log':
        plt.yscale('log')
        
    plt.xlabel(r'$n^{th}$-percentile', fontsize=16)
    
    if rescale:
        plt.ylabel('rescaled KL div.', fontsize=16)
    else:
        plt.ylabel('KL div.', fontsize=16)
    
    title_txt = 'Topic '+str(topic)+ ' Threshold Heuristic'
    plt.title(title_txt, fontsize=16)
    plt.legend(loc="upper center")
    plt.show()
    plt.clf()
    return