"""
@author: Frank Gao
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
            
'''
Convert cells to documents for tomotopy
'''
def cells_to_documents(X_i, gene_names):
    doc = []
    for j in range(len(gene_names)):
        n = X_i[j]
        for k in range(n):
            doc.append(gene_names[j])
    return doc

'''
Identify discrete clusters and topic genes from fastTopics
'''
def FastTopics_cluster_assign(adata, L, t_type = 'fastTopics'):
    '''
    topic_model: a tomotopy topic model object
    '''
    n, K = L.shape
    topic_clusters = np.zeros(n, dtype=np.int8)
    
    #add topic weights of cells to adata
    for k in range(K):
        t_str = t_type + '_' + str(k)
        adata.obs[t_str] = L[:, k]   
    #assign cells to a topic based on the highest weight
    for i in range(n):
        topic_clusters[i] =  np.argmax(L[i])
    adata.obs[t_type+'_cluster'] = topic_clusters

def TopicGeneFiltering(lfc, lfsr, lfc_up_th = 0.5, lfc_down_th=-0.1, lfsr_up_th = 0.001, lfsr_down_th = 0.001):
    '''
    lfc: the postmean log fold change
    lfsr: the local false sigh rate
    Select all top genes above (for up) or below (for down) the mean lfc and below the associated lfsr 
    '''
    M, K = lfc.shape
    assert lfc_up_th > 0
    assert lfc_down_th < 0
    top_genes_up = np.zeros((M,K), dtype=bool)
    top_genes_down = np.zeros((M,K), dtype=bool)
    top_genes = np.zeros((M,K), dtype=bool)
    for k in range(K):
        lfc_k = lfc[:,k]
        lfsr_k = lfsr[:,k]
        top_genes_up_k = (lfc_k > lfc_up_th) & (lfsr_k < lfsr_up_th)
        top_genes_down_k = (lfc_k < lfc_down_th) & (lfsr_k < lfsr_down_th) 
        top_genes_up[:,k] = top_genes_up_k
        top_genes_down[:,k] = top_genes_down_k
        top_genes_k = top_genes_up_k | top_genes_down_k
        top_genes[:,k] = top_genes_k
    return top_genes, top_genes_up, top_genes_down

def TopicGenesQC(adata, top_genes, xkey = 'spliced', ukey = 'unspliced',
                          quant_thr = 98, spliced_thr = 1, unspliced_thr = 1,
                          sup_ratio_lb = 1/10, sup_ratio_ub= 10,
                          var_ratio_lb = 0.03, var_ratio_ub = 3):
    '''
    Filter genes by KL, gamma, and/or expression threshold.
    
    gamma needs to be in (gamma_lb, gamma_ub)
    KL must be between than KL_threshold
    
    For any gene, the quant_thr of spliced count > spliced_thr
                  the quant_thr of unspliced count > unspliced_thr
    '''
    reasonable_genes = []
    gene_names = adata.var_names.tolist()
    for i in range(adata.n_vars):   
        #not enough spliced
        if np.percentile(adata.layers[xkey][:,i].A, quant_thr) <= spliced_thr:
            continue;
        #not enough unspliced
        if np.percentile(adata.layers[ukey][:,i].A, quant_thr) <= unspliced_thr:
            continue;
        #compute sup ratio
        sup_ratio_i = np.max(adata.layers[xkey][:, i ])/np.max(adata.layers[ukey][:, i ])
        #check sup ratio
        if sup_ratio_i < sup_ratio_lb or sup_ratio_i > sup_ratio_ub:
            continue;
        #check variance ratio
        var_ratio_i = np.std(adata.layers[xkey][:, i].A)/np.std(adata.layers[ukey][:, i ].A)
        if var_ratio_i < var_ratio_lb or var_ratio_i > var_ratio_ub:
            continue; 
        reasonable_genes.append(gene_names[i])
    #pick intersection of topic genes and high quality genes
    reasonable_top_genes = []
    for i in range(len(top_genes)):
        reasonable_top_genes.append(list(set(top_genes[i]).intersection(set(reasonable_genes))))
    return reasonable_top_genes

def remove_U(top_genes, gene_names):
    '''
    remove the trailing '_U' from a list of genes when applicable. Also remove duplicates
    '''
    M, K = top_genes.shape
    ttg = []
    
    for k in range(K):
        ttg_k = gene_names[top_genes[:, k]]
        
        ttg_k_cleaned = []
        #make sure the list of genes is not empty
        if len(ttg_k) == 0:
            ttg.append([])
            continue;
        else:
            for i in range(len(ttg_k)):
                #remove the '_U'
                if ttg_k[i][-2:] == '_U':
                    ttg_k_cleaned.append(ttg_k[i][0:-2])
                else:
                    ttg_k_cleaned.append(ttg_k[i])
        ttg.append(list(set(ttg_k_cleaned)))
    return ttg

'''
For post-processing topic modeling results such as cluster aggregation for visualizing rMFPT
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

def aggregate_clusters(adata, clusters_key, excludes, new_clusters_key, other_names='others'):
    '''
    adata: (AnnData) object
    obs_key: (str) field specification to extract clusters from
    excludes: (list of str) these clusters will not be aggregated 
    new_clusters_key: (str) field names to store new clusters in
    other_names: (str): the names for the aggregate
    
    aggregate everything other than "excludes" into one cluster
    '''
    n_obs = adata.n_obs
    clusters_annotations = adata.obs[clusters_key].to_numpy()
    new_clusters_annotations = np.zeros(n_obs, dtype=object)
    for i in range(n_obs):
        ann_i = clusters_annotations[i]
        if ann_i in excludes:
            new_clusters_annotations[i] = ann_i
        else:
            new_clusters_annotations[i] = other_names     
    adata.obs[new_clusters_key] = new_clusters_annotations
    return


'''
Plotting and Utilities for topic number selections
'''
def CaoJuan2009(lda):
    k = lda.k
    num_vocabs = lda.num_vocabs
    rm_top = len(lda.removed_top_words)
    #extract the topic by gene matrix theta
    theta = np.zeros((num_vocabs-rm_top, k))
    for i in range(k):
        theta[:,i] = lda.get_topic_word_dist(i)
    sum_cosine = 0
    #compute topic-topic cosine correlation
    for i in range(k):
        norm_i = np.linalg.norm(theta[:,i], ord=2)
        for j in range(i+1, k):
            sum_cosine += np.dot(theta[:,i], theta[:,j]) / ( norm_i * np.linalg.norm(theta[:,j], ord=2) )
    metric = sum_cosine / (k*(k-1)/2)
    return metric

def Deveaud2014(lda):
    k = lda.k
    num_vocabs = lda.num_vocabs
    rm_top = len(lda.removed_top_words)
    #extract the topic by gene matrix
    theta = np.zeros((num_vocabs-rm_top, k))
    for i in range(k):
        theta[:,i] = lda.get_topic_word_dist(i)
    jsd = 0
    #compute Jensen-Shannon distrance
    for i in range(k):
        for j in range(i+1, k):
            x = theta[:,i]
            y = theta[:,j]
            jsd += np.sum(x*np.log(x/y)) + np.sum(y*np.log(y/x)) 
    jsd = jsd/2/(k*(k-1))
    return jsd

def AIC(lda):
    return 2*(lda.k-1)*lda.num_vocabs - 2*lda.ll_per_word*lda.num_words

def BIC(lda):
    return (lda.k-1)*lda.num_vocabs*np.log(len(lda.docs)) - 2*lda.ll_per_word*lda.num_words

def rescale(x):
    '''
    rescale a positive x to (0,1)
    '''
    return (x-np.min(x))/(np.max(x)-np.min(x))

def ldatuning_plot(lda_models, name = 'topic_modeling_scores.png', dpi = 200):
    ks = [lda.k for lda in lda_models]
    AICs = []
    BICs = []
    CaoJuan2009_scores = []
    Deveaud2014_scores = []
    for lda in lda_models:
        AICs.append(AIC(lda))
        BICs.append(BIC(lda))
        CaoJuan2009_scores.append(CaoJuan2009(lda))
        Deveaud2014_scores.append(Deveaud2014(lda))
    plt.plot(ks, rescale(AICs), 'o-')
    plt.plot(ks, rescale(BICs), 's--')
    plt.plot(ks, rescale(CaoJuan2009_scores), 'D-')
    plt.plot(ks, rescale(Deveaud2014_scores), 'X-' )
    plt.ylabel('Rescaled Scores', size=16)
    plt.xlabel('Number of Topics', size=16)
    plt.title('Topic Modeling Scores', size=16)
    plt.legend(['AIC', 'BIC', 'CaoJuan2009', 'Deveaud2014'])
    plt.savefig(name, dpi=dpi)
    
    