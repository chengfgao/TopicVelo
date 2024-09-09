import numpy as np
import scipy
import scvelo as scv
from scipy.sparse import csr_matrix
from os.path import exists
import matplotlib.pyplot as plt
from scipy.sparse import save_npz, load_npz
from deeptime.markov.tools.analysis import stationary_distribution, mfpt


from .inference_tools import burst_inference, progressBar
'''
Utility tools for transition matrix
1. Cell selection
2. Gene selection
3. adata subset selection
'''
def get_cells_indices(adata, topics, 
                      topic_weights_th_percentile = None, 
                      above_or_below = 'above', topic_type = 'fastTopics'):
    '''
    'above_or_below': pick cells above the th or below the threshold
    '''
    ttc_indices = []
    #if topic_weights_th_percentile is a scalar, all topics will have the threshold
    if type(topic_weights_th_percentile) is not list:
        topic_weights_th_percentile = np.ones(len(topics))*topic_weights_th_percentile
    for i in range(len(topics)):
        if topic_weights_th_percentile is None:
            ttc_indices.append([j for j in range(adata.n_obs) if adata.obs[topic_type+'_cluster'][j] == topics[i]])
        else:
            #get the threshold for topic k 
            k_str = topic_type+'_'+str(topics[i])
            th_k = np.percentile(adata.obs[k_str], topic_weights_th_percentile[i])
            if above_or_below == 'above':
                ttc_indices.append([j for j in range(adata.n_obs) if adata.obs[k_str][j] >= th_k])
            elif above_or_below == 'below':
                ttc_indices.append([j for j in range(adata.n_obs) if adata.obs[k_str][j] < th_k])
            else:
                print('Error: Please choose if the percentiles are for above or below')
    other_cells_indices = np.array(list(set(np.arange(adata.n_obs))-set([x for xs in ttc_indices for x in xs])))
    return ttc_indices, other_cells_indices

def filter_velocity_genes(adata, xkey = 'spliced', ukey = 'unspliced', vkey = 'burst',
                          KL_lb = 0, KL_ub = 100, 
                          gamma_lb = 0.001, gamma_ub = 100):
    '''
    Filter genes by KL and gamma.
    
    gamma needs to be in (gamma_lb, gamma_ub)
    KL must be between than KL_threshold
    '''
    reasonable_genes = []
    gene_names = adata.var_names.tolist()
    gamma_key = vkey + '_velocity_gamma'
    KL_key = vkey + '_velocity_KLdiv'
    
    for i in range(adata.n_vars):
        #check the gamma
        gamma_i = adata.var[gamma_key][i]
        KL_i = adata.var[KL_key][i]
        
        #gamma value is unreasonable
        if gamma_i < gamma_lb or gamma_i > gamma_ub:
            continue;
        if KL_i < KL_lb or KL_i > KL_ub:
            continue;
        reasonable_genes.append(gene_names[i])
    return reasonable_genes

def get_adata_subset(adata, topic, save_path, topic_weights_th_percentile = None):
    '''
    Get adata subset to a topic
    1. get topic cells and genes
    2. load velocity inference
    '''
    if topic_weights_th_percentile is None:
        #extract and parse the topicvelo params
        x = adata.uns['topicVelo_params']['topics'].index(topic)
        topic_weights_th_percentile = adata.uns['topicVelo_params']['topic_weights_th_percentile'][x]
    ttc_indices, other_cells_indices = get_cells_indices(adata, [topic], topic_weights_th_percentile = topic_weights_th_percentile)
    if len(other_cells_indices) > 0:
        adata_other_cells = adata[other_cells_indices,:] 
    #get the top genes for topic k
    ttg = adata.uns['top_genes'][topic]
    ttg_indices = [adata.var.index.get_loc(gene_name) for gene_name in ttg]
    #subset the data and recompute neighbor list in this subset
    adata_subset = adata[ttc_indices[0], ttg_indices]
    scv.pp.neighbors(adata_subset)
    inferredParams = np.load(save_path)
    #add the burst_velocity_gamma values
    adata_subset.var['burst_velocity_gamma'] = inferredParams['Optimzal Parameters'][:, 2]
    adata_subset.var['burst_velocity_KLdiv'] = inferredParams['KLdiv']
    velocity_graph(adata_subset, 
                xkey = adata.uns['topicVelo_params']['embed_xkey'], 
                ukey = adata.uns['topicVelo_params']['embed_ukey'],
                gene_subset = ttg)  
    return adata_subset

'''
Computing the transition matrix 
'''
def velocity_graph(adata, vkey = 'burst_velocity', 
                   xkey = 'spliced', ukey = 'unspliced', 
                   gene_subset = None, n_jobs = 1, round_size_normalized = True, 
                   transition_matrix_mode='count'):
    '''
    Velocity has been inferred. The adata file contains the kinetic parameters
    xkey, ukey: the spliced/unspliced count matrices to use for computing velocity and transition matrix
    gene_subset: the subset of genes used for computing the transition matrix
    '''  
    def velocity_vectors(U_matrix, S_matrix, gamma_vector):
        #compute u-gamma*s for every (cell,gene)
        return U_matrix - S_matrix.multiply(gamma_vector)
    gamma_key = vkey+'_gamma'
    if xkey == 'Ms':
        adata.layers[vkey] = adata.layers['Mu']-np.multiply(adata.layers['Ms'], list(adata.var[gamma_key]))  
    elif xkey == 'spliced' and round_size_normalized:
        B_velocity_vectors = velocity_vectors(np.round(adata.layers[ukey]),
                    np.round(adata.layers[xkey]), 
                    adata.var[gamma_key])
        adata.layers[vkey] = B_velocity_vectors.A
    else:
        B_velocity_vectors = velocity_vectors(adata.layers[ukey],
                    adata.layers[xkey], 
                    adata.var[gamma_key])
        adata.layers[vkey] = B_velocity_vectors.A
    if transition_matrix_mode == 'count':
        try:
            scv.tl.velocity_graph(adata, vkey=vkey, xkey=xkey, gene_subset = gene_subset, n_jobs = n_jobs)
        except ValueError:
            scv.pp.neighbors(adata)
            scv.tl.velocity_graph(adata, vkey=vkey, xkey=xkey, gene_subset = gene_subset, n_jobs = n_jobs)


def combined_topics_transitions(adata, topics = None, 
                                topic_weights_th_percentile = None,
                                recompute = True,
                                recompute_matrix = True,
                                steady_state_perc = 95, QC_on_topic_genes = False,
                                velocity_type = 'burst',
                                infer_xkey='spliced', infer_ukey='unspliced',
                                embed_xkey = 'Ms', embed_ukey ='Mu',
                                topic_type = 'fastTopics', top_genes_key = 'top_genes', params_key = 'topicVelo_params',
                                transition_matrix_mode = 'count', transition_matrix_name = None,
                                subset_save_prefix = '', save = None):
    '''
    Main function for TopicVelo
    Compute topic-specific transition matrices then integrate them according to topic weights. 

    Construct a global transition matrix from topic transition matrix
    Topic cells are selected. There need to cells overlapping from all the topics. 
   
    Each topic transition matrix is computed using the top topic genes using FastTopics and user-specfied parameters.
    
    The topic transition matrices are combined using cell weights
    (e.g. if a cell is only assigned to topic 1, that cell has weight 1 in topic 1.
    if a cell is assigned to topic 1 with weight 0.6 and topic 2 with weight 0.3, that cell's transition
    matrix will be 2/3*t2 + 1/3*t1 )
    
    The global transition matrix will be row-normalized to 1. 
    
    ***the adata object should contain the top genes from user-specified criteria in adata.uns['top_genes'] 
    
    topics: the topics we want to piece up together 
        default: None, all of the topics will be used
        otherwise, must be a list. A list of one topic is permitted. 
    
    topic_weights_th_percentile: if a list, must be the same dimensions as topics specified (to be implemented. Take a scalar for now)
            
    '''
    #extract the number of cells
    n = adata.n_obs
    #extract the number of topics:
    K = len(adata.uns[top_genes_key])
    #get the top genes
    top_genes = adata.uns[top_genes_key]
    if QC_on_topic_genes:
        try:
            reasonable_top_genes = adata.uns['reasonable_top_genes']
        except:
            print('Need to perform quality control on topic genes')
            return 
    
    #use all topics if topic is none
    use_all_topics = False
    if topics is None:
        use_all_topics = True
        topics = list(range(K))
        
    #add params to adata
    topicVelo_params = {}
    topicVelo_params['topics'] = topics
    topicVelo_params['topic_weights_th_percentile'] = topic_weights_th_percentile 
    topicVelo_params['steady_state_perc'] = steady_state_perc
    topicVelo_params['infer_xkey'] = infer_xkey
    topicVelo_params['infer_ukey'] = infer_ukey
    topicVelo_params['embed_xkey'] = embed_xkey
    topicVelo_params['embed_ukey'] = embed_ukey 
    adata.uns[params_key] = topicVelo_params
    
    #for computing the confidence (coherence within neighborhood of velocity
    topics_cells_velocity_confidence = np.zeros((len(topics),n))
    
    #get topic cells
    ttc_indices, other_cells_indices = get_cells_indices(adata, topics, 
                                                         topic_weights_th_percentile = topic_weights_th_percentile,
                                                         topic_type = topic_type)
    if len(other_cells_indices) > 0:
        adata_other_cells = adata[other_cells_indices,:] 
    #get topic steady state cells
    ttc_ss_indices, transitient_cells_indices = get_cells_indices(adata, topics, 
                                                                  topic_weights_th_percentile = steady_state_perc, 
                                                                  topic_type = topic_type)
    
    #compute with scVelo
    scv.tl.velocity(adata, vkey='velocity')
    
    #store the transition matrices from topics
    TMs = []
    #compute transition matrix for each topic
    for x in range(len(topics)):
        k = topics[x]
        #get the top genes for topic k
        ttg_k = top_genes[k]
        ttg_indices = [adata.var.index.get_loc(gene_name) for gene_name in ttg_k]
        #subset the data and recompute neighbor list in this subset
        adata_subset = adata[ttc_indices[x], ttg_indices]
        scv.pp.neighbors(adata_subset)
        #subset to steady-state cells
        adata_subset_ss = adata[ttc_ss_indices[x], ttg_indices]
        topic_cell_vel_confidence =  np.zeros(n)
        scv.tl.velocity(adata_subset, vkey='velocity')
        
        if QC_on_topic_genes:
            reasonable_genes = reasonable_top_genes[k]
        else:
            reasonable_genes = ttg_k
                
        if velocity_type == 'stochastic':
            #compute velocity, velocity_graph and the transition matrix
            scv.tl.velocity_graph(adata_subset, vkey='velocity', gene_subset=reasonable_genes)
            scv.tl.velocity_confidence(adata_subset, vkey='velocity')
            topic_cell_vel_confidence =  np.zeros(n)
            topic_cell_vel_confidence[ttc_indices[x]] = adata_subset.obs['velocity_confidence']
            topics_cells_velocity_confidence[x] = topic_cell_vel_confidence
            TMs.append(scv.utils.get_transition_matrix(adata_subset, vkey='velocity'))
            
        elif velocity_type == 'burst':
            #burst topic genes and cells. Do not recompute if no need
            save_infer = subset_save_prefix+'T'+str(k)+'_'+infer_xkey+'_'+embed_xkey
            save_path = save_infer + '.npz'
            #recompute if the file does not exist or forced to recompute
            if not exists(save_path) or recompute:
                burst_inference(adata_subset_ss, savestring = save_path, report_freq = 50,
                            xkey = infer_xkey, ukey = infer_ukey,
                            vkey = 'burst_velocity')
            inferredParams = np.load(save_path)
            #add the burst_velocity_gamma values
            adata_subset.var['burst_velocity_gamma'] = inferredParams['Optimzal Parameters'][:, 2]
            adata_subset.var['burst_velocity_KLdiv'] = inferredParams['KLdiv']
            if QC_on_topic_genes:
                reasonable_genes = reasonable_top_genes[k]
            else:
                reasonable_genes = ttg_k
            velocity_graph(adata_subset, 
                        transition_matrix_mode=transition_matrix_mode, 
                        xkey = embed_xkey, ukey = embed_ukey,
                        gene_subset = reasonable_genes)  
            
            #add topic cell velocity coherence
            scv.tl.velocity_confidence(adata_subset, vkey='burst_velocity')
            topic_cell_burst_vel_confidence =  np.zeros(n)
            topic_cell_burst_vel_confidence[ttc_indices[x]] = adata_subset.obs['burst_velocity_confidence']
            topics_cells_velocity_confidence[x] = topic_cell_burst_vel_confidence
                
        
            #add topic gene kl divergence (from simulated with MLE to experimental)
            topic_gene_kl_str = 'Topic'+str(k)+'_KLdiv'
            topic_gene_kl = np.zeros(adata.n_vars)
            topic_gene_kl[ttg_indices] = inferredParams['KLdiv']
            adata.var[topic_gene_kl_str] = topic_gene_kl
            #add topic-specific transition 
            TMs.append(scv.utils.get_transition_matrix(adata_subset, vkey='burst_velocity'))

    TM_save_path = subset_save_prefix + 'TransitionMatrix.npz'
    
    #Use topic-specific matrices and topic weights to construct integrated transition matrix
    if not exists(TM_save_path) or recompute_matrix:
        #compute the cell weights for transition matrix
        cells_weights_for_tm = np.zeros((n,len(topics)))
        #iterate through number of topics consider
        for x in range(len(topics)):
            k = topics[x]
            cells_k_indices =  ttc_indices[x]
            cells_weights_k = adata.obs[topic_type+'_'+str(k)]
            #iterate the cells within the topics
            for i in range(len(cells_k_indices)):
                #get the index (in the global adata) of the topic cell
                cell_ki_ind = cells_k_indices[i]
                cells_weights_for_tm[cell_ki_ind, x] = cells_weights_k[cell_ki_ind]
        #row normalize to 1
        for i in range(n):
            weight_sum = np.sum(cells_weights_for_tm[i])
            if weight_sum > 0:
                cells_weights_for_tm[i] = cells_weights_for_tm[i]/weight_sum

        #construct the combined transition matrix
        combined_TM = csr_matrix((n,n), dtype=np.float32)
        for x in range(len(topics)):
            #track number of topics
            print(x)
            TM_k = TMs[x]
            #iterate through cells in the topic
            cells_k_indices =  ttc_indices[x] 
            for i in range(len(cells_k_indices)):
                #get the indices (in the global adata) of the topic cell
                cell_ki_ind = cells_k_indices[i]
                #get the cells transitioned into
                temp_js = TM_k[i].nonzero()[1]
                TM_ki = np.zeros(n)
                for l in range(len(temp_js)):
                    #get the transition probability for the i-th topic cell to l-th topic cell in topic k
                    TM_kij = TM_k[i, temp_js[l]]
                    TM_ki[cells_k_indices[temp_js[l]]] = TM_kij
                #add this to the combined transition matrix weighted by cell weights
                combined_TM[cell_ki_ind] = combined_TM[cell_ki_ind]+ TM_ki * cells_weights_for_tm[cell_ki_ind, x]  
        #clean up the transition matrix
        #row normalize to 1
        combined_TM = csr_matrix(combined_TM / combined_TM.sum(axis=1) )
        combined_TM = np.round(combined_TM,6)
        combined_TM = combined_TM / combined_TM.sum(axis=1)
        combined_TM = np.squeeze(np.asarray(combined_TM))
        #if a cell is not included, that cell has uniformly random transitions to nearest neighbors
        for i in range(adata.n_obs):
            if np.isnan(np.sum(combined_TM[i])):
                neighbors = list(adata.obsp['connectivities'][i].nonzero()[1])
                n_neigh = len(neighbors)
                combined_TM[i] = np.zeros(adata.n_obs)
                for j in neighbors:
                    combined_TM[i][j] = 1/n_neigh
        #save transition matrix 
        combined_TM = csr_matrix(combined_TM)
        save_npz(TM_save_path, combined_TM)
        #compute aggregate velocity confidence with cellweights 
        if velocity_type == 'burst':
            adata.obs['topicVelo_velocity_confidence_by_topicWeights'] = np.diag(np.matmul(cells_weights_for_tm, 
                                                                                           topics_cells_velocity_confidence))
            adata.obs['topicVelo_velocity_confidence'] = topics_cells_velocity_confidence.max(axis=0)
            adata.obsm['topicVelo_velocity_confidence_by_topics'] = topics_cells_velocity_confidence.T
        elif velocity_type == 'stochastic':
            adata.obs['scVelo+TM_velocity_confidence_by_topicWeights'] = np.diag(np.matmul(cells_weights_for_tm, 
                                                                                           topics_cells_velocity_confidence))
            adata.obs['scVelo+TM_velocity_confidence'] = topics_cells_velocity_confidence.max(axis=0)
            adata.obsm['scVelo+TM_velocity_confidence_by_topics'] = topics_cells_velocity_confidence.T
    else:
        combined_TM = load_npz(TM_save_path)
        combined_TM = csr_matrix(combined_TM)
    if transition_matrix_name:
        adata.obsp[transition_matrix_name+'_T'] = combined_TM
    elif velocity_type == 'burst':
        adata.obsp['topicVelo_T'] = combined_TM
    elif velocity_type == 'stochastic':
        adata.obsp['topic_modeling+scvelo_stochastic_T'] = combined_TM

    return combined_TM, ttc_indices