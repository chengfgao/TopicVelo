"""
@author: Frank Gao
"""
import numpy as np
import matplotlib.pyplot as plt
import scvelo as scv
import scanpy as scp
import matplotlib as mpl
import matplotlib
import seaborn as sns
import matplotlib.colors as colors
from TM_Utils import get_cells_indices


plt.rcParams["font.family"] = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
 
'''
Plot subroutine for generic scatter plot 
'''
def scatter_plot(adata, obs_key, 
              basis='umap', cmap='RdPu',
              point_size=5, figsize=(7,5),
              perc = [2, 98], labelsize=30,
              vmax=None, vmin=None, vcenter=None,
              savefile=None):
    '''
    Make a plot for the obs_key
    
    parameters
    ----------
    adata: (Adata) object containing scRNAseq information
    obskey : (str) field in adata.obs to plot
    
    basis: (str) the projection for plotting
    cmap: (str) colorbar for plotting
    point_size: (float)
    
    returns
    ----------
    None
    '''
    basis='X_'+basis
    data  = adata.obs[obs_key].to_numpy()
    order = np.argsort(np.abs(data))
    #change the aspect ratio
    plt.figure(figsize=figsize)
    if vmin is None:
        vmin = np.percentile(data, perc[0])
    if vmax is None:
        vmax = np.percentile(data, perc[1])
    if vcenter is None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    #plot the rest with colorbar
    im=plt.scatter(adata.obsm[basis][order][:,0], adata.obsm[basis][order][:,1],
                    s=point_size, c = data[order], cmap=cmap, norm=norm) 
    plt.axis('off')
    if savefile:
        plt.savefig(savefile, format='png', dpi=300, transparent=True)
    else:
        plt.show()
    plt.clf()
    # draw a new figure and replot the colorbar there
    fig,ax = plt.subplots(figsize=figsize)
    cbar = plt.colorbar(im)    
    cbar.ax.tick_params(labelsize=labelsize) 
    cbar.ax.set_yscale('linear')
    ax.remove()
    plt.tight_layout()
    #save the colorbar
    if savefile:
        cbar_save = 'cb_'+savefile[0:-3]+'svg'
        plt.savefig(cbar_save, format='svg', dpi=300, transparent=True)
    return


'''
Basic Plotting functions
1. For topic weights
2. For gene expression
3. For estimated RNA velocity
'''
def Plot_Topics(adata, topic, t_type = 'fastTopics',
                basis='umap', perc = [2, 98], point_size=2,
                savefile = False, shrink=0.35, labelsize=10, figsize=(7,5), cbar_save=None):
    '''
    Topic Y plots 
    In each Topic Y plot, cells are colored by their Topic Y weight, 
    and the continuous color scale is shifted so as to highlight the variability appropriately 
    (may be different for different topics).
    Cells should be plotted such that those with lower topic values are plotted 
    under those with higher topic values.
    
    topic is a string. '0', '1' etc
    '''
    plt.figure(figsize=figsize)
    topic_label = t_type + ' Topic ' + topic
    #order the topics by weight
    topic_access = t_type + '_'+topic
    topic_weights = adata.obs[topic_access]
    topic_weights_order = np.argsort(topic_weights)
    topic_weights = topic_weights[topic_weights_order]
    title_txt = topic_label + ' Weights\n'
    obsm_key ='X_'+basis
    fig = plt.axes()
    #plot all cells
    top_plot = plt.scatter(adata.obsm[obsm_key][topic_weights_order][:,0], 
                adata.obsm[obsm_key][topic_weights_order][:,1], 
                s=point_size, c = topic_weights, cmap = 'viridis_r', 
                vmin=np.percentile(topic_weights, perc[0]), vmax=np.percentile(topic_weights, perc[1]))
    plt.axis('off')
    if savefile:
        save_str = t_type+'_Topic'+topic+'_'+basis+'_'+'.png'
        fig.figure.savefig(save_str, format='png', dpi=300, transparent=False, facecolor='white')
    plt.tight_layout()
    plt.show()
    plt.clf()
    # draw a new figure and replot the colorbar there
    fig,ax = plt.subplots(figsize=figsize)
    cbar = plt.colorbar(top_plot, shrink=shrink)    
    cbar.ax.tick_params(labelsize=labelsize) 
    ax.remove()
    plt.tight_layout()
    #save the colorbar
    if savefile:
        plt.savefig(cbar_save, format='svg', dpi=300, transparent=True)

def Plot_Genes(adata, gene, layers, expr_percentile = 90, log_scale = False, basis='umap', 
               title ='', savefile = False, s = 2, cbar_save=None,
               aspect_ratio=1, shrink=1, figsize=(7,5), labelsize=30):
    '''
    Gene plots, where in a plot of gene G, cells are colored by their normalized expression for gene G, 
    with the continuous scale adjusted so that variability is appropriately highlighted. 
    '''
    plt.figure(figsize=figsize)
    #get the normalized gene expression 
    gene_ind = adata.var.index.get_loc(gene)
    obsm_key ='X_'+basis
    if layers=='Ms' or layers=='Mu':
        normalized_gene_expr = adata.layers[layers][:, gene_ind]
    else:
        normalized_gene_expr = adata.layers[layers].toarray()[:, gene_ind]
    expr_order = np.argsort(normalized_gene_expr)
    normalized_gene_expr = normalized_gene_expr[expr_order]
    if log_scale:
        normalized_gene_expr = np.log(normalized_gene_expr+1)
    
    plt.scatter(adata.obsm[obsm_key][:,0], 
                adata.obsm[obsm_key][:,1], 
                s=s, c = 'silver', zorder = 1)
    exp_plot = plt.scatter(adata.obsm[obsm_key][expr_order][:,0], 
                adata.obsm[obsm_key][expr_order][:,1], 
                s=s, c = normalized_gene_expr, cmap = 'plasma_r', zorder=2, 
                vmax=np.percentile(normalized_gene_expr,expr_percentile))
    plt.axis('off')
    plt.savefig(savefile, format='png', dpi=300, transparent=True)
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    # draw a new figure and replot the colorbar there
    fig,ax = plt.subplots(figsize=figsize)
    cbar = plt.colorbar(exp_plot, shrink=shrink)    
    cbar.ax.tick_params(labelsize=labelsize) 
    ax.remove()
    plt.tight_layout()
    #save the colorbar
    if savefile:
        plt.savefig(cbar_save, format='svg', dpi=300, transparent=True, bbox_inches='tight')

    
def velocity_bounds(adata, gene, vkeys = ['burst_velocity', 'velocity'], perc=[2,98]):
        gene_ind = adata.var.index.get_loc(gene)
        lb = np.inf
        ub = -np.inf 
        for vkey in vkeys:
            lb = np.min([lb, np.percentile(adata.layers[vkey][:, gene_ind], perc[0])])
            ub = np.max([ub, np.percentile(adata.layers[vkey][:, gene_ind], perc[1])])
        print(ub)    
        return lb, ub

def Plot_Velocity(adata, gene, lb=None, ub=None, perc = [2, 98], 
                  adata_subset = None, highlight_sign_diff = False,
                  vkey='burst_velocity', basis='umap', 
                  s=2, cmap = 'RdYlBu', sort_by_magnitude = False,
                  title_txt = '', savefile = False, cbar_save=None, 
                  aspect_ratio = 1.0, shrink=1, figsize=(7,5),labelsize=50):
    '''
    plot the velocity for a gene on a 2D embedding
    if adata_subset is not None: only the subset velocity will be plotted 
    '''
    plt.figure(figsize=figsize)
    
    n_obs = adata.n_obs if adata_subset is None else adata_subset.n_obs
    #get the velocity
    if adata_subset is None:
        gene_ind = adata.var.index.get_loc(gene)
        gene_vel = adata.layers[vkey][:, gene_ind]
    else:
        gene_ind = adata_subset.var.index.get_loc(gene)
        gene_vel = adata_subset.layers[vkey][:, gene_ind]
    #sort the velocity
    if sort_by_magnitude:
        vel_order = np.argsort(np.abs(gene_vel))
        gene_vel = gene_vel[vel_order]
    else:
        vel_order = np.arange(0, n_obs)
    obsm_key = 'X_'+basis
    if lb is None and ub is None:
        #get the lower and upper bound for plotting
        lb = np.percentile(gene_vel, perc[0])
        ub = np.percentile(gene_vel, perc[1])

    if lb < 0 and ub > 0 or highlight_sign_diff:
        if highlight_sign_diff:
            ub = np.max([np.abs(lb), np.abs(ub)])
            lb = -ub
        norm = colors.TwoSlopeNorm(vmin=lb, vcenter=0, vmax=ub)
    elif ub <= 0 and not highlight_sign_diff:
        cmap ='Reds_r'
        norm = colors.TwoSlopeNorm(vmin=lb, vcenter=lb/2, vmax=0)
    elif lb >= 0 and not highlight_sign_diff:
        cmap = 'Blues'
        norm = colors.TwoSlopeNorm(vmin=0, vcenter = ub/2, vmax=ub)
    print(lb, ub)
    
    if adata_subset is None:
        vel_plot = plt.scatter(adata.obsm[obsm_key][:,0][vel_order], 
                adata.obsm[obsm_key][:,1][vel_order], 
                s=s, c = gene_vel, cmap = cmap,
                norm=norm)
    else:
        #plot background
        plt.scatter(adata.obsm[obsm_key][:,0], 
                adata.obsm[obsm_key][:,1], 
                s=s, c = 'silver', zorder = 1)
        #plot velocity
        vel_plot = plt.scatter(adata_subset.obsm[obsm_key][:,0][vel_order], 
                adata_subset.obsm[obsm_key][:,1][vel_order], 
                s=s, c = gene_vel, cmap = cmap,
                norm=norm, zorder=2)
    #Remove grid and tickers but keep labels
    plt.axis('off')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, format='png', dpi=300, transparent=True)
        plt.show()
    else:
        plt.show()
    # draw a new figure and replot the colorbar there
    fig,ax = plt.subplots(figsize=figsize)
    cbar = plt.colorbar(vel_plot, shrink=shrink)    
    cbar.set_ticks([np.around(lb*0.8, 2), 0, np.around(ub*0.8, 2)])
    cbar.ax.tick_params(labelsize=labelsize) 
    ax.remove()
    plt.tight_layout()
    #save the colorbar
    if savefile:
        plt.savefig(cbar_save, format='svg', dpi=300, transparent=True, bbox_inches='tight')
    
    
'''
Horizontal Bar chart for visualizing top genes within a topic
'''
def pl_TopTopicGenes(n, topic, gene_names, top_genes_indices, log_fold_change, z, figsize=(15,10),
                     save='', up_or_down = 'up', xticks=[], save_type = 'svg', ticksize=20):
    '''
    n = number of top genes. The order is determined by selecting a lfsr threshold 
    then sort based on log-fold change
    and color by z score
    Must provide the up_genes and down_genes independently 
    '''
    indices = top_genes_indices[:, topic].nonzero()[0]
    ttg_names = gene_names[indices]
    log_fold_change_ttg = np.take(log_fold_change[:,topic], indices)
    z_ttg = np.take(z[:,topic], indices)
    
    #if only one gene
    if len(indices) == 1:
        plt.barh(ttg_names, log_fold_change_ttg, color = 'yellow',  height=0.2)
        plt.yticks(fontsize=ticksize)
        plt.xticks(fontsize=ticksize, ticks=xticks)
        plt.tight_layout()
        plt.savefig(save, format=save_type, transparent=False, facecolor='white', bbox_inches='tight', pad_inches=0)
        return ttg_names, log_fold_change_ttg, z_ttg 
    
    #sorted by lfc
    if up_or_down == 'up':
        sorted_indices = np.argsort(-1*log_fold_change_ttg)
    elif up_or_down == 'down':
        sorted_indices = np.argsort(log_fold_change_ttg)
    
    ttg_names_sorted = np.array(ttg_names)[sorted_indices][:n]
    log_fold_change_ttg_sorted = np.array(log_fold_change_ttg)[sorted_indices][:n]
    z_ttg_sorted = np.array(z_ttg)[sorted_indices][:n]
    #square root the absolute value of z-scores
    z_ttg_sorted = np.abs(z_ttg_sorted)
    #change the aspect ratio
    plt.figure(figsize=figsize)
    #color by z score
    data_color_normalized = [(x-min(z_ttg_sorted)) / (max(z_ttg_sorted)- min(z_ttg_sorted)) for x in z_ttg_sorted]
    my_cmap = plt.cm.get_cmap('plasma_r')
    colors = my_cmap(data_color_normalized)
    plt.barh(np.flip(ttg_names_sorted), np.flip(log_fold_change_ttg_sorted), color = np.flip(colors, axis=0),  height=0.5)
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize, ticks=xticks)
    #plt.xlabel('Log Fold Change', fontsize=30)
    sm = matplotlib.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(np.min(z_ttg_sorted), np.max(z_ttg_sorted)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    #cbar.set_label('|Z-Scores|', rotation=270, labelpad=25, fontsize=30)
    cbar.ax.tick_params(labelsize=ticksize) 
    plt.tight_layout()
    plt.savefig(save, format=save_type, transparent=False, facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.show()
    return ttg_names, log_fold_change_ttg, z_ttg 


'''
Plot topic-specific streamlines
1. no topic weights, just cells (can be from different topics)
2. With topic weights for a given topic
'''
def topic_transitions_plot(adata, adata_subset, topics, topic_weights_th_percentile = None,
                           vkey = 'burst_velocity',
                           basis = 'X_draw_graph_fa', save_fig_s = None, 
                           s = 10, V=None,
                           title='', color_key = 'fastTopics_cluster', 
                           plot_type ='stream', figsize=(7.5,5)):
    '''
    Must have the transition matrix
    Take in multiple topics
    '''
    #get topic cells
    ttc_indices, other_cells_indices = get_cells_indices(adata, topics, topic_weights_th_percentile = topic_weights_th_percentile)
    if len(other_cells_indices) > 0:
        if plot_type == 'stream':
            fig = scv.pl.velocity_embedding_stream(adata_subset, vkey=vkey, basis=basis[2:], V=V,
                  dpi=300, show=False, size=0, density=2, figsize=figsize)
            
        elif plot_type == 'grid':
            fig = scv.pl.velocity_embedding_grid(adata_subset, vkey=vkey, basis=basis[2:], scale=1, arrow_size=1, V=V,
                                 dpi=300, show=False, size=0, figsize=figsize)
            
        fig.scatter(adata.obsm[basis][other_cells_indices,0], adata.obsm[basis][other_cells_indices,1], s=s, c = "#DDDDDD", zorder=1, label = 'Other Cells', alpha=0.5)
        for x in range(len(topics)):
            fig.scatter(adata[ttc_indices[x]].obsm[basis][:,0], adata[ttc_indices[x]].obsm[basis][:,1], s=s, zorder=2, alpha=0.5, linewidths=0,
                    label = 'Topic '+str(topics[x])+' Cells', c = adata.uns[color_key+'_colors'][topics[x]])
    else:
        fig = scv.pl.velocity_embedding_stream(adata_subset, vkey=vkey, basis=basis[2:], color=color_key,
                                 dpi=300, show=False, V=V,figsize=figsize)
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if save_fig_s is not None:
        fig.figure.savefig(save_fig_s, format='png', dpi=300, transparent=False, facecolor='white')
    plt.clf()
    return fig
    
    
def topic_transitions_with_weights_plot(adata, adata_subset, topic, 
                                  topic_weights_th_percentile = None,
                                  t_type = 'fastTopics', 
                                  vkey = 'burst_velocity',
                                  basis='umap_ori', perc = [2, 98],
                                  density=1.5,
                                  point_size=2,
                                  cmap = 'viridis_r',
                                  savefile = False, figsize=(7.5,5), 
                                 arrow_color='red', arrow_size=1):
    '''
    Topic Y plots 
    In each Topic Y plot, cells are colored by their Topic Y weight, 
    and the continuous color scale is shifted so as to highlight the variability appropriately 
    (may be different for different topics).
    Cells should be plotted such that those with lower topic values are plotted 
    under those with higher topic values.
    
    topic is a string. '0', '1' etc
    '''
    if topic_weights_th_percentile is None:
        #extract and parse the topicvelo params
        x = adata.uns['topicVelo_params']['topics'].index(topic)
        topic_weights_th_percentile = adata.uns['topicVelo_params']['topic_weights_th_percentile'][x]

    #order the topics by weight
    topic_access = t_type + '_'+ str(topic)
    topic_weights = adata.obs[topic_access]
 
    #get topic cells
    ttc_indices, other_cells_indices = get_cells_indices(adata, [topic], topic_weights_th_percentile = topic_weights_th_percentile)
    ttc_indices = ttc_indices[0]
    
    #must have computed velocity_graph in adata_subset
    fig = scv.pl.velocity_embedding_stream(adata_subset, vkey=vkey, basis=basis, 
                  dpi=300, show=False, size=0, density=density, arrow_color=arrow_color, arrow_size=arrow_size, figsize=figsize)
    basis = 'X_'+basis
    if len(other_cells_indices) > 0:
        fig.scatter(adata.obsm[basis][other_cells_indices,0], adata.obsm[basis][other_cells_indices,1],
                    s=point_size, c = "#DDDDDD", zorder=1, label = 'Other Cells')
        #plot topic cells
        im = plt.scatter(adata.obsm[basis][ttc_indices][:,0], 
                adata.obsm[basis][ttc_indices][:,1], 
                s=point_size, c = topic_weights[ttc_indices], cmap = cmap,
                vmin=np.percentile(topic_weights, perc[0]), vmax=np.percentile(topic_weights, perc[1]), zorder=2)
    else:
        #plot all cells
        im = plt.scatter(adata.obsm[basis][:,0], 
                adata.obsm[basis][:,1], 
                s=point_size, c = topic_weights, cmap = cmap, alpha =0.5,  
                vmin=np.percentile(topic_weights, perc[0]), vmax=np.percentile(topic_weights, perc[1]))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20) 
    plt.axis('off')
    if savefile:
        fig.figure.savefig(savefile, format='png', dpi=300, transparent=False, facecolor='white')
        return fig
    else:
        plt.clf()
        return fig
    
'''
Plot for mfpt to a group of target cells
'''    
def mfpt_plot(adata, dest, Tkey='topicVelo', 
              basis='umap', cmap='magma_r', dest_color =  "deepskyblue", 
              point_size=5, figsize=(7,5),
              perc = [2, 98], labelsize=20, 
              vmax=None, vmin=None, vcenter=None,
              savefile=None):
    '''
    Make a plot for the mean first passage time 
    
    parameters
    ----------
    adata: (Adata) object containing scRNAseq information
    dest: (str) name of the target cells
    Tkey: (str) name of the transition matrix used for computing mfpt
    basis: (str) the project for plotting
    cmap: (str) colorbar for plotting
    point_size: (float)
    
    returns
    ----------
    None
    '''
    basis='X_'+basis
    
    mfpt_str = Tkey+'_mfpt_'+dest
    mfpt = adata.obs[mfpt_str].to_numpy()

    #separate into zeros and nonzeros
    other_indices = np.nonzero(mfpt)
    other_mfpt = mfpt[other_indices]
    
    target_indices = np.where(mfpt == 0)[0]
    
    #change the aspect ratio
    plt.figure(figsize=figsize)

    
    #plot the zeros in the background
    plt.scatter(adata.obsm[basis][target_indices,0], adata.obsm[basis][target_indices,1],
                    s=point_size, c =dest_color, zorder=1, label = 'Target Cells')
    if vmin is None:
        vmin = np.percentile(other_mfpt, perc[0])
    if vmax is None:
        vmax = np.percentile(other_mfpt, perc[1])
    if vcenter is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        #plot the rest with colorbar
        im=plt.scatter(adata.obsm[basis][other_indices,0], adata.obsm[basis][other_indices,1],
                    s=point_size, c = other_mfpt, zorder=2, cmap=cmap,norm=norm)
    else:
        #plot the rest with colorbar
        im=plt.scatter(adata.obsm[basis][other_indices,0], adata.obsm[basis][other_indices,1],
                        s=point_size, c = other_mfpt, zorder=2, cmap=cmap, 
                    vmin=vmin, vmax=vmax)
    plt.axis('off')
    if savefile:
        plt.savefig(savefile, format='png', dpi=300, transparent=True)
    else:
        plt.show()
    plt.clf()
    # draw a new figure and replot the colorbar there
    fig,ax = plt.subplots(figsize=figsize)
    cbar = plt.colorbar(im)    
    cbar.ax.tick_params(labelsize=labelsize) 
    cbar.ax.set_yscale('linear')
    ax.remove()
    plt.tight_layout()
    #save the colorbar
    if savefile:
        cbar_save = 'cb_'+savefile[0:-3]+'svg'
        plt.savefig(cbar_save, format='svg', dpi=300, transparent=True)
    return

'''
Comparision for visualzing different contributions to stationary distributions
1. heatmap 
2. stacked bar plot 
aggregated by cell types
'''
def cell_categorical_annotation_indices(adata, annotations, categories):
    cells = []
    for i in range(adata.n_obs):
        if adata.obs[annotations][i] in categories:
            cells.append(i)
    return cells

def comparison_heatmap(adata, keys, labels=None, groupby='cell_type', categories=None,
                        fontsize=16, title=None,
                        figsize=(15,5), savefile=None, 
                       cmap='viridis', labelsize=32):
    if labels is None:
        labels = keys
    n_keys = len(keys)
    x = np.linspace(0, n_keys-1, num=n_keys)
    if categories is None:
        categories = adata.obs[groupby].unique()
        n_categories = len(categories)
    else:
        n_categories = len(categories) 
    data = np.zeros((n_keys, n_categories))
    for i in range(n_categories):
        cells_i = cell_categorical_annotation_indices(adata, groupby, categories[i])
        for j in range(n_keys):
            data[j,i] = np.sum(adata.obs[keys[j]].to_numpy()[cells_i])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap)
    # Show all ticks and label them with the respective list entries
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xticks(np.arange(len(categories)), labels=categories)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    cbar = plt.colorbar(im)    
    cbar.ax.tick_params(labelsize=labelsize) 
    cbar.ax.set_yscale('linear')
    ax.set_title(title)
    fig.tight_layout()
    if savefile:
        plt.savefig(savefile, format='svg', dpi=300, transparent=True)
    else:
        plt.show()

    
def comparision_stacked_bar_plot(adata, keys, labels=None, groupby='cell_type', categories=None,
                        fontsize=16, title=None,
                        figsize=(15,5), savefile=None):
    '''
    make a stacked bar plot for sum of individual categories in keys->categories (i.e. adata.obs[keys].cat.categories).
    
    '''
    if labels is None:
        labels = keys  
    n_keys = len(keys)
    x = np.linspace(0, n_keys-1, num=n_keys)
    
    if categories is None:
        categories = adata.obs[groupby].unique()
        n_categories = len(categories)
    else:
        n_categories = len(categories) 
    data = np.zeros((n_keys, n_categories))
    for i in range(n_categories):
        cells_i = cell_categorical_annotation_indices(adata, groupby, categories[i])
        for j in range(n_keys):
            data[j,i] = np.sum(adata.obs[keys[j]].to_numpy()[cells_i])
    
    def horizontal_stacked_bar_chart(labels, data, category_names, category_colors, figsize, fontsize):
        '''
        Parameters
        ----------
        results : dict
            A mapping from question labels to a list of answers per category.
            It is assumed all lists contain the same number of entries and that
            it matches the length of *category_names*.
        category_names : list of str
            The category labels.
       '''
        fig, ax = plt.subplots(figsize=figsize)
        ax.invert_yaxis()
        ax.set_xlim(0, np.sum(data, axis=1).max())
        data_cum = data.cumsum(axis=1)
        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.75,
                            label=colname, color=color)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=fontsize)
        return fig, ax
    ccs = groupby+'_colors'
    category_colors = np.array(adata.uns[ccs])[np.array([list(adata.obs[groupby].cat.categories.to_numpy()).index(i) for i in categories])]
    horizontal_stacked_bar_chart(labels, data, categories, category_colors, figsize, fontsize)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, format='svg', dpi=300, transparent=True)
    else:
        plt.show()
    

'''
The violin plot used to visualize difference in distributions of mfpt
'''
def comparision_violin_plot(adata, keys, groupby='cell_type', categories=None,
                        key_spacing = 1, groupby_spacing=2.75, figsize=(15,5),
                        facecolors = None, edgecolor = 'black', 
                        fontsize=16, ylabel=None, title=None,
                        qs=[25,75], ylim=None,
                        savefile=None):
    '''
    parameters
    ---------------
    adata: (Adata) object containing scRNAseq information
    keys: (list of str) in adata.obs that contains the data
    groupby: (str) the name of grouping used to divide up the values in keys
    categories: (list of str) the categroies within a group to be used
    key_spacing: (float) spacing between violins within each key
    groupby_spacing: (float) spacing between violins among keys (between categories)
    facecolors; (list of str) specify the colors for each key
    
    
    
    returns
    ---------------
    None
    
    '''
    from matplotlib.pyplot import violinplot
    #import matplotlib.patches as mpatches
    
    n_keys = len(keys)
    
    if categories is None:
        categories = adata.obs[groupby].unique()
        n_categories = len(categories)
    else:
        n_categories = len(categories)
        
    n_violins = int(len(keys)*len(categories))
    
    #positions of the violins
    ps = np.linspace(0, key_spacing*(n_keys-1), num=n_keys)
    positions = [ps]
    tick_ps = [np.mean(ps)]
    for i in range(1, n_categories):
        new_ps = ps+i*groupby_spacing
        positions.append(new_ps)
        tick_ps.append(np.mean(new_ps))
    positions = np.array(positions).flatten()
    
    #extract data for each key->categories
    data = []
    medians=[]
    qs1 = []
    qs2 = []
    
    for i in range(n_categories):
        cells_i = cell_categorical_annotation_indices(adata, groupby, categories[i])
        for j in range(n_keys):
            to_add = adata.obs[keys[j]].to_numpy()[cells_i]
            data.append(to_add )
            medians.append(np.median(to_add))
            qs1.append(np.percentile(to_add, qs[0]))
            qs2.append(np.percentile(to_add, qs[1]))
    data=np.array(data)
    
    #set dimension
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    violin_plot = ax.violinplot(data, positions, widths=0.85, showmeans=False, showmedians=False, showextrema=False)
    
    ax.vlines(positions, qs1, qs2, color='k', linestyle='-', lw=2.5, zorder=1)
    ax.scatter(positions, medians, marker='o', color='white', zorder=2, s=15)
    
    if facecolors is None:
        facecolors = sns.color_palette("husl", n_keys)
        
    for i, pc in enumerate(violin_plot["bodies"], 1):
        pc.set_facecolor(facecolors[int(i%n_keys)])
        pc.set_edgecolor(edgecolor)
    
    #add x ticks and labels
    ax.set_xticks(tick_ps)
    ax.set_xticklabels(categories, fontsize=fontsize)
    #add y labels
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(keys, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    plt.ylim(ylim)
    plt.tight_layout()
    
    
    if savefile:
        plt.savefig(savefile, format='svg', dpi=300, transparent=True)
    else:
        plt.show()
        
        
def comparision_violin_plot_v2(adata, keys, categories,
                            groupby='cell_type',
                            key_spacing = 1, groupby_spacing=2.75, figsize=(15,5),
                            edgecolor = 'black', color_by = 'cell_type',
                            fontsize=16, xlabels=None, ylabel=None, title=None,
                            qs=[25,75], ylim=None, savefile=None):
    '''
    parameters
    ---------------
    adata: 
        (Adata) object containing scRNAseq information
    keys: 
        (list of str) adata.obs that contains the data 
    categories: 
        (list of str) the annotations within a group to be used
    
    groupby: 
        (str) the name of grouping used to divide up the values in keys
    key_spacing: 
        (float) spacing between violins within each key
    groupby_spacing: 
        (float) spacing between violins among keys (between categories)
    
    
    
    returns
    ---------------
    None
    
    '''
    from matplotlib.pyplot import violinplot
    
    
    n_keys = len(keys)
    n_categories = len(categories)
    n_violins = int(len(keys)*len(categories))
    
    #positions of the violins
    ps = np.linspace(0, key_spacing*(n_categories-1), num=n_categories)
    positions = [ps]
    tick_ps = [np.mean(ps)]
    for i in range(1, n_keys):
        new_ps = ps+i*groupby_spacing
        positions.append(new_ps)
        tick_ps.append(np.mean(new_ps))
    positions = np.array(positions).flatten()
    
    #extract data for each categories->key
    data = []
    medians=[]
    qs1 = []
    qs2 = []
    
    for i in range(n_keys):
        for j in range(n_categories):
            to_add = adata.obs[keys[i]].to_numpy()[np.where(adata.obs[groupby]==categories[j])[0]]
            data.append(to_add)
            medians.append(np.median(to_add))
            qs1.append(np.percentile(to_add, qs[0]))
            qs2.append(np.percentile(to_add, qs[1]))
    data=np.array(data)
    
    #set dimension
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    violin_plot = ax.violinplot(data, positions, widths=0.85, showmeans=False, showmedians=False, showextrema=False)
    
    ax.vlines(positions, qs1, qs2, color='white', linestyle='-', lw=2.5, zorder=1)
    ax.scatter(positions, medians, marker='o', color='k', zorder=2, s=15)
    
    facecolors = []
    all_categories = list(adata.obs[color_by].cat.categories.to_numpy())
    ccs = color_by+'_colors'
    colors = np.array(adata.uns[ccs])
    for c in categories:
        if c in all_categories:
            #print(colors[all_categories.index(c)])
            facecolors.append(colors[all_categories.index(c)])
        else:
            facecolors.append('#000000')
    
    for i, pc in enumerate(violin_plot["bodies"], 0):
        pc.set_facecolor(facecolors[int(i%n_categories)])
        pc.set_edgecolor(edgecolor)
    
    #add x ticks and labels
    ax.set_xticks(tick_ps)
    if xlabels:
        ax.set_xticklabels(xlabels, fontsize=fontsize)
    else:
        ax.set_xticklabels(keys, fontsize=fontsize)
    #add y labels
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(categories, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    plt.ylim(ylim)
    plt.tight_layout()
    
    
    if savefile:
        plt.savefig(savefile, format='svg', dpi=300, transparent=True)
    else:
        plt.show()
        
def relative_flux_plot(
    adata, 
    k_transition_matrices, 
    markersize=50,
    fontsize=20, legends=None,
    title=None, ylim=[-1,1], figsize=(10,5), savefile=None):
    """Plot Relative Flux Direction Correctness Score (A->B) from transition matrices
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_transition_matrices (list of str): 
            keys to the transition matrix in adata.obs.
        cluster_transitions (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
            
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    cluster_transitions = adata.uns[k_transition_matrices[0]+'_rel_flux'].keys()
    positions = np.arange(len(cluster_transitions))
    for k_tr_mat in k_transition_matrices:          
        ax.scatter(positions, adata.uns[k_tr_mat+'_rel_flux'].values(), marker='o', s=markersize)                    
    xlabels = [A+'\n ↓ \n'+B for (A,B) in cluster_transitions]
    
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=fontsize)
 
    #add y labels
    ax.set_ylabel('Relative Flux', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    if legends:
        ax.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    else:
        ax.legend(k_transition_matrices, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    plt.ylim(ylim)
    plt.tight_layout()
    
    if savefile:
        plt.savefig(savefile, format='svg', dpi=300, transparent=True)
    else:
        plt.show()
