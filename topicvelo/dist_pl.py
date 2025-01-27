"""
@author: Frank Gao
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot

from .transcription_simulation import geometric_burst_transcription, joint_distribution_analysis, joint_distribution_analysis_exper, os_ss_jd
from .inference_tools import KL_divergence

plt.rcParams["font.family"] = "Arial"
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

def proportion_pie_chart(adata, xkey='spliced', ukey='unspliced',
                    fontsize=20,title=None, save=False):
    '''
    Plot the proportion of spliced vs unspliced in a pie chart
    
    Args:
        adata (Anndata): 
            Anndata object.
        xkey: 
            key to the spliced count matrix in adata.layers
        ukey: 
            key to the unspliced count matrix in adata.layers
        fontsize:
            
        title:
        
        save: (str if need to save)
            
    Returns:
        None
    '''
    
    S = np.sum(np.round(adata.layers[xkey].toarray().flatten()))
    U = np.sum(np.round(adata.layers[ukey].toarray().flatten()))
    plt.pie([S,U], labels=['spliced', 'unspliced'], autopct='%1.1f%%', textprops={'fontsize': fontsize})
    plt.title(title, size=fontsize)
    plt.tight_layout()
    if save:
        plt.savefig(save, format='svg', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()

        
def proportion_bar_chart(adata, raw_xkey='raw_spliced', raw_ukey='raw_unspliced',
                         xkey='spliced', ukey='unspliced',
                         fontsize=20,title=None, figsize=(15,3), save=False):
    S_raw = np.sum(adata.layers[raw_xkey].toarray().flatten())
    U_raw = np.sum(adata.layers[raw_ukey].toarray().flatten())
    raw_total = S_raw+U_raw
    S = np.sum(np.round(adata.layers[xkey].toarray().flatten()))
    U = np.sum(np.round(adata.layers[ukey].toarray().flatten()))
    SN_total = S+U
    S_prop = [S_raw/raw_total, S/SN_total]
    U_prop = [U_raw/raw_total, U/SN_total]
    counts = ['Raw', 'Size Normalized']
    
    plt.figure(figsize=figsize)
    b1 = plt.barh(counts, S_prop, left=U_prop)
    b2 = plt.barh(counts, U_prop)
    
    plt.yticks(fontsize=fontsize)
    plt.legend([b1, b2], ["Spliced", "Unspliced"], fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title, size=fontsize)
    plt.tight_layout()
    plt.xlim([0,1])
    
    
    plt.text(U_prop[0]/2, 0, str(U_prop[0]*100)[0:4],color='white', fontsize=fontsize, ha='center', va='center')
    plt.text(U_prop[1]/2, 1, str(U_prop[1]*100)[0:4],color='white', fontsize=fontsize, ha='center', va='center')
    plt.text(S_prop[0]/2+U_prop[0], 0, str(S_prop[0]*100)[0:4],color='white', fontsize=fontsize, ha='center', va='center')
    plt.text(S_prop[1]/2+U_prop[1], 1, str(S_prop[1]*100)[0:4],color='white', fontsize=fontsize, ha='center', va='center')
            
    if save:
        plt.savefig(save, format='svg', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()

'''
Joint distribution from experiments for all cells
1. provided the gene JD
2. Specify a given gene in an adata object
'''
def plot_jd(gene_JD, 
            x_cutoff = None, y_cutoff = None,
            vmax = None, vmin = None, log_scale_cb = False, label =False,
            save = False, title='', title_size=16, cbar_orientation='vertical'):
    fig, ax = plt.subplots()
    if x_cutoff is not None: 
        if log_scale_cb:
            im = ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap='YlOrBr', aspect='equal',
                           norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin))
        else:
            im = ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap='YlOrBr', aspect='equal',
                      vmax = vmax, vmin = vmin)
    else: 
        im = ax.imshow(gene_JD, cmap='YlOrBr', aspect='equal', vmax = vmax, vmin = vmin)
    plt.gca().invert_yaxis()
    fig.colorbar(im, orientation=cbar_orientation)
    if label:
        plt.xlabel('Spliced', size=20)
        plt.ylabel('Unspliced', size=20)
        plt.title('', size=title_size)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    if save:
        plt.savefig(save, format='svg', dpi=300, bbox_inches='tight')
    
def plot_exper_jd(adata, gene_name, 
                        xkey = 'spliced', ukey='unspliced', 
                        x_cutoff = None, y_cutoff = None, cmap='YlOrBr',
                        vmin = None, vmax = None, log_scale_cb = True, save=False):
    gene_id = adata.var.index.get_loc(gene_name)
    gene_S = np.round(adata.layers[xkey][:, gene_id ].toarray().flatten()).astype(np.uint64)
    gene_U = np.round(adata.layers[ukey][:, gene_id ].toarray().flatten()).astype(np.uint64)
    gene_JD = joint_distribution_analysis_exper(gene_U, gene_S)
    fig, ax = plt.subplots()
    if x_cutoff is not None: 
        if log_scale_cb:
            im = ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap=cmap, aspect='equal', norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin))
        else:
            im = ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap=cmap, aspect='equal',
                      vmax = vmax, vmin = vmin)
    else: 
        im = ax.imshow(gene_JD, cmap='YlOrBr', aspect='equal', vmax = vmax, vmin = vmin)
    plt.gca().invert_yaxis()
    fig.colorbar(im)
    plt.xlabel('Spliced', size=20)
    plt.ylabel('Unspliced', size=20)
    tt= 'Experimental JD ' + 'for '+ gene_name
    plt.title(tt, size=16)
    plt.tight_layout()
    if save:
        plt.savefig(save, edgecolor='black', dpi=300, bbox_inches = "tight", facecolor='white')
    return gene_JD

'''
Joint distribution from simulation and analytical computations
1. Simulation for JD from the burst model
2. Calculation for JD from the one state model
'''
def plot_burst_sim_jd(gene_name, params, 
                          num_reactions = 1000000, burnin = 100000,
                          x_cutoff = None, y_cutoff = None,
                          vmax = None, vmin = None, log_scale_cb = False, 
                          save = False, file_type='svg'):
    #Two State Parameters
    kon, b, gamma = params
    #splicing rate
    beta = 1
    U, S, dt = geometric_burst_transcription(kon, b, beta, gamma, num_reactions)
    U = U[burnin:]
    S = S[burnin:]
    dt = dt[burnin:]
    gene_JD = joint_distribution_analysis(U, S, dt)
    fig, ax = plt.subplots()
    if x_cutoff is not None: 
        if log_scale_cb:
            im = ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap='YlOrBr', aspect='equal',
                           norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin))
        else:
            im = ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap='YlOrBr', aspect='equal',
                      vmax = vmax, vmin = vmin)
    else: 
        im = ax.imshow(gene_JD, cmap='YlOrBr', aspect='equal', vmax = vmax, vmin = vmin)
    plt.gca().invert_yaxis()
    fig.colorbar(im)
    plt.xlabel('Spliced', size=20)
    plt.ylabel('Unspliced', size=20)
    tt= 'Simulated Burst Joint Distribution ' + 'for '+ gene_name
    plt.title(tt, size=16)
    plt.tight_layout()
    if save:
        plt.savefig(save, format=file_type, dpi=300)
    return gene_JD


def plot_os_analytical_jd(gene_name, params, 
                          umax, smax,
                          vmax = None, vmin = None, log_scale_cb = False, 
                          save = False):
    #One State Parameters
    alpha, gamma = params
    #splicing rate
    beta = 1
    gene_JD = os_ss_jd(alpha, beta, gamma, umax, smax)
    fig, ax = plt.subplots()
    if log_scale_cb:
        im = ax.imshow(gene_JD, cmap='YlOrBr', aspect='equal',
                       norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin))
    else:
        im = ax.imshow(gene_JD, cmap='YlOrBr', aspect='equal', vmax = vmax, vmin = vmin)   
    plt.gca().invert_yaxis()
    fig.colorbar(im)
    plt.xlabel('Spliced', size=20)
    plt.ylabel('Unspliced', size=20)
    tt= 'One State Distribtuion' + 'for '+ gene_name
    plt.title(tt, size=16)
    plt.tight_layout()
    if save:
        save_str = 'OS_JD_'+gene_name+'.svg'
        plt.savefig(save_str, format='svg', dpi=300)
    return gene_JD

# #FIX: this code is not working. Need to specify the spacing between the scatters
# def plot_exper_jd_clusters(adata, gene_name, color_by = 'lda_cluster',
#                             layers = 'raw_spliced', x_scatters = 2, sp =0.15, all_cells = False, 
#                             x_cutoff = None, y_cutoff = None, markersize = 10, alpha =1, 
#                             vmin = None, vmax = None):
#     '''
#     For visualizing joint distributions within/across clusters 
#     For a given gene:
#     1. Scatter plot highlighting Proportions from different clusters for each point in the discrete distribution
#     2. Heatmap of the joint distributions in different clusters 
#     '''
#     gene_id = adata.var.index.get_loc(gene_name)
#     gene_S = adata.layers['raw_spliced'][:, gene_id ].toarray().flatten().astype(np.uint64)
#     gene_U = adata.layers['raw_unspliced'][:, gene_id ].toarray().flatten().astype(np.uint64)
    
#     #get the cell type names, number of types, and number of cells for each type
#     types = adata.obs[color_by].unique()
#     num_types = adata.obs[color_by].nunique()
#     n_cells_types = adata.obs[color_by].value_counts()
    
#     #allocate to scatters based on color_by
#     #Put x_scatter in a row with center of the x-coordinates as [x-a, x+a]
#     #limit the center of y-coordinates between [y-a, y+a]
#     n_row = int(np.ceil(num_types/x_scatters))
#     last_row_size = num_types%x_scatters
#     y_shift = np.linspace(-sp, sp, num=n_row)
#     x_shift = np.linspace(-sp, sp, num=x_scatters)
#     x_last_row_shift = np.linspace(-a, a, num=last_row_size)
#     shifts = []
#     for i in range(n_row):
#         for j in range(x_scatters):
#             shifts.append((x_shift[j], y_shift[i]))
#     for i in range(last_row_size):
#         shifts.append((x_last_row_shift[i], y_shift[-1]))
#     fig, ax = plt.subplots()
    
#     #for each cell type, tally the distribution
#     for i in range(num_types):
#         ci = types[i]
#         ci_indices = [i for i, x in enumerate(list(adata.obs[color_by])) if x == ci]
#         gene_U_ci = gene_U[ci_indices]
#         gene_S_ci = gene_S[ci_indices]
#         s_max_ci = int(np.max(gene_S_ci))
#         u_max_ci = int(np.max(gene_U_ci))
#         gene_JD_ci = joint_distribution_analysis_exper(gene_U_ci, gene_S_ci)
#         min_JD = np.min(gene_JD_ci[np.nonzero(gene_JD_ci)])
#         U = []
#         S = []
#         dist_ci = []
#         f = lambda a: (np.log10(a/min_JD*10))*(markersize)         # function to calculate size from data
#         g = lambda b: np.float_power(10, b/markersize )/10*min_JD  # inverse function to calc. data from size
        
#         for u in range(u_max_ci+1):
#             for s in range(s_max_ci+1):
#                 dist_ci_us = gene_JD_ci[u,s]
#                 if dist_ci_us> 0:
#                     U.append(u)
#                     S.append(s)
#                     dist_ci.append(dist_ci_us)
#         dist_ci = np.array(dist_ci)
#         #dist_ci = 1-np.exp(-rate*dist_ci)
#         S = S+shifts[i][0]
#         U = U+shifts[i][1]
#         label = color_by+' '+ci
#         sc = plt.scatter(S , U, s=f(dist_ci), label=label, alpha = alpha)
#     if all_cells:
#         s_max = int(np.max(gene_S))
#         u_max = int(np.max(gene_U))
#         U = []
#         S = []
#         dist = []
#         gene_JD = joint_distribution_analysis_exper(gene_U, gene_S)
#         for u in range(u_max+1):
#             for s in range(s_max+1):
#                 dist_us = gene_JD[u,s]
#                 if dist_us> 0:
#                     U.append(u)
#                     S.append(s)
#                     dist.append(dist_us)
#         dist = np.array(dist)
#         sc = plt.scatter(S, U, s=f(dist), label = 'All Cells', alpha=0.5)
#     size_legend = pyplot.legend(*sc.legend_elements("sizes", color='k', func=g), 
#                                 labelspacing=0.5, frameon=False,
#                                 loc='upper left', bbox_to_anchor=(0.994, 0.55))
#     color_legend = pyplot.legend(loc='upper right', bbox_to_anchor=(1.32, 1.03),  frameon=False, labelspacing=1)
#     plt.xlabel('Spliced', size=20)
#     plt.ylabel('Unspliced', size=20)
#     tt= 'Experimental JD ' + 'for '+ gene_name
#     plt.title(tt, size=16)
#     ax.add_artist(color_legend)
#     ax.add_artist(size_legend)
#     savestring = gene_name + '_by_'+ color_by+'.png'
#     plt.savefig(savestring, edgecolor='black', dpi=300, bbox_inches = "tight", facecolor='white')
    
def heatmap_exper_cluster_focus_jd(adata, gene_name, 
                                   clusters='lda_cluster', focuses=['0'], gamma=None, 
                                   S_layer='raw_spliced', U_layer='raw_unspliced', 
                                   cmap='YlOrRd',
                                   x_cutoff=None, y_cutoff=None, height=10, 
                                   vmin=1e-4, vmax=None):
    """
    Creates a heatmap visualization comparing joint distributions of spliced and unspliced RNA counts
    for a specific gene across different cell clusters.

    Parameters:
    -----------
    adata : AnnData object
        Annotated data matrix containing RNA sequencing data
    gene_name : str
        Name of the gene to analyze
    clusters : str, optional (default='lda_cluster')
        Column name in adata.obs containing cluster assignments
    focuses : list, optional (default=['0'])
        List of cluster IDs to focus on for comparison. Each focus will get its own subplot.
    gamma : float, optional (default=None)
        Slope (inferred degradation constant) for reference line in plot
    S_layer : str, optional (default='raw_spliced')
        Layer name containing spliced RNA counts
    U_layer : str, optional (default='raw_unspliced')
        Layer name containing unspliced RNA counts
    cmap : str or matplotlib.colors.Colormap, optional (default='YlOrRd')
        Colormap for visualization. Can be either a string name of a matplotlib 
        colormap or a matplotlib.colors.Colormap object
    x_cutoff : int, optional (default=None)
        Maximum x-axis value to display
    y_cutoff : int, optional (default=None)
        Maximum y-axis value to display
    height : int, optional (default=10)
        Figure height in inches
    vmin : float, optional (default=1e-4)
        Minimum value for color scaling
    vmax : float, optional (default=None)
        Maximum value for color scaling

    Returns:
    --------
    None
        Saves the generated plot to a PNG file
    """
    # Convert single focus to list if necessary
    if not isinstance(focuses, list):
        focuses = [focuses]

    # Get gene index and extract spliced/unspliced counts
    gene_id = adata.var.index.get_loc(gene_name)
    gene_S = adata.layers[S_layer][:, gene_id].toarray().flatten().astype(np.uint64)
    gene_U = adata.layers[U_layer][:, gene_id].toarray().flatten().astype(np.uint64)

    # Calculate range for plotting
    U_max = int(np.max(gene_U)+1)
    S_max = int(np.max(gene_S)+1)

    # Set up figure layout parameters
    Cols = len(focuses) + 2  # Number of focuses + other cells + all cells
    Position = range(1, Cols+1)
    max_JDs = []
    gene_JD_cis = []
    
    # Calculate figure width based on height and cutoffs
    width = np.ceil(Cols*height)
    # if y_cutoff is not None:
    #     width = np.ceil(x_cutoff/y_cutoff*height*Cols)
    
    # Create figure and remove unnecessary spines and ticks
    fig = plt.figure(1, figsize=(width, height))
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    
    # Set title
    tt = 'Experimental JD ' + 'for '+ gene_name + '\n'
    plt.title(tt, size=16)

    # Get indices for all focused clusters
    focus_indices_list = []
    for focus in focuses:
        f_indices = np.array([i for i, x in enumerate(list(adata.obs[clusters])) 
                            if str(x) == str(focus)])
        focus_indices_list.append(f_indices)
    
    # Get indices for other cells (cells not in any focus)
    all_focus_indices = np.concatenate(focus_indices_list)
    other_indices = np.array(list(set(np.arange(adata.n_obs)) - set(all_focus_indices)))
    
    # Calculate joint distributions for each focus cluster
    for f_indices in focus_indices_list:
        gene_U_ci = gene_U[f_indices]
        gene_S_ci = gene_S[f_indices]
        
        # Calculate joint distribution for current cluster
        gene_JD_ci = joint_distribution_analysis_exper(gene_U_ci, gene_S_ci)
        
        # Create matrix of appropriate size and store distribution
        Ui_max = int(np.max(gene_U_ci)+1)
        Si_max = int(np.max(gene_S_ci)+1)
        M = np.zeros((U_max, S_max))
        M[0:Ui_max, 0:Si_max] = gene_JD_ci
        gene_JD_cis.append(M)
        max_JDs.append(np.max(gene_JD_ci))
    
    # Calculate joint distribution for other cells
    gene_U_other = gene_U[other_indices]
    gene_S_other = gene_S[other_indices]
    gene_JD_other = joint_distribution_analysis_exper(gene_U_other, gene_S_other)
    M = np.zeros((U_max, S_max))
    M[0:int(np.max(gene_U_other)+1), 0:int(np.max(gene_S_other)+1)] = gene_JD_other
    gene_JD_cis.append(M)
    max_JDs.append(np.max(gene_JD_other))
    
    # Calculate joint distribution for all cells
    gene_JD = joint_distribution_analysis_exper(gene_U, gene_S)
    max_JDs.append(np.max(gene_JD))
    vmax = np.max(max_JDs)
    
    # Create subplots for each focus cluster and other cells
    for i in range(len(focuses) + 1):    
        gene_JD_ci = gene_JD_cis[i]
        ax = fig.add_subplot(1, Cols, Position[i])
        
        # Apply cutoffs if specified and create heatmap
        if x_cutoff is not None: 
            ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap='Greys', 
                     norm=matplotlib.colors.LogNorm(vmax=vmax, vmin=vmin), aspect='auto')
            im1 = ax.imshow(gene_JD_ci[0:y_cutoff, 0:x_cutoff], cmap=cmap, 
                          norm=matplotlib.colors.LogNorm(vmax=vmax, vmin=vmin), aspect='auto')
        else: 
            ax.imshow(gene_JD, cmap='Greys', 
                     norm=matplotlib.colors.LogNorm(vmax=vmax, vmin=vmin), aspect='auto')
            im1 = ax.imshow(gene_JD_ci, cmap=cmap, 
                          norm=matplotlib.colors.LogNorm(vmax=vmax, vmin=vmin), aspect='auto')
        
        # Configure axis labels and add gamma line if specified
        ax.invert_yaxis()
        ax.set_xlabel('Spliced', size=14)
        ax.set_ylabel('Unspliced', size=14)
        if gamma is not None:
            ax.axline((0,0), slope=gamma)
        
        # Set subplot titles
        if i < len(focuses):
            t = clusters+'_'+str(focuses[i])
            ax.set_title(t, size=14)
        else:
            ax.set_title('Other Cells', size=14)
    
    # Create subplot for all cells
    ax = fig.add_subplot(1,Cols,Position[-1])
    if x_cutoff is not None: 
        im = ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap='Greys', 
                      norm=matplotlib.colors.LogNorm(vmax=vmax, vmin=vmin), aspect='auto')
    else: 
        im = ax.imshow(gene_JD, cmap='Greys', 
                      norm=matplotlib.colors.LogNorm(vmax=vmax, vmin=vmin), aspect='auto')    
    
    # Configure final subplot
    ax.invert_yaxis()
    ax.set_xlabel('Spliced', size=14)
    ax.set_ylabel('Unspliced', size=14)
    ax.set_title('All Cells', size=14)
    
    # Add colorbars
    cb_ax = fig.add_axes([1.0,.164,.01,.554])
    fig.colorbar(im1,orientation='vertical',cax=cb_ax)
    cb_ax = fig.add_axes([1.07,.164,.01,.554])
    fig.colorbar(im,orientation='vertical',cax=cb_ax)
    
    # Add gamma/degradation constant line if specified
    if gamma is not None:
        ax.axline((0,0), slope=gamma)    
    
    # Adjust layout and save figure
    fig.tight_layout(pad=1)
    focus_str = '_'.join(str(f) for f in focuses)
    savestring = f"{gene_name}_{focus_str}_HeatMap_by_{clusters}.png"
    plt.savefig(savestring, edgecolor='black', dpi=300, bbox_inches="tight", facecolor='white')