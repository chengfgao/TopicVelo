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

'''
For visualizing joint distributions within/across clusters 
For a given gene:
1. Scatter plot highlighting Proportions from different clusters for each point in the discrete distribution
2. Heatmap of the joint distributions in different clusters 
'''
def plot_exper_jd_clusters(adata, gene_name, color_by = 'lda_cluster',
                            layers = 'raw_spliced', x_scatters = 2, sp =0.15, all_cells = False, 
                            x_cutoff = None, y_cutoff = None, markersize = 10, alpha =1, 
                            vmin = None, vmax = None):
    
    gene_id = adata.var.index.get_loc(gene_name)
    gene_S = adata.layers['raw_spliced'][:, gene_id ].toarray().flatten().astype(np.uint64)
    gene_U = adata.layers['raw_unspliced'][:, gene_id ].toarray().flatten().astype(np.uint64)
    
    #get the cell type names, number of types, and number of cells for each type
    types = adata.obs[color_by].unique()
    num_types = adata.obs[color_by].nunique()
    n_cells_types = adata.obs[color_by].value_counts()
    
    #allocate to scatters based on color_by
    #Put x_scatter in a row with center of the x-coordinates as [x-a, x+a]
    #limit the center of y-coordinates between [y-a, y+a]
    n_row = int(np.ceil(num_types/x_scatters))
    last_row_size = num_types%x_scatters
    y_shift = np.linspace(-sp, sp, num=n_row)
    x_shift = np.linspace(-sp, sp, num=x_scatters)
    x_last_row_shift = np.linspace(-a, a, num=last_row_size)
    shifts = []
    for i in range(n_row):
        for j in range(x_scatters):
            shifts.append((x_shift[j], y_shift[i]))
    for i in range(last_row_size):
        shifts.append((x_last_row_shift[i], y_shift[-1]))
    fig, ax = plt.subplots()
    
    #for each cell type, tally the distribution
    for i in range(num_types):
        ci = types[i]
        ci_indices = [i for i, x in enumerate(list(adata.obs[color_by])) if x == ci]
        gene_U_ci = gene_U[ci_indices]
        gene_S_ci = gene_S[ci_indices]
        s_max_ci = int(np.max(gene_S_ci))
        u_max_ci = int(np.max(gene_U_ci))
        gene_JD_ci = joint_distribution_analysis_exper(gene_U_ci, gene_S_ci)
        min_JD = np.min(gene_JD_ci[np.nonzero(gene_JD_ci)])
        U = []
        S = []
        dist_ci = []
        f = lambda a: (np.log10(a/min_JD*10))*(markersize)         # function to calculate size from data
        g = lambda b: np.float_power(10, b/markersize )/10*min_JD  # inverse function to calc. data from size
        
        for u in range(u_max_ci+1):
            for s in range(s_max_ci+1):
                dist_ci_us = gene_JD_ci[u,s]
                if dist_ci_us> 0:
                    U.append(u)
                    S.append(s)
                    dist_ci.append(dist_ci_us)
        dist_ci = np.array(dist_ci)
        #dist_ci = 1-np.exp(-rate*dist_ci)
        S = S+shifts[i][0]
        U = U+shifts[i][1]
        label = color_by+' '+ci
        sc = plt.scatter(S , U, s=f(dist_ci), label=label, alpha = alpha)
    if all_cells:
        s_max = int(np.max(gene_S))
        u_max = int(np.max(gene_U))
        U = []
        S = []
        dist = []
        gene_JD = joint_distribution_analysis_exper(gene_U, gene_S)
        for u in range(u_max+1):
            for s in range(s_max+1):
                dist_us = gene_JD[u,s]
                if dist_us> 0:
                    U.append(u)
                    S.append(s)
                    dist.append(dist_us)
        dist = np.array(dist)
        sc = plt.scatter(S, U, s=f(dist), label = 'All Cells', alpha=0.5)
    size_legend = pyplot.legend(*sc.legend_elements("sizes", color='k', func=g), 
                                labelspacing=0.5, frameon=False,
                                loc='upper left', bbox_to_anchor=(0.994, 0.55))
    color_legend = pyplot.legend(loc='upper right', bbox_to_anchor=(1.32, 1.03),  frameon=False, labelspacing=1)
    plt.xlabel('Spliced', size=20)
    plt.ylabel('Unspliced', size=20)
    tt= 'Experimental JD ' + 'for '+ gene_name
    plt.title(tt, size=16)
    ax.add_artist(color_legend)
    ax.add_artist(size_legend)
    savestring = gene_name + '_by_'+ color_by+'.png'
    plt.savefig(savestring, edgecolor='black', dpi=300, bbox_inches = "tight", facecolor='white')
    
def heatmap_exper_cluster_focus_jd(adata, gene_name, clusters= 'lda_cluster', focus = '0', gamma = None, 
                            S_layer = 'raw_spliced', U_layer = 'raw_unspliced', cmap = 'YlOrRd',
                            x_cutoff = None, y_cutoff = None, height =10, 
                            vmin = 1e-4, vmax = None, log_scale = True):
    
    gene_id = adata.var.index.get_loc(gene_name)
    gene_S = adata.layers[S_layer][:, gene_id ].toarray().flatten().astype(np.uint64)
    gene_U = adata.layers[U_layer][:, gene_id ].toarray().flatten().astype(np.uint64)
    #range for plotting
    U_max = int(np.max(gene_U)+1)
    S_max = int(np.max(gene_S)+1)
    types = adata.obs[clusters].unique()

    #number of columns is the number of clusters/topics +(optional) all cells
    Cols = 3
    # Create a Position index
    Position = range(1,Cols+1)
    max_JDs = []
    gene_JD_cis = []
    #width by height
    #width = np.ceil(S_max/U_max*height*Cols)
    width = np.ceil(Cols*height)
    
    if y_cutoff is not None:
        width = np.ceil(x_cutoff/y_cutoff*height*Cols)
    fig = plt.figure(1, figsize=(width, height))

 
    # Selecting the axis-X making the bottom and top axes False.
    plt.tick_params(axis='x', which='both', bottom=False,
                top=False, labelbottom=False)
    # Selecting the axis-Y making the right and left axes False
    plt.tick_params(axis='y', which='both', right=False,
                left=False, labelleft=False)
  
    # Iterating over all the axes in the figure
    # and make the Spines Visibility as False
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    tt= 'Experimental JD ' + 'for '+ gene_name + '\n'
    plt.title(tt, size=16)
    #cells in the topic of interest
    f_indices = np.array([i for i, x in enumerate(list(adata.obs[clusters])) if str(x) == focus])  
    #Other cells
    other_indices = np.array(list(set(np.arange(adata.n_obs))-set(f_indices)))
    
    
    for i in range(2):
        if i == 0:
            ci_indices = f_indices
        else:
            ci_indices = other_indices
        gene_U_ci = gene_U[ci_indices]
        gene_S_ci = gene_S[ci_indices]
        #get jd for the cluster
        gene_JD_ci = joint_distribution_analysis_exper(gene_U_ci, gene_S_ci)
        
        #put the cluster jd into a large box
        Ui_max = int(np.max(gene_U_ci)+1)
        Si_max = int(np.max(gene_S_ci)+1)
        M = np.zeros((U_max, S_max))
        M[0:Ui_max, 0:Si_max] = gene_JD_ci
        gene_JD_cis.append(M)
        #get the nonzero min and max of the joint distribution
        max_JDs.append(np.max(gene_JD_ci))
        
    #for all cells
    gene_JD = joint_distribution_analysis_exper(gene_U, gene_S)  
    #get the nonzero min and max of the joint distribution
    max_JDs.append(np.max(gene_JD))
    
    vmax = np.max(max_JDs)
    
    for i in range(2):    
        gene_JD_ci = gene_JD_cis[i]
        # add every single subplot to the figure with a for loop
        ax = fig.add_subplot(1,Cols,Position[i])
        #ax.set_aspect('auto')
        if x_cutoff is not None: 
            ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap='Greys', norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin), aspect='auto')
            im1 = ax.imshow(gene_JD_ci[0:y_cutoff, 0:x_cutoff], cmap=cmap, norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin), aspect='auto')
        else: 
            ax.imshow(gene_JD, cmap='Greys', norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin), aspect='auto')
            im1 = ax.imshow(gene_JD_ci, cmap=cmap, norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin), aspect='auto')
        ax.invert_yaxis()
        ax.set_xlabel('Spliced', size=14)
        ax.set_ylabel('Unspliced', size=14)
        
        if gamma is not None:
            ax.axline((0,0), slope=gamma)
        if i == 0:
            t = clusters+'_'+focus
            ax.set_title(t, size=14)
        else:
            ax.set_title('Other Cells', size=14)
    
    ax = fig.add_subplot(1,Cols,Position[-1])
    if x_cutoff is not None: 
        im = ax.imshow(gene_JD[0:y_cutoff, 0:x_cutoff], cmap='Greys', norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin), aspect='auto')
    else: 
        im = ax.imshow(gene_JD, cmap='Greys', norm=matplotlib.colors.LogNorm(vmax = vmax, vmin = vmin), aspect='auto')    
    ax.invert_yaxis()
    ax.set_xlabel('Spliced', size=14)
    ax.set_ylabel('Unspliced', size=14)
    ax.set_title('All Cells', size=14)
    
    #add the two color bars
    cb_ax = fig.add_axes([1.0,.164,.01,.554])
    fig.colorbar(im1,orientation='vertical',cax=cb_ax)
    cb_ax = fig.add_axes([1.07,.164,.01,.554])
    fig.colorbar(im,orientation='vertical',cax=cb_ax)
    if gamma is not None:
        ax.axline((0,0), slope=gamma)    
    #fig.colorbar(im, shrink=0.85)
    fig.tight_layout(pad=1)
    savestring = gene_name+'_'+focus + '_HeatMap_by_'+ clusters+'.png'
    plt.savefig(savestring, edgecolor='black', dpi=300, bbox_inches = "tight", facecolor='white')
    