
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from itertools import permutations
import networkx as nx
import pandas as pd
import scipy.io
from scipy.signal import correlate
import os
import math
import shutil
import random
import warnings
warnings.filterwarnings('ignore')
import pickle
import time

# from HSIC.causality.utils.models import GAM
# from HSIC.causality.utils.independence import HSIC, dHSIC

import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

import seaborn as snspe
from sklearn.covariance import GraphicalLasso



# df = pd.read_excel('exp_1_preprocessed_may_2.xlsx')
filename = r'C:\Eminds pros\E-minds projects\fuhrlander\causal-data-engine\pipeline\windows\wt80_seg0_win000.csv'
df = pd.read_csv(filename)
filename_to_save = 'wt80_seg0_win000'  #G2R16
# df = df.head(10000)
print(df.columns)
print(df.shape)
print('min timestamp: ', df['timestamp'].min())
print('max timestamp: ', df['timestamp'].max())

column_names_to_select = [
    'timestamp',
    'wrot_avg_A_ValBl1',
    'wrot_avg_A_ValBl2',
    'wrot_avg_V_ValBl1',
    'wrot_min_TmpPwrSply_ValBl1',
    'wrot_min_A_ValBl1',
    'wrot_max_A_ValBl2',
    'wrot_sdv_TmpHtSinkPco_Bl1'
]

df = df[column_names_to_select]

column_names_final = list(df.columns)
print('column_names_final: ', column_names_final)

# RESULTS_DIRECTORY = f'{filename_to_save}'+"_"+str(df.shape[1])
RESULTS_DIRECTORY = "results"+"_"+str(df.shape[1])



M = df.values
ica_mask_data = df.T
num_nodes, samplesize = ica_mask_data.shape
print('num_nodes:', num_nodes)
print('samplesize: ',samplesize)
# Parameters
Ni = 0
Nt = df.shape[0] + Ni
N = Nt - Ni
Lmin =6  # CONFIG
Lmax =15   # CONFIG
Linc = 1
kurt = np.zeros((num_nodes, 1))
hsim = 0
Bstore = np.zeros((num_nodes, 1))
print('shape of Bstore:', Bstore.shape)
Lp = []
HSIC_dict = {}
hsic_overall_results = {}

ITER=5
bootstrap_exp=False

 



SET_LAMBDA = 0.1
ICA_lambda = np.log(samplesize) * SET_LAMBDA  # usually lamda is set to a constant times log(T), where T is sample size.
ICA_regu = 0.05
stablize_tol = 0.02
stablize_sparsify = True



@nb.njit('(int_[:,::1], float64[:,::1], int_, int_)')
def get_pr(idx, r, mmax, n):
    samplesize, numvars = idx.shape
    res = np.zeros((mmax, numvars), dtype=np.float64)
    for i in range(samplesize):
        for j in range(numvars):
            res[idx[i, j] - 1, j] += (1 - r[i, j]) ** 2 / 2
            res[idx[i, j], j] += 0.5 + r[i, j] * (1 - r[i, j])
            res[idx[i, j] + 1, j] += r[i, j] ** 2 / 2
    return res / n

@nb.njit('(int_[:,::1], float64[:,::1], float64[:,::1], float64)')
def get_psi(idx, logp, r, bandwidth):
    samplesize, numvars = idx.shape
    res = np.zeros((numvars, samplesize), dtype=np.float64)
    for i in range(samplesize):
        for j in range(numvars):
            res[j, i] += logp[idx[i, j] - 1, j] * (1 - r[i, j]) + \
                         logp[idx[i, j], j] * (2 * r[i, j] - 1) - \
                         logp[idx[i, j] + 1, j] * r[i, j]

    return res / bandwidth


def scorecond(data):
    n, numvars = data.shape
    bdwidth = 2 * (11 * np.sqrt(np.pi) / 20) ** (1 / 5) * (4 / (3 * n)) ** (1 / 5)  # repeated calcuearlyd for many times though

    # prewhitening
    data = data - data.mean(axis=0)
    T = np.sqrt((data * data).mean(axis=0)) # in shape (p,), same as data.std(axis=0)
    data = data / T

    # # Grouping the data into cells, idx gives the index of the cell
    # # % containing a datum, r gives its relative distance to the leftmost
    # # % border of the cell
    r = data / bdwidth
    idx = np.floor(r).astype(int)
    r = r - idx
    idx = idx - idx.min(axis=0) + 1  # 0 <= idx-1

    pr = get_pr(idx, r, idx.max() + 2, n)

    logp = np.log(pr, out=np.zeros_like(pr), where=(pr != 0))  # to contain log(cond. prob.)
    # entropy = np.log(bdwidth * T) - (pr * logp).sum(axis=0)  # in shape (numvars,)

    psi = get_psi(idx, logp, r, bdwidth)
    psi = psi - psi.mean(axis=1)[:, None]  # center psi, in shape (numvars, n)
    lam = (psi.T * data).sum(axis=0) / n - 1  # correction, lam in shape (numvars,)
    psi = ((psi.T - data * lam) / T).T

    return psi

def estim_beta_pham(x):
    '''
    @param x: data rows (k, T), k is numvars, T is sample size
    @return: beta in shape (k, T), the same shape as psi, and data
    '''
    return -1. * scorecond(np.copy(x.T, order='C'))


def adaptive_size(grad_new, grad_old, eta_old, z_old):
    alpha = 0 # 0.7
    up = 1.05 # 1.1 1.05
    down = 0.8 # 0.4 0.5 0.34 0.5
    z = grad_new + alpha * z_old
    etaup = (grad_new * grad_old) >= 0
    eta = eta_old * (up * etaup + down * (1 - etaup))
    eta[eta >= 0.03] = 0.03 # min(eta, 0.03)
    return eta, z


def natural_grad_Adasize_Mask_regu(X, Mask, regu, init_W=None):
    N, T = X.shape
    mu = 3e-3 # 3e-3 # original matlab code: 3e-3
    itmax = 5000 # 10000 #18000 # 18000
    Tol = 1e-6 # 1e-4, now smaller. otherwise early stopped
    num_edges = Mask.sum()

    # initilization of W
    if init_W is None:
        WW = np.eye(N, N)
        for i in range(N):
            Ind_i = np.where(Mask[i] != 0)[0]
            X_Ind_i = X[Ind_i]
            WW[i, Ind_i] = -0.5 * (X[i] @ X_Ind_i.T) @ np.linalg.pinv(X_Ind_i @ X_Ind_i.T) # regress each Xi on unmasked nodes
        W = 0.5 * (WW + WW.T)
    else:
        W = np.copy(init_W)
    W[np.diag_indices(N)] = 1   # just to make sure

    z = np.zeros((N, N))
    eta = mu * np.ones_like(W)
    y_psi = np.zeros_like(X)
    y_psi0 = np.zeros_like(X)
    Grad_W_o = None

    init_avg_gradient_curve = []
    init_loss_curve = []

    for iter in range(itmax):
        # if iter % 100 == 0: print(f'natural_grad_Adasize_Mask_regu: ======== {iter}/{itmax} ========')
        y = W @ X
        argsort_y = np.argsort(y, axis=1)
        # update W: linear ICA with marginal score function estimated from data...
        if iter % 12 == 0:
            y_psi = np.copy(estim_beta_pham(y))
            y_psi0 = np.take_along_axis(y_psi, argsort_y, axis=1)
        else:
            y_psi[(np.tile(np.arange(N), (T, 1)).T, argsort_y)] = np.copy(y_psi0)
        ##################################################################

        # with regularization to make W small
        Grad_W_n = y_psi @ X.T / float(T) + np.linalg.inv(W.T) - 2 * regu * W
        if iter == 0: Grad_W_o = np.copy(Grad_W_n)
        eta, z = adaptive_size(Grad_W_n, Grad_W_o, eta, z)
        delta_W = eta * z
        W = W + delta_W * Mask

        avg_gradient = np.abs(Grad_W_n * Mask).sum() / num_edges
        init_avg_gradient_curve.append(avg_gradient)
        if avg_gradient < Tol: break

        Grad_W_o = np.copy(Grad_W_n)

    return W, np.array(init_avg_gradient_curve), np.array(init_loss_curve)



def sparseica_W_adasize_Alasso_mask_regu(lamda, Mask, X, regu, init_W=None):
    ''' ICA with SCAD penalized entries of the de-mixing matrix
    @param lamda: float, usually lamda is set to a constant times log(T), where T is sample size
    @param Mask: N*N 0 1 matrix, only updates the 1 entries on gradient
        in 2-step CD, if no mask, it's set to ones(N,N) - eye(N)
    @param X: data matrix in shape N*T, where N is the number of nodes, T is sample size. don't need to be whitened
    @param regu: float, e.g., 0.00, 0.002, 0.01, 0.05
    @param init_W: N*N matrix, initial value of W (usually as None)
    @return:
    '''
    N, T = X.shape
    XX = X - X.mean(axis=1)[:, None]
    # To avoid instability
    std_XX = XX.std(axis=1, ddof=1) # note that the std function in matlab is sample stddev, so ddof=1 here
    XX = np.diag(1. / std_XX) @ XX # it should be @XX here, not @X, in case X is not zero-meaned. (bug in matlab code)
    Refine = True
    num_edges = Mask.sum()

    # learning rate
    mu = 1e-3 # 1e-6
    beta = 0 # 1
    m = 60 # for approximate the derivative of |.|
    itmax = 15000 # i.e., now we don't use penal. 8000 # 10000 # 15000 # 15000 # 10000
    Tol = 1e-6

    # initiliazation
    # print('Initialization....')
    WW, init_avg_gradient_curve, init_loss_curve = natural_grad_Adasize_Mask_regu(XX, Mask, regu, init_W=init_W)

    omega1 = 1. / np.abs(WW[Mask != 0])
    # to avoid instability
    Upper = 3 * omega1.mean()
    omega1[omega1 > Upper] = Upper
    omega = np.zeros((N, N))
    omega[Mask != 0] = omega1
    W = np.copy(WW)

    z = np.zeros((N, N))
    eta = mu * np.ones_like(W)
    W_old = W + np.eye(N)
    grad_new = np.copy(W_old)
    y_psi = np.zeros_like(XX)
    y_psi0 = np.zeros_like(XX)
    grad_old = None
    y = np.zeros_like(XX)

    penal_avg_gradient_curve = []
    penal_loss_curve = []

    # print('Penalization....')
    for iter in range(itmax):
        # if iter % 100 == 0: print(f'sparseica_W_adasize_Alasso_mask_regu: ======== {iter}/{itmax} ========')
        y = W @ XX
        avg_gradient = np.abs(grad_new * Mask).sum() / num_edges
        penal_avg_gradient_curve.append(avg_gradient)
        if avg_gradient < Tol:
            if Refine:
                Mask = np.abs(W) > 0.01
                Mask[np.diag_indices(N)] = 0
                lamda = 0.
                Refine = False
            else:
                break

        # update W: linear ICA with marginal score function estimated from data...
        argsort_y = np.argsort(y, axis=1)
        if iter % 8 == 0:
            y_psi = np.copy(estim_beta_pham(y))
            y_psi0 = np.take_along_axis(y_psi, argsort_y, axis=1)
        else:
            y_psi[(np.tile(np.arange(N), (T, 1)).T, argsort_y)] = np.copy(y_psi0)

        dev = omega * np.tanh(m * W)

        # with additional regularization
        regu_l1 = regu / 2.
        grad_new = y_psi @ XX.T / T + np.linalg.inv(W.T) - \
                     4 * beta * (np.diag(np.diag(y @ y.T / T)) - np.eye(N)) * (y @ XX.T / T) - \
                        dev * lamda / T - \
                            2 * regu_l1 * W # seems that it should be 0-mean-1-std XX here. in original code, it's X here?
        if iter == 0: grad_old = np.copy(grad_new)

        # adaptive size
        eta, z = adaptive_size(grad_new, grad_old, eta, z)
        delta_W = eta * z
        W = W + 0.9 * delta_W * Mask
        grad_old = np.copy(grad_new)

    # re-scaling
    W = np.diag(std_XX) @ W @ np.diag(1. / std_XX)
    WW = np.diag(std_XX) @ WW @ np.diag(1. / std_XX)    # WW is returned by initialization
    y = np.diag(std_XX) @ y
    Score = omega * np.abs(W)
    return y, W, WW, Score, \
           init_avg_gradient_curve, init_loss_curve, \
           np.array(penal_avg_gradient_curve), np.array(penal_loss_curve)



def from_W_to_B(W, tol=0.02, sparsify=True):
    '''
    find the best (row) permutation among nodes s.t. the system is stable. python codes transearlyd by Minghao Fu.
    @param W: the demixing matrix returned by the above `sparseica_W_adasize_Alasso_mask_regu`
    @param tol: tolerance, for sparsity thresholding
    @param sparsify: whether to set too small values in adjmat to 0
    @return:
        B: the adjacency matrix
        perm: the nodes permutation
    '''
    dd = W.shape[0]
    W_max = np.max(np.abs(W))
    if sparsify:
        W = W * (np.abs(W) >= W_max * tol)

    P_all = np.array(list(permutations(range(dd))))
    Num_P = len(P_all)
    EyeI = np.eye(dd)

    Loop_strength_bk = np.inf
    B, perm = None, None
    for i in range(Num_P):
        W_p = W[P_all[i], :]
        if np.min(np.abs(np.diag(W_p))) != 0:
            W_p1 = np.diag(1 / np.diag(W_p)) @ W_p
            W_p2 = EyeI - W_p1
            Loop_strength = 0
            B_prod = W_p2
            for jj in range(dd - 1):
                B_prod = B_prod @ W_p2
                Loop_strength += np.sum(np.abs(np.diag(B_prod)))

            if Loop_strength < Loop_strength_bk:
                Loop_strength_bk = Loop_strength
                B = W_p2
                perm = P_all[i]

    return B, perm


# estimate function
def two_step_CD(data, allowed_directed_edges=None, forbidden_directed_edges=None, init_mask_by_lasso=False, init_W=None):
    """
    Estimate the directed LiNGAM graph from data.
    Parameters
    ----------
    data : array, shape (n_features, n_samples). The data to estimate the graph from.
    allowed_directed_edges : List<Tuple>, optional. The allowed directed edges. Default as None.
    forbidden_directed_edges : List<Tuple>, optional. The forbidden directed edges. Default as None.
    init_mask_by_lasso : bool, optional. If False (default), Mask is set to all ones except for diagonals. If True (default),
            use Lasso, i.e., an edge X-Y is allowed only if X and Y are conditionally dependent given all other variables.
    Returns
    -------
    B : array, shape (n_features, n_features). The estimated directed graph with edge weights.
    """

    # 1. Hyper-parameters setting for sparseica_W_adasize_Alasso_mask_regu.
    # num_nodes, samplesize = data.shape
    # ICA_lambda = np.log(samplesize) * 4  # usually lamda is set to a constant times log(T), where T is sample size.
    # ICA_regu = 0.05
    # stablize_tol = 0.02
    # stablize_sparsify = True

    # 2. Set init Mask.
    if init_mask_by_lasso:  # TODO: use the alasso in matlab code
        gl = GraphicalLasso()
        gl.fit(data.T)
        ICA_Mask = np.abs(gl.precision_) > 0.05 * np.max(np.abs(gl.precision_))
    else: ICA_Mask = np.ones((num_nodes, num_nodes))
    ICA_Mask[np.diag_indices(num_nodes)] = 0

    if allowed_directed_edges:
        for pa, ch in allowed_directed_edges: ICA_Mask[ch, pa] = ICA_Mask[pa, ch] = 1
    if forbidden_directed_edges:
        for pa, ch in forbidden_directed_edges:
            if (ch, pa) in forbidden_directed_edges: # only disable at Mask when it is 2-way forbidden.
                ICA_Mask[ch, pa] = ICA_Mask[pa, ch] = 0

    # 3. Run 2-step ICA and get the estimated demixing matrix W.
    print('ICA_lambda: ', ICA_lambda)
    if init_W:
        print('Stage: Apriori Matrix is used')
        _, W, _, _, _, _, _, _ = sparseica_W_adasize_Alasso_mask_regu(ICA_lambda, ICA_Mask, data, ICA_regu, init_W)
    else:
        _, W, _, _, _, _, _, _ = sparseica_W_adasize_Alasso_mask_regu(ICA_lambda, ICA_Mask, data, ICA_regu)

    # 4. Further process the estimated demixing matrix W so that the corresponding causal system is stable.
    adjacency_matrix, nodes_permutation = from_W_to_B(W, tol=stablize_tol, sparsify=stablize_sparsify)

    # 5. Check if the forbidden directed edges are present.
    forbidden_edge_presented = forbidden_directed_edges is not None and \
                               any([adjacency_matrix[ch, pa] != 0 for pa, ch in forbidden_directed_edges])
    if forbidden_edge_presented:
        new_Mask = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
        # note: we cannot use lasso mask now. Need other possible edges (even cond.inds are vioearlyd) to explain the data.
        for pa, ch in forbidden_directed_edges: new_Mask[ch, pa] = 0    # the inverse direction (ch->pa) is allowed.
        init_W = np.eye(num_nodes) - adjacency_matrix * new_Mask    # trust the estimated causal system except for the forbidden edges.
        print('ICA_lambda: ', ICA_lambda)
        _, W, _, _, _, _, _, _ = sparseica_W_adasize_Alasso_mask_regu(ICA_lambda, new_Mask, data, ICA_regu, init_W)
        adjacency_matrix, nodes_permutation = from_W_to_B(W, tol=stablize_tol, sparsify=stablize_sparsify)

    return adjacency_matrix, W, nodes_permutation



def dg_plot(AdS, iteration_number, num_nodes, column_names):

    Lwmult = 5.0  # Select link width
    sigLs = 0.1  # Value below which Structural Causal Factors are considered NOT significant

    # Create a directed graph
    G = nx.DiGraph()


    names = column_names


    s = [i for i in range(1, num_nodes)]
    s_expanded = [node for node in range(1, num_nodes + 1) for _ in range(num_nodes)]
    t_expanded = [(i % num_nodes) + 1 for i in range(len(s_expanded))]
    # print(s_expanded, len(s_expanded))
    # print(t_expanded, len(t_expanded))




    # weightsS = np.hstack((AdS[:,0], AdS[:,1], AdS[:,2], AdS[:,3]))
    weightsS = np.hstack([AdS[:, i] for i in range(num_nodes)])
    # print(weightsS)
    # print(len(weightsS))



    NodeTable = pd.DataFrame({'DGnodes': names})

    EdgeTable = pd.DataFrame({'EndNodes': list(zip(s_expanded, t_expanded)), 'Weight': weightsS})
    EdgeTableZ=EdgeTable
    weightsZ=weightsS
    LWidthsZ=Lwmult*abs(weightsZ)

    for i in range(len(weightsS)-1, -1, -1):
        if abs(EdgeTable.iloc[i, 1]) < sigLs:
            EdgeTableZ = EdgeTableZ.drop(i)
            LWidthsZ = np.delete(LWidthsZ, i)

    
    # Add nodes from NodeTable
    for index, row in NodeTable.iterrows():
        G.add_node(row['DGnodes'])

    # Add edges from EdgeTableZ
    for index, row in EdgeTableZ.iterrows():
        source, target = row['EndNodes']
        G.add_edge(names[source - 1], names[target - 1], weight=row['Weight'])


    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    

    # Positions for nodes
    num_rows = math.ceil(math.sqrt(num_nodes))
    num_cols = math.ceil(num_nodes / num_rows)
    pos = {column_names[i]: (j % num_cols, -j // num_cols) for i, j in enumerate(range(num_nodes))}



    # Plot graph with curved edges and adjusted label positions
    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue')

    node_labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=14, font_color='red')

    # Draw curved edges
    arcs = nx.draw_networkx_edges(G, pos, ax=ax, width=LWidthsZ, edge_color='blue', connectionstyle='arc3,rad=0.1')
    edge_labels = {edge: f"{weight:.2f}" for edge, weight in nx.get_edge_attributes(G, 'weight').items()}
    print("edge_labels")
    print(edge_labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, rotate=False)
    plt.axis('off')
    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(iteration_number))
    os.makedirs(iteration_folder_path, exist_ok=True)
    plot_filename = os.path.join(iteration_folder_path, f'NodeGraph_{iteration_number}.png')
    plt.savefig(plot_filename)
    

def HSIC_independence_test(results_dict):

    # Initialize variables to track the closest key and its corresponding absolute difference
    closest_key = None
    closest_difference = float('inf')

    # Iterate through the dictionary
    for key, values in results_dict.items():
        # Calcuearly the sum of absolute differences between each value and zero
        abs_diff_sum = sum(abs(value) for value in values.values())
        
        # Update closest key if the current key has a smaller absolute difference
        if abs_diff_sum < closest_difference:
            closest_key = key
            closest_difference = abs_diff_sum

    return closest_key


def HSIC_Agg_Plot(data, x_ticks_min_range,x_ticks_max_range):
    # Initialize a list of colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'navy', 'teal', 'gold']
    
    # Get all unique key pairs as y-axis labels
    y_labels = sorted(set(pair for d in data.values() for pair in d.keys()))
    
    # Initialize an empty dictionary to store plots
    plots = {}
    
    # Create a subplot
    fig, ax = plt.subplots()
    
    # Iterate over each key pair and plot the corresponding data
    for idx, pair in enumerate(y_labels):
        # Extract x and y values for the current pair
        x_values = []
        y_values = []
        for key, value in data.items():
            if pair in value:
                x_values.append(key)
                y_values.append(value[pair])
            else:
                x_values.append(None)
                y_values.append(None)
        
        # Plot the data for the current pair
        plots[pair] = ax.plot(x_values, y_values, color=colors[idx % len(colors)], label=f"Pair: {pair}", marker='o', linestyle='-')
    
    # Set custom x-axis labels
    x_ticks_max_range = x_ticks_max_range + 1
    plt.xticks(range(x_ticks_min_range, x_ticks_max_range))
    
    # Add labels and title to the plot
    plt.xlabel('Aggregation Key')
    plt.ylabel('HSIC')
    plt.title('Line Plot for Key Pairs')
    
    # Move the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Show the plot
    plt.grid(True)
    
    # Show the plot
    # plt.figure(figsize=(12, 8))
    # plt.tight_layout()
    plt.savefig(RESULTS_DIRECTORY+"/HSIC_Agg_Plot.png")


    # save the HSIC dict to excel file (optional)
    data_list = [(key1, key2[0], key2[1], value) for key1, subdict in data.items() for key2, value in subdict.items()]

    # Convert list of tuples to DataFrame
    df = pd.DataFrame(data_list, columns=['Aggregation_Factor', 'row_A', 'row_B', 'Value'])

    # Write DataFrame to Excel file
    df.to_excel(RESULTS_DIRECTORY+"/HSIC.xlsx", index=False)

def movmean(x, window, endpoints='shrink', nanflag='includenan'):
    """
    Compute the moving mean of an array.
    
    Parameters:
    x : array-like
        Input data (1D array).
    window : int or list of two ints
        If int, it specifies the length of the moving window.
        If list of two ints, [NB, NF] specifies the number of previous and next elements.
    endpoints : str
        Controls how to calcuearly means at the endpoints:
        'shrink' (default), 'fill', or 'discard'.
    nanflag : str
        Specifies how NaN values are treated:
        'includenan' (default) or 'omitnan'.
    
    Returns:
    numpy.ndarray
        The moving mean of the input data.
    """
    
    x = np.asarray(x)  # Convert input to numpy array
    n = len(x)
    
    if isinstance(window, int):
        nb = window // 2
        nf = window - nb - 1
    elif isinstance(window, list) and len(window) == 2:
        nb, nf = window
    else:
        raise ValueError("Window must be an integer or a list of two integers.")
    
    # Prepare output array
    y = np.full(n, np.nan)  # Initialize with NaN
    
    for i in range(n):
        start_idx = max(0, i - nb)
        end_idx = min(n, i + nf + 1)
        
        window_data = x[start_idx:end_idx]
        
        if nanflag == 'omitnan':
            window_data = window_data[~np.isnan(window_data)]
        
        if len(window_data) == 0:
            continue
        
        if endpoints == 'shrink':
            y[i] = np.mean(window_data)
        elif endpoints == 'fill':
            if len(window_data) < (nb + nf + 1):
                padded_window = np.pad(window_data, (0, (nb + nf + 1) - len(window_data)), constant_values=np.nan)
                y[i] = np.mean(padded_window)
            else:
                y[i] = np.mean(window_data)
        elif endpoints == 'discard':
            if len(window_data) < (nb + nf + 1):
                y[i] = np.nan  # Keep as NaN if not enough elements
            else:
                y[i] = np.mean(window_data)

    return y


def zero_mean_unit_variance(ndarray):
    # Create a copy of the original ndarray
    ndarray_std = ndarray.copy()
    
    # Calcuearly the mean and standard deviation for each column
    means = np.mean(ndarray_std, axis=0)
    std_devs = np.std(ndarray_std, axis=0)
    
    # Normalize each column
    for i in range(ndarray_std.shape[1]):
        ndarray_std[:, i] = (ndarray_std[:, i] - means[i]) / std_devs[i]
    
    return ndarray_std

def add_noise_to_duplicates(arr, noise_scale=0.01):
    # Add non guassian noise
    noise = np.random.uniform(0, noise_scale, arr.shape)
    return arr + noise



def get_bootstrap_data(arr):
    """
    Generate multiple bootstrap samples from the original dataset.
    
    Parameters:
    - arr: The original dataset.
  
    Returns:
    - noisy bootstrap sample.
    """
    
    # Perform bootstrapping with replacement along the specified axis
    col_indices = np.random.choice(arr.shape[1], size=arr.shape[1], replace=True)
        
       
    bootstrap_sample = arr[:, col_indices]
        
    # Add noise to the bootstrap sample (optional)
    noisy_sample = add_noise_to_duplicates(bootstrap_sample, noise_scale=0.01)
        
    
    return noisy_sample


# expanding 5*500 to 5*5000 by adding noise
# 5*500, bootstrap again to the same shape by adding noise

def rbf_kernel(X, sigma):
    """Compute the RBF (Gaussian) kernel matrix."""
    G = np.sum(X**2, axis=1)
    Q = np.tile(G, (len(G), 1))
    R = Q.T
    dists = Q + R - 2 * np.dot(X, X.T)
    dists = np.maximum(dists, 0)  # Ensure no negative distances
    return np.exp(-dists / (2 * sigma**2))

def hsic(X, Y, alpha=0.05):
    """Compute the HSIC test statistic and threshold."""
    m = X.shape[0]

    # Median heuristic for kernel bandwidth
    def median_heuristic(data):
        G = np.sum(data**2, axis=1)
        Q = np.tile(G, (len(G), 1))
        R = Q.T
        dists = Q + R - 2 * np.dot(data, data.T)
        dists = dists[np.triu_indices_from(dists, k=1)]  # Upper triangle distances
        return np.sqrt(0.5 * np.median(dists[dists > 0]))

    sigma_x = median_heuristic(X)
    sigma_y = median_heuristic(Y)

    # Centering matrix
    H = np.eye(m) - np.ones((m, m)) / m

    # Compute kernel matrices
    K = rbf_kernel(X, sigma_x)
    L = rbf_kernel(Y, sigma_y)

    # Centered kernel matrices
    Kc = H @ K @ H
    Lc = H @ L @ H

    # Test statistic
    test_stat = (1 / m) * np.sum(Kc * Lc)

    # Variance under H0
    var_hsic = (1 / 6) * (Kc * Lc)**2
    var_hsic = (1 / m / (m - 1)) * (np.sum(var_hsic) - np.trace(var_hsic))
    var_hsic *= 72 * (m - 4) * (m - 5) / m / (m - 1) / (m - 2) / (m - 3)

    # Mean under H0
    K -= np.diag(np.diag(K))
    L -= np.diag(np.diag(L))
    mu_x = (1 / m / (m - 1)) * np.sum(K)
    mu_y = (1 / m / (m - 1)) * np.sum(L)
    m_hsic = (1 / m) * (1 + mu_x * mu_y - mu_x - mu_y)

    # Gamma distribution parameters
    alpha_param = m_hsic**2 / var_hsic
    beta_param = var_hsic * m / m_hsic

    # Threshold
    thresh = gamma.ppf(1 - alpha, alpha_param, scale=beta_param)

    return test_stat, thresh


def compute_hsic_for_all_pairs(data):
    """Compute HSIC for all distinct pairs of rows in the dataset."""
    num_nodes = data.shape[0]
    hsic_results = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Upper triangular part
            X = data[i, :].reshape(-1, 1)
            Y = data[j, :].reshape(-1, 1)
            test_stat, thresh = hsic(X, Y)
            hsic_results.append((i, j, test_stat, thresh))

    return hsic_results



def visualize_hsic_results(hsic_results, num_nodes, L):
    """Visualize HSIC results on a grid with color coding."""
    grid = np.zeros((num_nodes, num_nodes))

    for i, j, test_stat, thresh in hsic_results:
        if test_stat < thresh:
            grid[i, j] = 1  # Green for testStat < thresh
        else:
            grid[i, j] = -1  # Red for testStat >= thresh

    # Create a custom colormap with red and green
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "black", "green"])

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=cmap, interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(label="HSIC Test Result")
    plt.title("HSIC Test Results for All Pairs")
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    plot_filename_hsic = os.path.join(iteration_folder_path, f'hsic_{L}.png')
    plt.savefig(plot_filename_hsic)


def count_green_blocks(hsic_results, num_nodes):
    """Count green blocks (test_stat < thresh) for each node."""
    green_counts = np.zeros(num_nodes, dtype=int)
    
    for i, j, test_stat, thresh in hsic_results:
        if test_stat < thresh:
            green_counts[i] += 1
            
    total_green = green_counts.sum()
    return total_green, green_counts

def plot_green_blocks_over_iterations(iteration_results, num_nodes):
    iterations = []
    total_greens = []

    for iteration, hsic_list in iteration_results.items():
        total_green, _ = count_green_blocks(hsic_list, num_nodes)
        iterations.append(iteration)
        total_greens.append(total_green)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, total_greens, marker='o', linestyle='-', color='green')
    plt.xlabel('Iteration Number')
    plt.ylabel('Total Number of Green Blocks')
    plt.title('Green Blocks Across Iterations')
    plt.grid(True)
    plt.xticks(iterations)
    plot_filename_hsic = os.path.join(RESULTS_DIRECTORY, 'hsic_overall.png')
    plt.savefig(plot_filename_hsic)

def plot_green_blocks_stem(iteration_results, num_nodes):
    iterations = []
    total_greens = []

    for iteration, hsic_list in sorted(iteration_results.items()):
        total_green, _ = count_green_blocks(hsic_list, num_nodes)
        iterations.append(iteration)
        total_greens.append(total_green)


    # Normalize total_greens to [0,1]
    max_green = max(total_greens) if total_greens else 1  # Avoid division by zero
    normalized_greens = [tg / max_green for tg in total_greens]

    plt.figure(figsize=(10, 6))
    # Stem plot: vertical lines (“stems”) with markers at the tips
    plt.stem(
        iterations,
        normalized_greens,
        linefmt='-',       # stem line style
        markerfmt='o',     # marker style
        basefmt=' '        # no baseline
    )

    plt.xlabel('Iteration Number')
    plt.ylabel('Normalized Total Number of Green Blocks')
    plt.title('Normalized Green Blocks Across Iterations (Stem Plot)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(iterations)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plot_filename_hsic = os.path.join(RESULTS_DIRECTORY, 'hsic_overall.png')
    plt.savefig(plot_filename_hsic)
    plt.close()
    # Return the iteration number with the highest number of green blocks
    max_index = total_greens.index(max(total_greens))
    return iterations[max_index]

    # plt.xlabel('Iteration Number')
    # plt.ylabel('Total Number of Green Blocks')
    # plt.title('Green Blocks Across Iterations (Stem Plot)')
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.xticks(iterations)
    # plot_filename_hsic = os.path.join(RESULTS_DIRECTORY, 'hsic_overall.png')
    # plt.savefig(plot_filename_hsic)




def edge_weights_to_dataframe(edge_weights, keys):
    # Initialize a dictionary to store values for each key
    data = {key: 0 for key in keys}  # Default value is 0
    
    # Popuearly the dictionary with edge_weights values
    for key, value in edge_weights.items():
        data[key] = float(value)  # Convert string to float
    
    # Convert the dictionary to a DataFrame (single row)
    df = pd.DataFrame([data])
    return df

def store_mean_std_dict_for_non_zero_entries(b_matrices, nodes):
    """
    Calcuearly the mean and standard deviation of each non-zero entry 
    across a list of 7x7 numpy ndarrays and store them in a dictionary.
    
    Args:
        b_matrices (list of np.ndarray): A list of 7x7 numpy ndarrays.
        nodes (int): Dimension of the matrices (7 in this case).
        
    Returns:
        dict: A dictionary where keys are (i, j) indices of non-zero entries, 
              and values are tuples (mean, std).
    """
    # Stack matrices along a new axis to create a 3D array
    stacked_matrices = np.stack(b_matrices, axis=0)
    
    # Create a mask for non-zero entries across all matrices
    non_zero_mask = np.any(stacked_matrices != 0, axis=0)
    
    # Initialize an empty dictionary to store results
    result_dict = {}
    
    # Iterate over each entry in the 7x7 matrix
    for i in range(nodes):
        for j in range(nodes):
            if non_zero_mask[i, j]:
                # Extract values from all matrices for the current entry
                values = stacked_matrices[:, i, j]
                # Filter out zero entries
                non_zero_values = values[values != 0]
                # Calcuearly mean and std
                mean = np.mean(non_zero_values)
                std = np.std(non_zero_values)
                # Store the result in the dictionary
                result_dict[(i, j)] = (mean, std)
    
    return result_dict


def calcuearly_mean_std_for_non_zero_entry(b_matrices, nodes):
    """
    Calcuearly the mean and standard deviation of each non-zero entry 
    across a list of 7x7 numpy ndarrays.
    
    Args:
        b_matrices (list of np.ndarray): A list of 7x7 numpy ndarrays.
        
    Returns:
        np.ndarray: A 7x7 matrix where each entry is a tuple (mean, std) for non-zero entries
                    and 0 for entries that are zero in all matrices.
    """
    # Stack matrices along a new axis to create a 3D array
    stacked_matrices = np.stack(b_matrices, axis=0)
    
    # Create a mask for non-zero entries across all matrices
    non_zero_mask = np.any(stacked_matrices != 0, axis=0)
    
    # Initialize an empty 7x7 matrix to store results
    result = np.zeros((nodes, nodes), dtype=object)
    
    # Iterate over each entry in the 7x7 matrix
    for i in range(nodes):
        for j in range(nodes):
            if non_zero_mask[i, j]:
                # Extract values from all matrices for the current entry
                values = stacked_matrices[:, i, j]
                # Filter out zero entries
                non_zero_values = values[values != 0]
                # Calcuearly mean and std
                mean = np.mean(non_zero_values)
                std = np.std(non_zero_values)
                # Store the result as a tuple (mean, std)
                result[i, j] = (mean, std)
            else:
                # Keep the entry as 0 for always-zero positions
                result[i, j] = 0
    
    return result

def calcuearly_figure_of_merit(mean_std_matrix, nodes):
    """
    Calcuearly the Figure of Merit (FoM) for each entry in a matrix 
    of (mean, std) tuples. FoM is defined as std / mean.
    
    Args:
        mean_std_matrix (np.ndarray): A 7x7 matrix with entries as (mean, std) tuples 
                                      or 0 for zero entries.
        
    Returns:
        np.ndarray: A 7x7 matrix with the Figure of Merit (FoM) for each entry.
    """
    # Initialize an empty 7x7 matrix to store the FoM results
    fom_matrix = np.zeros((nodes, nodes))
    
    # Iterate over each entry in the mean_std_matrix
    for i in range(nodes):
        for j in range(nodes):
            entry = mean_std_matrix[i, j]
            if entry != 0:  # Check if the entry is non-zero
                mean, std = entry
                if mean != 0:  # Avoid division by zero
                    fom_matrix[i, j] = std / np.abs(mean)
                else:
                    fom_matrix[i, j] = 0  # Set FoM to 0 if mean is 0
            else:
                fom_matrix[i, j] = 0  # Keep zero for always-zero entries
    
    return fom_matrix

def save_b_matrix_to_excel(b_matrix, path, L, fog=False, scale=False):
    df = pd.DataFrame(b_matrix, columns=column_names_to_select)
    print(df)

    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    os.makedirs(iteration_folder_path, exist_ok=True)
        

    if fog is False and scale is False:
        file_name_b_matrix = os.path.join(path, f'{filename_to_save}_B_Matrix_{L}.xlsx')
        df.to_excel(file_name_b_matrix, index=False)
    elif fog is True and scale is False:
        file_name_b_matrix = os.path.join(path, f'FOG_{filename_to_save}_B_Matrix_{L}.xlsx')
        df.to_excel(file_name_b_matrix, index=False)
    else:
        print('scale is true')
        file_name_b_matrix = os.path.join(path, f'scaled_{filename_to_save}_B_Matrix_{L}.xlsx')
        df.to_excel(file_name_b_matrix, index=False)




def save_yy_matrix_to_excel(yy_matrix, path, L):
    df = pd.DataFrame(yy_matrix, columns=column_names_to_select)

    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    os.makedirs(iteration_folder_path, exist_ok=True)

    file_name_b_matrix = os.path.join(path, f'yy_Matrix_{L}.xlsx')
    df.to_excel(file_name_b_matrix, index=False)
def save_Hn_matrix_to_excel(Hn_matrix, path, L):
    df = pd.DataFrame(Hn_matrix, columns=column_names_to_select)

    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    os.makedirs(iteration_folder_path, exist_ok=True)

    file_name_b_matrix = os.path.join(path, f'Hn_Matrix_{L}.xlsx')
    df.to_excel(file_name_b_matrix, index=False)

def save_csv(df, path, L):
    # df = pd.DataFrame(Hn_matrix, columns=column_names_to_select)

    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    os.makedirs(iteration_folder_path, exist_ok=True)

    file_name_b_matrix = os.path.join(path, f'agg_data_{L}.csv')
    df.to_csv(file_name_b_matrix, index=False)


def plot_mean_std_for_non_zero_entries(b_matrices, nodes, output_dir):
    """
    Calcuearly the mean and standard deviation of each non-zero entry 
    across a list of 5x5 numpy ndarrays and save the plots for the mean 
    and std heatmaps.
    
    Args:
        b_matrices (list of np.ndarray): A list of 5x5 numpy ndarrays.
        nodes (int): Dimension of the matrices (5 in this case).
        output_dir (str): Directory where the plots will be saved.
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    # Stack matrices along a new axis to create a 3D array
    stacked_matrices = np.stack(b_matrices, axis=0)
    
    # Create a mask for non-zero entries across all matrices
    non_zero_mask = np.any(stacked_matrices != 0, axis=0)
    
    # Initialize mean and std matrices
    mean_matrix = np.zeros((nodes, nodes))
    std_matrix = np.zeros((nodes, nodes))
    
    # Calcuearly mean and std for non-zero entries
    for i in range(nodes):
        for j in range(nodes):
            if non_zero_mask[i, j]:
                # Extract values from all matrices for the current entry
                values = stacked_matrices[:, i, j]
                # Filter out zero entries
                non_zero_values = values[values != 0]
                # Calcuearly mean and std
                mean_matrix[i, j] = np.mean(non_zero_values)
                std_matrix[i, j] = np.std(non_zero_values)
    
    # Plotting heatmaps for mean and std
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_matrix, annot=True, cmap="coolwarm", cbar=True)
    plt.title("Mean of Non-Zero Entries")
    plt.savefig(f"{output_dir}/mean_heatmap.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(std_matrix, annot=True, cmap="coolwarm", cbar=True)
    plt.title("Standard Deviation of Non-Zero Entries")
    plt.savefig(f"{output_dir}/std_heatmap.png")
    plt.close()

    print(f"Plots saved in: {output_dir}")
    
def smooth_dataframe(df, iqr_multiplier=1.5):
        """
        Smooths numeric columns by replacing outliers with column medians.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        iqr_multiplier (float): Controls outlier detection sensitivity (default: 1.5)
        
        Returns:
        pd.DataFrame: Smoothed DataFrame with original column names
        """
        df_smoothed = df.copy()
        
        for col in df_smoothed.columns:
            if pd.api.types.is_numeric_dtype(df_smoothed[col]):
                # Calcuearly statistics
                median = df_smoothed[col].median()
                q1 = df_smoothed[col].quantile(0.25)
                q3 = df_smoothed[col].quantile(0.75)
                iqr = q3 - q1
                
                # Define outlier boundaries
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr
                
                # Identify and replace outliers
                is_outlier = (df_smoothed[col] < lower_bound) | (df_smoothed[col] > upper_bound)
                df_smoothed.loc[is_outlier, col] = median
        
        return df_smoothed

def zero_mean(ndarray):
    """
    Normalize an ndarray to have zero mean but preserve the original scale (no unit variance).
    
    Parameters:
    -----------
    ndarray : numpy.ndarray
        Input array to be normalized
    
    Returns:
    --------
    numpy.ndarray
        Array with zero mean for each column but original scale preserved
    """
    # Create a copy of the original ndarray
    ndarray_normalized = ndarray.copy()
    
    # Calcuearly the mean for each column
    means = np.mean(ndarray_normalized, axis=0)
    
    # Subtract the mean from each column to center the data at zero
    for i in range(ndarray_normalized.shape[1]):
        ndarray_normalized[:, i] = ndarray_normalized[:, i] - means[i]
    
    return ndarray_normalized


def get_plots(df, path, L, exp_name=None):

    cols = list(df.columns)
    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    print('iteration_folder_path: ', iteration_folder_path)
    os.makedirs(iteration_folder_path, exist_ok=True)

    for i in cols:
        df[i].plot()
        
        # Adding title and labels
        plt.title(f'Trend of Column {i}')
        plt.xlabel('Index')
        plt.ylabel('Values')
        
        plt.xticks(np.arange(0, 18000, 1000))  
        plt.xticks(rotation=90)
        
        file_name = str(i) +"_"+exp_name+'.png'
        file_name_full_path = os.path.join(path, file_name)
        print('file_name_full_path: ', file_name_full_path)
        plt.savefig(file_name_full_path)

def plot_pairplot(df, path, L):
    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    os.makedirs(iteration_folder_path, exist_ok=True)
    # Drop timestamp for pairplot
    df_plot = df.drop(columns=['timestamp'], errors='ignore')
    sns.pairplot(df_plot)
    plt.suptitle(f"{str(L)} - Pairplot", y=1.02, fontsize=16)
    plt.tight_layout()

    file_name = os.path.join(path, f'scatter_plot_{L}.png')
    plt.savefig(file_name)

def plot_ecg_graph(filename):
    import matplotlib.dates as mdates
    df = pd.read_csv(filename, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)

    column_names_to_select = [
    "Gearbox Temperature",
    "Gearbox Oil Temperature",
    "Gearbox Bearing 151 Temperature",
    "Gearbox Bearing 152 Temperature",
    "Gearbox Bearing 450 Temperature",
    "Gearbox Bearing 452 Temperature",
    "Gearbox Oil Pressure",
    "Bearing Oil Pressure",
    "Bearing Inlet Oil Pressure",
    "Gearbox Cooling Water Forward Temperature",
    "Gearbox Cooling Water Backward Temperature",
    ]
    
    
    columns_to_plot = column_names_to_select

    fig, axs = plt.subplots(len(columns_to_plot), 1, figsize=(14, 3 * len(columns_to_plot)), sharex=True)
    if len(columns_to_plot) == 1:
        axs = [axs]

    for idx, col in enumerate(columns_to_plot):
        ax = axs[idx]
        ax.plot(df.index, df[col], label=col)
        ax.set_title(f"{col}", fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.tick_params(axis='x', rotation=30, labelsize=10)

    plt.xlabel("timestamp")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    file_name = os.path.join(RESULTS_DIRECTORY, f'ecg_plot_{filename_to_save}.png')
    plt.savefig(file_name)


def scale_graph_weights(edge_labels, b_matrix):
    """
    Scales graph edge weights based on the provided adjacency matrix.

    The function identifies the largest absolute weight in the entire matrix and
    calcuearlys a scaling factor to map this weight to 2.0. All other weights
    in both the matrix and the edge labels dictionary are then scaled
    proportionally.

    Args:
        edge_labels (dict): A dictionary where keys are tuples representing
                            edges and values are their corresponding weights as strings.
        b_matrix (np.ndarray): A NumPy array representing the adjacency
                               matrix of the graph.

    Returns:
        tuple: A tuple containing:
            - scaled_edge_labels (dict): The new dictionary of edge labels with
                                         scaled weights.
            - scaled_b_matrix (np.ndarray): The new adjacency matrix with
                                            scaled weights.
    """
    # Find the maximum absolute weight from the adjacency matrix to determine
    # the scaling factor.
    max_abs_weight = np.max(np.abs(b_matrix))

    # If the max weight is zero, avoid division by zero. The scaling factor is 0.
    scaling_factor = 2.0 / max_abs_weight if max_abs_weight != 0 else 0

    # Apply the scaling factor to the entire adjacency matrix.
    scaled_b_matrix = b_matrix * scaling_factor

    # Create a new dictionary for the scaled edge labels.
    scaled_edge_labels = {}
    for edge, weight_str in edge_labels.items():
        # Convert the string weight to a float for calculation.
        original_weight = float(weight_str)
        # Scale the weight and format it as a string with two decimal places.
        scaled_weight = original_weight * scaling_factor
        scaled_edge_labels[edge] = f"{scaled_weight:.2f}"

    return scaled_edge_labels, scaled_b_matrix


def scale_b_matrix_func(b_matrix):
    """
    Scales a graph's adjacency matrix.

    The function identifies the largest absolute weight in the entire matrix and
    calcuearlys a scaling factor to map this weight to 2.0. All other weights
    in the matrix are then scaled proportionally.

    Args:
        b_matrix (np.ndarray): A NumPy array representing the adjacency
                               matrix of the graph.

    Returns:
        np.ndarray: The new adjacency matrix with scaled weights.
    """
    # Find the maximum absolute weight from the adjacency matrix to determine
    # the scaling factor.
    max_abs_weight = np.max(np.abs(b_matrix))

    # If the max weight is zero, avoid division by zero. The scaling factor is 0.
    scaling_factor = 2.0 / max_abs_weight if max_abs_weight != 0 else 0

    # Apply the scaling factor to the entire adjacency matrix.
    scaled_b_matrix = b_matrix * scaling_factor

    return scaled_b_matrix



def generate_index_colored_pairplot(df, iteration_folder_path, L):
    """
    Loads data from a CSV, creates a pair plot colored by row index,
    ensures proper diagonal distributions, and removes the color bar.
    """


    df['Progression'] = df.index
    plot_vars = [col for col in df.columns if col not in ['Unnamed: 0', 'Progression']]
    print(f"\nColumns selected for plotting: {plot_vars}")

    # --- Generate the Pair Plot using PairGrid for more control ---
    print("\nGenerating the pair plot...")

    # 1. Initialize an empty PairGrid
    g = sns.PairGrid(df, vars=plot_vars)

    # 2. Map the diagonal plots: A standard histogram WITHOUT the hue
    g.map_diag(sns.histplot, color='skyblue')

    # 3. Map the off-diagonal plots: A scatter plot WITH the hue
    g.map_offdiag(sns.scatterplot,
                  hue=df['Progression'], # Pass the hue data directly
                  palette='viridis',
                  s=15,
                  alpha=0.6)


    # --- Customize and Save the Final Plot ---
    g.fig.suptitle("Pair Plot with Proper Distributions (No Legend)", y=1.02)
    file_name = os.path.join(iteration_folder_path, f'color_scatter_plot_{L}.png')
    
    plt.savefig(file_name, bbox_inches='tight')
    print(f"\nPlot has been generated and saved as '{file_name}'")



def transpose_values_keep_headers(b_matrix, path, L):
    
    df = pd.DataFrame(b_matrix, columns=column_names_to_select)
    # Transpose only the values (excluding headers)
    transposed_values = df.values.T
    # Create a new DataFrame with the same headers
    transposed_df = pd.DataFrame(transposed_values, columns=df.columns)
    # Save to a new Excel file
    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    os.makedirs(iteration_folder_path, exist_ok=True)
    file_name_b_matrix = os.path.join(path, f'B_Matrix_Transpose_{L}.xlsx')
    transposed_df.to_excel(file_name_b_matrix, index=False)

# Main loop
for L in range(Lmin, Lmax + 1, Linc):
    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))

    start_time = time.perf_counter()

    print("loop: ", L)
    
    Nw = N // L

    print('Total number of points: ', Nw)
 

    B_list = []
    for i in range(1, num_nodes+1):
        B_ = M[Ni:Nt, i - 1]  # Indexing for B1, B2, B3, ..., B8
        B_list.append(B_)


    Bg_list = [[] for _ in range(num_nodes)]


    for nw in range(1, Nw + 1):
        nst = (nw - 1) * L
        for i in range(num_nodes):
            Bg = np.sum(B_list[i][nst:nst+L]) / L
            Bg_list[i].append(Bg)

    
    # zero mean unit variance
    Hn = np.column_stack(Bg_list)


    #Hn = zero_mean_unit_variance(Hn)

    X = Hn.T

    scatter_plot_nd_array = X.T


    save_Hn_matrix_to_excel(Hn,iteration_folder_path, L)

    agg_df = pd.DataFrame(scatter_plot_nd_array, columns=column_names_to_select)

    # plot_pairplot(agg_df, iteration_folder_path, L)
    # print('stage: generate_index_colored_pairplot')
    # generate_index_colored_pairplot(agg_df, iteration_folder_path, L)

    try:
        plot_pairplot(agg_df, iteration_folder_path, L)
        generate_index_colored_pairplot(agg_df, iteration_folder_path, L)
    except:
        pass
        
    save_csv(agg_df, iteration_folder_path, L)


    print("stage: two_step_CD")

    B_adjacency_matrix, W_m, nodes_permutation = two_step_CD(X)
    # print("B Matrix")
    # print(B_adjacency_matrix)

    edge_weights_list = []
    b_matrices_list = []

    if bootstrap_exp is True:
        for _ in tqdm(range(ITER)): 
            data = get_bootstrap_data(X)
            print('data shape after get_bootstrap_data')
            print(data.shape)
            B_adjacency_matrix, W, nodes_permutation = two_step_CD(data)

            print('B shape: ', B_adjacency_matrix.shape)
            b_matrices_list.append(B_adjacency_matrix)


        iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
        plot_mean_std_for_non_zero_entries(b_matrices_list, num_nodes, iteration_folder_path)
        b_matrix_with_mean_std = calcuearly_mean_std_for_non_zero_entry(b_matrices_list, num_nodes)
        b_matrix_std_mean_dict = store_mean_std_dict_for_non_zero_entries(b_matrices_list, num_nodes)
        figure_of_merit_b_matrix = calcuearly_figure_of_merit(b_matrix_with_mean_std, num_nodes)
        # print('b_matrix_std_mean_dict')
        # print(b_matrix_std_mean_dict)

        # print('figure_of_merit')
        # print(figure_of_merit_b_matrix)

        iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
        save_b_matrix_to_excel(figure_of_merit_b_matrix, iteration_folder_path, L, fog=True)


    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))

    # print('B_adjacency_matrix')
    # print(B_adjacency_matrix)
    save_b_matrix_to_excel(B_adjacency_matrix, iteration_folder_path, L, fog=False, scale=False)
    transpose_values_keep_headers(B_adjacency_matrix, iteration_folder_path, L)
    scale_b_matrix = scale_b_matrix_func(B_adjacency_matrix)
    # print('scale_b_matrix')
    # print(scale_b_matrix)
    save_b_matrix_to_excel(scale_b_matrix, iteration_folder_path, L, fog=False, scale=True)



    Bstore = np.hstack([Bstore, B_adjacency_matrix])

    
    yy = np.dot(W_m, X)
    iteration_folder_path_b_matrix = os.path.join(RESULTS_DIRECTORY, str(L))
    file_name_yy_matrix = os.path.join(iteration_folder_path, f'yy_Matrix_{L}.pkl')
    save_yy_matrix_to_excel(yy.T, iteration_folder_path, L) 
    import torch
    torch.save(yy, file_name_yy_matrix)  # Uses PyTorch's serialization
   
    # print(yy)
    # print(yy.shape)


    # print('yy here')
    kt = scipy.stats.kurtosis(yy, axis=1, fisher=True)  # changed here yy to Hn-X

    
    kurt = np.column_stack((kurt, kt))

    Lp.append(L)
    Bopt = B_adjacency_matrix
    # print("bopt",Bopt)
    # print("b",B_adjacency_matrix)
    print("stage: dg_plot")

    dg_plot(Bopt,L, num_nodes, column_names_final)
    print('scaling')
    dg_plot(scale_b_matrix,L, num_nodes, column_names_final)



    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---NEW HSIC CODE----%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    print("stage: HSIC new")
    hsic_results = compute_hsic_for_all_pairs(yy)
    # print('hsic_results')
    # print(type(hsic_results))
    # print(hsic_results)
    visualize_hsic_results(hsic_results, yy.shape[0], L)
    hsic_overall_results[L] = hsic_results
    # plot_green_blocks_over_iterations(iteration_results_example, yy.shape[0])


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---NEW HSIC CODE----%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    # Cross AND Auto-correlation functions of SEM noise, yy
    # numpy.correlate is used as the equivalent of MATLAB's xcorr with 'full' mode

    print("stage: Autocorrelation with yy")


    yy = yy.T



    # Assuming `yy` is a NumPy array with shape (2000, 4)
    Nw = yy.shape[0]  # or any other appropriate value for Nw

    # Calcuearly cross and auto-correlation functions
    xrf_list = []
    for i in range(yy.shape[1]):
        for j in range(yy.shape[1]):
            corr = correlate(yy[:, i], yy[:, i], mode='full')
            center = len(corr) // 2
            xrf_list.append(corr[center - 31:center + 32])  # Extract 63 values centered at zero lag

    # Convert list to NumPy array
    xrf = np.array(xrf_list).T

    # Normalize by Nw
    xrf = xrf / Nw



    # Plotting xrf
    plt.figure()
    plt.plot(xrf)
    # plt.show()
    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    plot_filename_corr = os.path.join(iteration_folder_path, f'yy_plot_iteration_autoCorr_{L}.png')
    plt.savefig(plot_filename_corr)
    # print("*"*100)

    mean_curve = np.mean(xrf, axis=1)

    # Create the stem plot
    plt.figure(figsize=(10, 6))
    plt.stem(mean_curve, linefmt='b-', markerfmt='bo', basefmt='r-', label='ACF')

    # Add confidence bounds (example: ±0.2 for illustration)
    confidence_bound = 0.2
    plt.axhline(y=confidence_bound, color='orange', linestyle='--', label='Confidence Bound')
    plt.axhline(y=-confidence_bound, color='orange', linestyle='--')
    # Add labels and title
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(' Autocorrelation Function with yy')
    plt.legend()
    plt.grid(True)
    plot_filename_corr_stem = os.path.join(iteration_folder_path, f'yy_stem_autoCorr_{L}.png')
    plt.savefig(plot_filename_corr_stem)

    print("stage: Autocorrelation with X")


    auto_x = X.T


    # Assuming `yy` is a NumPy array with shape (2000, 4)
    Nw = auto_x.shape[0]  # or any other appropriate value for Nw

    # Calcuearly cross and auto-correlation functions
    xrf_list_x = []
    for i in range(auto_x.shape[1]):
        for j in range(auto_x.shape[1]):
            corr = correlate(auto_x[:, i], auto_x[:, i], mode='full')
            center = len(corr) // 2
            xrf_list_x.append(corr[center - 31:center + 32])  # Extract 63 values centered at zero lag

    # Convert list to NumPy array
    xrf_x = np.array(xrf_list_x).T

    # Normalize by Nw
    xrf_x = xrf_x / Nw


    # Plotting xrf
    plt.figure()
    plt.plot(xrf_x)
    # plt.show()
    iteration_folder_path = os.path.join(RESULTS_DIRECTORY, str(L))
    plot_filename_corr = os.path.join(iteration_folder_path, f'X_plot_iteration_autoCorr_{L}.png')
    plt.savefig(plot_filename_corr)
    # print("*"*100)

    mean_curve = np.mean(xrf_x, axis=1)

    # Create the stem plot
    plt.figure(figsize=(10, 6))
    plt.stem(mean_curve, linefmt='b-', markerfmt='bo', basefmt='r-', label='ACF')

    # Add confidence bounds (example: ±0.2 for illustration)
    confidence_bound = 0.2
    plt.axhline(y=confidence_bound, color='orange', linestyle='--', label='Confidence Bound')
    plt.axhline(y=-confidence_bound, color='orange', linestyle='--')
    # Add labels and title
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function with X')
    plt.legend()
    plt.grid(True)
    plot_filename_corr_stem_x = os.path.join(iteration_folder_path, f'x_stem_autoCorr_{L}.png')
    plt.savefig(plot_filename_corr_stem_x)


    end_time = time.perf_counter()
    loop_time = end_time - start_time
    print(f"Loop {L} took {loop_time:.4f} seconds")

    import pickle
    iteration_folder_path_b_matrix = os.path.join(RESULTS_DIRECTORY, str(L))
    file_name_b_matrix = os.path.join(iteration_folder_path, f'B_Matrix_{L}.pkl')
    
    with open(file_name_b_matrix, 'wb') as f:
        pickle.dump(B_adjacency_matrix, f)

    print('*'*100)


print('Stage: HSIC Normalized StemPlot')

highest_agg_factor = plot_green_blocks_stem(hsic_overall_results, yy.shape[0])
print("highest_agg_factor: ", highest_agg_factor)
print('Stage: Bucket ECG Plot')
plot_ecg_graph(filename)



Bstore = np.delete(Bstore, 0, axis=1)
kurt = np.delete(kurt, 0, axis=1)
ksort = np.sort(np.abs(kurt), axis=0)
ksort = np.delete(ksort, 0, axis=0)
ksum = np.sum(ksort, axis=0)

# print('k sum: ', ksum)
agg_numbers = list(range(1, len(ksum)+1))
df_ksum = pd.DataFrame({'agg_numbers': agg_numbers, 'kurtosis_value': ksum})
df_ksum.to_excel(RESULTS_DIRECTORY+"/kurtosis_data.xlsx", index=False)



plt.figure(211)
plt.subplot(2, 1, 1)

bar_width = 0.2
positions = np.arange(len(Lp))


bar_width = 0.35
index = np.arange(len(Lp))

# Plotting the Magnitude of Kurtosis of each Channel
plt.figure(211)
plt.subplot(2, 1, 1)

for i, k in enumerate(kurt):
    plt.bar(index + i * bar_width, np.abs(k), width=bar_width, label=f'Channel {i+1}')

plt.title('Magnitude of Kurtosis of each Channel')
plt.xticks(index + 1.5 * bar_width, Lp)
plt.legend()

plt.figure(212)
# Plotting the Sum of Magnitude of top 3 Kurtosis
plt.subplot(2, 1, 2)
plt.stem(Lp, ksum)
# plt.stem(Lp, ksum, use_line_collection=True)  # use_line_collection is recommended for modern versions of matplotlib
plt.title('Sum of Magnitude of top 3 Kurtosis')
# plt.show()

plt.savefig(RESULTS_DIRECTORY+"/Kurtosis.png")
# Aggregate data cross-correlation matrix
xrm = np.dot(Hn.T, Hn) / Nw




# Create a folder called "run1" and move the "Graphs" folder into it
# internal_dir = "L_10_acc_10k_pts_5_nodes_feb_2025"
# internal_dir = f"{filename_to_save}"
internal_dir = "WT_80_Causal_Analysis_Results"


os.makedirs(internal_dir, exist_ok=True)
shutil.move(RESULTS_DIRECTORY, os.path.join(internal_dir, RESULTS_DIRECTORY))




