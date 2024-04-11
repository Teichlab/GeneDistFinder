"""
Implemented by: Dinithi Sumanaweera
Date: April 11, 2024
Description: NNDistFinder module computes distributional distances of gene expression for each query cell 
in terms of its own neighbourhood cells vs. reference neighbourhood cells. Cell neighbourhoods are queried using the  
data structures available in BBKNN package (https://github.com/Teichlab/bbknn). 
Acknowledgement: Krzysztof Polanski
"""

import numpy as np
import pandas as pd 
import bbknn
import multiprocessing
from multiprocessing import Pool
from scipy.spatial import distance
from scipy.special import softmax
from scipy.sparse import csr_matrix
from scipy.stats import wasserstein_distance
import Utils
from tqdm import tqdm
import sys
#from tqdm.notebook import tqdm_notebook
import warnings
warnings.filterwarnings("ignore")

def main(*args, **kwargs):

    global adata_ref, adata_query, embedding_basis, N_NEIGHBOURS, distance_metric, n_threads
    
    adata_ref = args[0]
    adata_query = args[1]
    embedding_basis = args[2]
    N_NEIGHBOURS = kwargs.get('n_neighbours',None) if ('n_neighbours' in kwargs) else 25
    distance_metric = kwargs.get('distance_metric',None) if('distance_metric' in kwargs) else 'wasserstein'
    if(distance_metric not in ['wasserstein','mml']):
        print('Note: only wasserstein, mml distances are available. Returning. ')
        return 
    n_threads = kwargs.get('n_threads',None) if('n_threads' in kwargs) else multiprocessing.cpu_count() 
    print('n_neighbours: ', N_NEIGHBOURS)
    print('distance metric: ', distance_metric)
    print('n_processors: ', n_threads)
    print('NNDist computation ======')
    
    global Q2R_knn_indices, Q2Q_knn_indices, R_weights, Q_weights , gene_list
    Q2R_knn_indices, Q2Q_knn_indices, R_weights, Q_weights = construct_RQ_tree() 
    gene_list= adata_ref.var_names
    dists = [] 
    nQcells = adata_query.shape[0]

    with Pool(n_threads) as p:
            dists = list(tqdm(p.imap(run_main, np.arange(0, nQcells)), total= nQcells))
    
    gene_diffs_df = pd.DataFrame(dists)
    gene_diffs_df.columns = gene_list
    gene_diffs_df.index = adata_query.obs_names
    
    print('Normalizing output ======')
    if(distance_metric != 'mml'):
        normalize_column = lambda col: (col - col.min()) / (col.max() - col.min()) # min-max normalization
        df_normalized = gene_diffs_df.apply(normalize_column, axis=0)
    else:
        gene_diffs_df = np.log1p(gene_diffs_df)
        gene_diffs_df.fillna(0, inplace=True)
        gene_diffs_df[gene_diffs_df < 0] = 0
        df_normalized = gene_diffs_df
        
    return df_normalized

def construct_RQ_tree():
        
        params = {}
        params['computation'] = 'cKDTree'
        params['neighbors_within_batch'] = N_NEIGHBOURS 
        Q2R_ckd = bbknn.matrix.create_tree(adata_ref.obsm[embedding_basis] , params)
        Q2R_knn_distances, Q2R_knn_indices = bbknn.matrix.query_tree(adata_query.obsm[embedding_basis], Q2R_ckd, params) 
        # from each query cell to n neighbouring ref cells
        
        Q2Q_ckd = bbknn.matrix.create_tree(adata_query.obsm[embedding_basis] , params)
        Q2Q_knn_distances, Q2Q_knn_indices = bbknn.matrix.query_tree(adata_query.obsm[embedding_basis], Q2Q_ckd, params) 
        # from each query cell to its n neighbouring query cells
        
        R_weights = pd.DataFrame(Q2R_knn_distances).apply(lambda row: 1/softmax(row), axis=1)
        Q_weights = pd.DataFrame(Q2Q_knn_distances).apply(lambda row: 1/softmax(row), axis=1)
        
        return Q2R_knn_indices, Q2Q_knn_indices, R_weights, Q_weights
          
def run_main(i):
        
        adata_ref_neighbours = adata_ref[Q2R_knn_indices[i]]   # get the ref neighbour cells
        adata_query_neighbours = adata_query[Q2Q_knn_indices[i]] # get the query neighbour cells
        Qmat = csr_matrix(adata_query_neighbours.X.todense().transpose())
        Rmat = csr_matrix(adata_ref_neighbours.X.todense().transpose())

        gene_dists = []
        for j in range(len(gene_list)):
            Q_gene_vec = csr_mat_col_densify(Qmat, j) 
            R_gene_vec = csr_mat_col_densify(Rmat, j)

            if(distance_metric == 'wasserstein'):
                dist = wasserstein_distance(u_values=R_gene_vec, v_values=Q_gene_vec, u_weights= R_weights[i], v_weights= Q_weights[i])
            else:
                dist = compute_mmldist(R_gene_vec, Q_gene_vec, R_weights[i], Q_weights[i])
                
            gene_dists.append(dist)

        return gene_dists

def csr_mat_col_densify(csr_matrix, j): 
        start_ptr = csr_matrix.indptr[j]
        end_ptr = csr_matrix.indptr[j + 1]
        data = csr_matrix.data[start_ptr:end_ptr]
        dense_column = np.zeros(csr_matrix.shape[1])
        dense_column[csr_matrix.indices[start_ptr:end_ptr]] = data
        return dense_column
    
def compute_mmldist(R_gene_vec, Q_gene_vec, R_weights, Q_weights):
    
        n = len(R_weights)
        Q = np.dot(Q_weights, Q_gene_vec)/np.sum(Q_weights)
        R = np.dot(R_weights, R_gene_vec)/np.sum(R_weights)
        # weighted variance 
        Q_std = np.sqrt(n* np.dot(Q_weights,  np.power( (Q_gene_vec - Q), 2)) / ((n-1)*np.sum(Q_weights)) )
        R_std = np.sqrt(n* np.dot(R_weights,  np.power( (R_gene_vec - R), 2)) / ((n-1)*np.sum(R_weights)) )
        if(np.count_nonzero(R_gene_vec)<=3 and np.count_nonzero(Q_gene_vec)<=3): # if both are almost 0 (less than 3 counts overall)
            return 0.0
        elif(np.count_nonzero(R_gene_vec)<=3): # if only ref is almost 0 expressed
            R_std = Q_std
        elif(np.count_nonzero(Q_gene_vec)<=3): # if only query is almost 0 expressed
            Q_std = R_std 
        
        gex1 = R_gene_vec; gex2 = Q_gene_vec; μ_S = R; μ_T = Q; σ_S = R_std; σ_T = Q_std
        ref_data = gex1; query_data = gex2
        I_ref_model, I_refdata_g_ref_model = Utils.run_dist_compute_v3(ref_data, μ_S, σ_S) 
        I_query_model, I_querydata_g_query_model = Utils.run_dist_compute_v3(query_data, μ_T, σ_T) 
        I_ref_model, I_querydata_g_ref_model = Utils.run_dist_compute_v3(query_data, μ_S, σ_S) 
        I_query_model, I_refdata_g_query_model = Utils.run_dist_compute_v3(ref_data, μ_T, σ_T) 
        match_encoding_len1 = I_ref_model + I_querydata_g_ref_model + I_refdata_g_ref_model
        match_encoding_len1 = match_encoding_len1/(len(query_data)+len(ref_data))
        match_encoding_len2 = I_query_model + I_refdata_g_query_model + I_querydata_g_query_model
        match_encoding_len2 = match_encoding_len2/(len(query_data)+len(ref_data))
        match_encoding_len = (match_encoding_len1 + match_encoding_len2 )/2.0 
        null = (I_ref_model + I_refdata_g_ref_model + I_query_model + I_querydata_g_query_model)/(len(query_data)+len(ref_data))
        match_compression =   match_encoding_len - null 
        return round(float(match_compression.numpy()),4) 

    
if __name__ == "__main__":
    args = sys.argv[1:] 
    main(*args)