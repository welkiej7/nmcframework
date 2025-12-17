import numpy as np 
from numpy.typing import NDArray, ArrayLike
from typing import Union
import warnings as wrn
from copy import deepcopy
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import scipy.stats as stats


"""
Title: User space generations for polarization measures testing. 
Date Created: 29.11.2024
Date Last Updated: 04.12.2025
"""

def gaussian_mixture_model(mu_1, mu_2, sigma_1, sigma_2, n_1, n_2):
    """
    Create samples from a Gaussian mixture model with normalization to [-1,1].
    
    Parameters:
    -----------
    mu_1, mu_2: float
        Means of the two Gaussian components
    sigma_1, sigma_2: float
        Standard deviations of the two Gaussian components
    n_1, n_2: int
        Number of samples to draw from each component
        
    Returns:
    --------
    samples: ndarray
        Combined samples from both distributions normalized to [-1,1]
    mixing_weights: tuple
        The mixing weights (π₁, π₂)
    """

    total_samples = n_1 + n_2
    mixing_weight_1 = n_1 / total_samples
    mixing_weight_2 = n_2 / total_samples

    samples_1 = np.random.normal(mu_1, sigma_1, size=n_1)
    samples_2 = np.random.normal(mu_2, sigma_2, size=n_2)
    

    all_samples = np.concatenate((samples_1, samples_2))

    min_val = np.min(all_samples)
    max_val = np.max(all_samples)
    

    normalized_samples = (all_samples - min_val) / (max_val - min_val) * 2 - 1

    return normalized_samples, (mixing_weight_1, mixing_weight_2)


def generate_similarity_matrix(opinion_space:NDArray) -> NDArray:

    """
    Compute a pairwise similarity matrix based on normalized Euclidean distance.
    
    This function takes an input matrix where each row represents a point in 
    an opinion space and computes the similarity between all pairs of rows.
    The similarity metric used is:
    
        s(a, b) = 1 / (1 + ||a - b||₂)
    
    where ||a - b||₂ is the Euclidean norm between rows a and b.  
    The output is an n × n matrix where n is the number of rows in `opinion_space`.
    
    Parameters:
    -----------
    opinion_space : ndarray of shape (n_samples, n_features)
        Input matrix where each row corresponds to a point (e.g., a user's opinion
        on multiple dimensions).
        
    Returns:
    --------
    similarity : ndarray of shape (n_samples, n_samples)
        Pairwise similarity matrix where entry (i, j) is 
        1 / (1 + Euclidean distance between row i and row j).
    """
    diff = opinion_space[:,None,:] - opinion_space[None,:,:] # Take the difference for euclidean norm
    dist = np.linalg.norm(diff, axis = -1) # For each pair

    similarity = 1 / (1 + dist)

    return similarity

def bimodality_coefficient(data):
    """
    Calculate the bimodality coefficient for a distribution.
    BC > 0.555 indicates bimodality.
    
    Args:
        data: numpy array of data points
    
    Returns:
        bc: bimodality coefficient
        is_bimodal: boolean indicating if distribution is bimodal
    """
    n = len(data)
    

    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    

    bc = (skewness**2 + 1) / (kurtosis + 3 * ((n-1)**2) / ((n-2)*(n-3)))
    

    is_bimodal = bc > 0.555
    
    return bc, is_bimodal




def calculate_similarity_on_single_topic(opinion_array):

    x_min = opinion_array.min()
    x_max = opinion_array.max()
    x_norm = (opinion_array - x_min) / (x_max - x_min)


    X = x_norm.reshape(-1, 1)      # shape (n, 1)
    D = np.abs(X - X.T)            # broadcasting -> (n, n)
    
    return (1 / (1 + D))



# -- Utils are below, not directly related to the opinion space generation. -- 

def _calculate_heirarchical_clustering(similarity_matrix:NDArray) -> NDArray:

    """
    Orders the given user space with hierarchical clustering. This function created is not to be imported but rather 
    to showcase and demonstrate how generative functions work.
    
    Parameters
    -----------
    data: NDArray
      2D array with values between 0 and 1

    Returns
    -----------
    data_sorted: NDArray
      Hierarchically clustered data.

    """

    row_linkage = hierarchy.linkage(pdist(similarity_matrix), method='complete')
    col_linkage = hierarchy.linkage(pdist(similarity_matrix.T), method='complete')
    
    row_order = hierarchy.leaves_list(row_linkage)
    col_order = hierarchy.leaves_list(col_linkage)
    
    
    data_sorted = similarity_matrix[row_order, :][:, col_order]
    
    return data_sorted


