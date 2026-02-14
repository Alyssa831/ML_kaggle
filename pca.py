#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from numpy import genfromtxt
import csv
from sklearn.preprocessing import StandardScaler

def PCA(data,k):
    """
    Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
    Rows of A correspond to observations (i.e., wines), columns to variables.
    Reduces the dimensionality to k.
    """
    # Standardize the data (mean=0, variance=1)
    A=data
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(A)
    #A_scaled = 0 if NaNs or infs
    A_scaled = np.nan_to_num(A_scaled)

    # Calculate the covariance matrix of the standardized data
    cov_mat = np.cov(A_scaled , rowvar = False)
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues , eigenvectors = np.linalg.eig(cov_mat)

    # Sort the eigenvalues and their corresponding eigenvectors
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:,sorted_index]

    # Select the top k eigenvectors
    eigenvector_subset = sorted_eigenvectors[:,0:k]

    # Transform the data 
    newData = np.dot(A_scaled, eigenvector_subset)

    return newData








#=============================================================================


# %%
