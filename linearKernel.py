import numpy as np

def linearKernel(X1, X2):
    # Computes the linear Kernel between two set of features
    
    m = X1.shape[0] #number of samples in X1
    K = np.ones((m,X2.shape[0])) #k has the size of number of samples in X1 x number of samples in X2
    #it gives the similarity between each sample in X1 to each sample in X2 
    #by doting the features of the samples
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the linear kernel (see the assignment
    #				for more details).
    
    for i in range(m):
        #K[i,:] = X1[i,:].dot(X2.T)
        K[i,:] = np.dot(X1[i,:], X2)
        
        ##K[i,:] is a row vector representing the similarity of sample i in X1 to all samples in X2
        #it is calculated by dotting the features of sample i in X1 with all samples in X2

    # =============================================================
        
    return K
