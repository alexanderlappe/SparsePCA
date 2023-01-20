from SparsePCA import *
import pandas as pd

# Load data (change path to location of data file)
Y = pd.read_excel(r"C:\Users\Alex\Documents\Uni\VojtaPCA\Data\keyPointsAllVideosED.xlsx")
# Convert to numpy
X = Y.to_numpy(copy=True)

# Normalize columns
X -= np.mean(X, axis=0)
# Might make sense to also standardize the features but I left them as they were for now

# Number of components to compute
n_components = 5
# Set hyperparameter. Larger value means stronger sparsification. Need to play around with this value.
lambdas = 10000
# Set learning rate, number of iterations and alpha value to control the approximation of the absolute value.
# Can probably be left unchanged
lr = 1e-6
steps = 10000
alpha = 100

# Instantiate the SparsePCA class.
sparse_pca = SparsePCA(n_components=n_components, alpha=alpha, verbose=True)

# Fit the sparse PCA. W will be a matrix that contains the components.
W = sparse_pca.fit(X, lambdas=lambdas)
# Print some useful properties of the found components
print("Explained variance:", sparse_pca.explained_variance)
print("Total Explained variance", np.sum(sparse_pca.explained_variance))
print("W.T @ W:", W.T @ W)
print("L1 norm of components:", np.linalg.norm(W, ord=1, axis=0))