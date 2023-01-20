from SparsePCA import *
import pandas as pd
Y = pd.read_excel(r"C:\Users\Alex\Documents\Uni\VojtaPCA\Data\keyPointsAllVideosED.xlsx")
X = Y.to_numpy(copy=True)
vars = np.expand_dims((np.var(X, axis=1)), 1)
print(vars)

# Normalize columns
# X = np.random.randn(1770, 21) * np.sqrt(vars)
X -= np.mean(X, axis=0)
print(np.min(X))
# print(np.mean(X, axis=0))
# print(np.var(X, axis=1))
# print(X.shape)

n_components = 5
lambd = 10000
lr = 1e-6
steps = 10000
alpha = 100

sparse_pca = SparsePCA(n_components=n_components, alpha=alpha, verbose=True)

W = sparse_pca.fit(X, lambdas=10000)
print("Explained variance:", sparse_pca.explained_variance)
print("Total Explained variance", np.sum(sparse_pca.explained_variance))
print("W.T @ W:", W.T @ W)
print("L1 norm of components:", np.linalg.norm(W, ord=1, axis=0))

# [3.66100924 2.61949157 3.10564342 2.57092353 3.66520359]