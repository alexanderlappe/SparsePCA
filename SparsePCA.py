import numpy as np
from sklearn.decomposition import PCA
class SparsePCA():
    """
    Class object that contains all code necessary to run the sparse PCA. The main method that calls the other methods is
    self.fit(). The logic is:
    1. Take the data matrix and compute the first principal component without constraints. (method: run_regular_pca_())
    2. Iteratively sparsify the component using gradient descent (method: sparsify_w())
    3. Remove the variance of the found component from the data matrix.

    Then repeat 1.-3. until all desired components are found.
    """
    def __init__(self, n_components, alpha=100, verbose=False):
        self.n_components = n_components
        self.alpha = alpha
        self.verbose = verbose
        self.explained_variance = None

    def run_regular_pca_(self, X):
        """
        Initializes a principal component without sparsity constraint
        Args:
            X: data matrix of shape (n, k)

        Returns: Principal component of shape (k)

        """

        u, s, vt = np.linalg.svd(X)
        v = np.squeeze(vt[0, :])
        if self.verbose:
            print(v @ np.cov(X.T) @ v)
            print("l1 norm of unconstrained component:", np.linalg.norm(v, 1))
        return v

    ### Methods for gradient calculation. Don't need to be called outside thise file!
    def abs(self, w):
        return w * np.tanh(self.alpha * w)

    def abs_prime(self, w):
        diag = np.tanh(self.alpha * w) + self.alpha * w * (1 - np.tanh(self.alpha * w) ** 2)
        return diag

    def u1(self, X, w):
        return w @ np.cov(X.T) @ w

    def u2(self, w):
        return - np.sum(self.abs(w))

    def v1(self, w):
        return np.linalg.norm(w) ** 2

    def v2(self, w):
        return np.linalg.norm(w) ** 2

    def u1_prime(self, X, w):
        u_prime = 2 * w.T @ np.cov(X.T)
        return u_prime

    def u2_prime(self, w):
        return - self.abs_prime(w)

    def v1_prime(self, w):
        return 2 * w

    def v2_prime(self, w):
        return 2 * w

    def compute_first_grad(self, X, w):
        enum = self.u1_prime(X, w) * self.v1(w) - self.u1(X, w) * self.v1_prime(w)
        denom = self.v1(w) ** 2
        return enum / denom

    def compute_reg_grad(self, w):
        enum = self.u2_prime(w) * self.v2(w) - self.u2(w) * self.v2_prime(w)
        denom = self.v2(w) ** 2
        grad = enum / denom
        return grad

    def compute_grad(self, X, w, lambd):
        grad = self.compute_first_grad(X, w) + lambd * self.compute_reg_grad(w)
        # Orthogonalize
        grad = grad - (np.dot(grad, w) / np.dot(w, w)) * w
        return grad

    def compute_loss(self, X, w, lambd):
        return self.u1(X, w) / self.v1(w) + lambd * self.u2(w) / self.v2(w)

    def sparsify_w(self, X, w, lambd, lr, steps):
        """
        Iteratively sparsifies a principal component w
        Args:
            X: Data matrix
            w: Starting point of the optimization. Should be an unconstrained PC
            lambd: hyperparameter
            lr: Learning rate
            steps: Number of steps of iteration

        Returns: The sparse principal component of shape (k)

        """
        if self.verbose == True:
            print("starting loss:", self.compute_loss(X, w, lambd))
        for step in range(steps):
            grad = self.compute_grad(X, w, lambd)
            w += lr * grad
            w = w / np.linalg.norm(w)
            if step % 1000 == 0 and self.verbose == True:
                print("step:", step)
                print("grad norm:", np.linalg.norm(grad))
                print("loss:", self.compute_loss(X, w, lambd))
                print("var loss:", self.u1(X, w) / self.v1(w), "reg loss:", lambd * self.u2(w) / self.v2(w))
                print()
        return w

    def find_component(self, X, lambd, lr, steps):
        """
        Finds the sparse principal component for a data matrix
        Args:
            X: Data matrix
            lambd: hyperparameter
            lr: Learning rate
            steps: Number of steps

        Returns: The sparse principal component of shape (k)

        """
        w = self.run_regular_pca_(X)
        w_out = self.sparsify_w(X, w, lambd, lr, steps)
        return w_out

    def remove_variance(self, X, w):
        """
        Remove the component variance from the data to prepare matrix to find the next component
        Args:
            X: Data matrix used to compute w
            w: First principal component of matrix X

        Returns: The data matrix with variance removed. Shape is same as X

        """
        X = X - X @ np.outer(w, w)
        return X

    def fit(self, X, lambdas, lr=1e-6, steps=1000):
        """
        Finds all sparse principal components in one run.
        Args:
            X: Data matrix
            lambdas: Int or List of length self.n_components containing where entry i is hyperparameter for component i
            lr: Learning rate
            steps: Number of steps to be used

        Returns: Matrix where the i-th column contains the i-th sparse PC

        """
        # Keep track of the original X
        X_original = np.copy(X)
        if not isinstance(lambdas, list):
            lambdas = self.n_components * [lambdas]
        # Initialize component matrix
        W = np.zeros((X.shape[1], self.n_components))
        # Loop over number of desired components
        for component_id in range(self.n_components):
            print("Computing component", component_id, "...")
            w = self.find_component(X, lambdas[component_id], lr, steps)
            W[:, component_id] = w
            # Remove the variance of the found component from the data to find the next one
            X = self.remove_variance(X, w)
        # Compute explained variance
        self.compute_explained_variance(X_original, W)
        return W

    def compute_explained_variance(self, X, W):
        """
        Computes the explained variance (done automatically after fitting)
        Args:
            X: Data matrix
            W: Matrix containing the principal components (n_features, n_components)

        Returns: nd.array where entry i is the ratio of explained variance of component i

        """
        self.explained_variance = np.diag(W.T @ np.cov(X.T) @ W) / np.trace(np.cov(X.T))
        return None




