import math
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Ellipse
from itertools import chain
import time
from tqdm import tqdm
from operator import attrgetter
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from scipy.stats import chi2
from scipy.spatial.distance import pdist

# surface functions, merge function, and other helper functions

def surface_single(x, U, A, p=2, k=1):
    """
    Compute the hyperellipsoid surface function for a single point.

    Parameters
    ----------
    x : np.ndarray, shape (d,)
        The query point.
    U : np.ndarray, shape (d, d)
        Eigenvectors (columns) defining the ellipsoid orientation.
    A : np.ndarray, shape (d,)
        Semi-axis lengths along each eigenvector direction.
    p : float or np.ndarray, shape (d)
        power/exponent of the ellipsoid axes.
    k : float
        confidence value

    Returns
    -------
    float
        < 0 if x is inside the ellipsoid
        = 0 if x is on the surface
        > 0 if x is outside the ellipsoid
    """
    x_local = U.T @ x        # Project into ellipsoid's local frame, shape (d,)
    return np.sum(np.abs(x_local / A) ** p) - (k ** 2)

def surface_multi(xs, Us, As, p=2, k=1):
    """
    Compute the hyperellipsoid surface function for n points,
    each against its own ellipsoid.

    Parameters
    ----------
    xs : np.ndarray, shape (n, d)
        Query points.
    Us : np.ndarray, shape (n, d, d)
        Per-ellipsoid eigenvector matrices.
    As : np.ndarray, shape (n, d)
        Per-ellipsoid semi-axis lengths.

    Returns
    -------
    np.ndarray, shape (n,)
        Surface function value for each (point, ellipsoid) pair.
        < 0 inside, = 0 on surface, > 0 outside.
    """
    xs_local = np.einsum('ndi, ni -> nd', Us.swapaxes(1,2), xs)  # (n, d)
    return np.sum(np.abs(xs_local / As) ** p, axis=1) - (k ** 2)              # (n,)

def merge_nodes(x, y): # NOTE: New P is currently avg of the 2 that is (x.p + y.p) / 2
    new_n = x.n + y.n
    dist = x.M - y.M
    new_M = ((x.n * x.M) + (y.n * y.M)) / (new_n)
    new_S = ((x.n / new_n) * x.S) + ((y.n / new_n) * y.S) + (((x.n * y.n)/ (new_n ** 2)) * (np.outer(dist, dist)))
    eigen_value, eigen_vector = np.linalg.eigh(new_S)
    # reverse the order of the eigenvalue/vectors to be in DESCENDING order
    eigen_value = eigen_value[::-1]
    eigen_vector = eigen_vector[:, ::-1]
    # eigen_vector = eigen_vector.T
    # calculate new width that based on confidence ellipsoid
    confidence = new_n / (new_n + x.dim)
    chi = chi2.ppf(confidence, x.dim)
    new_A = np.sqrt(np.abs(eigen_value) * chi)
    new_A[new_A == 0] = x.eps

    return Node(dimension=x.dim, label=x.label, A=new_A, n=new_n, M=new_M, S=new_S, U=eigen_vector, eps=x.eps, alpha=x.alpha, p=(x.p + y.p) / 2)

def train_test_split(ds, test_ratio=0.2):
    ds = ds[:]  # copy
    random.shuffle(ds)

    split_idx = int(len(ds) * (1 - test_ratio))
    return ds[:split_idx], ds[split_idx:]

class Node():
    def __init__(self,
        dimension, label,
        A, p, eps, alpha,
        n=1,
        M=None,
        S=None,
        U=None,
        boundary=None):
        """
        A node contain information about a single ellipsoid such as its centre, cov matrix, and the axis length andd direction.
        It can be "updated" with a point to move it to learn and cover more data point.
        
        Parameters
        A (np.ndarray) (dim) : The length of each axis. Not necessary sorted.  np array 
        p (np.ndarray) (dim) : The power/exponent the nth term of the ellipsoid will be raised to when calculating surface function.
        eps (float) : A small value to add to the axis length when calculating surface function to avoid div by zero.
        alpha (float) : A hyperparameter between [0, 1] to control how the axis length is updated.
                        It specify the weight between "fixed" and "dynamic" update rule.
                        For more information see (Wongsriphisant, et al. 2026) https://doi.org/10.1016/j.eswa.2025.129818
        n (int) : The number of data points this node has learnt from.
        U (np.ndarray) (dim, dim) : The eigenvectors of the cov matrix. NOT transposed that is U[i] contain the ith eigenvector.
        M (np.ndarray) (dim) : The mean/center of the ellipsoid
        S (np.ndarray) (dim, dim) : The cov matrix of the ellipsoid. 
        boundary (dict) : A dictionary containing the boundary point and a special key, _max containing the maximum value.
        max_boundary (int) : The maximum number of points allowed in the boundary.
        """

        self.dim = dimension
        self.label = label
        self.A = A
        self.p = p
        self.eps = eps
        self.alpha = alpha
        self.n = n
        self.confidence = n / (n + self.dim)
        self.chi = chi2.ppf(self.confidence, self.dim)

        self.U = np.eye(dimension) if U is None else U
        self.M = np.array([0.] * dimension) if M is None else M
        self.S = np.zeros((dimension, dimension)) if S is None else S

        if boundary is not None:
            self.boundary = boundary
        else:
            self.boundary = []
        self.max_boundary = 2 * dimension

    def update_boundary(self, x, xd):
        """
        Update the boundary that contain the points nearest to the edge.
        This function is called when the ellipsoid is moved/updated.
        The point x is also considered whether it is a boundary point.
        """
        #print(x, self.boundary)
        # First, update the points in the boundary because the ellipsoid might have moved.
        for p in self.boundary:
            d = surface_single(p - self.M, self.U, self.A, self.p)

            if d > 0:
                for i, x in enumerate(self.boundary):
                    if np.allclose(x, p):
                        self.boundary.pop(i)
                        break

        # If the number of points in the boundary is small we simply add the new point
        d = surface_single(x - self.M, self.U, self.A, self.p)
        if d > 0:
            return None
        if len(self.boundary) < self.max_boundary:
            self.boundary.append(x)
            return None

        # Create copies of the boundary
        A = np.array(self.boundary)
        As = np.repeat(A[None, :, :], A.shape[0], axis=0)
        # Replace one row in each copy of A with x
        As[np.arange(A.shape[0]), np.arange(A.shape[0])] = x

        def f(mats):
            # mats has shape (k, n, m)
            gram = np.matmul(mats.transpose(0, 2, 1), mats)  # shape (k, m, m)
            return np.linalg.det(gram)
        det = f(np.array([A]))
        dets = f(As)

        if det > dets.max():
            pass
        else:
            self.boundary[np.argmax(dets)] = x

        """# Else we remove the point closest to the center (this might be x),        
        if d > self.boundary["_min"][1]:
            self.boundary.pop(self.boundary["_min"][0])
            self.boundary[tuple(x)] = d
            if d > self.boundary["_max"][1]:
                self.boundary["_max"] = (p, d)
            if d < self.boundary["_min"][1]:
                self.boundary["_min"] = (p, d)"""

    def update_exponent(self):
        if len(self.boundary) < self.max_boundary:
            return None
        print("pass")
        x0 = self.p
        cons = []
        for b_point in self.boundary:
            def con(x):
                return - surface_single(b_point - self.M, self.U, self.A, x)
            cons.append({'type': 'ineq', 'fun': con})
            
        solution = minimize(np.sum, x0, method='SLSQP', constraints=cons, bounds=[(1, 10) for _ in range(len(self.p))])
        print("replace",self.p,solution.x)
        self.p = solution.x

    def update(self, x, parent):
        """
        Update the node with a new data point x.
        Performing the following steps.
        1. Calculate the new mean and cov matrix of the node including the new point
        2. Calculate the eigenvalue/vector of the new cov matrix
        3. Update the width of the axes of the ellipsoid
        4. Calculate the growth criterion to check whether the updated ellipsoid cover the new data point
        5. If the growth criterion is acceptable the node is allowed to grow/update to include this new data point,
           else the node is not updated and a new node is added to cover the new datapoint instead.

        Parameter
        x (np.ndarray) (dim) : A new data point the model need to learn/update from.
        parent (VEBF) : The VEBF model this node belong to. We only need this in case we need to add a new node to the VEBF model. 
        """

        # Calculate new mean and cov matrix
        n = self.n
        M = self.M
        alpha = n / (n + 1)
        beta = x / (n + 1)
        M_new = (alpha * M) + beta
        k_1 = (np.outer(x, x)/(n + 1)) - np.outer(M_new, M_new) + np.outer(M, M)
        k_2 = - (np.outer(M, M) / (n + 1))
        k = k_1 + k_2
        S_new = (alpha * self.S) + k

        eigen_value, eigen_vector = np.linalg.eigh(S_new)
        # reverse the order of the eigenvalue/vectors to be in DESCENDING order
        eigen_value = eigen_value[::-1]
        eigen_vector = eigen_vector[:, ::-1]

        # calculate new width with adaptively by combining fixed and dynamic width update rules
        """beta = 1 - self.alpha
        fixed = np.sqrt(np.pi * 2 * np.abs(eigen_value))
        dynamic = self.A + ((M_new - self.M) @ eigen_vector.T)
        new_A = (self.alpha * fixed) + (beta * dynamic)
        new_A[new_A == 0] = self.eps"""

        # calculate new width that based on confidence ellipsoid
        new_A = np.sqrt(np.abs(eigen_value) * self.chi)
        new_A[new_A == 0] = self.eps
        
        gc = surface_single(x - M_new, eigen_vector, self.A, self.p)
        if gc <= 0:
            # update the current node
            self.M = M_new
            self.U = eigen_vector
            self.S = S_new
            self.A = new_A
            self.n += 1
            self.confidence = n / (n + self.dim)
            self.chi = chi2.ppf(self.confidence, self.dim)
            # self.update_boundary(x, gc)
            # self.update_exponent()
            return self
        else:
            # add new node
            new_node = Node(self.dim, self.label, M=x, eps=self.eps, A=parent.widths[self.label], alpha=self.alpha, p=self.p)
            parent.nodes[self.label].append(new_node)
            return new_node

    def __str__(self):
        return f"A node covering {self.n} points."

class VEBF():
    def __init__(self, dimension, merge_parameter=0, eps=0.00001, alpha=0.55, delta=1., p=None):
        """
        The Versatile Ellipsoid Basis Function model.
        Geometrically, the model learn by covering the datapoints with ellipsoids to learn the distribution of the data.
        
        Parameter
        dimension (int) : Number of dimension of input feature.
        merge_parameter (float) : The hyperparameter that controll when two ellipsoids/nodes will be merged.
        p (np.ndarray) (dim) : The power/exponent the nth term of the ellipsoid will be raised to when calculating surface function. (dim) np array
        eps (float) : A small value to add to the axis length when calculating surface function to avoid div by zero.
        alpha (float) : A hyperparameter between [0, 1] to control how the axis length is updated.
                        It specify the weight between "fixed" and "dynamic" update rule.
                        For more information see (Wongsriphisant, et al. 2026) https://doi.org/10.1016/j.eswa.2025.129818
        delta (float) : A hyperparameter that is a multiplier for the initial width.
        widths (dict) : Dictionary containing the initial width for each label class.
                        The key is the label (we assume it is hashable).
                        The value is the average pairwise euclidean distance of examples in that class multiply with delta.
        """
        self.dim = dimension
        self.merge_parameter = merge_parameter
        self.nodes = {}
        self.eps = eps
        self.alpha = alpha
        self.delta = delta
        self.p = np.array([2.] * dimension) if p is None else p # default to "normal" hyperellipsoid with p=2
        self.widths = {}

    def update_width(self, X, Y, overwrite=False):
        """
        Update the widths dictionary for each class.
        If the overwrite flag is set to true, we recompute everything,
        else we only compute the initial width for unseen labels.
        """
        dists = {}
        labels = np.unique(Y)
        for x, label in zip(X, Y):
            if not overwrite and label in self.widths:
                continue
            if label in points:
                points[label].append(x)
            else:
                points[label] = [x]
        for label in points:
            self.widths[label] = pdist(np.array(points[label]), metric="euclidean").mean()

    def find_shortest(self, x, y):
        # TODO: vectorize this
        min_dist = float('inf')
        min_node = None
        for node in self.nodes[y]:
            dist = np.linalg.norm(node.M - x)
            if dist < min_dist:
                min_dist = dist
                min_node = node
        return min_node

    # NOTE: can combine this with find_shortest by using y=None as default value
    def find_shortest_all(self, x, nodes=None):
        min_dist = float('inf')
        min_node = None
        nodes = self.get_nodes() if nodes is None else nodes
        for node in nodes:
            dist = np.linalg.norm(node.M - x)
            if dist < min_dist:
                min_dist = dist
                min_node = node
        return min_node

    def check_merge(self, x):
        """
        Check for pair of nodes which can be merged.
        Only consider the "latest" node, x

        Parameters:
        x (Node): The node which has just been updated or created.
        """
        nodes = self.nodes[x.label]
        if len(nodes) == 1:
            return
        nodes.remove(x)
        
        Us = np.array([node.U for node in nodes])
        Ms = np.array([node.M for node in nodes])
        As = np.array([node.A for node in nodes])
        Ps = np.array([node.p for node in nodes])
        
        scores = surface_multi(x.M - Ms, Us, As, Ps)
        merge_candidates = [node for node, score in zip(nodes, scores) if score <= self.merge_parameter]
        if len(merge_candidates) > 0:
            y = merge_candidates[0]
            nodes.remove(y)
            new_node = merge_nodes(x, y)
            nodes.append(new_node)
            self.check_merge(new_node)
            return

        n = len(nodes)
        U = x.U
        A = x.A
        P = x.p
        Us = np.broadcast_to(U, (n,) + U.shape)
        As = np.broadcast_to(A, (n,) + A.shape)
        Ps = np.broadcast_to(P, (n,) + P.shape)

        scores = surface_multi(Ms - x.M, Us, As, Ps)
        merge_candidates = [node for node, score in zip(nodes, scores) if score <= self.merge_parameter]
        if len(merge_candidates) > 0:
            y = merge_candidates[0]
            nodes.remove(y)
            new_node = merge_nodes(y, x)
            nodes.append(new_node)
            self.check_merge(new_node)
            return
            
        nodes.append(x)

    def get_nodes(self):
        return list(chain.from_iterable(self.nodes.values()))

    def predict(self, x):

        nodes = self.get_nodes()
        assert len(nodes) > 0, "Can't predict with empty VEBF."
        Us = np.array([node.U for node in nodes])
        Ms = np.array([node.M for node in nodes])
        As = np.array([node.A for node in nodes])
        Ps = np.array([node.p for node in nodes])

        scores = surface_multi(x - Ms, Us, As, Ps)
        idx = np.argsort(scores)
        result = nodes[idx[0]].label
        return result
        """ OLD prediction by euclidean

        candidates = [node for node, score in zip(nodes, scores) if score <= 0]

        if len(candidates) == 0:
            closest = self.find_shortest_all(x)
            return closest.label
        return self.find_shortest_all(x, candidates).label"""
            
    def fit_single(self, x, y):
        if y not in self.nodes: # Add a new node for unseen label.
            self.nodes[y] = [Node(dimension=self.dim, label=y, M=x, eps=self.eps, A=self.widths[y], alpha=self.alpha, p=self.p)]
        else: # Update closest existing node for existing label.
            shortest = self.find_shortest(x, y)
            latest = shortest.update(x, self) # latest is the node with latest change, which need to checked for merge
            self.check_merge(latest)

    def fit(self, X, Y, batch=False):
        self.update_width(X, Y)
        if not batch:
            for x, y in zip(X, Y):
                self.fit_single(x, y)
        else:
            raise NotImplementedError

    def __str__(self):
        total = sum(len(x) for x in self.nodes.values())
        string = f"A vebf with {total} nodes.\n"
        for label, nodes in self.nodes.items():
            for node in nodes:
                string += f"Label {label}: {str(node)}\n"
        return string