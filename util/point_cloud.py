import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree


def filter_pointcloud(x, distance_threshold=0.2, max_points=30000, K=16):
    """ Filter outliers from point cloud.
    Removes point clusters or singletons further than distance_threshold from biggest cluster.
    Args:
        x: point cloud with shape (N, 3)
        distance_threshold: maximum Euclidean distance between connected clusters
        max_points: point clouds larger than max_points will be randomly downsampled
    Returns:
        inlier_indices: indices of inlier points with shape (N,)
    """
    N, D = x.shape
    if N > max_points:
        indices = np.random.permutation(N)[:max_points]
        x = x[indices, :]

    # 1. Form distance matrix and apply threshold:
    metric = 'sqeuclidean'
    pairwise_distances = pdist(x, metric) # Condensed distance matrix; dist(u=X[i], v=X[j]) is computed and stored in entry m * i + j - ((i + 2) * (i + 1)) // 2
    pairwise_distances = squareform(pairwise_distances) # Convert to square matrix
    edges = pairwise_distances < distance_threshold ** 2 # Apply threshold

    # 2. Compute connected component labels
    graph = csr_matrix(edges)
    num_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    mode_label, num_modal_points = mode(labels)
    mode_label, num_modal_points = mode_label[0], num_modal_points[0] # Take first in case of tie
    inlier_indices = (labels == mode_label)

    # 3. Cull points with fewer than K neighbours within distance_threshold
    results = KDTree(x).query_ball_tree(KDTree(x[inlier_indices, :]), r=distance_threshold)
    counts = np.array([len(r) for r in results])
    # print([len(r) for r in results])
    inlier_indices2 = counts >= K
    inlier_indices *= inlier_indices2

    if N > max_points:
        inlier_indices = indices[inlier_indices] # Convert back into original indices

    print(f'num_components: {num_components}')
    print(f'num_modal_points: {num_modal_points}')
    print(f'num_filtered_points: {len(inlier_indices)}')
    print(f'inlier fraction: {100. * len(inlier_indices) / x.shape[0]} \%')

    return inlier_indices