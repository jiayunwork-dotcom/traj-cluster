import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.cluster import OPTICS as SKOPTICS, SpectralClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from .preprocessing import Trip
from .similarity import distance_to_similarity


@dataclass
class ClusterResult:
    labels: np.ndarray
    cluster_ids: List[int]
    n_clusters: int
    n_noise: int
    silhouette_score: Optional[float]
    algorithm: str
    cluster_stats: Dict[int, Dict] = field(default_factory=dict)
    extra_data: Dict = field(default_factory=dict)


def compute_k_distance(matrix: np.ndarray, k: int = 4) -> np.ndarray:
    sorted_dists = np.sort(matrix, axis=1)[:, 1:k + 1]
    k_dists = sorted_dists[:, -1]
    return np.sort(k_dists)[::-1]


def find_eps_knuckle(dist_sorted: np.ndarray) -> float:
    if len(dist_sorted) < 3:
        return float(dist_sorted[0]) if len(dist_sorted) > 0 else 0.0

    x = np.arange(len(dist_sorted))
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (dist_sorted - dist_sorted.min()) / (dist_sorted.max() - dist_sorted.min() + 1e-10)

    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec) + 1e-10
    line_vec = line_vec / line_len

    dist_to_line = np.zeros(len(x_norm))
    for i in range(len(x_norm)):
        point = np.array([x_norm[i], y_norm[i]])
        vec = point - start
        dist_to_line[i] = np.linalg.norm(vec - np.dot(vec, line_vec) * line_vec)

    idx = int(np.argmax(dist_to_line))
    return float(dist_sorted[idx])


def cluster_dbscan(
    dist_matrix: np.ndarray,
    trips: List[Trip],
    eps: Optional[float] = None,
    min_samples: int = 5
) -> ClusterResult:
    if eps is None:
        k_dists = compute_k_distance(dist_matrix, k=min_samples)
        eps = find_eps_knuckle(k_dists)

    eps = max(eps, 1.0)

    model = SKDBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = model.fit_predict(dist_matrix)

    return _make_result(labels, dist_matrix, trips, 'dbscan', {
        'eps': eps,
        'min_samples': min_samples,
        'k_distance_plot': compute_k_distance(dist_matrix, k=min_samples)
    })


def cluster_optics(
    dist_matrix: np.ndarray,
    trips: List[Trip],
    min_samples: int = 5,
    max_eps: float = np.inf,
    xi: float = 0.05
) -> ClusterResult:
    sorted_idx = np.argsort(dist_matrix, axis=1)
    dist_matrix_copy = dist_matrix.copy()

    model = SKOPTICS(
        min_samples=min_samples,
        max_eps=max_eps if np.isfinite(max_eps) else np.max(dist_matrix_copy) * 1.1,
        xi=xi,
        metric='precomputed'
    )
    labels = model.fit_predict(dist_matrix_copy)

    ordering = model.ordering_
    reachability = model.reachability_[ordering]

    return _make_result(labels, dist_matrix, trips, 'optics', {
        'min_samples': min_samples,
        'reachability_plot': {
            'ordering': ordering.tolist(),
            'reachability': reachability.tolist()
        }
    })


def cluster_spectral(
    dist_matrix: np.ndarray,
    trips: List[Trip],
    n_clusters: Optional[int] = None,
    gamma: Optional[float] = None
) -> ClusterResult:
    sim_matrix = distance_to_similarity(dist_matrix, gamma=gamma)
    sim_matrix = (sim_matrix + sim_matrix.T) / 2.0
    np.fill_diagonal(sim_matrix, 0.0)

    if n_clusters is None:
        n_clusters = _estimate_k_by_eigengap(sim_matrix)

    n_clusters = max(2, min(n_clusters, len(trips) - 1))

    try:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(sim_matrix)
    except Exception:
        labels = np.zeros(len(trips), dtype=int)

    eigvals = None
    try:
        L = laplacian(sim_matrix, normed=True)
        eigvals, _ = eigh(L.todense() if hasattr(L, 'todense') else L)
        eigvals = eigvals.tolist()
    except Exception:
        eigvals = []

    return _make_result(labels, dist_matrix, trips, 'spectral', {
        'n_clusters': n_clusters,
        'eigenvalues': eigvals,
        'gamma': gamma
    })


def _estimate_k_by_eigengap(sim_matrix: np.ndarray, max_k: int = 20) -> int:
    try:
        L = laplacian(sim_matrix, normed=True)
        if hasattr(L, 'todense'):
            L_dense = L.todense()
        else:
            L_dense = np.array(L)
        eigvals, _ = eigh(L_dense)
        eigvals = np.sort(eigvals)
        eigvals = eigvals[:max_k]

        if len(eigvals) < 3:
            return 2

        gaps = np.diff(eigvals)
        valid_mask = eigvals[1:] < 0.95
        if valid_mask.sum() > 0:
            gaps_masked = gaps.copy()
            gaps_masked[~valid_mask[:-1]] = -np.inf
            return int(np.argmax(gaps_masked) + 2)
        return int(np.argmax(gaps) + 2)
    except Exception:
        return min(5, len(sim_matrix) // 3)


def _make_result(
    labels: np.ndarray,
    dist_matrix: np.ndarray,
    trips: List[Trip],
    algorithm: str,
    extra: Dict
) -> ClusterResult:
    unique_labels = sorted(set(labels))
    n_noise = int(np.sum(labels == -1))
    cluster_ids = [l for l in unique_labels if l != -1]
    n_clusters = len(cluster_ids)

    sil_score = None
    if n_clusters >= 2 and len(trips) > n_clusters * 2:
        try:
            valid_mask = labels != -1
            if valid_mask.sum() > 1:
                sil_score = float(silhouette_score(
                    dist_matrix[valid_mask][:, valid_mask],
                    labels[valid_mask],
                    metric='precomputed'
                ))
        except Exception:
            pass

    stats = {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        cluster_trips = [trips[i] for i in idx]
        dists = np.array([t.distance_m for t in cluster_trips])
        durations = np.array([t.duration_s for t in cluster_trips])
        stats[cid] = {
            'count': len(idx),
            'avg_distance_km': float(np.mean(dists) / 1000.0) if len(dists) > 0 else 0.0,
            'avg_duration_min': float(np.mean(durations) / 60.0) if len(durations) > 0 else 0.0,
            'trip_indices': idx.tolist()
        }

    return ClusterResult(
        labels=labels,
        cluster_ids=cluster_ids,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette_score=sil_score,
        algorithm=algorithm,
        cluster_stats=stats,
        extra_data=extra
    )


def apply_clustering(
    dist_matrix: np.ndarray,
    trips: List[Trip],
    method: str = 'dbscan',
    **kwargs
) -> ClusterResult:
    if method == 'dbscan':
        return cluster_dbscan(dist_matrix, trips, **kwargs)
    elif method == 'optics':
        return cluster_optics(dist_matrix, trips, **kwargs)
    elif method == 'spectral':
        return cluster_spectral(dist_matrix, trips, **kwargs)
    else:
        raise ValueError(f"未知聚类方法: {method}")
