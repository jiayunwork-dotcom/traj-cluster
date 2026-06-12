import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Polygon
from .preprocessing import Trip
from .clustering import ClusterResult
from .utils import haversine_vectorized, centroid, haversine_pair, latlon_to_grid


@dataclass
class Hotspot:
    hotspot_id: str
    center_lat: float
    center_lon: float
    area_km2: float
    point_count: int
    density_per_km2: float
    unique_users: int
    peak_hour: int
    convex_hull: Optional[List[Tuple[float, float]]]


@dataclass
class HotspotResult:
    hotspots: List[Hotspot]
    heatmap_data: np.ndarray
    heatmap_bbox: Tuple[float, float, float, float]
    heatmap_resolution: int


@dataclass
class Anomaly:
    trip_index: int
    trip_id: str
    anomaly_types: List[str]
    details: Dict


def detect_hotspots(
    all_lats: np.ndarray,
    all_lons: np.ndarray,
    all_user_ids: Optional[np.ndarray] = None,
    all_timestamps: Optional[np.ndarray] = None,
    eps_m: float = 200.0,
    min_samples: int = 50,
    heatmap_resolution: int = 100
) -> HotspotResult:
    if len(all_lats) == 0:
        return HotspotResult(
            hotspots=[],
            heatmap_data=np.zeros((heatmap_resolution, heatmap_resolution)),
            heatmap_bbox=(0, 0, 0, 0),
            heatmap_resolution=heatmap_resolution
        )

    eps_km = eps_m / 1000.0
    lat_km_per_deg = 111.0
    lon_km_per_deg = 111.0 * np.cos(np.radians(np.mean(all_lats)))
    eps_lat = eps_km / lat_km_per_deg
    eps_lon = eps_km / max(lon_km_per_deg, 0.001)

    X = np.column_stack([all_lats, all_lons])
    X_scaled = X.copy()
    X_scaled[:, 0] *= lat_km_per_deg / eps_km
    X_scaled[:, 1] *= lon_km_per_deg / eps_km

    model = DBSCAN(eps=np.sqrt(2), min_samples=min_samples, metric='euclidean')
    labels = model.fit_predict(X_scaled)

    unique_labels = sorted(set(labels))
    hotspots = []
    user_set_per_cluster: Dict[int, set] = defaultdict(set)
    hour_counter: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    if all_user_ids is None:
        all_user_ids = np.array([''] * len(all_lats))
    if all_timestamps is None:
        all_timestamps = np.array([pd.Timestamp.now()] * len(all_lats))

    for idx, lbl in enumerate(labels):
        if lbl != -1:
            user_set_per_cluster[lbl].add(str(all_user_ids[idx]))
            ts = pd.Timestamp(all_timestamps[idx])
            hour_counter[lbl][ts.hour] += 1

    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        cluster_lats = all_lats[mask]
        cluster_lons = all_lons[mask]
        if len(cluster_lats) < min_samples:
            continue

        c_lat, c_lon = centroid(cluster_lats, cluster_lons)
        hull = None
        area_km2 = 0.0

        try:
            points = list(zip(cluster_lats, cluster_lons))
            mp = MultiPoint(points)
            convex_hull_poly = mp.convex_hull
            hull = [(p[1], p[0]) for p in list(convex_hull_poly.exterior.coords)]
            hull = [(p[0], p[1]) for p in list(convex_hull_poly.exterior.coords)]
            lat_deg_width = np.max(cluster_lats) - np.min(cluster_lats)
            lon_deg_width = np.max(cluster_lons) - np.min(cluster_lons)
            lat_km = lat_deg_width * 111.0
            lon_km = lon_deg_width * 111.0 * np.cos(np.radians(c_lat))
            area_km2 = lat_km * lon_km * 0.7854
        except Exception:
            pass

        unique_users_count = len(user_set_per_cluster[lbl])
        peak_hour = max(hour_counter[lbl].items(), key=lambda x: x[1])[0] if hour_counter[lbl] else 12
        density = len(cluster_lats) / max(area_km2, 0.001)

        hotspots.append(Hotspot(
            hotspot_id=f"hotspot_{lbl}",
            center_lat=round(c_lat, 6),
            center_lon=round(c_lon, 6),
            area_km2=round(area_km2, 4),
            point_count=int(mask.sum()),
            density_per_km2=round(density, 1),
            unique_users=unique_users_count,
            peak_hour=peak_hour,
            convex_hull=hull
        ))

    hotspots.sort(key=lambda h: h.point_count, reverse=True)

    bbox = (
        float(np.min(all_lons)), float(np.min(all_lats)),
        float(np.max(all_lons)), float(np.max(all_lats))
    )
    lon_min, lat_min, lon_max, lat_max = bbox
    heatmap = np.zeros((heatmap_resolution, heatmap_resolution), dtype=int)

    lat_bins = np.linspace(lat_min, lat_max, heatmap_resolution + 1)
    lon_bins = np.linspace(lon_min, lon_max, heatmap_resolution + 1)

    for lat, lon in zip(all_lats, all_lons):
        i = min(heatmap_resolution - 1, max(0, int(np.digitize(lat, lat_bins) - 1)))
        j = min(heatmap_resolution - 1, max(0, int(np.digitize(lon, lon_bins) - 1)))
        heatmap[heatmap_resolution - 1 - i, j] += 1

    return HotspotResult(
        hotspots=hotspots,
        heatmap_data=heatmap,
        heatmap_bbox=bbox,
        heatmap_resolution=heatmap_resolution
    )


def detect_anomalies(
    trips: List[Trip],
    cluster_result: ClusterResult,
    dist_matrix: Optional[np.ndarray] = None,
    od_pairs: Optional[Dict] = None
) -> List[Anomaly]:
    anomalies = []
    labels = cluster_result.labels

    od_avg_duration: Dict[Tuple, List[float]] = defaultdict(list)
    if od_pairs is not None:
        for idx, t in enumerate(trips):
            o_g = latlon_to_grid(t.origin[0], t.origin[1])
            d_g = latlon_to_grid(t.destination[0], t.destination[1])
            od_avg_duration[(o_g, d_g)].append(t.duration_s)

    od_mean = {k: float(np.mean(v)) for k, v in od_avg_duration.items() if len(v) > 0}

    cluster_avg_dists = {}
    cluster_std_dists = {}
    for cid in cluster_result.cluster_ids:
        stats = cluster_result.cluster_stats.get(cid, {})
        idx_list = stats.get('trip_indices', [])
        if len(idx_list) >= 2 and dist_matrix is not None:
            within_dists = []
            for i in idx_list:
                for j in idx_list:
                    if i < j:
                        within_dists.append(dist_matrix[i, j])
            if within_dists:
                cluster_avg_dists[cid] = float(np.mean(within_dists))
                cluster_std_dists[cid] = float(np.std(within_dists))

    for idx, t in enumerate(trips):
        anomaly_types = []
        details = {}
        label = labels[idx] if idx < len(labels) else -1

        if label == -1:
            anomaly_types.append('cluster_noise')
            details['reason'] = 'DBSCAN标记为噪声轨迹'

        if label != -1 and label in cluster_avg_dists and dist_matrix is not None:
            stats = cluster_result.cluster_stats.get(label, {})
            idx_list = stats.get('trip_indices', [])
            if len(idx_list) >= 2:
                mean_dists = []
                for other_idx in idx_list:
                    if other_idx != idx:
                        mean_dists.append(dist_matrix[idx, other_idx])
                if mean_dists:
                    avg_dist = float(np.mean(mean_dists))
                    threshold = cluster_avg_dists[label] + 2 * cluster_std_dists[label]
                    if avg_dist > threshold:
                        anomaly_types.append('path_deviation')
                        details['path_deviation'] = f"距簇中心平均距离 {avg_dist:.1f}m > 阈值 {threshold:.1f}m"

        o_g = latlon_to_grid(t.origin[0], t.origin[1])
        d_g = latlon_to_grid(t.destination[0], t.destination[1])
        key = (o_g, d_g)
        if key in od_mean:
            avg_dur = od_mean[key]
            if t.duration_s > avg_dur * 3.0:
                anomaly_types.append('detour')
                details['detour'] = f"时长 {t.duration_s/60:.1f}min > 同OD平均 {avg_dur/60:.1f}min 的3倍"

        if anomaly_types:
            anomalies.append(Anomaly(
                trip_index=idx,
                trip_id=t.trip_id,
                anomaly_types=anomaly_types,
                details=details
            ))

    return anomalies
