import numpy as np
from geopy.distance import geodesic
from typing import Tuple, Optional

EARTH_RADIUS_KM = 6371.0088


def haversine_vectorized(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c * 1000


def haversine_pair(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_vectorized(
        np.array([lat1]), np.array([lon1]), np.array([lat2]), np.array([lon2])
    )[0]


def consecutive_distances(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    if len(lats) < 2:
        return np.array([])
    return haversine_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])


def total_trajectory_distance(lats: np.ndarray, lons: np.ndarray) -> float:
    return float(np.sum(consecutive_distances(lats, lons)))


def point_to_segment_distance(
    p_lat: float, p_lon: float,
    a_lat: float, a_lon: float,
    b_lat: float, b_lon: float
) -> float:
    d_ap = haversine_pair(p_lat, p_lon, a_lat, a_lon)
    d_bp = haversine_pair(p_lat, p_lon, b_lat, b_lon)
    d_ab = haversine_pair(a_lat, a_lon, b_lat, b_lon)

    if d_ab == 0:
        return d_ap

    t = ((p_lat - a_lat) * (b_lat - a_lat) + (p_lon - a_lon) * (b_lon - a_lon)) / (d_ab ** 2)
    t = max(0.0, min(1.0, t))

    proj_lat = a_lat + t * (b_lat - a_lat)
    proj_lon = a_lon + t * (b_lon - a_lon)
    return haversine_pair(p_lat, p_lon, proj_lat, proj_lon)


def compute_bbox(lats: np.ndarray, lons: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(np.min(lons)),
        float(np.min(lats)),
        float(np.max(lons)),
        float(np.max(lats))
    )


def centroid(lats: np.ndarray, lons: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(lats)), float(np.mean(lons))


def latlon_to_grid(lat: float, lon: float, cell_size_m: float = 500.0) -> Tuple[int, int]:
    lat_cell = int(lat / (cell_size_m / 111000.0))
    lon_scale = 111000.0 * np.cos(np.radians(lat))
    lon_cell = int(lon / (cell_size_m / max(lon_scale, 1.0)))
    return lat_cell, lon_cell


def smooth_trajectory(lats: np.ndarray, lons: np.ndarray, window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    if window < 2 or len(lats) < window:
        return lats, lons
    kernel = np.ones(window) / window
    lats_smooth = np.convolve(lats, kernel, mode='same')
    lons_smooth = np.convolve(lons, kernel, mode='same')
    lats_smooth[:window // 2] = lats[:window // 2]
    lats_smooth[-window // 2:] = lats[-window // 2:]
    lons_smooth[:window // 2] = lons[:window // 2]
    lons_smooth[-window // 2:] = lons[-window // 2:]
    return lats_smooth, lons_smooth


def parse_timestamp(ts_str: str) -> Optional[np.datetime64]:
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y%m%d %H:%M:%S',
        '%d-%m-%Y %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
    ]
    for fmt in formats:
        try:
            return np.datetime64(pd.Timestamp(ts_str, format=fmt))
        except:
            continue
    try:
        return np.datetime64(pd.Timestamp(ts_str))
    except:
        try:
            return np.datetime64(int(ts_str), 's')
        except:
            return None


import pandas as pd
