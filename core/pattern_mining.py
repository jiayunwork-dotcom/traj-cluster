import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from .preprocessing import Trip
from .utils import latlon_to_grid, haversine_pair


@dataclass
class ODPair:
    origin_grid: Tuple[int, int]
    dest_grid: Tuple[int, int]
    origin_centroid: Tuple[float, float]
    dest_centroid: Tuple[float, float]
    count: int
    avg_distance_km: float
    avg_duration_min: float


@dataclass
class CommutePattern:
    user_id: str
    home_grid: Tuple[int, int]
    work_grid: Tuple[int, int]
    home_loc: Tuple[float, float]
    work_loc: Tuple[float, float]
    morning_od_count: int
    evening_od_count: int
    total_workdays: int
    confidence: float


@dataclass
class FrequentPath:
    grid_sequence: Tuple[Tuple[int, int], ...]
    support: int
    support_ratio: float
    representative_trip_indices: List[int]
    centroid_path: List[Tuple[float, float]]


@dataclass
class PatternMiningResult:
    od_pairs: List[ODPair]
    od_matrix: np.ndarray
    grid_index: Dict[Tuple[int, int], int]
    commute_patterns: List[CommutePattern]
    frequent_paths: List[FrequentPath]
    hour_distribution: Dict[int, Dict[int, int]]
    weekday_distribution: Dict[int, Dict[int, int]]
    duration_distribution: Dict[int, np.ndarray]
    cell_size_m: float


def recommend_grid_size(trips: List[Trip]) -> float:
    all_spacings = []
    for t in trips:
        for i in range(1, len(t.lats)):
            d = haversine_pair(t.lats[i - 1], t.lons[i - 1], t.lats[i], t.lons[i])
            all_spacings.append(d)
    if not all_spacings:
        return 500.0
    avg_spacing = float(np.mean(all_spacings))
    return max(200.0, min(1000.0, avg_spacing * 3))


def analyze_od(trips: List[Trip], cell_size_m: float = 500.0) -> Tuple[List[ODPair], np.ndarray, Dict[Tuple[int, int], int]]:
    od_counter: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict] = defaultdict(
        lambda: {'count': 0, 'dists': [], 'durations': [], 'orig_lats': [], 'orig_lons': [], 'dest_lats': [], 'dest_lons': []}
    )

    all_grids: Set[Tuple[int, int]] = set()

    for t in trips:
        o_lat, o_lon = t.origin
        d_lat, d_lon = t.destination
        o_grid = latlon_to_grid(o_lat, o_lon, cell_size_m)
        d_grid = latlon_to_grid(d_lat, d_lon, cell_size_m)
        all_grids.add(o_grid)
        all_grids.add(d_grid)

        entry = od_counter[(o_grid, d_grid)]
        entry['count'] += 1
        entry['dists'].append(t.distance_m / 1000.0)
        entry['durations'].append(t.duration_s / 60.0)
        entry['orig_lats'].append(o_lat)
        entry['orig_lons'].append(o_lon)
        entry['dest_lats'].append(d_lat)
        entry['dest_lons'].append(d_lon)

    grid_list = sorted(all_grids)
    grid_index = {g: i for i, g in enumerate(grid_list)}
    n = len(grid_list)
    od_matrix = np.zeros((n, n), dtype=int)

    od_pairs = []
    for (o_g, d_g), data in od_counter.items():
        if o_g in grid_index and d_g in grid_index:
            od_matrix[grid_index[o_g], grid_index[d_g]] = data['count']

        od_pairs.append(ODPair(
            origin_grid=o_g,
            dest_grid=d_g,
            origin_centroid=(float(np.mean(data['orig_lats'])), float(np.mean(data['orig_lons']))),
            dest_centroid=(float(np.mean(data['dest_lats'])), float(np.mean(data['dest_lons']))),
            count=data['count'],
            avg_distance_km=float(np.mean(data['dists'])),
            avg_duration_min=float(np.mean(data['durations']))
        ))

    od_pairs.sort(key=lambda x: x.count, reverse=True)
    return od_pairs, od_matrix, grid_index


def detect_commute_patterns(trips: List[Trip], cell_size_m: float = 500.0,
                            threshold_ratio: float = 0.6) -> List[CommutePattern]:
    user_trips: Dict[str, List[Trip]] = defaultdict(list)
    for t in trips:
        user_trips[t.user_id].append(t)

    patterns = []

    for uid, user_trip_list in user_trips.items():
        workday_trips = []
        workdays: Set[str] = set()

        for t in user_trip_list:
            ts = pd.Timestamp(t.start_time)
            weekday = ts.weekday()
            if weekday < 5:
                workday_trips.append(t)
                workdays.add(ts.strftime('%Y-%m-%d'))

        total_workdays = len(workdays)
        if total_workdays < 3:
            continue

        morning_od: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = defaultdict(int)
        evening_od: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = defaultdict(int)
        morning_loc: Dict = defaultdict(lambda: {'lats': [], 'lons': []})
        evening_loc: Dict = defaultdict(lambda: {'lats': [], 'lons': []})

        for t in workday_trips:
            ts = pd.Timestamp(t.start_time)
            hour = ts.hour
            o_lat, o_lon = t.origin
            d_lat, d_lon = t.destination
            o_grid = latlon_to_grid(o_lat, o_lon, cell_size_m)
            d_grid = latlon_to_grid(d_lat, d_lon, cell_size_m)

            if 7 <= hour <= 9:
                key = (o_grid, d_grid)
                morning_od[key] += 1
                morning_loc[key]['lats'].append(o_lat)
                morning_loc[key]['lons'].append(o_lon)
                morning_loc[key]['lats'].append(d_lat)
                morning_loc[key]['lons'].append(d_lon)
            elif 17 <= hour <= 19:
                key = (o_grid, d_grid)
                evening_od[key] += 1
                evening_loc[key]['lats'].append(o_lat)
                evening_loc[key]['lons'].append(o_lon)
                evening_loc[key]['lats'].append(d_lat)
                evening_loc[key]['lons'].append(d_lon)

        if not morning_od or not evening_od:
            continue

        best_morning = max(morning_od.items(), key=lambda x: x[1])
        best_evening = max(evening_od.items(), key=lambda x: x[1])

        m_count = best_morning[1]
        e_count = best_evening[1]

        if m_count / total_workdays < threshold_ratio or e_count / total_workdays < threshold_ratio:
            continue

        m_home_g, m_work_g = best_morning[0]
        e_work_g, e_home_g = best_evening[0]

        if m_home_g != e_home_g or m_work_g != e_work_g:
            if m_home_g == e_work_g and m_work_g == e_home_g:
                pass
            else:
                continue

        confidence = min(m_count, e_count) / total_workdays

        m_data = morning_loc[best_morning[0]]
        home_loc = (float(np.mean(m_data['lats'][:m_count])), float(np.mean(m_data['lons'][:m_count])))
        work_loc = (float(np.mean(m_data['lats'][m_count:])), float(np.mean(m_data['lons'][m_count:])))

        patterns.append(CommutePattern(
            user_id=uid,
            home_grid=m_home_g,
            work_grid=m_work_g,
            home_loc=home_loc,
            work_loc=work_loc,
            morning_od_count=m_count,
            evening_od_count=e_count,
            total_workdays=total_workdays,
            confidence=round(confidence, 3)
        ))

    patterns.sort(key=lambda x: x.confidence, reverse=True)
    return patterns


def mine_frequent_paths(trips: List[Trip], cell_size_m: float = 500.0,
                        min_support: int = 3, min_len: int = 5) -> List[FrequentPath]:
    if len(trips) == 0:
        return []

    sequences = []
    grid_coords: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)

    for t in trips:
        seq = []
        prev_grid = None
        for lat, lon in zip(t.lats, t.lons):
            g = latlon_to_grid(lat, lon, cell_size_m)
            grid_coords[g].append((lat, lon))
            if g != prev_grid:
                seq.append(g)
                prev_grid = g
        if len(seq) >= min_len:
            sequences.append(tuple(seq))

    if not sequences:
        return []

    class TrieNode:
        def __init__(self):
            self.children: Dict = {}
            self.count: int = 0
            self.trip_indices: List[int] = []

    root = TrieNode()

    for trip_idx, seq in enumerate(sequences):
        for start in range(len(seq)):
            node = root
            for j in range(start, len(seq)):
                cell = seq[j]
                if cell not in node.children:
                    node.children[cell] = TrieNode()
                node = node.children[cell]
                node.count += 1
                if not node.trip_indices or node.trip_indices[-1] != trip_idx:
                    node.trip_indices.append(trip_idx)

    frequent = []

    def dfs(node: TrieNode, path: List[Tuple[int, int]]):
        if len(path) >= min_len and node.count >= min_support:
            path_tuple = tuple(path)
            centroid_path = []
            for g in path_tuple:
                coords = grid_coords.get(g, [])
                if coords:
                    lats = [c[0] for c in coords]
                    lons = [c[1] for c in coords]
                    centroid_path.append((float(np.mean(lats)), float(np.mean(lons))))
                elif len(path_tuple) > 0:
                    centroid_path.append((0.0, 0.0))

            unique_trips = len(set(node.trip_indices))
            if unique_trips >= min_support:
                frequent.append(FrequentPath(
                    grid_sequence=path_tuple,
                    support=unique_trips,
                    support_ratio=unique_trips / len(sequences),
                    representative_trip_indices=node.trip_indices[:10],
                    centroid_path=centroid_path
                ))

        for cell, child in sorted(node.children.items(), key=lambda x: -x[1].count):
            if child.count >= min_support:
                dfs(child, path + [cell])

    dfs(root, [])
    frequent.sort(key=lambda x: (-x.support, -len(x.grid_sequence)))
    return frequent[:50]


def analyze_temporal_distributions(trips: List[Trip], labels: Optional[np.ndarray] = None) -> Tuple[Dict, Dict, Dict]:
    if labels is None:
        labels = np.zeros(len(trips), dtype=int)

    hour_dist: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    weekday_dist: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    duration_dist: Dict[int, List[float]] = defaultdict(list)

    for idx, t in enumerate(trips):
        cluster_id = int(labels[idx]) if idx < len(labels) else 0
        ts = pd.Timestamp(t.start_time)
        hour_dist[cluster_id][ts.hour] += 1
        weekday_dist[cluster_id][ts.weekday()] += 1
        duration_dist[cluster_id].append(t.duration_s / 60.0)

    duration_dist_np = {cid: np.array(v) for cid, v in duration_dist.items()}
    return hour_dist, weekday_dist, duration_dist_np


def mine_all_patterns(
    trips: List[Trip],
    labels: Optional[np.ndarray] = None,
    cell_size_m: Optional[float] = None,
    min_support: Optional[int] = None
) -> PatternMiningResult:
    if cell_size_m is None:
        cell_size_m = recommend_grid_size(trips)
    if min_support is None:
        min_support = max(3, len(trips) // 20)

    od_pairs, od_matrix, grid_index = analyze_od(trips, cell_size_m)
    commute_patterns = detect_commute_patterns(trips, cell_size_m)
    frequent_paths = mine_frequent_paths(trips, cell_size_m, min_support)
    hour_dist, weekday_dist, duration_dist = analyze_temporal_distributions(trips, labels)

    return PatternMiningResult(
        od_pairs=od_pairs,
        od_matrix=od_matrix,
        grid_index=grid_index,
        commute_patterns=commute_patterns,
        frequent_paths=frequent_paths,
        hour_distribution={k: dict(v) for k, v in hour_dist.items()},
        weekday_distribution={k: dict(v) for k, v in weekday_dist.items()},
        duration_distribution=duration_dist,
        cell_size_m=cell_size_m
    )
