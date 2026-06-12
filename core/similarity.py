import numpy as np
from typing import List, Tuple, Callable, Optional
from .utils import haversine_vectorized, haversine_pair
from .preprocessing import Trip


def dtw_distance(trip1: Trip, trip2: Trip) -> float:
    n, m = len(trip1.lats), len(trip2.lats)
    if n == 0 or m == 0:
        return float('inf')

    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    lats1, lons1 = trip1.lats, trip1.lons
    lats2, lons2 = trip2.lats, trip2.lons

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = haversine_pair(lats1[i - 1], lons1[i - 1], lats2[j - 1], lons2[j - 1])
            cost_matrix[i, j] = d + min(
                cost_matrix[i - 1, j],
                cost_matrix[i, j - 1],
                cost_matrix[i - 1, j - 1]
            )

    return float(cost_matrix[n, m])


def fastdtw_distance(trip1: Trip, trip2: Trip, radius: int = 1) -> float:
    n, m = len(trip1.lats), len(trip2.lats)
    if n == 0 or m == 0:
        return float('inf')

    if min(n, m) < radius + 2:
        return dtw_distance(trip1, trip2)

    def _reduce_res(trip_lats, trip_lons, factor=2):
        new_len = len(trip_lats) // factor
        if new_len < 1:
            return trip_lats, trip_lons
        idx = (np.arange(new_len) * factor).astype(int)
        return trip_lats[idx], trip_lons[idx]

    def _expand_path(path, n, m, radius):
        path_set = set(path)
        expanded = set()
        for (i, j) in path_set:
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = i * 2 + di, j * 2 + dj
                    if 0 <= ni < n and 0 <= nj < m:
                        expanded.add((ni, nj))
        for i in range(n):
            expanded.add((i, 0))
            expanded.add((i, m - 1))
        for j in range(m):
            expanded.add((0, j))
            expanded.add((n - 1, j))
        return list(expanded)

    def _dtw_with_window(lats1, lons1, lats2, lons2, window):
        n, m = len(lats1), len(lats2)
        dist = np.full((n + 1, m + 1), np.inf)
        dist[0, 0] = 0.0
        for (i, j) in window:
            d = haversine_pair(lats1[i], lons1[i], lats2[j], lons2[j])
            dist[i + 1, j + 1] = d + min(
                dist[i, j + 1],
                dist[i + 1, j],
                dist[i, j]
            )
        return float(dist[n, m])

    def _dtw_simple(lats1, lons1, lats2, lons2):
        n, m = len(lats1), len(lats2)
        if n == 0 or m == 0:
            return float('inf'), []
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0.0
        came_from = {}
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = haversine_pair(lats1[i - 1], lons1[i - 1], lats2[j - 1], lons2[j - 1])
                options = []
                if cost_matrix[i - 1, j] < np.inf:
                    options.append((cost_matrix[i - 1, j], (i - 1, j)))
                if cost_matrix[i, j - 1] < np.inf:
                    options.append((cost_matrix[i, j - 1], (i, j - 1)))
                if cost_matrix[i - 1, j - 1] < np.inf:
                    options.append((cost_matrix[i - 1, j - 1], (i - 1, j - 1)))
                if options:
                    min_c, prev = min(options, key=lambda x: x[0])
                    cost_matrix[i, j] = d + min_c
                    came_from[(i, j)] = prev
        if cost_matrix[n, m] == np.inf:
            return float('inf'), []
        path = []
        cur = (n, m)
        while cur is not None and cur != (0, 0):
            if cur[0] > 0 and cur[1] > 0:
                path.append((cur[0] - 1, cur[1] - 1))
            cur = came_from.get(cur)
        path.reverse()
        return float(cost_matrix[n, m]), path

    radius = max(1, radius)
    min_size = radius + 2

    n1, n2 = len(trip1.lats), len(trip2.lats)
    if min(n1, n2) < min_size * 2:
        dist, _ = _dtw_simple(trip1.lats, trip1.lons, trip2.lats, trip2.lons)
        return dist

    lats1_r, lons1_r = trip1.lats, trip1.lons
    lats2_r, lons2_r = trip2.lats, trip2.lons

    reductions = []
    while len(lats1_r) > min_size and len(lats2_r) > min_size:
        reductions.append((len(lats1_r), len(lats2_r)))
        lats1_r, lons1_r = _reduce_res(lats1_r, lons1_r)
        lats2_r, lons2_r = _reduce_res(lats2_r, lons2_r)

    distance, path = _dtw_simple(lats1_r, lons1_r, lats2_r, lons2_r)

    for (orig_n, orig_m) in reversed(reductions):
        if not path:
            return dtw_distance(trip1, trip2)
        window = _expand_path(path, orig_n, orig_m, radius)
        cur1_lats = trip1.lats[:orig_n]
        cur1_lons = trip1.lons[:orig_n]
        cur2_lats = trip2.lats[:orig_m]
        cur2_lons = trip2.lons[:orig_m]
        if len(window) < orig_n * orig_m * 0.3:
            try:
                dist_matrix = {}
                cf = {}
                dist_matrix[(0, 0)] = 0.0
                for (i, j) in sorted(set(window), key=lambda x: (x[0] + x[1], x[0])):
                    if i == 0 and j == 0:
                        dist_matrix[(i, j)] = haversine_pair(cur1_lats[0], cur1_lons[0], cur2_lats[0], cur2_lons[0])
                        cf[(i, j)] = None
                        continue
                    opts = []
                    for di, dj in [(-1, 0), (0, -1), (-1, -1)]:
                        ni, nj = i + di, j + dj
                        if (ni, nj) in dist_matrix:
                            opts.append((dist_matrix[(ni, nj)], (ni, nj)))
                    if opts:
                        mc, prev = min(opts, key=lambda x: x[0])
                        d = haversine_pair(cur1_lats[i], cur1_lons[i], cur2_lats[j], cur2_lons[j])
                        dist_matrix[(i, j)] = mc + d
                        cf[(i, j)] = prev
                end_key = (orig_n - 1, orig_m - 1)
                if end_key in dist_matrix and dist_matrix[end_key] < np.inf:
                    new_path = []
                    cur = end_key
                    while cur is not None:
                        new_path.append(cur)
                        cur = cf.get(cur)
                    new_path.reverse()
                    distance = dist_matrix[end_key]
                    path = new_path
                    continue
            except Exception:
                pass
        distance, path = _dtw_simple(cur1_lats, cur1_lons, cur2_lats, cur2_lons)

    return distance


def _fastdtw_core(lats1, lons1, lats2, lons2, window):
    return None, None


def frechet_distance(trip1: Trip, trip2: Trip) -> float:
    n, m = len(trip1.lats), len(trip2.lats)
    if n == 0 or m == 0:
        return float('inf')

    lats1, lons1 = trip1.lats, trip1.lons
    lats2, lons2 = trip2.lats, trip2.lons

    dist = np.full((n, m), -1.0)

    def c(i, j):
        if dist[i, j] >= 0:
            return dist[i, j]
        d = haversine_pair(lats1[i], lons1[i], lats2[j], lons2[j])
        if i == 0 and j == 0:
            res = d
        elif i > 0 and j == 0:
            res = max(c(i - 1, 0), d)
        elif i == 0 and j > 0:
            res = max(c(0, j - 1), d)
        else:
            res = max(min(c(i - 1, j), c(i, j - 1), c(i - 1, j - 1)), d)
        dist[i, j] = res
        return res

    return float(c(n - 1, m - 1))


def lcss_similarity(trip1: Trip, trip2: Trip, epsilon_m: float = 100.0,
                    delta_min: Optional[float] = None) -> float:
    n, m = len(trip1.lats), len(trip2.lats)
    if n == 0 or m == 0:
        return 0.0

    lats1, lons1 = trip1.lats, trip1.lons
    lats2, lons2 = trip2.lats, trip2.lons
    times1 = trip1.timestamps.astype('datetime64[ns]')
    times2 = trip2.timestamps.astype('datetime64[ns]')

    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = haversine_pair(lats1[i - 1], lons1[i - 1], lats2[j - 1], lons2[j - 1])
            time_ok = True
            if delta_min is not None:
                time_diff = abs((times1[i - 1] - times2[j - 1]).astype('timedelta64[m]').astype(float))
                time_ok = time_diff <= delta_min

            if d <= epsilon_m and time_ok:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

    lcss_len = int(dp[n, m])
    return lcss_len / min(n, m) if min(n, m) > 0 else 0.0


def edr_distance(trip1: Trip, trip2: Trip, epsilon_m: float = 100.0) -> float:
    n, m = len(trip1.lats), len(trip2.lats)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    lats1, lons1 = trip1.lats, trip1.lons
    lats2, lons2 = trip2.lats, trip2.lons

    dp = np.zeros((n + 1, m + 1), dtype=float)
    dp[0, :] = np.arange(m + 1)
    dp[:, 0] = np.arange(n + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = haversine_pair(lats1[i - 1], lons1[i - 1], lats2[j - 1], lons2[j - 1])
            sub_cost = 0.0 if d <= epsilon_m else 1.0
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + sub_cost
            )

    return float(dp[n, m])


def compute_avg_point_spacing(trips: List[Trip]) -> float:
    all_dists = []
    for t in trips:
        for i in range(1, len(t.lats)):
            d = haversine_pair(t.lats[i - 1], t.lons[i - 1], t.lats[i], t.lons[i])
            all_dists.append(d)
    if not all_dists:
        return 100.0
    return float(np.mean(all_dists))


def compute_distance_matrix(
    trips: List[Trip],
    method: str = 'dtw',
    use_fastdtw: bool = False,
    fastdtw_radius: int = 1,
    epsilon_m: Optional[float] = None,
    delta_min: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> np.ndarray:
    n = len(trips)
    matrix = np.zeros((n, n), dtype=float)

    if n > 500 and method == 'dtw' and not use_fastdtw:
        use_fastdtw = True

    if epsilon_m is None and method in ['lcss', 'edr']:
        epsilon_m = compute_avg_point_spacing(trips) * 3.0

    total = (n * (n - 1)) // 2
    done = 0

    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = trips[i], trips[j]
            if method == 'dtw':
                if use_fastdtw:
                    d = fastdtw_distance(t1, t2, radius=fastdtw_radius)
                else:
                    d = dtw_distance(t1, t2)
            elif method == 'frechet':
                d = frechet_distance(t1, t2)
            elif method == 'lcss':
                sim = lcss_similarity(t1, t2, epsilon_m=epsilon_m, delta_min=delta_min)
                d = 1.0 - sim
            elif method == 'edr':
                d = edr_distance(t1, t2, epsilon_m=epsilon_m)
            else:
                raise ValueError(f"未知方法: {method}")

            matrix[i, j] = d
            matrix[j, i] = d
            done += 1
            if progress_callback and (done % max(1, total // 100) == 0):
                progress_callback(done, total)

    return matrix


def distance_to_similarity(matrix: np.ndarray, gamma: float = None) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / (np.median(matrix[matrix > 0]) + 1e-10)
    return np.exp(-gamma * matrix)
