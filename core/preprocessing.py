import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from .utils import haversine_vectorized, consecutive_distances, compute_bbox, smooth_trajectory

REQUIRED_COLUMNS = ['user_id', 'timestamp', 'longitude', 'latitude']
OPTIONAL_COLUMNS = ['speed', 'heading', 'mode']

MAX_SPEED_KMH = 200.0
MAX_DISTANCE_M = 5000.0
MIN_TIME_INTERVAL_S = 10.0
TRIP_SPLIT_MINUTES = 30.0


@dataclass
class DatasetInfo:
    total_users: int
    total_trips: int
    total_points: int
    time_span_start: str
    time_span_end: str
    bbox: Tuple[float, float, float, float]
    columns: List[str]
    has_mode: bool


@dataclass
class Trip:
    trip_id: str
    user_id: str
    points_df: pd.DataFrame
    start_time: np.datetime64
    end_time: np.datetime64
    distance_m: float
    duration_s: float
    avg_speed_kmh: float

    @property
    def lats(self) -> np.ndarray:
        return self.points_df['latitude'].values

    @property
    def lons(self) -> np.ndarray:
        return self.points_df['longitude'].values

    @property
    def timestamps(self) -> np.ndarray:
        return self.points_df['timestamp'].values

    @property
    def origin(self) -> Tuple[float, float]:
        return (self.lats[0], self.lons[0])

    @property
    def destination(self) -> Tuple[float, float]:
        return (self.lats[-1], self.lons[-1])


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    col_mapping = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    aliases = {
        'user_id': ['user_id', 'userid', 'uid', 'id', 'vehicle_id', 'driver_id'],
        'timestamp': ['timestamp', 'time', 'datetime', 'date_time', 'ts', 't'],
        'longitude': ['longitude', 'lon', 'lng', 'long', 'x'],
        'latitude': ['latitude', 'lat', 'y'],
        'speed': ['speed', 'velocity', 'v', 'spd'],
        'heading': ['heading', 'direction', 'bearing', 'angle', 'dir'],
        'mode': ['mode', 'transport_mode', 'transportation', 'traj_mode', 'type'],
    }

    for std_name, alias_list in aliases.items():
        for alias in alias_list:
            if alias in lower_cols:
                col_mapping[std_name] = lower_cols[alias]
                break

    return col_mapping


def validate_columns(col_mapping: Dict[str, str]) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in col_mapping]
    return len(missing) == 0, missing


def parse_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce', utc=True)


def compute_speed_from_positions(lats: np.ndarray, lons: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(lats) < 2:
        return np.zeros(len(lats))
    dists = consecutive_distances(lats, lons)
    time_diffs = np.diff(times.astype('datetime64[ns]')).astype('timedelta64[s]').astype(float)
    time_diffs = np.maximum(time_diffs, 1.0)
    speeds = (dists / 1000.0) / (time_diffs / 3600.0)
    speeds = np.concatenate([[speeds[0]], speeds])
    return speeds


def filter_noise(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    stats = {
        'total_points': len(df),
        'removed_high_speed': 0,
        'removed_long_jump': 0,
        'kept_points': 0,
    }

    if len(df) == 0:
        return df, stats

    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    if 'speed' not in df.columns or df['speed'].isna().all():
        all_speeds = np.zeros(len(df))
        for uid in df['user_id'].unique():
            mask = df['user_id'] == uid
            idx = np.where(mask)[0]
            if len(idx) >= 2:
                all_speeds[idx] = compute_speed_from_positions(
                    df.loc[idx, 'latitude'].values,
                    df.loc[idx, 'longitude'].values,
                    df.loc[idx, 'timestamp'].values
                )
        df['speed'] = all_speeds
    else:
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce').fillna(0.0)

    high_speed_mask = df['speed'] > MAX_SPEED_KMH

    lats = df['latitude'].values
    lons = df['longitude'].values
    times = df['timestamp'].values.astype('datetime64[ns]')
    user_ids = df['user_id'].values

    long_jump_mask = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if user_ids[i] == user_ids[i - 1]:
            dist = haversine_vectorized(
                np.array([lats[i]]), np.array([lons[i]]),
                np.array([lats[i - 1]]), np.array([lons[i - 1]])
            )[0]
            time_diff = (times[i] - times[i - 1]).astype('timedelta64[s]').astype(float)
            if dist > MAX_DISTANCE_M and time_diff < MIN_TIME_INTERVAL_S:
                long_jump_mask[i] = True

    remove_mask = high_speed_mask | long_jump_mask
    stats['removed_high_speed'] = int(high_speed_mask.sum())
    stats['removed_long_jump'] = int(long_jump_mask.sum())

    df_clean = df[~remove_mask].reset_index(drop=True)
    stats['kept_points'] = len(df_clean)

    return df_clean, stats


def split_into_trips(df: pd.DataFrame, split_minutes: float = TRIP_SPLIT_MINUTES) -> List[Trip]:
    trips = []
    split_seconds = split_minutes * 60.0

    for user_id, group in df.groupby('user_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        if len(group) < 2:
            continue

        times = group['timestamp'].values.astype('datetime64[ns]')
        time_diffs = np.diff(times).astype('timedelta64[s]').astype(float)
        split_indices = np.where(time_diffs > split_seconds)[0] + 1
        split_indices = np.concatenate([[0], split_indices, [len(group)]])

        for i in range(len(split_indices) - 1):
            start, end = split_indices[i], split_indices[i + 1]
            if end - start < 2:
                continue

            trip_df = group.iloc[start:end].reset_index(drop=True)
            lats = trip_df['latitude'].values
            lons = trip_df['longitude'].values
            t_start = times[start]
            t_end = times[end - 1]
            duration_s = (t_end - t_start).astype('timedelta64[s]').astype(float)
            distance_m = float(np.sum(consecutive_distances(lats, lons)))
            avg_speed = (distance_m / 1000.0) / (duration_s / 3600.0) if duration_s > 0 else 0.0

            trip_id = f"{user_id}_{i}"
            trip = Trip(
                trip_id=trip_id,
                user_id=str(user_id),
                points_df=trip_df,
                start_time=t_start,
                end_time=t_end,
                distance_m=distance_m,
                duration_s=duration_s,
                avg_speed_kmh=avg_speed
            )
            trips.append(trip)

    return trips


def apply_smoothing(trips: List[Trip], window_size: int = 3) -> List[Trip]:
    if window_size < 2:
        return trips

    smoothed_trips = []
    for trip in trips:
        lats_smooth, lons_smooth = smooth_trajectory(trip.lats, trip.lons, window_size)
        new_df = trip.points_df.copy()
        new_df['latitude'] = lats_smooth
        new_df['longitude'] = lons_smooth

        lats = new_df['latitude'].values
        lons = new_df['longitude'].values
        distance_m = float(np.sum(consecutive_distances(lats, lons)))
        duration_s = trip.duration_s
        avg_speed = (distance_m / 1000.0) / (duration_s / 3600.0) if duration_s > 0 else 0.0

        smoothed_trips.append(Trip(
            trip_id=trip.trip_id,
            user_id=trip.user_id,
            points_df=new_df,
            start_time=trip.start_time,
            end_time=trip.end_time,
            distance_m=distance_m,
            duration_s=duration_s,
            avg_speed_kmh=avg_speed
        ))
    return smoothed_trips


def compute_dataset_info(df: pd.DataFrame, trips: List[Trip]) -> DatasetInfo:
    bbox = compute_bbox(df['latitude'].values, df['longitude'].values)
    return DatasetInfo(
        total_users=df['user_id'].nunique(),
        total_trips=len(trips),
        total_points=len(df),
        time_span_start=str(df['timestamp'].min()),
        time_span_end=str(df['timestamp'].max()),
        bbox=bbox,
        columns=list(df.columns),
        has_mode='mode' in df.columns
    )


def import_csv(filepath: str, smooth: bool = False, smooth_window: int = 3,
               split_minutes: float = TRIP_SPLIT_MINUTES) -> Tuple[List[Trip], DatasetInfo, Dict]:
    raw_df = pd.read_csv(filepath)
    col_mapping = detect_columns(raw_df)
    ok, missing = validate_columns(col_mapping)

    if not ok:
        raise ValueError(f"缺少必需列: {missing}")

    rename_map = {v: k for k, v in col_mapping.items()}
    df = raw_df.rename(columns=rename_map)

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_COLUMNS + OPTIONAL_COLUMNS].copy()
    df['timestamp'] = parse_timestamps(df['timestamp'])
    df = df.dropna(subset=['timestamp', 'latitude', 'longitude']).reset_index(drop=True)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['user_id'] = df['user_id'].astype(str)
    df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

    df_clean, noise_stats = filter_noise(df)
    trips = split_into_trips(df_clean, split_minutes)

    if smooth:
        trips = apply_smoothing(trips, smooth_window)

    info = compute_dataset_info(df_clean, trips)
    return trips, info, noise_stats
