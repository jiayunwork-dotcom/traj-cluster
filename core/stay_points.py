import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from .utils import haversine_vectorized, centroid
from .preprocessing import Trip

MICRO_STOP_MINUTES = 2.0


@dataclass
class StayPoint:
    stay_id: str
    user_id: str
    trip_id: str
    lat: float
    lon: float
    start_time: np.datetime64
    end_time: np.datetime64
    duration_min: float
    point_count: int
    radius_m: float
    is_transit_stop: bool = False


@dataclass
class UserSequence:
    user_id: str
    events: List[Dict] = field(default_factory=list)


def detect_stay_points_for_trip(
    trip: Trip,
    dist_threshold_m: float = 200.0,
    time_threshold_min: float = 20.0,
    micro_stop_min: float = MICRO_STOP_MINUTES
) -> Tuple[List[StayPoint], List[Tuple[int, int]]]:
    lats = trip.lats
    lons = trip.lons
    times = trip.timestamps.astype('datetime64[ns]')
    n = len(lats)

    stay_points = []
    move_segments = []
    i = 0
    stay_counter = 0

    while i < n:
        j = i + 1
        found_stay = False

        while j < n:
            window_lats = lats[i:j + 1]
            window_lons = lons[i:j + 1]
            c_lat, c_lon = centroid(window_lats, window_lons)
            dists_to_centroid = haversine_vectorized(
                window_lats, window_lons,
                np.full_like(window_lats, c_lat),
                np.full_like(window_lons, c_lon)
            )
            max_dist = float(np.max(dists_to_centroid))

            if max_dist > dist_threshold_m:
                break

            time_span = (times[j] - times[i]).astype('timedelta64[m]').astype(float)
            if time_span >= time_threshold_min:
                point_count = j - i + 1
                c_lat, c_lon = centroid(lats[i:j + 1], lons[i:j + 1])
                duration_min = (times[j] - times[i]).astype('timedelta64[m]').astype(float)
                is_transit = duration_min < micro_stop_min

                sp = StayPoint(
                    stay_id=f"{trip.trip_id}_sp{stay_counter}",
                    user_id=trip.user_id,
                    trip_id=trip.trip_id,
                    lat=c_lat,
                    lon=c_lon,
                    start_time=times[i],
                    end_time=times[j],
                    duration_min=duration_min,
                    point_count=point_count,
                    radius_m=max_dist,
                    is_transit_stop=is_transit
                )
                stay_points.append(sp)
                stay_counter += 1

                if i > 0:
                    move_segments.append((move_segments[-1][1] if move_segments else 0, i))
                move_segments.append((-1, -1))

                i = j + 1
                found_stay = True
                break

            j += 1

        if not found_stay:
            i += 1

    if not stay_points and n >= 2:
        move_segments.append((0, n - 1))
    else:
        real_moves = []
        last_end = 0
        for sp in stay_points:
            pass
        all_indices = sorted(set([0, n - 1] + [
            idx for sp in stay_points
            for idx in [
                np.where(times == sp.start_time)[0][0] if np.where(times == sp.start_time)[0].size > 0 else 0,
                np.where(times == sp.end_time)[0][0] if np.where(times == sp.end_time)[0].size > 0 else n - 1
            ]
        ]))
        for k in range(len(all_indices) - 1):
            s, e = all_indices[k], all_indices[k + 1]
            if e > s:
                real_moves.append((s, e))
        move_segments = real_moves

    return stay_points, move_segments


def detect_all_stay_points(
    trips: List[Trip],
    dist_threshold_m: float = 200.0,
    time_threshold_min: float = 20.0
) -> Tuple[List[StayPoint], Dict[str, UserSequence]]:
    all_stays = []
    user_sequences: Dict[str, UserSequence] = {}

    for trip in trips:
        stays, moves = detect_stay_points_for_trip(trip, dist_threshold_m, time_threshold_min)
        all_stays.extend(stays)

        uid = trip.user_id
        if uid not in user_sequences:
            user_sequences[uid] = UserSequence(user_id=uid)

        seq = user_sequences[uid]
        for start_idx, end_idx in moves:
            if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
                seq.events.append({
                    'type': 'move',
                    'trip_id': trip.trip_id,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': trip.timestamps[start_idx],
                    'end_time': trip.timestamps[end_idx],
                    'start_lat': trip.lats[start_idx],
                    'start_lon': trip.lons[start_idx],
                    'end_lat': trip.lats[end_idx],
                    'end_lon': trip.lons[end_idx],
                })

        for sp in stays:
            if not sp.is_transit_stop:
                seq.events.append({
                    'type': 'stay',
                    'stay_id': sp.stay_id,
                    'lat': sp.lat,
                    'lon': sp.lon,
                    'start_time': sp.start_time,
                    'end_time': sp.end_time,
                    'duration_min': sp.duration_min,
                })

    for uid in user_sequences:
        user_sequences[uid].events.sort(key=lambda e: e['start_time'])

    return all_stays, user_sequences


def stays_to_dataframe(stays: List[StayPoint]) -> pd.DataFrame:
    if not stays:
        return pd.DataFrame(columns=[
            'stay_id', 'user_id', 'trip_id', 'latitude', 'longitude',
            'start_time', 'end_time', 'duration_min', 'point_count',
            'radius_m', 'is_transit_stop'
        ])

    data = [{
        'stay_id': s.stay_id,
        'user_id': s.user_id,
        'trip_id': s.trip_id,
        'latitude': s.lat,
        'longitude': s.lon,
        'start_time': str(s.start_time),
        'end_time': str(s.end_time),
        'duration_min': round(s.duration_min, 2),
        'point_count': s.point_count,
        'radius_m': round(s.radius_m, 2),
        'is_transit_stop': s.is_transit_stop,
    } for s in stays]
    return pd.DataFrame(data)
