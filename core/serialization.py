import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from .preprocessing import Trip
from .stay_points import StayPoint


def trips_to_serializable(trips: List[Trip]) -> List[Dict]:
    if not trips:
        return []
    return [{
        'trip_id': t.trip_id,
        'user_id': t.user_id,
        'lats': t.lats.tolist(),
        'lons': t.lons.tolist(),
        'timestamps': [str(ts) for ts in t.timestamps],
        'start_time': str(t.start_time),
        'end_time': str(t.end_time),
        'distance_m': t.distance_m,
        'duration_s': t.duration_s,
        'avg_speed_kmh': t.avg_speed_kmh,
        'mode': t.points_df['mode'].iloc[0] if 'mode' in t.points_df.columns and pd.notna(t.points_df['mode'].iloc[0]) else None,
        'point_count': len(t.lats),
    } for t in trips]


def serializable_to_trips(data: List[Dict]) -> List[Trip]:
    trips = []
    for d in data:
        df = pd.DataFrame({
            'latitude': d['lats'],
            'longitude': d['lons'],
            'timestamp': pd.to_datetime(d['timestamps'], utc=True),
            'speed': np.nan,
            'heading': np.nan,
            'mode': d.get('mode'),
            'user_id': d['user_id'],
        })
        trips.append(Trip(
            trip_id=d['trip_id'],
            user_id=d['user_id'],
            points_df=df,
            start_time=np.datetime64(d['start_time']),
            end_time=np.datetime64(d['end_time']),
            distance_m=d['distance_m'],
            duration_s=d['duration_s'],
            avg_speed_kmh=d['avg_speed_kmh']
        ))
    return trips


def stays_to_serializable(stays: List[StayPoint]) -> List[Dict]:
    return [{
        'stay_id': s.stay_id,
        'user_id': s.user_id,
        'trip_id': s.trip_id,
        'lat': s.lat,
        'lon': s.lon,
        'start_time': str(s.start_time),
        'end_time': str(s.end_time),
        'duration_min': s.duration_min,
        'point_count': s.point_count,
        'radius_m': s.radius_m,
        'is_transit_stop': s.is_transit_stop,
    } for s in stays]


def serializable_to_stays(data: List[Dict]) -> List[StayPoint]:
    return [StayPoint(
        stay_id=s['stay_id'],
        user_id=s['user_id'],
        trip_id=s['trip_id'],
        lat=s['lat'],
        lon=s['lon'],
        start_time=np.datetime64(s['start_time']),
        end_time=np.datetime64(s['end_time']),
        duration_min=s['duration_min'],
        point_count=s['point_count'],
        radius_m=s['radius_m'],
        is_transit_stop=s['is_transit_stop']
    ) for s in data]
