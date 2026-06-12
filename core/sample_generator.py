import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta


CITY_CENTER_LAT = 31.2304
CITY_CENTER_LON = 121.4737


def generate_workday_route(user_idx: int, day: datetime, commute_type: str = 'normal') -> pd.DataFrame:
    np.random.seed(user_idx * 1000 + day.day)

    home_lat = CITY_CENTER_LAT + np.random.uniform(-0.08, 0.08)
    home_lon = CITY_CENTER_LON + np.random.uniform(-0.08, 0.08)
    work_lat = CITY_CENTER_LAT + np.random.uniform(-0.03, 0.03)
    work_lon = CITY_CENTER_LON + np.random.uniform(-0.03, 0.03)

    all_rows = []
    modes = ['car', 'bus', 'subway', 'bike']
    mode = modes[user_idx % 4]

    def sample_points(start_lat, start_lon, end_lat, end_lon, start_time, duration_min, noise=0.0005):
        n_points = max(10, int(duration_min * 2))
        t = np.linspace(0, 1, n_points)
        mid_lat = (start_lat + end_lat) / 2 + np.random.uniform(-noise * 3, noise * 3)
        mid_lon = (start_lon + end_lon) / 2 + np.random.uniform(-noise * 3, noise * 3)

        lats = (1 - t) ** 2 * start_lat + 2 * (1 - t) * t * mid_lat + t ** 2 * end_lat
        lons = (1 - t) ** 2 * start_lon + 2 * (1 - t) * t * mid_lon + t ** 2 * end_lon
        lats += np.random.normal(0, noise, n_points)
        lons += np.random.normal(0, noise, n_points)

        for i in range(n_points):
            speed_kmh = np.random.uniform(15, 50) if mode == 'car' else np.random.uniform(10, 30)
            heading = np.random.uniform(0, 360)
            ts = start_time + timedelta(minutes=(duration_min * i / n_points))
            all_rows.append({
                'user_id': f'user_{user_idx:03d}',
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'longitude': round(lons[i], 6),
                'latitude': round(lats[i], 6),
                'speed': round(speed_kmh, 2),
                'heading': round(heading, 1),
                'mode': mode
            })

    morning_start = day.replace(hour=7 + (user_idx % 2), minute=(user_idx * 13) % 60)
    morning_dur = 25 + np.random.randint(-5, 15)
    sample_points(home_lat, home_lon, work_lat, work_lon, morning_start, morning_dur)

    for _ in range(2):
        if np.random.random() < 0.7:
            err_lat = work_lat + np.random.uniform(-0.005, 0.005)
            err_lon = work_lon + np.random.uniform(-0.005, 0.005)
            lunch_start = day.replace(hour=12, minute=np.random.randint(0, 30))
            lunch_dur = 10 + np.random.randint(-3, 8)
            sample_points(work_lat, work_lon, err_lat, err_lon, lunch_start, lunch_dur)
            back_start = lunch_start + timedelta(minutes=lunch_dur + 60)
            sample_points(err_lat, err_lon, work_lat, work_lon, back_start, lunch_dur)

    evening_start = day.replace(hour=17 + (user_idx % 3), minute=(user_idx * 17) % 60)
    evening_dur = 30 + np.random.randint(-5, 20)

    if np.random.random() < 0.6:
        shop_lat = (work_lat + home_lat) / 2 + np.random.uniform(-0.02, 0.02)
        shop_lon = (work_lon + home_lon) / 2 + np.random.uniform(-0.02, 0.02)
        sample_points(work_lat, work_lon, shop_lat, shop_lon, evening_start, evening_dur // 2)
        shop_end = evening_start + timedelta(minutes=evening_dur // 2 + 45)
        sample_points(shop_lat, shop_lon, home_lat, home_lon, shop_end, evening_dur // 2)
    else:
        sample_points(work_lat, work_lon, home_lat, home_lon, evening_start, evening_dur)

    return pd.DataFrame(all_rows)


def generate_weekend_route(user_idx: int, day: datetime) -> pd.DataFrame:
    np.random.seed(user_idx * 2000 + day.day * 7)

    home_lat = CITY_CENTER_LAT + np.random.uniform(-0.08, 0.08)
    home_lon = CITY_CENTER_LON + np.random.uniform(-0.08, 0.08)

    all_rows = []
    modes = ['car', 'walk', 'bike', 'subway']
    mode = modes[user_idx % 4]

    def sample_points(start_lat, start_lon, end_lat, end_lon, start_time, duration_min, noise=0.0006):
        n_points = max(8, int(duration_min * 1.5))
        t = np.linspace(0, 1, n_points)
        lats = start_lat + t * (end_lat - start_lat) + np.random.normal(0, noise, n_points)
        lons = start_lon + t * (end_lon - start_lon) + np.random.normal(0, noise, n_points)

        for i in range(n_points):
            speed_kmh = np.random.uniform(5, 40) if mode in ['car', 'bike'] else np.random.uniform(3, 10)
            ts = start_time + timedelta(minutes=(duration_min * i / max(n_points - 1, 1)))
            all_rows.append({
                'user_id': f'user_{user_idx:03d}',
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'longitude': round(lons[i], 6),
                'latitude': round(lats[i], 6),
                'speed': round(speed_kmh, 2),
                'heading': round(np.random.uniform(0, 360), 1),
                'mode': mode
            })

    if np.random.random() < 0.7:
        mall_lat = CITY_CENTER_LAT + np.random.uniform(-0.05, 0.05)
        mall_lon = CITY_CENTER_LON + np.random.uniform(-0.05, 0.05)
        morning_start = day.replace(hour=10, minute=np.random.randint(0, 60))
        dur = 35 + np.random.randint(0, 25)
        sample_points(home_lat, home_lon, mall_lat, mall_lon, morning_start, dur)

        back_start = morning_start + timedelta(minutes=dur + 180 + np.random.randint(0, 60))
        sample_points(mall_lat, mall_lon, home_lat, home_lon, back_start, dur)

    if np.random.random() < 0.5:
        park_lat = CITY_CENTER_LAT + np.random.uniform(-0.06, 0.06)
        park_lon = CITY_CENTER_LON + np.random.uniform(-0.06, 0.06)
        evening_start = day.replace(hour=18, minute=np.random.randint(0, 60))
        dur = 25 + np.random.randint(0, 20)
        sample_points(home_lat, home_lon, park_lat, park_lon, evening_start, dur)
        back_start = evening_start + timedelta(minutes=dur + 90 + np.random.randint(0, 30))
        sample_points(park_lat, park_lon, home_lat, home_lon, back_start, dur)

    return pd.DataFrame(all_rows)


def generate_random_route(user_idx: int, day: datetime) -> pd.DataFrame:
    np.random.seed(user_idx * 5000 + day.dayofyear)

    all_rows = []
    start_lat = CITY_CENTER_LAT + np.random.uniform(-0.1, 0.1)
    start_lon = CITY_CENTER_LON + np.random.uniform(-0.1, 0.1)

    hour = np.random.randint(6, 23)
    start_time = day.replace(hour=hour, minute=np.random.randint(0, 60))

    n_segments = np.random.randint(2, 6)
    modes = ['car', 'bus', 'bike', 'walk']
    mode = modes[np.random.randint(0, 4)]

    cur_lat, cur_lon = start_lat, start_lon

    for seg in range(n_segments):
        end_lat = CITY_CENTER_LAT + np.random.uniform(-0.1, 0.1)
        end_lon = CITY_CENTER_LON + np.random.uniform(-0.1, 0.1)
        dur = 15 + np.random.randint(5, 45)
        n_points = max(8, int(dur * 2))
        t = np.linspace(0, 1, n_points)

        lats = cur_lat + t * (end_lat - cur_lat) + np.random.normal(0, 0.0005, n_points)
        lons = cur_lon + t * (end_lon - cur_lon) + np.random.normal(0, 0.0005, n_points)

        for i in range(n_points):
            ts = start_time + timedelta(minutes=(dur * i / max(n_points - 1, 1)))
            speed = np.random.uniform(10, 45)
            all_rows.append({
                'user_id': f'user_{user_idx:03d}',
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'longitude': round(lons[i], 6),
                'latitude': round(lats[i], 6),
                'speed': round(speed, 2),
                'heading': round(np.random.uniform(0, 360), 1),
                'mode': mode
            })

        cur_lat, cur_lon = end_lat, end_lon
        start_time += timedelta(minutes=dur + np.random.randint(10, 60))

    return pd.DataFrame(all_rows)


def generate_sample_data(n_users: int = 30, n_days: int = 7) -> pd.DataFrame:
    base_date = datetime(2024, 6, 3)
    all_dfs = []

    for user_idx in range(n_users):
        for day_offset in range(n_days):
            day = base_date + timedelta(days=day_offset)
            is_weekend = day.weekday() >= 5

            if user_idx < 20:
                if is_weekend:
                    df = generate_weekend_route(user_idx, day)
                else:
                    df = generate_workday_route(user_idx, day)
            else:
                if np.random.random() < 0.3 or is_weekend:
                    df = generate_weekend_route(user_idx, day)
                elif np.random.random() < 0.6:
                    df = generate_workday_route(user_idx, day, 'irregular')
                else:
                    df = generate_random_route(user_idx, day)

            if len(df) > 0:
                all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    n_noise = int(len(full_df) * 0.02)
    if n_noise > 0:
        noise_idx = np.random.choice(len(full_df), n_noise, replace=False)
        for idx in noise_idx:
            if np.random.random() < 0.5:
                full_df.loc[idx, 'speed'] = np.random.uniform(250, 400)
            else:
                full_df.loc[idx, 'latitude'] = CITY_CENTER_LAT + np.random.uniform(-1, 1)
                full_df.loc[idx, 'longitude'] = CITY_CENTER_LON + np.random.uniform(-1, 1)

    return full_df.sample(frac=1, random_state=42).reset_index(drop=True)
