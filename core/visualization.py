import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
from .preprocessing import Trip


CLUSTER_COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
    '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
]

NOISE_COLOR = '#888888'
ANOMALY_COLOR = '#E74C3C'


def get_cluster_color(cluster_id: int) -> str:
    if cluster_id == -1:
        return NOISE_COLOR
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def create_empty_map(bbox: Tuple[float, float, float, float] = None,
                     basemap: str = 'carto-positron') -> go.Figure:
    fig = go.Figure()
    if bbox is None:
        lon_min, lat_min, lon_max, lat_max = 121.35, 31.15, 121.60, 31.32
    else:
        lon_min, lat_min, lon_max, lat_max = bbox

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    lat_range = max(lat_max - lat_min, 0.01)
    lon_range = max(lon_max - lon_min, 0.01)
    zoom = 11 - max(0, int(np.log2(max(lat_range, lon_range) * 60)))

    fig.update_layout(
        mapbox=dict(
            style=basemap,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=max(9, min(15, zoom)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)"
        ),
        dragmode='select',
        newshape=dict(line=dict(color='#667eea', width=2)),
    )
    return fig


def add_trajectories_to_map(fig: go.Figure, trips: List[Trip],
                            labels: Optional[np.ndarray] = None,
                            anomaly_indices: Optional[set] = None,
                            selected_indices: Optional[List[int]] = None,
                            max_render: int = 2000) -> go.Figure:
    if not trips:
        return fig

    n_trips = len(trips)
    if n_trips > max_render:
        step = n_trips // max_render
        render_idx = list(range(0, n_trips, step))
    else:
        render_idx = list(range(n_trips))

    anomaly_set = anomaly_indices if anomaly_indices else set()

    for idx in render_idx:
        trip = trips[idx]
        if labels is not None and idx < len(labels):
            color = get_cluster_color(int(labels[idx]))
        else:
            color = '#2563EB'

        if idx in anomaly_set:
            color = ANOMALY_COLOR

        width = 2 if idx in anomaly_set else 1.5
        opacity = 0.85 if idx in anomaly_set else 0.6

        show_legend = False
        name = f"轨迹 {trip.trip_id}"
        if labels is not None and idx < len(labels):
            cid = int(labels[idx])
            name = f"簇 {cid}" if cid != -1 else "噪声"
            first_in_cluster = True
            if selected_indices is not None and idx < len(render_idx):
                for prev_i in render_idx[:render_idx.index(idx)]:
                    if prev_i < len(labels) and int(labels[prev_i]) == cid:
                        first_in_cluster = False
                        break
            show_legend = first_in_cluster

        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=trip.lons.tolist(),
            lat=trip.lats.tolist(),
            line=dict(width=width, color=color),
            opacity=opacity,
            name=name,
            showlegend=show_legend,
            hovertemplate=f"<b>{trip.trip_id}</b><br>" +
                          f"用户: {trip.user_id}<br>" +
                          f"距离: {trip.distance_m/1000:.2f} km<br>" +
                          f"时长: {trip.duration_s/60:.1f} min<br>" +
                          f"<extra></extra>",
            customdata=[trip.trip_id] * len(trip.lats),
        ))

    return fig


def add_stay_points_to_map(fig: go.Figure, stays: List[Dict]) -> go.Figure:
    if not stays:
        return fig

    real_stays = [s for s in stays if not s.get('is_transit_stop', False)]
    if not real_stays:
        return fig

    lons = [s['lon'] for s in real_stays]
    lats = [s['lat'] for s in real_stays]
    durations = np.array([s['duration_min'] for s in real_stays])
    sizes = np.log2(durations + 1) * 4 + 4

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=lons,
        lat=lats,
        marker=dict(
            size=sizes.tolist(),
            color='#FF6B35',
            opacity=0.7,
        ),
        name="停留点",
        showlegend=True,
        hovertemplate="<b>停留点</b><br>" +
                      "时段: %{customdata[0]} ~ %{customdata[1]}<br>" +
                      "时长: %{customdata[2]:.1f} 分钟<br>" +
                      "GPS点数: %{customdata[3]}<br>" +
                      "<extra></extra>",
        customdata=[[
            s['start_time'], s['end_time'], s['duration_min'], s['point_count']
        ] for s in real_stays]
    ))
    return fig


def add_heatmap_to_map(fig: go.Figure, heatmap_data: np.ndarray,
                       bbox: Tuple[float, float, float, float]) -> go.Figure:
    if heatmap_data is None or heatmap_data.size == 0:
        return fig

    lon_min, lat_min, lon_max, lat_max = bbox
    resolution = heatmap_data.shape[0]

    lats = []
    lons = []
    intensities = []

    for i in range(resolution):
        for j in range(resolution):
            val = int(heatmap_data[i, j])
            if val > 0:
                lat = lat_min + (resolution - 1 - i) / (resolution - 1) * (lat_max - lat_min)
                lon = lon_min + j / (resolution - 1) * (lon_max - lon_min)
                lats.append(lat)
                lons.append(lon)
                intensities.append(val)

    if not intensities:
        return fig

    fig.add_trace(go.Densitymapbox(
        lat=lats,
        lon=lons,
        z=intensities,
        radius=15,
        colorscale='Viridis',
        opacity=0.6,
        showscale=True,
        colorbar=dict(title="密度", thickness=15, len=0.6),
        name="热力图",
        showlegend=True,
        hovertemplate="GPS点数: %{z}<br><extra></extra>",
    ))
    return fig


def add_od_flows_to_map(fig: go.Figure, od_pairs: List[Dict], top_n: int = 20) -> go.Figure:
    if not od_pairs:
        return fig

    sorted_pairs = sorted(od_pairs, key=lambda x: x['count'], reverse=True)[:top_n]
    if not sorted_pairs:
        return fig

    max_count = max(p['count'] for p in sorted_pairs)

    lons = []
    lats = []
    for p in sorted_pairs:
        o_lat, o_lon = p['origin_centroid']
        d_lat, d_lon = p['dest_centroid']
        mid_lat = (o_lat + d_lat) / 2
        mid_lon = (o_lon + d_lon) / 2
        curve_offset = 0.008 if (o_lat, o_lon) != (d_lat, d_lon) else 0
        curve_lat = mid_lat + curve_offset
        curve_lon = mid_lon + curve_offset

        lons.extend([o_lon, curve_lon, d_lon, None])
        lats.extend([o_lat, curve_lat, d_lat, None])

    widths = []
    for p in sorted_pairs:
        w = 1 + (p['count'] / max_count) * 6
        widths.extend([w, w, w, None])

    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=lons,
        lat=lats,
        line=dict(
            color='#9B59B6',
            width=2,
        ),
        opacity=0.7,
        name=f"OD流向 (Top{top_n})",
        showlegend=True,
        hoverinfo='skip',
    ))

    origin_lons = [p['origin_centroid'][1] for p in sorted_pairs]
    origin_lats = [p['origin_centroid'][0] for p in sorted_pairs]
    dest_lons = [p['dest_centroid'][1] for p in sorted_pairs]
    dest_lats = [p['dest_centroid'][0] for p in sorted_pairs]
    counts = [p['count'] for p in sorted_pairs]
    marker_sizes = [6 + (c / max_count) * 14 for c in counts]

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=origin_lons + dest_lons,
        lat=origin_lats + dest_lats,
        marker=dict(
            size=marker_sizes + marker_sizes,
            color=['#3498DB'] * len(sorted_pairs) + ['#E74C3C'] * len(sorted_pairs),
            opacity=0.8,
        ),
        name="起终点",
        showlegend=False,
        hovertemplate=["起: %{customdata}" for _ in sorted_pairs] + ["终: %{customdata}" for _ in sorted_pairs],
        customdata=counts + counts,
    ))

    return fig


def add_hotspots_to_map(fig: go.Figure, hotspots: List[Dict], heatmap_mode: bool = False) -> go.Figure:
    if not hotspots:
        return fig

    for i, h in enumerate(hotspots):
        color = f'rgba(231, 76, 60, 0.25)'

        if h.get('convex_hull') and not heatmap_mode:
            hull = h['convex_hull']
            lons = [p[0] for p in hull]
            lats = [p[1] for p in hull]
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=lons,
                lat=lats,
                fill="toself",
                fillcolor=color,
                line=dict(color='#E74C3C', width=2),
                name=f"热点 {i+1}",
                showlegend=(i == 0),
                legendgroup="hotspots",
                hovertemplate=f"<b>热点 {i+1}</b><br>" +
                              f"中心: ({h['center_lat']:.4f}, {h['center_lon']:.4f})<br>" +
                              f"点数: {h['point_count']}<br>" +
                              f"用户: {h['unique_users']}<br>" +
                              f"高峰时段: {h['peak_hour']}:00<br>" +
                              f"<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scattermapbox(
                mode="markers",
                lon=[h['center_lon']],
                lat=[h['center_lat']],
                marker=dict(
                    size=min(30, 10 + np.log2(h['point_count'] + 1) * 3),
                    color='#E74C3C',
                    opacity=0.5,
                ),
                name=f"热点 {i+1}",
                showlegend=(i == 0),
                legendgroup="hotspots",
            ))

    return fig


def add_animation_frame(fig: go.Figure, trip: Trip, frame: int) -> go.Figure:
    if len(trip.lats) == 0:
        return fig
    idx = min(frame, len(trip.lats) - 1)

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[trip.lons[idx]],
        lat=[trip.lats[idx]],
        marker=dict(size=14, color='#F39C12', symbol='circle', allowoverlap=True),
        name="当前位置",
        showlegend=True,
        hovertemplate=f"<b>动画</b><br>时间: {trip.timestamps[idx]}<br><extra></extra>",
    ))
    return fig


def create_hour_dist_fig(hour_dist: Dict[int, Dict[int, int]]) -> go.Figure:
    hours = list(range(24))

    fig = go.Figure()
    if not hour_dist:
        fig.add_bar(x=hours, y=[0] * 24, marker_color='#667eea')
    else:
        all_cluster_ids = sorted(hour_dist.keys())
        for cid in all_cluster_ids:
            data = hour_dist.get(cid, {})
            y = [data.get(h, 0) for h in hours]
            if cid == 0 and len(all_cluster_ids) == 1:
                fig.add_bar(x=hours, y=y, name="全部", marker_color='#667eea')
            else:
                fig.add_bar(x=hours, y=y, name=f"簇 {cid}" if cid != -1 else "噪声",
                            marker_color=get_cluster_color(cid))

    fig.update_layout(
        barmode='stack',
        margin=dict(l=40, r=10, t=10, b=30),
        xaxis=dict(title="小时", tickmode='array', tickvals=hours, ticktext=[f"{h:02d}" for h in hours]),
        yaxis=dict(title="出行次数"),
        showlegend=(len(hour_dist) > 1),
        legend=dict(font=dict(size=9), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=180,
    )
    return fig


def create_weekday_dist_fig(weekday_dist: Dict[int, Dict[int, int]]) -> go.Figure:
    days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    fig = go.Figure()
    if not weekday_dist:
        fig.add_bar(x=days, y=[0] * 7, marker_color='#00CC96')
    else:
        all_cluster_ids = sorted(weekday_dist.keys())
        for cid in all_cluster_ids:
            data = weekday_dist.get(cid, {})
            y = [data.get(i, 0) for i in range(7)]
            if cid == 0 and len(all_cluster_ids) == 1:
                fig.add_bar(x=days, y=y, name="全部", marker_color='#00CC96')
            else:
                fig.add_bar(x=days, y=y, name=f"簇 {cid}" if cid != -1 else "噪声",
                            marker_color=get_cluster_color(cid))

    fig.update_layout(
        barmode='stack',
        margin=dict(l=40, r=10, t=10, b=30),
        yaxis=dict(title="出行次数"),
        showlegend=(len(weekday_dist) > 1),
        legend=dict(font=dict(size=9), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=160,
    )
    return fig


def create_distance_dist_fig(distances_km: List[float]) -> go.Figure:
    fig = go.Figure()
    if not distances_km:
        return fig
    fig.add_histogram(
        x=distances_km,
        nbinsx=25,
        marker_color='#AB63FA',
        opacity=0.8,
        name="距离分布"
    )
    fig.update_layout(
        margin=dict(l=40, r=10, t=10, b=30),
        xaxis=dict(title="出行距离 (km)"),
        yaxis=dict(title="频数"),
        showlegend=False,
        height=160,
    )
    return fig


def create_duration_dist_fig(durations_min: List[float]) -> go.Figure:
    fig = go.Figure()
    if not durations_min:
        return fig
    fig.add_histogram(
        x=durations_min,
        nbinsx=25,
        marker_color='#19D3F3',
        opacity=0.8,
        name="时长分布"
    )
    fig.update_layout(
        margin=dict(l=40, r=10, t=10, b=30),
        xaxis=dict(title="出行时长 (min)"),
        yaxis=dict(title="频数"),
        showlegend=False,
        height=160,
    )
    return fig


def create_cluster_pie_fig(labels: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if labels is None or len(labels) == 0:
        return fig

    unique, counts = np.unique(labels, return_counts=True)
    label_names = [f"簇 {l}" if l != -1 else "噪声" for l in unique]
    colors = [get_cluster_color(int(l)) for l in unique]

    fig.add_pie(
        labels=label_names,
        values=counts,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=10),
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(font=dict(size=9), orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5),
        height=200,
    )
    return fig


def create_mode_pie_fig(modes: List[str]) -> go.Figure:
    fig = go.Figure()
    valid_modes = [m for m in modes if m and str(m) != 'nan']
    if not valid_modes:
        return fig
    from collections import Counter
    counter = Counter(valid_modes)
    fig.add_pie(
        labels=list(counter.keys()),
        values=list(counter.values()),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=10),
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(font=dict(size=9)),
        height=180,
    )
    return fig


def create_distance_matrix_heatmap(matrix: np.ndarray) -> go.Figure:
    if matrix is None or matrix.size == 0:
        return go.Figure()

    n = min(100, matrix.shape[0])
    display = matrix[:n, :n]

    fig = go.Figure(data=go.Heatmap(
        z=display,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="距离", thickness=12, len=0.85),
    ))
    fig.update_layout(
        margin=dict(l=30, r=10, t=10, b=30),
        xaxis=dict(title="轨迹ID"),
        yaxis=dict(title="轨迹ID"),
        height=300,
    )
    return fig


def create_k_distance_plot(k_dists: np.ndarray, eps_value: float = None) -> go.Figure:
    fig = go.Figure()
    if k_dists is None or len(k_dists) == 0:
        return fig

    x = list(range(len(k_dists)))
    fig.add_scatter(x=x, y=k_dists, mode='lines', line=dict(color='#667eea'))

    if eps_value is not None:
        fig.add_hline(y=eps_value, line_dash="dash", line_color="red",
                      annotation_text=f"推荐 eps={eps_value:.1f}m",
                      annotation_position="bottom right")

    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=30),
        title="k-距离图 (排序)",
        xaxis_title="轨迹 (按k-距离降序)",
        yaxis_title="第k近邻距离 (m)",
        height=220,
    )
    return fig
