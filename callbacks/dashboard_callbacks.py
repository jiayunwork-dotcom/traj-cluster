import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, no_update, callback, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from core.preprocessing import Trip
from core.serialization import serializable_to_trips, serializable_to_stays
from core.visualization import (
    create_empty_map, add_trajectories_to_map, add_stay_points_to_map,
    add_heatmap_to_map, add_od_flows_to_map, add_hotspots_to_map,
    add_animation_frame,
    create_hour_dist_fig, create_weekday_dist_fig, create_distance_dist_fig,
    create_duration_dist_fig, create_cluster_pie_fig, create_mode_pie_fig
)


def register_map_callbacks(app):
    @app.callback(
        Output("main-map", "figure"),
        Input("store-trips", "data"),
        Input("store-stays", "data"),
        Input("store-cluster-result", "data"),
        Input("store-patterns", "data"),
        Input("store-hotspots", "data"),
        Input("store-anomalies", "data"),
        Input("dropdown-basemap", "value"),
        Input("check-layers", "value"),
        Input("slider-top-od", "value"),
        Input("check-heatmap-mode", "value"),
        Input("store-animation-frame", "data"),
        Input("store-selected-trip-indices", "data"),
        State("main-map", "figure"),
        prevent_initial_call=False,
    )
    def update_map(trips_data, stays_data, cluster_data, patterns_data,
                   hotspot_data, anomalies_data, basemap, layers, top_od,
                   heatmap_mode, anim_frame, selected_idx, current_fig):
        if not trips_data:
            return create_empty_map(None, basemap or 'carto-positron')

        trips = serializable_to_trips(trips_data)
        if not trips:
            return create_empty_map(None, basemap or 'carto-positron')

        bbox = None
        all_lats = np.concatenate([t.lats for t in trips])
        all_lons = np.concatenate([t.lons for t in trips])
        if len(all_lats) > 0:
            bbox = (float(np.min(all_lons)), float(np.min(all_lats)),
                    float(np.max(all_lons)), float(np.max(all_lats)))

        fig = create_empty_map(bbox, basemap or 'carto-positron')
        layers_set = set(layers or [])

        labels = None
        anomaly_idx_set = None

        if cluster_data:
            full_labels = np.zeros(len(trips), dtype=int) - 1
            selected_idx_list = cluster_data.get('selected_indices', [])
            sel_labels = cluster_data.get('labels', [])
            for i, si in enumerate(selected_idx_list):
                if si < len(full_labels) and i < len(sel_labels):
                    full_labels[si] = sel_labels[i]
            labels = full_labels

        if anomalies_data:
            anomaly_idx_set = set(a['global_index'] for a in anomalies_data)

        if 'trajectories' in layers_set:
            fig = add_trajectories_to_map(fig, trips, labels, anomaly_idx_set, selected_idx)

        if stays_data and 'stays' in layers_set:
            fig = add_stay_points_to_map(fig, stays_data)

        if hotspot_data and 'hotspots' in layers_set:
            hs_list = hotspot_data.get('hotspots', [])
            use_hm = heatmap_mode and 'heatmap' in (heatmap_mode or [])
            fig = add_hotspots_to_map(fig, hs_list, use_hm)

        if hotspot_data and 'heatmap' in layers_set and not (heatmap_mode and 'heatmap' in heatmap_mode):
            hm = np.array(hotspot_data.get('heatmap', []))
            hb = tuple(hotspot_data.get('bbox', bbox if bbox else (0,0,0,0)))
            if hm.size > 0:
                fig = add_heatmap_to_map(fig, hm, hb)

        if patterns_data and 'od' in layers_set:
            od_list = patterns_data.get('od_pairs', [])
            fig = add_od_flows_to_map(fig, od_list, top_n=top_od or 20)

        if selected_idx and anim_frame is not None:
            target_idx = selected_idx[0] if isinstance(selected_idx, list) and len(selected_idx) > 0 else None
            if target_idx is not None and target_idx < len(trips):
                fig = add_animation_frame(fig, trips[target_idx], int(anim_frame or 0))

        return fig

    @app.callback(
        Output("time-range-slider", "min"),
        Output("time-range-slider", "max"),
        Output("time-range-slider", "value"),
        Output("time-range-slider", "disabled"),
        Output("time-range-label", "children"),
        Input("store-trips", "data"),
        Input("time-range-slider", "value"),
        prevent_initial_call=False,
    )
    def update_time_slider(trips_data, current_val):
        if not trips_data:
            return 0, 100, [0, 100], True, "无数据"

        trips = serializable_to_trips(trips_data)
        if not trips:
            return 0, 100, [0, 100], True, "无数据"

        times = [t.start_time for t in trips] + [t.end_time for t in trips]
        min_t = min(times)
        max_t = max(times)
        min_ts = int(pd.Timestamp(min_t).timestamp())
        max_ts = int(pd.Timestamp(max_t).timestamp())
        span = max(1, max_ts - min_ts)

        v = current_val if (current_val and len(current_val) == 2) else [0, 100]
        start_ts = min_ts + v[0] / 100 * span
        end_ts = min_ts + v[1] / 100 * span
        label = f"{pd.Timestamp(start_ts, unit='s').strftime('%m/%d %H:%M')} ~ {pd.Timestamp(end_ts, unit='s').strftime('%m/%d %H:%M')}"

        return 0, 100, v, False, label

    @app.callback(
        Output("store-animation-frame", "data"),
        Output("animation-interval", "n_intervals"),
        Output("animation-interval", "disabled"),
        Output("btn-animate", "style"),
        Output("dropdown-animation-speed", "style"),
        Input("btn-animate", "n_clicks"),
        Input("animation-interval", "n_intervals"),
        State("store-selected-trip-indices", "data"),
        State("store-trips", "data"),
        State("store-animation-frame", "data"),
        State("dropdown-animation-speed", "value"),
        prevent_initial_call=True,
    )
    def handle_animation(n_clicks, n_intervals, selected_idx, trips_data, current_frame, speed):
        if not selected_idx or not trips_data:
            return no_update, 0, True, {'width': 'auto', 'display': 'none'}, {'display': 'none'}

        trips = serializable_to_trips(trips_data)
        target = selected_idx[0] if isinstance(selected_idx, list) and len(selected_idx) > 0 else None
        if target is None or target >= len(trips):
            return no_update, 0, True, {'width': 'auto'}, {'display': 'none'}

        n_points = len(trips[target].lats)
        trigger = ctx.triggered_id

        show_btn = {'width': 'auto'}
        show_speed = {'fontSize': '11px'}

        if trigger == "btn-animate":
            if current_frame and current_frame > 0:
                return 0, 0, True, show_btn, show_speed
            return 0, 0, False, show_btn, show_speed

        if trigger == "animation-interval" and n_intervals and n_intervals > 0:
            next_frame = int(current_frame or 0) + max(1, (speed or 1) * 2)
            if next_frame >= n_points:
                return 0, 0, True, show_btn, show_speed
            return next_frame, n_intervals, False, show_btn, show_speed

        return no_update, no_update, no_update, show_btn, show_speed

    @app.callback(
        Output("btn-animate", "style", allow_duplicate=True),
        Output("dropdown-animation-speed", "style", allow_duplicate=True),
        Output("store-selected-trip-indices", "data"),
        Input("main-map", "clickData"),
        State("store-trips", "data"),
        State("store-selected-trip-indices", "data"),
        prevent_initial_call=True,
    )
    def handle_map_click(click_data, trips_data, current_sel):
        if not click_data or not trips_data:
            return {'width': 'auto', 'display': 'none'}, {'display': 'none'}, no_update

        trips = serializable_to_trips(trips_data)
        trip_ids = [t.trip_id for t in trips]
        pts = click_data.get('points', [])
        for p in pts:
            cd = p.get('customdata')
            if cd and cd in trip_ids:
                idx = trip_ids.index(cd)
                btn_style = {'width': 'auto'}
                sp_style = {'fontSize': '11px', 'width': '100px'}
                return btn_style, sp_style, [idx]

        return no_update, no_update, no_update


def register_dashboard_callbacks(app):
    @app.callback(
        Output("stat-avg-dist", "children"),
        Output("stat-avg-dur", "children"),
        Output("stat-daily-trips", "children"),
        Output("stat-silhouette", "children"),
        Output("stat-silhouette", "style"),
        Output("graph-hour-dist", "figure"),
        Output("graph-weekday-dist", "figure"),
        Output("graph-dist-dist", "figure"),
        Output("graph-cluster-pie", "figure"),
        Output("graph-mode-dist", "figure"),
        Output("dashboard-hour", "figure"),
        Output("dashboard-distance", "figure"),
        Output("dashboard-duration", "figure"),
        Output("dashboard-cluster-pie", "figure"),
        Input("store-trips", "data"),
        Input("store-cluster-result", "data"),
        Input("store-patterns", "data"),
        prevent_initial_call=False,
    )
    def update_dashboard(trips_data, cluster_data, patterns_data):
        if not trips_data:
            return ["--"] * 4 + [{}] + [go.Figure()] * 9

        trips = serializable_to_trips(trips_data)
        if not trips:
            return ["--"] * 4 + [{}] + [go.Figure()] * 9

        distances = [t.distance_m / 1000.0 for t in trips]
        durations = [t.duration_s / 60.0 for t in trips]
        modes = [t.points_df['mode'].iloc[0] if 'mode' in t.points_df.columns else None for t in trips]

        avg_dist = np.mean(distances) if distances else 0
        avg_dur = np.mean(durations) if durations else 0

        start_dates = set()
        for t in trips:
            d = str(t.start_time)[:10]
            start_dates.add(d)
        n_days = max(1, len(start_dates))
        daily = len(trips) / n_days

        labels = None
        sil_text = "--"
        sil_style = {}
        if cluster_data:
            sil = cluster_data.get('silhouette_score')
            sil_text = f"{sil:.3f}" if sil is not None else "N/A"
            if sil is not None and sil < 0.2:
                sil_style = {'color': '#dc3545', 'fontWeight': '600'}
            else:
                sil_style = {'color': '#28a745', 'fontWeight': '600'}

            full_labels = np.zeros(len(trips), dtype=int) - 1
            sel_idx = cluster_data.get('selected_indices', [])
            sel_labels = cluster_data.get('labels', [])
            for i, si in enumerate(sel_idx):
                if si < len(full_labels) and i < len(sel_labels):
                    full_labels[si] = sel_labels[i]
            labels = full_labels

        hour_dist = {}
        weekday_dist = {}
        if patterns_data:
            raw_hd = patterns_data.get('hour_dist', {})
            hour_dist = {int(k): {int(k2): v2 for k2, v2 in v.items()} for k, v in raw_hd.items()}
            raw_wd = patterns_data.get('weekday_dist', {})
            weekday_dist = {int(k): {int(k2): v2 for k2, v2 in v.items()} for k, v in raw_wd.items()}
        else:
            hd: dict = {0: {}}
            wd: dict = {0: {}}
            for i, t in enumerate(trips):
                cid = int(labels[i]) if (labels is not None and i < len(labels)) else 0
                ts = pd.Timestamp(t.start_time)
                if cid not in hd:
                    hd[cid] = {}
                if cid not in wd:
                    wd[cid] = {}
                hd[cid][ts.hour] = hd[cid].get(ts.hour, 0) + 1
                wd[cid][ts.weekday()] = wd[cid].get(ts.weekday(), 0) + 1
            hour_dist = hd
            weekday_dist = wd

        stat_vals = (
            f"{avg_dist:.2f}", f"{avg_dur:.1f}", f"{daily:.1f}", sil_text, sil_style,
        )

        hour_fig = create_hour_dist_fig(hour_dist)
        weekday_fig = create_weekday_dist_fig(weekday_dist)
        dist_fig = create_distance_dist_fig(distances)
        cluster_pie = create_cluster_pie_fig(labels) if labels is not None else go.Figure()
        mode_fig = create_mode_pie_fig(modes)
        dur_fig = create_duration_dist_fig(durations)

        return stat_vals + (
            hour_fig, weekday_fig, dist_fig, cluster_pie, mode_fig,
            hour_fig, dist_fig, dur_fig, cluster_pie
        )

    @app.callback(
        Output("dropdown-trip-detail", "options"),
        Input("store-trips", "data"),
    )
    def update_trip_detail_options(trips_data):
        if not trips_data:
            return []
        trips = serializable_to_trips(trips_data)
        return [
            {"label": f"{t.trip_id} ({t.user_id})", "value": i}
            for i, t in enumerate(trips)
        ]

    @app.callback(
        Output("trip-detail-info", "children"),
        Input("dropdown-trip-detail", "value"),
        State("store-trips", "data"),
        State("store-cluster-result", "data"),
    )
    def show_trip_detail(idx, trips_data, cluster_data):
        if idx is None or not trips_data:
            return []
        trips = serializable_to_trips(trips_data)
        if idx >= len(trips):
            return []
        t = trips[idx]

        cluster_text = "未聚类"
        if cluster_data:
            sel_idx = cluster_data.get('selected_indices', [])
            sel_labels = cluster_data.get('labels', [])
            if idx in sel_idx:
                pos = sel_idx.index(idx)
                if pos < len(sel_labels):
                    cid = sel_labels[pos]
                    cluster_text = f"簇 {cid}" if cid != -1 else "噪声"

        return [
            html.Div(className="section-card", children=[
                html.Div("🚗 轨迹详情", className="section-title"),
                html.Div(style={'fontSize': '12px', 'lineHeight': '1.8'}, children=[
                    html.Div([html.B("轨迹ID: "), t.trip_id]),
                    html.Div([html.B("用户ID: "), t.user_id]),
                    html.Div([html.B("所属聚类: "), cluster_text]),
                    html.Hr(style={'margin': '8px 0'}),
                    html.Div([html.B("起点: "), f"({t.origin[0]:.5f}, {t.origin[1]:.5f})"]),
                    html.Div([html.B("终点: "), f"({t.destination[0]:.5f}, {t.destination[1]:.5f})"]),
                    html.Div([html.B("距离: "), f"{t.distance_m/1000:.2f} km"]),
                    html.Div([html.B("时长: "), f"{t.duration_s/60:.1f} min"]),
                    html.Div([html.B("平均速度: "), f"{t.avg_speed_kmh:.1f} km/h"]),
                    html.Div([html.B("GPS点数: "), str(len(t.lats))]),
                    html.Hr(style={'margin': '8px 0'}),
                    html.Div([html.B("开始: "), str(t.start_time)]),
                    html.Div([html.B("结束: "), str(t.end_time)]),
                ])
            ])
        ]

    @app.callback(
        Output("download-trip-csv", "data"),
        Input("btn-export-trip-csv", "n_clicks"),
        State("store-trips", "data"),
        State("store-cluster-result", "data"),
        prevent_initial_call=True,
    )
    def export_trip_csv(n_clicks, trips_data, cluster_data):
        if not trips_data:
            raise PreventUpdate()
        trips = serializable_to_trips(trips_data)

        cluster_map = {}
        if cluster_data:
            sel_idx = cluster_data.get('selected_indices', [])
            sel_labels = cluster_data.get('labels', [])
            for i, si in enumerate(sel_idx):
                if i < len(sel_labels):
                    cluster_map[si] = sel_labels[i]

        rows = []
        for i, t in enumerate(trips):
            rows.append({
                'trip_id': t.trip_id,
                'user_id': t.user_id,
                'start_time': str(t.start_time),
                'end_time': str(t.end_time),
                'origin_lat': t.origin[0],
                'origin_lon': t.origin[1],
                'dest_lat': t.destination[0],
                'dest_lon': t.destination[1],
                'distance_km': round(t.distance_m / 1000.0, 3),
                'duration_min': round(t.duration_s / 60.0, 2),
                'avg_speed_kmh': round(t.avg_speed_kmh, 2),
                'point_count': len(t.lats),
                'cluster': cluster_map.get(i, None),
            })
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, "trip_statistics.csv", index=False)

    @app.callback(
        Output("download-stay-csv", "data"),
        Input("btn-export-stay-csv", "n_clicks"),
        State("store-stays", "data"),
        prevent_initial_call=True,
    )
    def export_stay_csv(n_clicks, stays_data):
        if not stays_data:
            raise PreventUpdate()
        stays = serializable_to_stays(stays_data)
        df = stays_to_dataframe(stays)
        return dcc.send_data_frame(df.to_csv, "stay_points.csv", index=False)

    from core.stay_points import stays_to_dataframe

    @app.callback(
        Output("download-cluster-csv", "data"),
        Input("btn-export-cluster-csv", "n_clicks"),
        State("store-cluster-result", "data"),
        prevent_initial_call=True,
    )
    def export_cluster_csv(n_clicks, cluster_data):
        if not cluster_data:
            raise PreventUpdate()
        rows = []
        stats = cluster_data.get('cluster_stats', {})
        for k, v in stats.items():
            rows.append({
                'cluster_id': k,
                'trip_count': v.get('count'),
                'avg_distance_km': round(v.get('avg_distance_km', 0), 3),
                'avg_duration_min': round(v.get('avg_duration_min', 0), 2),
                'trip_indices': str(v.get('trip_indices', [])),
            })
        extra = {
            'algorithm': cluster_data.get('algorithm'),
            'n_clusters': cluster_data.get('n_clusters'),
            'n_noise': cluster_data.get('n_noise'),
            'silhouette_score': cluster_data.get('silhouette_score'),
        }
        rows.append(extra)
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, "cluster_result.csv", index=False)

    @app.callback(
        Output("download-od-csv", "data"),
        Input("btn-export-od-csv", "n_clicks"),
        State("store-patterns", "data"),
        prevent_initial_call=True,
    )
    def export_od_csv(n_clicks, patterns_data):
        if not patterns_data:
            raise PreventUpdate()
        od_list = patterns_data.get('od_pairs', [])
        rows = [{
            'origin_lat': p['origin_centroid'][0],
            'origin_lon': p['origin_centroid'][1],
            'dest_lat': p['dest_centroid'][0],
            'dest_lon': p['dest_centroid'][1],
            'origin_grid': str(p['origin_grid']),
            'dest_grid': str(p['dest_grid']),
            'trip_count': p['count'],
            'avg_distance_km': round(p['avg_distance_km'], 3),
            'avg_duration_min': round(p['avg_duration_min'], 2),
        } for p in od_list]
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, "od_matrix.csv", index=False)

    @app.callback(
        Output("download-hotspot-csv", "data"),
        Input("btn-export-hotspot-csv", "n_clicks"),
        State("store-hotspots", "data"),
        prevent_initial_call=True,
    )
    def export_hotspot_csv(n_clicks, hotspot_data):
        if not hotspot_data:
            raise PreventUpdate()
        hs_list = hotspot_data.get('hotspots', [])
        rows = [{
            'hotspot_id': h['hotspot_id'],
            'center_lat': h['center_lat'],
            'center_lon': h['center_lon'],
            'area_km2': h['area_km2'],
            'point_count': h['point_count'],
            'density_per_km2': h['density_per_km2'],
            'unique_users': h['unique_users'],
            'peak_hour': h['peak_hour'],
        } for h in hs_list]
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, "hotspots.csv", index=False)

    @app.callback(
        Output("export-status", "children"),
        Input("btn-export-hour", "n_clicks"),
        Input("btn-export-weekday", "n_clicks"),
        Input("btn-export-dist", "n_clicks"),
        Input("btn-export-cluster-pie", "n_clicks"),
        State("graph-hour-dist", "figure"),
        State("graph-weekday-dist", "figure"),
        State("graph-dist-dist", "figure"),
        State("graph-cluster-pie", "figure"),
        prevent_initial_call=True,
    )
    def export_charts_png(btn1, btn2, btn3, btn4, f1, f2, f3, f4):
        import os
        from datetime import datetime
        trigger = ctx.triggered_id
        mapping = {
            "btn-export-hour": (f1, "hour_distribution.png"),
            "btn-export-weekday": (f2, "weekday_distribution.png"),
            "btn-export-dist": (f3, "distance_distribution.png"),
            "btn-export-cluster-pie": (f4, "cluster_pie.png"),
        }
        if trigger not in mapping:
            raise PreventUpdate()
        fig_dict, fname = mapping[trigger]
        try:
            fig = go.Figure(fig_dict)
            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            path = os.path.join(export_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{fname}")
            fig.write_image(path, scale=2)
            return html.Div(className="success-box", children=f"✓ 图片已保存至: {path}")
        except Exception as e:
            return html.Div(className="warning-box", children=f"⚠️ 导出失败: {str(e)}. 请在图表上使用Plotly工具栏的相机按钮下载PNG")
