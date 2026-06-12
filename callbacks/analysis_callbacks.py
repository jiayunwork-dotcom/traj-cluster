import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, no_update, callback, dcc, html
from dash.exceptions import PreventUpdate

from core.preprocessing import Trip
from core.stay_points import detect_all_stay_points, stays_to_dataframe
from core.similarity import (
    compute_distance_matrix, compute_avg_point_spacing
)
from core.clustering import (
    apply_clustering, ClusterResult, compute_k_distance, find_eps_knuckle
)
from core.pattern_mining import mine_all_patterns, PatternMiningResult
from core.hotspot_anomaly import detect_hotspots, detect_anomalies, Hotspot, Anomaly
from core.serialization import (
    trips_to_serializable, serializable_to_trips,
    stays_to_serializable, serializable_to_stays
)
from core.visualization import (
    create_empty_map, add_trajectories_to_map, add_stay_points_to_map,
    add_heatmap_to_map, add_od_flows_to_map, add_hotspots_to_map,
    add_animation_frame, create_distance_matrix_heatmap, create_k_distance_plot
)


def register_analysis_callbacks(app):
    @app.callback(
        Output("store-stays", "data"),
        Output("store-sequences", "data"),
        Output("stay-stats", "children"),
        Output("stay-list", "children"),
        Input("btn-detect-stay", "n_clicks"),
        State("store-trips", "data"),
        State("input-stay-dist", "value"),
        State("input-stay-time", "value"),
        prevent_initial_call=True,
    )
    def do_detect_stay(n_clicks, trips_data, dist_m, time_min):
        if not trips_data:
            return no_update, no_update, html.Div("请先导入数据", className="warning-box"), no_update
        trips = serializable_to_trips(trips_data)
        dist_m = dist_m or 200
        time_min = time_min or 20

        stays, sequences = detect_all_stay_points(trips, dist_m, time_min)
        stays_data = stays_to_serializable(stays)

        seq_data = {}
        for uid, seq in sequences.items():
            seq_data[uid] = seq.events

        real_stays = [s for s in stays if not s.is_transit_stop]
        transit_stays = [s for s in stays if s.is_transit_stop]
        total_dur = sum(s.duration_min for s in real_stays)

        stats_children = html.Div(style={'fontSize': '12px'}, children=[
            html.Div(className="stat-grid", children=[
                html.Div(className="stat-item", children=[
                    html.Div(f"{len(real_stays)}", className="stat-value"),
                    html.Div("真实停留点", className="stat-label"),
                ]),
                html.Div(className="stat-item", children=[
                    html.Div(f"{len(transit_stays)}", className="stat-value"),
                    html.Div("微停留(过站)", className="stat-label"),
                ]),
            ]),
            html.Div(f"🕒 总停留时长: {total_dur:.0f} min ({total_dur/60:.1f} h)"),
            html.Div(f"📊 平均停留: {total_dur/max(1,len(real_stays)):.1f} min/次"),
            html.Div(f"👥 有停留用户: {len(sequences)}"),
        ])

        list_children = []
        for i, s in enumerate(real_stays[:30]):
            list_children.append(html.Div(
                className="hotspot-list-item",
                children=[
                    html.Div(f"📍 #{i+1} {s.user_id} · {s.duration_min:.0f}分钟",
                             style={'fontWeight': '600', 'marginBottom': '4px'}),
                    html.Div(f"🕐 {str(s.start_time)[11:16]} ~ {str(s.end_time)[11:16]}",
                             style={'color': '#666'}),
                    html.Div(f"🗺️ ({s.lat:.5f}, {s.lon:.5f}) · {s.point_count}点",
                             style={'color': '#666'}),
                ]
            ))
        if len(real_stays) > 30:
            list_children.append(html.Div(
                style={'textAlign': 'center', 'color': '#999', 'padding': '8px', 'fontSize': '11px'},
                children=f"... 还有 {len(real_stays)-30} 个停留点未显示"
            ))

        return stays_data, seq_data, stats_children, list_children

    @app.callback(
        Output("store-distance-matrix", "data"),
        Output("progress-bar-sim", "style"),
        Output("progress-text-sim", "children"),
        Output("sim-heatmap-container", "children"),
        Output("check-fastdtw", "options"),
        Output("check-fastdtw", "value"),
        Input("btn-compute-sim", "n_clicks"),
        State("store-trips", "data"),
        State("dropdown-sim-method", "value"),
        State("check-fastdtw", "value"),
        State("input-fastdtw-radius", "value"),
        State("input-lcss-epsilon", "value"),
        State("input-lcss-delta", "value"),
        State("input-edr-epsilon", "value"),
        State("dropdown-trip-range", "value"),
        prevent_initial_call=True,
    )
    def do_compute_similarity(n_clicks, trips_data, method, use_fastdtw, radius,
                              lcss_eps, lcss_delta, edr_eps, trip_range):
        if not trips_data:
            return no_update, {'width': '0%'}, "请先导入数据", no_update, no_update, no_update
        all_trips = serializable_to_trips(trips_data)

        if trip_range == "sample50":
            idx = list(range(min(50, len(all_trips))))
        elif trip_range == "sample100":
            rng = np.random.default_rng(42)
            idx = rng.choice(len(all_trips), min(100, len(all_trips)), replace=False).tolist()
        elif trip_range == "sample200":
            rng = np.random.default_rng(42)
            idx = rng.choice(len(all_trips), min(200, len(all_trips)), replace=False).tolist()
        else:
            idx = list(range(len(all_trips)))

        trips = [all_trips[i] for i in idx]

        n = len(trips)
        force_fastdtw = n > 500 and method == 'dtw'

        progress_list = [0]

        def progress_cb(done, total):
            progress_list[0] = done

        try:
            use_fast = (use_fastdtw and "fastdtw" in use_fastdtw) or force_fastdtw
            epsilon_val = None
            if method in ['lcss', 'edr']:
                epsilon_val = lcss_eps if method == 'lcss' else edr_eps
                if epsilon_val is None or epsilon_val <= 0:
                    epsilon_val = compute_avg_point_spacing(trips) * 3.0
            delta_val = lcss_delta if (lcss_delta and lcss_delta > 0) else None

            matrix = compute_distance_matrix(
                trips, method=method,
                use_fastdtw=use_fast,
                fastdtw_radius=radius or 1,
                epsilon_m=epsilon_val,
                delta_min=delta_val,
                progress_callback=progress_cb
            )

            matrix_list = matrix.tolist()
            heatmap_fig = create_distance_matrix_heatmap(matrix)
            heatmap_graph = dcc.Graph(figure=heatmap_fig, style={'height': '300px'}, config={'displayModeBar': False})

            fastdtw_opts = [{"label": " 使用FastDTW近似(加速)", "value": "fastdtw"}]
            forced_value = ["fastdtw"] if force_fastdtw else (use_fastdtw or [])

            msg = f"✓ {n}条轨迹 · {method.upper()} · 完成"
            if force_fastdtw:
                msg += " (已强制启用FastDTW, 轨迹>500)"

            return (
                {'matrix': matrix_list, 'selected_indices': idx},
                {'width': '100%', 'backgroundColor': '#28a745'},
                msg,
                heatmap_graph,
                fastdtw_opts,
                forced_value
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (
                no_update, {'width': '100%', 'backgroundColor': '#dc3545'},
                f"❌ 错误: {str(e)}", no_update, no_update, no_update
            )

    @app.callback(
        Output("k-distance-plot-container", "children"),
        Output("k-distance-plot-container", "style"),
        Input("btn-kdistance", "n_clicks"),
        State("store-distance-matrix", "data"),
        State("input-dbscan-minpts", "value"),
        Input("input-dbscan-eps", "value"),
        prevent_initial_call=True,
    )
    def show_k_distance(n_clicks, dm_data, minpts, input_eps):
        if not dm_data:
            return no_update, {'display': 'none'}
        matrix = np.array(dm_data['matrix'])
        k = (minpts or 5) - 1
        k_dists = compute_k_distance(matrix, k=k)
        eps = input_eps if input_eps else float(find_eps_knuckle(k_dists))
        fig = create_k_distance_plot(k_dists, eps)
        return dcc.Graph(figure=fig, config={'displayModeBar': False}), {'display': 'block'}

    @app.callback(
        Output("store-cluster-result", "data"),
        Output("cluster-result-stats", "children"),
        Output("cluster-legend", "children"),
        Input("btn-do-cluster", "n_clicks"),
        State("store-distance-matrix", "data"),
        State("store-trips", "data"),
        State("dropdown-cluster-method", "value"),
        State("input-dbscan-eps", "value"),
        State("input-dbscan-minpts", "value"),
        State("input-optics-minpts", "value"),
        State("input-spectral-k", "value"),
        prevent_initial_call=True,
    )
    def do_cluster(n_clicks, dm_data, trips_data, method, eps, db_minpts, opt_minpts, spec_k):
        if not dm_data or not trips_data:
            return no_update, html.Div("请先计算距离矩阵", className="warning-box"), no_update

        matrix = np.array(dm_data['matrix'])
        selected_idx = dm_data['selected_indices']
        all_trips = serializable_to_trips(trips_data)
        trips = [all_trips[i] for i in selected_idx]

        try:
            kwargs = {}
            if method == 'dbscan':
                kwargs = {'eps': float(eps) if eps else None, 'min_samples': db_minpts or 5}
            elif method == 'optics':
                kwargs = {'min_samples': opt_minpts or 5}
            elif method == 'spectral':
                kwargs = {'n_clusters': int(spec_k) if spec_k else None}

            result: ClusterResult = apply_clustering(matrix, trips, method, **kwargs)

            result_data = {
                'labels': result.labels.tolist(),
                'cluster_ids': result.cluster_ids,
                'n_clusters': result.n_clusters,
                'n_noise': result.n_noise,
                'silhouette_score': result.silhouette_score,
                'algorithm': result.algorithm,
                'cluster_stats': {str(k): v for k, v in result.cluster_stats.items()},
                'extra_data': result.extra_data,
                'selected_indices': selected_idx,
            }

            sil_score = result.silhouette_score
            sil_text = f"{sil_score:.3f}" if sil_score is not None else "N/A"
            sil_color = '#dc3545' if (sil_score is not None and sil_score < 0.2) else '#28a745'
            sil_warn = html.Div("⚠️ 轮廓系数 < 0.2，聚类质量较差，请调整参数", className="warning-box") if (
                sil_score is not None and sil_score < 0.2) else None

            stats_children = [
                html.Div(className="stat-grid", children=[
                    html.Div(className="stat-item", children=[
                        html.Div(f"{result.n_clusters}", className="stat-value"),
                        html.Div("聚类数", className="stat-label"),
                    ]),
                    html.Div(className="stat-item", children=[
                        html.Div(f"{result.n_noise}", className="stat-value"),
                        html.Div("噪声数", className="stat-label"),
                    ]),
                ]),
                html.Div(style={'marginTop': '8px', 'fontSize': '12px'}, children=[
                    html.Span("📐 轮廓系数: "),
                    html.Span(sil_text, style={'color': sil_color, 'fontWeight': '600'}),
                ]),
                sil_warn if sil_warn else html.Div(),
            ]

            legend_children = []
            for cid in sorted(result.cluster_ids + [-1] if result.n_noise > 0 else result.cluster_ids):
                stats = result.cluster_stats.get(str(cid), {}) if cid != -1 else {}
                count = stats.get('count', result.n_noise) if cid != -1 else result.n_noise
                avg_d = stats.get('avg_distance_km', 0)
                avg_t = stats.get('avg_duration_min', 0)
                name = f"簇 {cid}" if cid != -1 else "噪声"
                color = '#888888' if cid == -1 else [
                    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
                ][cid % 10]

                legend_children.append(html.Div(
                    className="cluster-legend-item",
                    children=[
                        html.Div(className="color-box", style={'backgroundColor': color}),
                        html.Div(style={'flex': 1}, children=[
                            html.Div(f"{name} · {count}条", style={'fontWeight': '600'}),
                            html.Div(f"{avg_d:.2f}km · {avg_t:.1f}min",
                                     style={'color': '#888', 'fontSize': '11px'}) if cid != -1 else html.Div(),
                        ])
                    ]
                ))

            return result_data, stats_children, legend_children

        except Exception as e:
            import traceback
            traceback.print_exc()
            return no_update, html.Div(f"❌ 聚类失败: {str(e)}", className="danger-box"), no_update

    @app.callback(
        Output("store-patterns", "data"),
        Output("commute-list", "children"),
        Output("hotspot-list", "children"),
        Output("anomaly-list", "children"),
        Output("store-hotspots", "data"),
        Output("store-anomalies", "data"),
        Input("btn-mine-pattern", "n_clicks"),
        State("store-trips", "data"),
        State("store-cluster-result", "data"),
        State("store-distance-matrix", "data"),
        State("input-grid-size", "value"),
        State("input-min-support", "value"),
        State("check-anomaly-filter", "value"),
        prevent_initial_call=True,
    )
    def do_mine_patterns(n_clicks, trips_data, cluster_data, dm_data,
                         grid_size, min_sup, anomaly_filter):
        if not trips_data:
            return no_update, [html.Div("请先导入数据", className="info-box")], no_update, no_update, no_update, no_update

        all_trips = serializable_to_trips(trips_data)

        try:
            labels = None
            selected_idx = None
            matrix = None
            anomaly_indices_map = {}

            if cluster_data:
                full_labels = np.zeros(len(all_trips), dtype=int) - 1
                selected_idx = cluster_data.get('selected_indices', [])
                sel_labels = cluster_data.get('labels', [])
                for i, si in enumerate(selected_idx):
                    if si < len(full_labels) and i < len(sel_labels):
                        full_labels[si] = sel_labels[i]
                labels = full_labels

                if dm_data:
                    matrix = np.array(dm_data['matrix'])

            patterns: PatternMiningResult = mine_all_patterns(
                all_trips, labels=labels,
                cell_size_m=float(grid_size) if grid_size else None,
                min_support=int(min_sup) if min_sup else None
            )

            all_lats = np.concatenate([t.lats for t in all_trips]) if all_trips else np.array([])
            all_lons = np.concatenate([t.lons for t in all_trips]) if all_trips else np.array([])
            all_uids = np.concatenate([np.array([t.user_id] * len(t.lats)) for t in all_trips]) if all_trips else np.array([])
            all_ts = np.concatenate([t.timestamps for t in all_trips]) if all_trips else np.array([])

            hotspot_result = detect_hotspots(
                all_lats, all_lons, all_uids, all_ts,
                eps_m=200, min_samples=max(30, len(all_lats) // 200)
            )

            anomalies = []
            if cluster_data:
                from core.hotspot_anomaly import detect_anomalies
                selected_trips = [all_trips[i] for i in (selected_idx or [])]
                class FakeCR:
                    pass
                cr = FakeCR()
                cr.labels = np.array(cluster_data.get('labels', []))
                cr.cluster_ids = cluster_data.get('cluster_ids', [])
                cr.cluster_stats = {int(k): v for k, v in cluster_data.get('cluster_stats', {}).items()}
                anomalies = detect_anomalies(selected_trips, cr, matrix)

            filter_set = set(anomaly_filter or [])

            patterns_data = {
                'od_pairs': [{
                    'origin_grid': list(p.origin_grid),
                    'dest_grid': list(p.dest_grid),
                    'origin_centroid': list(p.origin_centroid),
                    'dest_centroid': list(p.dest_centroid),
                    'count': p.count,
                    'avg_distance_km': p.avg_distance_km,
                    'avg_duration_min': p.avg_duration_min,
                } for p in patterns.od_pairs],
                'commute_patterns': [{
                    'user_id': p.user_id,
                    'home_loc': list(p.home_loc),
                    'work_loc': list(p.work_loc),
                    'morning_od_count': p.morning_od_count,
                    'evening_od_count': p.evening_od_count,
                    'total_workdays': p.total_workdays,
                    'confidence': p.confidence,
                } for p in patterns.commute_patterns],
                'frequent_paths': [{
                    'support': f.support,
                    'support_ratio': f.support_ratio,
                    'centroid_path': f.centroid_path,
                } for f in patterns.frequent_paths],
                'hour_dist': {str(k): {str(k2): v2 for k2, v2 in v.items()}
                              for k, v in patterns.hour_distribution.items()},
                'weekday_dist': {str(k): {str(k2): v2 for k2, v2 in v.items()}
                                 for k, v in patterns.weekday_distribution.items()},
                'duration_dist': {str(k): v.tolist() for k, v in patterns.duration_distribution.items()},
                'cell_size_m': patterns.cell_size_m,
            }

            hotspots_data = [{
                'hotspot_id': h.hotspot_id,
                'center_lat': h.center_lat,
                'center_lon': h.center_lon,
                'area_km2': h.area_km2,
                'point_count': h.point_count,
                'density_per_km2': h.density_per_km2,
                'unique_users': h.unique_users,
                'peak_hour': h.peak_hour,
                'convex_hull': [[p[0], p[1]] for p in h.convex_hull] if h.convex_hull else None,
            } for h in hotspot_result.hotspots]

            heatmap_extra = {
                'heatmap': hotspot_result.heatmap_data.tolist(),
                'bbox': list(hotspot_result.heatmap_bbox),
                'resolution': hotspot_result.heatmap_resolution,
            }

            anomalies_data = [{
                'trip_index': a.trip_index,
                'trip_id': a.trip_id,
                'anomaly_types': a.anomaly_types,
                'details': a.details,
                'global_index': selected_idx[a.trip_index] if (selected_idx and a.trip_index < len(selected_idx)) else a.trip_index,
            } for a in anomalies if set(a.anomaly_types) & filter_set]

            commute_children = []
            if patterns.commute_patterns:
                for i, p in enumerate(patterns.commute_patterns[:15]):
                    commute_children.append(html.Div(className="hotspot-list-item", children=[
                        html.Div(f"👤 {p.user_id}", style={'fontWeight': '600', 'marginBottom': '4px'}),
                        html.Div(f"🏠→🏢 早{p.morning_od_count}次 · 晚{p.evening_od_count}次",
                                 style={'color': '#666', 'fontSize': '11px'}),
                        html.Div(f"📊 置信度: {p.confidence:.0%} · {p.total_workdays}工作日",
                                 style={'color': '#666', 'fontSize': '11px'}),
                    ]))
            else:
                commute_children.append(html.Div("未识别到明确通勤模式", className="info-box"))

            hotspot_children = []
            for i, h in enumerate(hotspot_result.hotspots[:15]):
                hotspot_children.append(html.Div(className="hotspot-list-item", children=[
                    html.Div(f"🔥 #{i+1} · {h.point_count}点 · {h.unique_users}用户",
                             style={'fontWeight': '600', 'marginBottom': '4px'}),
                    html.Div(f"位置: ({h.center_lat:.4f}, {h.center_lon:.4f})",
                             style={'color': '#666', 'fontSize': '11px'}),
                    html.Div(f"面积: {h.area_km2:.3f}km² · 密度: {h.density_per_km2:.0f}/km² · 高峰{h.peak_hour}时",
                             style={'color': '#666', 'fontSize': '11px'}),
                ]))
            if not hotspot_children:
                hotspot_children.append(html.Div("未检测到显著热点区域", className="info-box"))

            anomaly_children = []
            for i, a in enumerate(anomalies_data[:30]):
                tags = []
                for at in a['anomaly_types']:
                    cls = 'tag-detour' if at == 'detour' else (
                        'tag-deviation' if at == 'path_deviation' else 'tag-noise')
                    tag_name = '绕路' if at == 'detour' else (
                        '偏离' if at == 'path_deviation' else '噪声')
                    tags.append(html.Span(tag_name, className=f"anomaly-tag {cls}"))

                anomaly_children.append(html.Div(className="anomaly-item", children=[
                    html.Div(style={'marginBottom': '4px'}, children=tags),
                    html.Div(f"🚗 {a['trip_id']}", style={'fontWeight': '600', 'fontSize': '11px'}),
                    html.Div([
                        html.Div(f"• {v}", style={'fontSize': '10px', 'color': '#c0392b'})
                        for v in a['details'].values()
                    ]),
                ]))
            if not anomaly_children:
                anomaly_children.append(html.Div("✓ 未检出异常轨迹", className="success-box"))

            return (
                patterns_data,
                commute_children,
                hotspot_children,
                anomaly_children,
                {'hotspots': hotspots_data, **heatmap_extra},
                anomalies_data
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return no_update, [html.Div(f"❌ {str(e)}", className="danger-box")], no_update, no_update, no_update, no_update
