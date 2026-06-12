import base64
import io
import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, no_update, callback, ALL, MATCH
from dash.exceptions import PreventUpdate

from core.preprocessing import import_csv, Trip
from core.sample_generator import generate_sample_data
from core.serialization import trips_to_serializable, serializable_to_trips
from layouts.app_layout import (
    create_data_panel, create_stay_panel, create_similarity_panel,
    create_cluster_panel, create_pattern_panel,
    create_stats_panel, create_detail_panel, create_export_panel
)


_temp_csv = None


def register_data_callbacks(app):
    @app.callback(
        Output("sidebar-content", "children"),
        Input("sidebar-tabs", "value")
    )
    def switch_sidebar_tab(tab_value):
        if tab_value == "tab-data":
            return create_data_panel()
        elif tab_value == "tab-stay":
            return create_stay_panel()
        elif tab_value == "tab-sim":
            return create_similarity_panel()
        elif tab_value == "tab-cluster":
            return create_cluster_panel()
        elif tab_value == "tab-pattern":
            return create_pattern_panel()
        return create_data_panel()

    @app.callback(
        Output("right-panel-content", "children"),
        Input("right-tabs", "value")
    )
    def switch_right_tab(tab_value):
        if tab_value == "tab-stats":
            return create_stats_panel()
        elif tab_value == "tab-detail":
            return create_detail_panel()
        elif tab_value == "tab-export":
            return create_export_panel()
        return create_stats_panel()

    @app.callback(
        Output("upload-filename", "children"),
        Output("store-trips", "data", allow_duplicate=True),
        Output("store-info", "data", allow_duplicate=True),
        Output("store-noise-stats", "data", allow_duplicate=True),
        Output("dataset-stats", "children", allow_duplicate=True),
        Output("noise-stats", "children", allow_duplicate=True),
        Output("status-badge", "children", allow_duplicate=True),
        Output("status-badge", "color", allow_duplicate=True),
        Input("btn-preprocess", "n_clicks"),
        State("upload-data", "contents"),
        State("upload-data", "filename"),
        State("slider-split-minutes", "value"),
        State("check-smooth", "value"),
        State("slider-smooth-window", "value"),
        prevent_initial_call=True,
    )
    def do_preprocess(n_clicks, contents, filename, split_min, check_smooth, smooth_win):
        if n_clicks is None:
            raise PreventUpdate()

        import tempfile
        import os

        global _temp_csv
        if _temp_csv is None and (contents is None or filename is None):
            return "请先导入CSV文件或生成示例数据", no_update, no_update, no_update, no_update, no_update, "无数据", "warning"

        try:
            smooth_flag = "smooth" in (check_smooth or [])
            window = smooth_win if smooth_flag else 3

            if _temp_csv and (contents is None or filename is None):
                filepath = _temp_csv
            else:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                tmp.write(decoded.decode('utf-8-sig' if decoded.startswith(b'\xef\xbb\xbf') else 'utf-8'))
                tmp.close()
                filepath = tmp.name

            trips, info, noise_stats = import_csv(
                filepath, smooth=smooth_flag, smooth_window=window, split_minutes=split_min
            )

            trips_data = trips_to_serializable(trips)
            info_data = {
                'total_users': info.total_users,
                'total_trips': info.total_trips,
                'total_points': info.total_points,
                'time_start': info.time_span_start,
                'time_end': info.time_span_end,
                'bbox': list(info.bbox),
                'has_mode': info.has_mode,
            }

            stats_children = [
                html.Div(className="stat-grid", children=[
                    html.Div(className="stat-item", children=[
                        html.Div(f"{info.total_users}", className="stat-value"),
                        html.Div("总用户数", className="stat-label"),
                    ]),
                    html.Div(className="stat-item", children=[
                        html.Div(f"{info.total_trips}", className="stat-value"),
                        html.Div("总轨迹段", className="stat-label"),
                    ]),
                    html.Div(className="stat-item", children=[
                        html.Div(f"{info.total_points:,}", className="stat-value"),
                        html.Div("总GPS点数", className="stat-label"),
                    ]),
                    html.Div(className="stat-item", children=[
                        html.Div(f"{len(trips)}", className="stat-value"),
                        html.Div("有效Trips", className="stat-label"),
                    ]),
                ]),
                html.Div(style={'fontSize': '11px', 'color': '#666', 'marginTop': '8px'}, children=[
                    html.Div(f"🕒 时间跨度: {str(info.time_span_start)[:10]} ~ {str(info.time_span_end)[:10]}"),
                    html.Div(f"🗺️ 空间范围: [{info.bbox[1]:.3f}, {info.bbox[0]:.3f}] ~ [{info.bbox[3]:.3f}, {info.bbox[2]:.3f}]"),
                ])
            ]

            noise_children = [
                html.Div(style={'fontSize': '12px'}, children=[
                    html.Div(f"🧹 原始点: {noise_stats.get('total_points', 0):,}"),
                    html.Div(f"⚡ 速度异常剔除: {noise_stats.get('removed_high_speed', 0)}"),
                    html.Div(f"🦘 跳跃异常剔除: {noise_stats.get('removed_long_jump', 0)}"),
                    html.Div(f"✅ 保留: {noise_stats.get('kept_points', 0):,}"),
                ])
            ]

            msg = f"已处理 {filename or '示例数据'} · {info.total_trips} trips"
            return msg, trips_data, info_data, noise_stats, stats_children, noise_children, "已加载数据", "success"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 错误: {str(e)}", no_update, no_update, no_update, no_update, no_update, "错误", "danger"

    @app.callback(
        Output("btn-preprocess", "n_clicks"),
        Input("btn-generate-sample", "n_clicks"),
        State("slider-split-minutes", "value"),
        State("check-smooth", "value"),
        State("slider-smooth-window", "value"),
        prevent_initial_call=True,
    )
    def generate_sample(n_clicks, split_min, check_smooth, smooth_win):
        if n_clicks is None:
            raise PreventUpdate()

        import tempfile
        import os
        global _temp_csv

        df = generate_sample_data(n_users=20, n_days=5)
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(tmp.name, index=False)
        tmp.close()
        _temp_csv = tmp.name

        return 1

    from dash import html

    @app.callback(
        Output("dtw-options", "style"),
        Output("lcss-options", "style"),
        Output("edr-options", "style"),
        Input("dropdown-sim-method", "value")
    )
    def toggle_similarity_options(method):
        dtw_style = {'display': 'block'} if method in ['dtw'] else {'display': 'none'}
        lcss_style = {'display': 'block'} if method == 'lcss' else {'display': 'none'}
        edr_style = {'display': 'block'} if method == 'edr' else {'display': 'none'}
        return dtw_style, lcss_style, edr_style

    @app.callback(
        Output("dbscan-options", "style"),
        Output("optics-options", "style"),
        Output("spectral-options", "style"),
        Input("dropdown-cluster-method", "value")
    )
    def toggle_cluster_options(method):
        return (
            {'display': 'block'} if method == 'dbscan' else {'display': 'none'},
            {'display': 'block'} if method == 'optics' else {'display': 'none'},
            {'display': 'block'} if method == 'spectral' else {'display': 'none'},
        )

    @app.callback(
        Output("bottom-panel", "style"),
        Input("btn-toggle-bottom", "n_clicks"),
        State("bottom-panel", "style"),
        prevent_initial_call=True,
    )
    def toggle_bottom_panel(n_clicks, current_style):
        if current_style.get('display') == 'none':
            return {'display': 'flex'}
        return {'display': 'none'}

    @app.callback(
        Output("btn-reset", "n_clicks"),
        Input("btn-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_app(n_clicks):
        global _temp_csv
        _temp_csv = None
        return no_update

    @app.callback(
        Output("store-trips", "data", allow_duplicate=True),
        Output("store-info", "data", allow_duplicate=True),
        Output("store-noise-stats", "data", allow_duplicate=True),
        Output("store-stays", "data", allow_duplicate=True),
        Output("store-distance-matrix", "data", allow_duplicate=True),
        Output("store-cluster-result", "data", allow_duplicate=True),
        Output("store-patterns", "data", allow_duplicate=True),
        Output("store-hotspots", "data", allow_duplicate=True),
        Output("store-anomalies", "data", allow_duplicate=True),
        Input("btn-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_all_stores(n_clicks):
        return [None] * 9


from dash import html
