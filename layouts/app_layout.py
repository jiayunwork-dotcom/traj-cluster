import dash
from dash import dcc, html


SIDEBAR_TABS = ["tab-data", "tab-stay", "tab-sim", "tab-cluster", "tab-pattern"]
RIGHT_TABS = ["tab-stats", "tab-detail", "tab-export"]


def create_header() -> html.Header:
    return html.Header(
        className="app-header",
        children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center'}, children=[
                html.H1("🚗 城市交通GPS轨迹分析平台"),
                html.Span("Trajectory Clustering & Pattern Mining", className="subtitle")
            ]),
            html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'}, children=[
                html.Span("就绪", id="status-badge",
                          style={'background': '#28a745', 'color': 'white',
                                 'padding': '3px 10px', 'borderRadius': '10px',
                                 'fontSize': '12px', 'fontWeight': '500'}),
                html.Button("⟳ 重置", id="btn-reset", className="btn-secondary",
                            style={'width': 'auto', 'padding': '5px 12px'})
            ])
        ]
    )


def create_data_panel() -> html.Div:
    return html.Div(id="panel-data", style={'display': 'block'}, children=[
        html.Div(className="section-card", children=[
            html.Div("📥 数据导入", className="section-title"),
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    '拖拽CSV文件到此处 或 ',
                    html.A('点击选择文件', style={'color': '#667eea', 'cursor': 'pointer'})
                ]),
                style={
                    'width': '100%', 'height': '80px', 'lineHeight': '80px',
                    'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px',
                    'textAlign': 'center', 'margin': '10px 0', 'fontSize': '13px',
                    'color': '#7f8c8d', 'backgroundColor': '#fafbfc'
                },
                multiple=False
            ),
            html.Div(id="upload-filename", style={'fontSize': '12px', 'color': '#666', 'marginBottom': '10px'}),
            html.Button("🎲 生成示例数据", id="btn-generate-sample", className="btn-secondary",
                        style={'width': '100%', 'marginBottom': '10px'}),
        ]),
        html.Div(className="section-card", children=[
            html.Div("⚙️ 预处理参数", className="section-title"),
            html.Div(className="form-group", children=[
                html.Label("轨迹分段时间阈值(分钟)", className="form-label"),
                dcc.Slider(id="slider-split-minutes", min=10, max=120, value=30, step=5,
                           marks={10: '10', 30: '30', 60: '60', 120: '120'}),
            ]),
            html.Div(className="form-group", children=[
                dcc.Checklist(
                    id="check-smooth",
                    options=[{"label": " 启用轨迹平滑滤波", "value": "smooth"}],
                    value=[],
                    style={'fontSize': '12px', 'marginBottom': '8px'}
                ),
                html.Label("平滑窗口大小", className="form-label"),
                dcc.Slider(id="slider-smooth-window", min=2, max=11, value=3, step=2,
                           marks={2: '2', 5: '5', 8: '8', 11: '11'}),
            ]),
            html.Button("▶ 执行预处理", id="btn-preprocess", className="btn-primary"),
        ]),
        html.Div(className="section-card", id="dataset-stats-card", children=[
            html.Div("📈 数据集统计", className="section-title"),
            html.Div(id="dataset-stats", children=[
                html.Div("请先导入或生成数据", className="info-box")
            ]),
        ]),
        html.Div(className="section-card", id="noise-stats-card", children=[
            html.Div("🧹 噪声过滤", className="section-title"),
            html.Div(id="noise-stats", children=[]),
        ]),
    ])


def create_stay_panel() -> html.Div:
    return html.Div(id="panel-stay", style={'display': 'none'}, children=[
        html.Div(className="section-card", children=[
            html.Div("📍 停留点检测参数", className="section-title"),
            html.Div(className="form-group", children=[
                html.Label("距离阈值 D (米)", className="form-label"),
                dcc.Input(id="input-stay-dist", type="number", value=200, min=50, max=1000, step=10,
                          style={'width': '100%', 'padding': '6px'}),
            ]),
            html.Div(className="form-group", children=[
                html.Label("时间阈值 T (分钟)", className="form-label"),
                dcc.Input(id="input-stay-time", type="number", value=20, min=5, max=120, step=5,
                          style={'width': '100%', 'padding': '6px'}),
            ]),
            html.Button("▶ 检测停留点", id="btn-detect-stay", className="btn-primary"),
        ]),
        html.Div(className="section-card", id="stay-stats-card", children=[
            html.Div("📊 停留点统计", className="section-title"),
            html.Div(id="stay-stats", children=[]),
        ]),
        html.Div(className="section-card", id="stay-list-card", children=[
            html.Div("🏷️ 停留点列表", className="section-title"),
            html.Div(id="stay-list", style={'maxHeight': '300px', 'overflowY': 'auto'}, children=[]),
        ]),
    ])


def create_similarity_panel() -> html.Div:
    return html.Div(id="panel-sim", style={'display': 'none'}, children=[
        html.Div(className="section-card", children=[
            html.Div("📐 相似度度量方法", className="section-title"),
            html.Div(className="form-group", children=[
                html.Label("选择度量方法", className="form-label"),
                dcc.Dropdown(
                    id="dropdown-sim-method",
                    options=[
                        {"label": "DTW (动态时间规整)", "value": "dtw"},
                        {"label": "Frechet距离", "value": "frechet"},
                        {"label": "LCSS (最长公共子序列)", "value": "lcss"},
                        {"label": "EDR (实序列编辑距离)", "value": "edr"},
                    ],
                    value="dtw", clearable=False,
                ),
            ]),
            html.Div(id="dtw-options", className="form-group", children=[
                dcc.Checklist(
                    id="check-fastdtw",
                    options=[{"label": " 使用FastDTW近似(加速)", "value": "fastdtw"}],
                    value=["fastdtw"],
                    style={'fontSize': '12px', 'marginBottom': '8px'}
                ),
                html.Label("FastDTW半径参数", className="form-label"),
                dcc.Input(id="input-fastdtw-radius", type="number", value=1, min=1, max=10, step=1,
                          style={'width': '100%', 'padding': '6px'}),
            ]),
            html.Div(id="lcss-options", className="form-group", style={'display': 'none'}, children=[
                html.Label("空间匹配阈值 ε (米)", className="form-label"),
                dcc.Input(id="input-lcss-epsilon", type="number", value=100, min=10, max=1000, step=10,
                          style={'width': '100%', 'padding': '6px'}),
                html.Label("时间阈值 δ (分钟, 可选0=不限制)", className="form-label"),
                dcc.Input(id="input-lcss-delta", type="number", value=0, min=0, max=120, step=5,
                          style={'width': '100%', 'padding': '6px', 'marginTop': '6px'}),
            ]),
            html.Div(id="edr-options", className="form-group", style={'display': 'none'}, children=[
                html.Label("空间匹配阈值 (米)", className="form-label"),
                dcc.Input(id="input-edr-epsilon", type="number", value=100, min=10, max=1000, step=10,
                          style={'width': '100%', 'padding': '6px'}),
            ]),
            html.Div(className="form-group", children=[
                html.Label("选择分析的轨迹范围", className="form-label"),
                dcc.Dropdown(
                    id="dropdown-trip-range",
                    options=[
                        {"label": "全部轨迹", "value": "all"},
                        {"label": "随机抽样50条", "value": "sample50"},
                        {"label": "随机抽样100条", "value": "sample100"},
                        {"label": "随机抽样200条", "value": "sample200"},
                    ],
                    value="sample100", clearable=False,
                ),
            ]),
            html.Button("▶ 计算距离矩阵", id="btn-compute-sim", className="btn-primary"),
        ]),
        html.Div(className="section-card", children=[
            html.Div("⏳ 计算进度", className="section-title"),
            html.Div(className="progress-bar-container", children=[
                html.Div(id="progress-bar-sim", className="progress-bar", style={'width': '0%'})
            ]),
            html.Div(id="progress-text-sim", className="progress-text", children="等待计算..."),
        ]),
        html.Div(className="section-card", id="heatmap-card", children=[
            html.Div("🗺️ 距离矩阵热力图", className="section-title"),
            html.Div(id="sim-heatmap-container", children=[]),
        ]),
    ])


def create_cluster_panel() -> html.Div:
    return html.Div(id="panel-cluster", style={'display': 'none'}, children=[
        html.Div(className="section-card", children=[
            html.Div("🎯 聚类算法选择", className="section-title"),
            html.Div(className="form-group", children=[
                html.Label("聚类方法", className="form-label"),
                dcc.Dropdown(
                    id="dropdown-cluster-method",
                    options=[
                        {"label": "DBSCAN (基于密度)", "value": "dbscan"},
                        {"label": "OPTICS (DBSCAN改进)", "value": "optics"},
                        {"label": "谱聚类 Spectral", "value": "spectral"},
                    ],
                    value="dbscan", clearable=False,
                ),
            ]),
            html.Div(id="dbscan-options", children=[
                html.Div(className="form-group", children=[
                    html.Label("邻域半径 eps (米)", className="form-label"),
                    dcc.Input(id="input-dbscan-eps", type="number", value=None,
                              placeholder="留空自动推荐",
                              style={'width': '100%', 'padding': '6px'}),
                ]),
                html.Div(className="form-group", children=[
                    html.Label("最小邻域点数 min_samples", className="form-label"),
                    dcc.Input(id="input-dbscan-minpts", type="number", value=5, min=2, max=50, step=1,
                              style={'width': '100%', 'padding': '6px'}),
                ]),
                html.Button("📊 查看k-距离图", id="btn-kdistance", className="btn-secondary",
                            style={'width': '100%', 'marginBottom': '10px'}),
                html.Div(id="k-distance-plot-container", style={'display': 'none'}),
            ]),
            html.Div(id="optics-options", style={'display': 'none'}, children=[
                html.Div(className="form-group", children=[
                    html.Label("最小样本数 min_samples", className="form-label"),
                    dcc.Input(id="input-optics-minpts", type="number", value=5, min=2, max=50, step=1,
                              style={'width': '100%', 'padding': '6px'}),
                ]),
            ]),
            html.Div(id="spectral-options", style={'display': 'none'}, children=[
                html.Div(className="form-group", children=[
                    html.Label("聚类数 K (留空自动推荐)", className="form-label"),
                    dcc.Input(id="input-spectral-k", type="number", value=None,
                              placeholder="留空=自动eigengap推荐",
                              style={'width': '100%', 'padding': '6px'}),
                ]),
            ]),
            html.Button("▶ 执行聚类", id="btn-do-cluster", className="btn-primary"),
        ]),
        html.Div(className="section-card", id="cluster-result-card", children=[
            html.Div("📊 聚类结果", className="section-title"),
            html.Div(id="cluster-result-stats", children=[]),
        ]),
        html.Div(className="section-card", id="cluster-legend-card", children=[
            html.Div("🎨 聚类图例", className="section-title"),
            html.Div(id="cluster-legend", children=[]),
        ]),
    ])


def create_pattern_panel() -> html.Div:
    return html.Div(id="panel-pattern", style={'display': 'none'}, children=[
        html.Div(className="section-card", children=[
            html.Div("🔍 出行模式挖掘", className="section-title"),
            html.Div(className="form-group", children=[
                html.Label("OD网格大小 (米)", className="form-label"),
                dcc.Input(id="input-grid-size", type="number", value=None,
                          placeholder="留空自动推荐",
                          style={'width': '100%', 'padding': '6px'}),
            ]),
            html.Div(className="form-group", children=[
                html.Label("频繁路径最小支持度", className="form-label"),
                dcc.Input(id="input-min-support", type="number", value=None,
                          placeholder="留空自动计算",
                          style={'width': '100%', 'padding': '6px'}),
            ]),
            html.Div(className="form-group", children=[
                html.Label("显示前N个OD对", className="form-label"),
                dcc.Slider(id="slider-top-od", min=5, max=50, value=20, step=5,
                           marks={5: '5', 20: '20', 35: '35', 50: '50'}),
            ]),
            html.Button("▶ 挖掘出行模式", id="btn-mine-pattern", className="btn-primary"),
        ]),
        html.Div(className="section-card", id="commute-card", children=[
            html.Div("🚇 通勤模式识别", className="section-title"),
            html.Div(id="commute-list", children=[]),
        ]),
        html.Div(className="section-card", id="hotspot-card", children=[
            html.Div("🔥 热点区域发现", className="section-title"),
            html.Div(className="form-group", children=[
                dcc.Checklist(
                    id="check-heatmap-mode",
                    options=[{"label": " 热力图模式(替代凸包)", "value": "heatmap"}],
                    value=[],
                    style={'fontSize': '12px', 'marginBottom': '8px'}
                ),
            ]),
            html.Div(id="hotspot-list", children=[]),
        ]),
        html.Div(className="section-card", id="anomaly-card", children=[
            html.Div("⚠️ 异常轨迹检测", className="section-title"),
            html.Div(className="form-group", children=[
                html.Label("异常类型筛选", className="form-label"),
                dcc.Checklist(
                    id="check-anomaly-filter",
                    options=[
                        {"label": " 绕路", "value": "detour"},
                        {"label": " 路径偏离", "value": "path_deviation"},
                        {"label": " DBSCAN噪声", "value": "cluster_noise"},
                    ],
                    value=["detour", "path_deviation", "cluster_noise"],
                    style={'fontSize': '12px'}
                ),
            ]),
            html.Div(id="anomaly-list", style={'maxHeight': '300px', 'overflowY': 'auto'}, children=[]),
        ]),
    ])


def create_left_sidebar() -> html.Aside:
    return html.Aside(
        className="left-sidebar",
        children=[
            dcc.Tabs(
                id="sidebar-tabs",
                value="tab-data",
                className="sidebar-tabs",
                children=[
                    dcc.Tab(label="📊 数据", value="tab-data"),
                    dcc.Tab(label="📍 停留点", value="tab-stay"),
                    dcc.Tab(label="📐 相似度", value="tab-sim"),
                    dcc.Tab(label="🎯 聚类", value="tab-cluster"),
                    dcc.Tab(label="🔍 模式挖掘", value="tab-pattern"),
                ]
            ),
            html.Div(className="sidebar-content", children=[
                create_data_panel(),
                create_stay_panel(),
                create_similarity_panel(),
                create_cluster_panel(),
                create_pattern_panel(),
            ])
        ]
    )


def create_map_toolbar() -> html.Div:
    return html.Div(className="map-toolbar", children=[
        html.Div(style={'display': 'flex', 'gap': '8px', 'alignItems': 'center'}, children=[
            html.Label("底图:", style={'fontSize': '12px', 'fontWeight': '600'}),
            dcc.Dropdown(
                id="dropdown-basemap",
                options=[
                    {"label": "路网图", "value": "open-street-map"},
                    {"label": "卫星图", "value": "satellite-streets"},
                    {"label": "浅色底图", "value": "carto-positron"},
                    {"label": "深色底图", "value": "carto-darkmatter"},
                ],
                value="carto-positron", clearable=False,
                style={'width': '150px', 'fontSize': '12px'}
            ),
        ]),
        html.Div(style={'display': 'flex', 'gap': '8px', 'alignItems': 'center'}, children=[
            html.Label("图层:", style={'fontSize': '12px', 'fontWeight': '600'}),
            dcc.Checklist(
                id="check-layers",
                options=[
                    {"label": " 轨迹", "value": "trajectories"},
                    {"label": " 停留点", "value": "stays"},
                    {"label": " 热力图", "value": "heatmap"},
                    {"label": " OD流向", "value": "od"},
                    {"label": " 热点", "value": "hotspots"},
                    {"label": " 异常", "value": "anomalies"},
                ],
                value=["trajectories", "stays"],
                labelStyle={'marginRight': '8px', 'fontSize': '11px'}
            ),
        ]),
        html.Div(style={'flex': '1'}, children=[
            html.Div(style={'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'justifyContent': 'flex-end'}, children=[
                html.Button("🎬 动画回放", id="btn-animate", className="btn-secondary",
                            style={'width': 'auto', 'display': 'none'}),
                dcc.Dropdown(
                    id="dropdown-animation-speed",
                    options=[
                        {"label": "1x", "value": 1},
                        {"label": "2x", "value": 2},
                        {"label": "5x", "value": 5},
                        {"label": "10x", "value": 10},
                    ],
                    value=2, clearable=False,
                    style={'fontSize': '11px', 'width': '80px', 'display': 'none'},
                ),
            ])
        ]),
    ])


def create_time_slider() -> html.Div:
    return html.Div(id="time-slider-container", style={
        'padding': '8px 16px', 'backgroundColor': 'white',
        'borderTop': '1px solid #e1e8ed', 'flexShrink': 0
    }, children=[
        html.Div(style={'display': 'flex', 'gap': '12px', 'alignItems': 'center'}, children=[
            html.Span("⏱️ 时间范围:", style={'fontSize': '12px', 'fontWeight': '600', 'flexShrink': 0}),
            html.Div(style={'flex': 1}, children=[
                dcc.RangeSlider(
                    id="time-range-slider",
                    min=0, max=100, value=[0, 100],
                    allowCross=False, disabled=True,
                )
            ]),
            html.Div(id="time-range-label",
                     style={'fontSize': '11px', 'color': '#7f8c8d', 'minWidth': '180px', 'textAlign': 'right'})
        ])
    ])


def create_stats_panel() -> html.Div:
    return html.Div(id="rpanel-stats", style={'display': 'block'}, children=[
        html.Div(className="section-card", children=[
            html.Div("📊 核心指标", className="section-title"),
            html.Div(className="stat-grid", children=[
                html.Div(className="stat-item", children=[
                    html.Div("--", id="stat-avg-dist", className="stat-value"),
                    html.Div("平均出行距离(km)", className="stat-label"),
                ]),
                html.Div(className="stat-item", children=[
                    html.Div("--", id="stat-avg-dur", className="stat-value"),
                    html.Div("平均出行时长(min)", className="stat-label"),
                ]),
                html.Div(className="stat-item", children=[
                    html.Div("--", id="stat-daily-trips", className="stat-value"),
                    html.Div("日均出行次数", className="stat-label"),
                ]),
                html.Div(className="stat-item", children=[
                    html.Div("--", id="stat-silhouette", className="stat-value"),
                    html.Div("聚类轮廓系数", className="stat-label"),
                ]),
            ]),
        ]),
        html.Div(className="section-card", children=[
            html.Div("⏰ 出行时段分布", className="section-title"),
            dcc.Graph(id="graph-hour-dist", style={'height': '200px'}, config={'displayModeBar': False}),
        ]),
        html.Div(className="section-card", children=[
            html.Div("📅 星期分布", className="section-title"),
            dcc.Graph(id="graph-weekday-dist", style={'height': '180px'}, config={'displayModeBar': False}),
        ]),
        html.Div(className="section-card", children=[
            html.Div("📏 出行距离分布", className="section-title"),
            dcc.Graph(id="graph-dist-dist", style={'height': '180px'}, config={'displayModeBar': False}),
        ]),
        html.Div(className="section-card", children=[
            html.Div("🎨 各簇占比", className="section-title"),
            dcc.Graph(id="graph-cluster-pie", style={'height': '220px'}, config={'displayModeBar': False}),
        ]),
        html.Div(className="section-card", id="mode-dist-card", children=[
            html.Div("🚕 交通模式占比", className="section-title"),
            dcc.Graph(id="graph-mode-dist", style={'height': '180px'}, config={'displayModeBar': False}),
        ]),
    ])


def create_detail_panel() -> html.Div:
    return html.Div(id="rpanel-detail", style={'display': 'none'}, children=[
        html.Div(className="section-card", children=[
            html.Div("🔍 轨迹详情查询", className="section-title"),
            html.Div(className="form-group", children=[
                html.Label("选择轨迹ID", className="form-label"),
                dcc.Dropdown(id="dropdown-trip-detail", placeholder="选择一条轨迹查看详情"),
            ]),
        ]),
        html.Div(id="trip-detail-info", children=[]),
    ])


def create_export_panel() -> html.Div:
    return html.Div(id="rpanel-export", style={'display': 'none'}, children=[
        html.Div(className="section-card", children=[
            html.Div("💾 图表导出PNG", className="section-title"),
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '8px'}, children=[
                html.Button("📥 导出时段分布图", id="btn-export-hour", className="btn-secondary"),
                html.Button("📥 导出星期分布图", id="btn-export-weekday", className="btn-secondary"),
                html.Button("📥 导出距离分布图", id="btn-export-dist", className="btn-secondary"),
                html.Button("📥 导出簇占比饼图", id="btn-export-cluster-pie", className="btn-secondary"),
            ]),
        ]),
        html.Div(className="section-card", children=[
            html.Div("📑 数据导出CSV", className="section-title"),
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '8px'}, children=[
                html.Button("📥 导出轨迹统计报表", id="btn-export-trip-csv", className="btn-secondary"),
                html.Button("📥 导出停留点数据", id="btn-export-stay-csv", className="btn-secondary"),
                html.Button("📥 导出聚类结果", id="btn-export-cluster-csv", className="btn-secondary"),
                html.Button("📥 导出OD矩阵", id="btn-export-od-csv", className="btn-secondary"),
                html.Button("📥 导出热点区域数据", id="btn-export-hotspot-csv", className="btn-secondary"),
            ]),
        ]),
        html.Div(id="export-status", children=[]),
    ])


def create_right_panel() -> html.Aside:
    return html.Aside(
        className="right-panel",
        children=[
            dcc.Tabs(
                id="right-tabs",
                value="tab-stats",
                className="right-panel-tabs",
                children=[
                    dcc.Tab(label="📊 统计", value="tab-stats"),
                    dcc.Tab(label="📁 轨迹明细", value="tab-detail"),
                    dcc.Tab(label="💾 导出", value="tab-export"),
                ]
            ),
            html.Div(className="right-panel-content", children=[
                create_stats_panel(),
                create_detail_panel(),
                create_export_panel(),
            ])
        ]
    )


def create_bottom_panel() -> html.Footer:
    return html.Footer(
        className="bottom-panel",
        id="bottom-panel",
        style={'display': 'none'},
        children=[
            html.Div(className="bottom-panel-header", children=[
                html.Span("📊 统计 Dashboard", style={'fontWeight': '600', 'fontSize': '14px'}),
                html.Button("▼ 折叠", id="btn-toggle-bottom", className="btn-secondary",
                            style={'width': 'auto', 'padding': '4px 10px'})
            ]),
            html.Div(className="bottom-panel-content", children=[
                html.Div(className="dashboard-card", children=[
                    html.Div("⏰ 出行时段分布", className="dashboard-card-title"),
                    dcc.Graph(id="dashboard-hour", style={'flex': 1},
                              config={'displayModeBar': True,
                                      'toImageButtonOptions': {'format': 'png'}})
                ]),
                html.Div(className="dashboard-card", children=[
                    html.Div("📏 出行距离分布", className="dashboard-card-title"),
                    dcc.Graph(id="dashboard-distance", style={'flex': 1},
                              config={'displayModeBar': True})
                ]),
                html.Div(className="dashboard-card", children=[
                    html.Div("⏱️ 出行时长分布", className="dashboard-card-title"),
                    dcc.Graph(id="dashboard-duration", style={'flex': 1},
                              config={'displayModeBar': True})
                ]),
                html.Div(className="dashboard-card", children=[
                    html.Div("🎨 聚类占比", className="dashboard-card-title"),
                    dcc.Graph(id="dashboard-cluster-pie", style={'flex': 1},
                              config={'displayModeBar': True})
                ]),
            ])
        ]
    )


def create_app_layout() -> html.Div:
    stores = [
        dcc.Store(id="store-trips"),
        dcc.Store(id="store-info"),
        dcc.Store(id="store-noise-stats"),
        dcc.Store(id="store-stays"),
        dcc.Store(id="store-sequences"),
        dcc.Store(id="store-distance-matrix"),
        dcc.Store(id="store-cluster-result"),
        dcc.Store(id="store-patterns"),
        dcc.Store(id="store-hotspots"),
        dcc.Store(id="store-anomalies"),
        dcc.Store(id="store-selected-trip-indices"),
        dcc.Store(id="store-animation-frame", data=0),
        dcc.Store(id="store-current-bounds"),

        dcc.Download(id="download-trip-csv"),
        dcc.Download(id="download-stay-csv"),
        dcc.Download(id="download-cluster-csv"),
        dcc.Download(id="download-od-csv"),
        dcc.Download(id="download-hotspot-csv"),

        dcc.Interval(id="animation-interval", interval=500, disabled=True, n_intervals=0),
    ]

    return html.Div(className="app-container", children=[
        create_header(),
        html.Div(className="main-content", children=[
            create_left_sidebar(),
            html.Div(className="map-container", children=[
                create_map_toolbar(),
                html.Div(className="map-wrapper", children=[
                    dcc.Graph(
                        id="main-map",
                        style={'width': '100%', 'height': '100%'},
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
                            'scrollZoom': True,
                        }
                    )
                ]),
                create_time_slider(),
            ]),
            create_right_panel(),
        ]),
        create_bottom_panel(),
    ] + stores)
